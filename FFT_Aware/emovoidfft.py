import torch
from torch.optim import Optimizer
import math

"""
EmoVoid v3.8.3+ (260216) Moment-Free Edition FFT適応版(CPU-GPUデータ転送対応)
shadow-system v3.1 -moment v3.1 emoPulse v3.8 FFT-Swap-Aware
これまでの emo系 のすべてを継承し、独自更新式の特徴を受け継ぐ完全オリジナル最適化器
The “geometric relationship” between "W"eight and "G"radient Method
幾何学的最適化アルゴリズム Approx W-Ref Geometry 近似アシスト更新にし負荷低減
完全1次2次モーメント廃止、さまざまなコストを極限まで低減、正確性と軽量性と快適性を向上
### 残念なことに FFT適応 のため中間テンソル(コピー) を生じることで最軽量ではなくなります ###
### これは cuFFT 等で「複素数バッファ」などの厳格なデータ配置の要求により生じる仕様です ###
"""

class EmoVoidFFT(Optimizer):
    # クラス定義＆初期化
    def __init__(self, params,
                 lr=1.0,
                 eps=1e-8,
                 betas=(0.9, 0.995),
                 weight_decay=0.01,
                 use_shadow:bool=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._init_lr = lr
        self.should_stop = False     # 停止フラグの初期化
        self.use_shadow = use_shadow # 🔸shadow 使用フラグを保存
        self.emoScope = lr           # 動的学習率の調和とリズム
        self.dNR_hist = 1.0          # emoPulse hist 初期化
        self.noise_est = 1.0         # emoPulse nest 初期化
        self.d_est = 0.02            # emoPulse dest 初期化

    # 感情EMA更新(緊張と安静)
    def _update_ema(self, state, loss_val):
        ema = state.setdefault('ema', {})
        ema['short'] = 0.3 * loss_val + 0.7 * ema.get('short', loss_val)
        ema['medium'] = 0.05 * loss_val + 0.95 * ema.get('medium', loss_val)
        ema['long'] = 0.01 * loss_val + 0.99 * ema.get('long', loss_val)
        return ema

    # 感情スカラー値生成(EMA差分、滑らかな非線形スカラー、tanh(diff) は ±1.0 で有界性)
    # 係数"1"：ema差分 のスケール調整処理に活用(感度調節係数)／通常は1(タスクに応じ調整可(非推奨))
    # scale_base：Loss値とema値の乖離を修正(分母 ema(long) ｢改善率｣共通化/loss種に非依存)
    # 1e-5(デフォルト)／1e-6(感度向上)／1e-4(安定性向上)：分母を０にせず安定させる
    # トラウマ的反応や慣れによる鈍化で安定性向上(ema-medium 安定と急変を信頼度で感知)
    def _compute_scalar(self, ema):
        scale_base_l = max(ema['long'], 1e-5)
        scale_base_m = max(ema['medium'], 1e-5)
        diff_base = ema['long'] - ema['short']
        diff_l = diff_base / scale_base_l
        diff_m = diff_base / scale_base_m
        # longが十分静かなら、常にlongを優先
        if abs(diff_l) < 0.05:
            return math.tanh(diff_l)
        # longが静かでない時のみ、mediumの静けさを条件付きで採用
        if abs(diff_m) * scale_base_m < abs(diff_l) * scale_base_l:
            return math.tanh(diff_m)
        else:
            return math.tanh(diff_l)

    # (重要)現在は shadow-effect を参考に得た動的フィルタ効果の近似により use_shadow=False です
    # しかし全機能は shadow なしで全て成立します／コード確認時はこの shadow を考慮外として無視してください
    # Shadow混合比 ３段階構成 タスクに応じ調整可、以下を参考に 開始値・範囲量･変化幅を調整
    # return 開始値 + ((scalar) - 閾値) / 範囲量 * 変化幅 も可能(特殊用途向け)
    def _decide_ratio(self, scalar):
        if not self.use_shadow:
            return 0.0  # 🔸use_shadow = False のとき常に比率を 0 にする
        if abs(scalar) > 0.625:
            return 1.0 - abs(scalar)  # 急変｜強抑制｜tanh 0.73(0.27)
        else:
            return 0.0  # return<0 の場合は leap 専用(書き戻しはしないが履歴更新のみ)

    # 損失取得(損失値 loss_val を数値化、感情判定に使用、存在しないパラメータ(更新不要)はスキップ)
    @torch.no_grad()
    def step(self, closure=None):
        loss = torch.enable_grad()(closure)() if closure is not None else None
        loss_val = loss.item() if loss is not None else 0.0

        # EMA更新・スカラー生成(EMA差分からスカラーを生成しスパイク比率等を決定)
        ema = self._update_ema(self.state, loss_val)
        scalar = self._compute_scalar(ema)
        ratio = self._decide_ratio(scalar)
        trust = math.copysign((1.0 - abs(scalar)), scalar)

        # --- Start emoPulse (完全自動LR生成) ---
        # emoPulse (loss 時系列から D / Noise を推定し完全自動LRを生成)
        # d / N 履歴 (時間的D推定)
        self.noise_est = 0.97 * self.noise_est + 0.03 * abs(scalar)
        self.d_est = 0.97 * self.d_est + 0.03 * abs(trust)
        noise = max(self.noise_est, 1e-10) # max:1e-12程度(変更後：要アーリーストップ見直し)
        d = self.d_est
        # scalar、trust、の差分(瞬間的D推定)と各時間軸の確度推定(疑念と信頼の綱引き)
        Noise_base = abs(scalar - trust) + 0.1
        d_base = abs(noise - d) + 0.1
        # SNRにより異なる時間的確度比率から更新力を導出し２乗で出力最大化
        dNR_now_val = (d_base / Noise_base) ** 2
        # db / Nb dNR(SNR) 履歴化と最大値の成長率の増減
        if dNR_now_val >= self.dNR_hist and trust >= 0.5:
            # 加速：どんなに SNR が高くても、1.50倍という｢歩幅｣の成長制限
            self.dNR_hist = min(dNR_now_val, self.dNR_hist * 1.50)
        elif -0.5 <= trust <= 0.5:
            # 減速：怪しい時は即座に比率を下げる(確実に信頼できない場合に下げ圧力を溜める)
            self.dNR_hist = dNR_now_val * 0.80
        # emoPulse 最終決定： emoScorp によるユーザー意思の反映と安全値による制限
        emoPulse = float(max(min(self.dNR_hist * (self.emoScope * 1e-5), 3e-4), 1e-8))
        # --- End emoPulse (完全自動LR生成) ---

        # --- Start Approx W-Ref Geometry 近似アシスト ---
        # Weight Reference Geometry ("W"eight and "G"radient Method)
        # 中間テンソルによるVRAM負荷やcos類似度測定の計算負荷を実質０にする
        with torch.no_grad():
            # 現在の全パラメータのL1ノルムを一括計算(計算負荷: 低)
            # foreach_norm は各層のノルムをリストで返す。sumで1つの数値に集約。
            params = self.param_groups[0]['params']
            point_gl1 = sum(torch._foreach_norm(params, 1))
            prev = getattr(self, "prev_gl1", None)
            curr_step = getattr(self, '_step_count', 0)
            self._step_count = curr_step + 1
            # ウォームアップ期間中のみ、前回のノルムと比較して「一括修正」
            if prev is not None and curr_step < 55:
                # 前回のエネルギーを維持するための比率(スライス的な全層一律係数)
                ratio = (prev / (point_gl1 + 1e-8)).item()
                # 全層の重みを一撃でスケーリング(中間テンソル作成なし、最速)
                torch._foreach_mul_(params, ratio)
                # 修正したので、現在のノルムも再計算(または近似)
                point_gl1 *= ratio
            # 今回のノルムを次回の比較用に保存
            self.prev_gl1 = point_gl1
        # --- End Approx W-Ref Geometry 近似アシスト ---

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in (p for p in group['params'] if p.grad is not None):

                grad = p.grad
                state = self.state[p]

                # 動的学習率補正により shadow 形成を信頼度で調整(trustは正値化(負にならない))
                # shadow：必要時のみ(スパイクp部分に現在値を最大10%追従させる動的履歴更新)
                # 混合比率：スカラーが閾値を超える場合にのみ計算される(信頼できる感情信号かどうかの選別)
                # 急変時は感情機構による shadow 混合で強く抑制する(急制動による安定性の確保)
                # 新 shadow-system は動的学習率と信頼度で協調し選択的スパース性も発揮する
                if self.use_shadow :
                    if 'shadow' not in state: # 🔸shadow = False (デフォルト)
                        state['shadow'] = p.clone()
                    if ratio > 0: # 書き戻しと履歴更新(急変時の強い抑制と弱めの履歴更新)
                        p.mul_(1-ratio).add_(state['shadow'], alpha=abs(trust))
                    else: # 書き戻しせず履歴更新のみ：10%×trust
                        leap_ratio = 0.1 * abs(trust)
                        state['shadow'].lerp_(p, leap_ratio)

                # --- Start Gradient Update Logic ---
                # --- EmoVoid (Approx W-Ref Geometry) ---
                # 更新：emoPulse「時間軸」、W-Ref-Geo「空間軸」でODE近似へ導く
                p.add_(grad.to(p.device).sign_(), alpha=-emoPulse)
                # --- End Gradient Update Logic ---

        # ユーザー指定初期LRを実効値(emoPulse)で可視化する(PyTorch標準)
        for group in self.param_groups:
            group['lr'] = emoPulse

        # 感情機構の穏やかさ"安定状態"を外部伝達する(自動停止ではない)
        # Early Stop：瞬間値と33step分の履歴の差分で True にするだけ
        # 誤判定防止をしないのは点灯頻度で停止準備(予兆)にするため
        if abs(scalar) <= 5e-6 and abs(Noise_base - d_base) <= 5e-7:
            if not self.should_stop:
                self.emoScope = 1.0   # ユーザー意思を目的の収束へ整える
                self._step_count = 0  # 幾何学的再調整の再始動
            self.should_stop = True   # 💡 外部からこれを見て判断可
        else:
            self.should_stop = False  # 💡 誤判定などの取り消し

        return

"""
 https://github.com/muooon/EmoSens
 A new-dimensional geometric optimization algorithm traversing the void.
 Taking decisive steps forward, Weight-Reference Optimizer.
"""
