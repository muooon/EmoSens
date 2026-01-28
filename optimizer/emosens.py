import torch
from torch.optim import Optimizer
import math

"""
EmoSens v3.7.6 (260109) shadow-system v3.1 -moment v3.1 emoPulse v3.7
EmoNavi v3.6 継承 emoDrive 機構を emoPulse へ統合し簡略化(循環器的機構)
emoPulse 機構により完全自動化を目指す(ユーザーによる emoScope 調整可／改善度反映率)
dNR係数により emoPulse に履歴を混ぜて安定させた(d / N 履歴 による信頼度の維持)
Early scalar、Early Stop、効率化しつつ精度向上させ負荷も軽減する等の改修と微調整
"""

class EmoSens(Optimizer):
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
    # scale_base：Loss値とema値の乖離を修正(分母 ema(long) 「改善率」共通化/loss種に非依存)
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
        loss = closure() if closure is not None else None
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
        noise = max(self.noise_est, 1e-8)
        d = self.d_est
        # scalar、trust、の差分(瞬間的D推定)と各時間軸の確度推定(疑念と信頼の綱引き)
        Noise_base = abs(scalar - trust) + 0.1
        d_base = abs(noise - d) + 0.1
        # SNRにより異なる時間的確度比率から更新力を導出し２乗で出力最大化
        dNR_now_val = (d_base / Noise_base) ** 2
        # db / Nb dNR(SNR) 履歴化と最大値の成長率の増減
        if dNR_now_val >= self.dNR_hist and trust >= 0.5:
            # 加速：どんなに SNR が高くても、1.05倍という｢歩幅｣の成長制限
            self.dNR_hist = min(dNR_now_val, self.dNR_hist * 1.05)
        elif -0.5 <= trust <= 0.5:
            # 減速：怪しい時は即座に比率を下げる(確実に信頼できない場合に下げ圧力を溜める)
            self.dNR_hist = dNR_now_val * 0.98
        # emoPulse 最終決定： emoScorp によるユーザー意思の反映と安全値による制限
        emoPulse = max(min(self.dNR_hist * (self.emoScope * 1e-4), 3e-3), 1e-6)
        # --- End emoPulse (完全自動LR生成) ---

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

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
                # 1次・2次モーメントを使った勾配補正(decoupled weight decay 構造に近い)
                exp_avg = state.setdefault('exp_avg', torch.zeros_like(p))
                exp_avg_sq = state.setdefault('exp_avg_sq', torch.zeros_like(p))
                beta1, beta2 = group['betas']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                if group['weight_decay']:
                    p.add_(p, alpha=-group['weight_decay'] * emoPulse)
                p.addcdiv_(exp_avg, denom, value=-emoPulse)
                # --- End Gradient Update Logic ---

        # ユーザー指定初期LRを実効値(emoPulse)で可視化する(PyTorch標準)
        self._init_lr = emoPulse
        for group in self.param_groups:
            group['lr'] = emoPulse

        # 感情機構の穏やかさ"安定状態"を外部伝達する(自動停止ではない)
        # Early Stop：瞬間値と33step分の履歴の差分で True にするだけ
        # 誤判定防止をしないのは点灯頻度で停止準備(予兆)にするため
        if abs(scalar) <= 5e-6 and abs(Noise_base - d_base) <= 5e-7:
            self.should_stop = True   # 💡 外部からこれを見て判断可
            self.emoScope = 1.0       # ユーザー意思を目的の収束へ整える
        else:
            self.should_stop = False  # 💡 誤判定などの取り消し

        return

"""
 https://github.com/muooon/EmoSens
 An emotion-driven optimizer that feels loss and navigates accordingly.
 Don't think. Feel. Don't stop. Keep running. Believe in what's beyond.
"""
