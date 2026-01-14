import torch
from torch.optim import Optimizer
import math
from typing import Tuple, Callable, Union
from collections import deque

"""
EmoCats v3.7.6 (260109) shadow-system v3.1 -moment v3.1 emoPulse v3.7
EmoLynx v3.6 継承 emoDrive 機構を emoPulse へ統合し簡略化(循環器的機構)
emoPulse 機構により完全自動化を目指す(ユーザーによる emoScope 調整可／改善度反映率)
dNR係数により emoPulse に履歴を混ぜて安定させた(d / N 履歴 による信頼度の維持)
"""

# Helper function
def exists(val):
    return val is not None

class EmoCats(Optimizer):
    # クラス定義＆初期化 ベータ･互換性の追加
    def __init__(self, params: Union[list, torch.nn.Module], 
                 lr=1.0, 
                 eps=1e-8,
                 betas=(0.9, 0.995), 
                 weight_decay=0.01, 
                 use_shadow: bool = False): 
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        # lynxに応じてウェイト減衰のため保存
        self._init_lr = lr
        self.should_stop = False     # 停止フラグの初期化
        self.use_shadow = use_shadow # 🔸shadow 使用フラグを保存
        self.emoScope = lr           # 動的学習率の調和とリズム
        self.noise_est = 1.0         # emoPulse nest 初期化
        self.d_est = 0.02            # emoPulse dest 初期化
        self.dNR_hist = None         # emoPulse hist 初期化

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
        diff_l = (ema['long'] - ema['short']) / scale_base_l
        diff_m = (ema['long'] - ema['short']) / scale_base_m
        # longが十分静かなら、常にlongを優先
        if abs(diff_l) < 0.05:
            return math.tanh(diff_l)
        # longが静かでない時のみ、mediumの静けさを条件付きで採用
        if abs(diff_m) * scale_base_m < abs(diff_l) * scale_base_l:
            return math.tanh(1 * diff_m)
        else:
            return math.tanh(1 * diff_l)

    # アーリーストップ専用(静けさ判定の感情スカラ生成)
    def _early_scalar(self, ema):
        scale_base_l = max(ema['long'], 1e-5)
        diff = (ema['long'] - ema['short']) / scale_base_l
        return math.tanh(1 * diff)

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
    def step(self, closure: Callable | None = None): # クロージャの型ヒントを追加
        loss = None
        if exists(closure): # 一貫性のためにexistsヘルパーを使う
            with torch.enable_grad():
                loss = closure()
        loss_val = loss.item() if loss is not None else 0.0

        # EMA更新・スカラー生成(EMA差分からスカラーを生成しスパイク比率等を決定)
        ema = self._update_ema(self.state, loss_val)
        early_scalar = self._early_scalar(ema)
        scalar = self._compute_scalar(ema)
        ratio = self._decide_ratio(scalar)
        trust = math.copysign((1.0 - abs(scalar)), scalar)

        # --- Start emoPulse (完全自動LR生成) ---
        # emoPulse (loss 時系列から D / Noise を推定し完全自動LRを生成)
        # d / N 履歴 (時間的D推定)  
        self.noise_est = 0.97 * self.noise_est + 0.03 * abs(scalar)
        self.d_est = 0.97 * self.d_est + 0.03 * abs(trust)
        noise = max(self.noise_est, 1e-3)
        d = self.d_est
        # scalar、trust、の差分(瞬間的D推定)と各時間軸の確度推定(疑念と信頼の綱引き)
        Noise_base = abs(scalar - trust) + 0.1
        d_base = abs(noise - d) + 0.1
        # SNRにより異なる時間的確度比率から更新力を導出し２乗で出力最大化
        dNR_now_val = (d_base / Noise_base) ** 2
        # db / Nb dNR(SNR) 履歴化と最大値の成長率の増減
        if self.dNR_hist is None:
            self.dNR_hist = 1.0
        else:
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
            # 共通パラメータ抽出
            _wd_actual, beta1, beta2 = group['weight_decay'], *group['betas']
            # PGチェックにフィルタ
            for p in filter(lambda p: exists(p.grad), group['params']):

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
                # exp_avg初期化
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']

                # Stepweight decay : decoupled_wd 
                p.mul_(1 - emoPulse * _wd_actual)
                beta1, beta2 = group['betas']

                # 勾配ブレンド
                blended_grad = grad.mul(1 - beta1).add_(exp_avg, alpha=beta1)

                # 最終的なパラメータ更新
                p.add_(blended_grad.sign(), alpha = -emoPulse)
                exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)
                # --- End Gradient Update Logic ---

        # ユーザー指定初期LRを実効値(emoPulse)で可視化する(PyTorch標準)
        self._init_lr = emoPulse
        for group in self.param_groups:
            group['lr'] = emoPulse

        # 感情機構の発火が収まり"十分に安定"していることを外部伝達できる(自動停止ロジックではない)
        # Early Stop用 scalar 記録(バッファ共通で管理/最大32件保持/動静評価)
        hist = self.state.setdefault('scalar_hist', deque(maxlen=32))
        hist.append(early_scalar)

        # Early Stop判断(静けさの合図)
        # 32ステップ分のスカラー値の静かな条件を満たした時"フラグ" should_stop = True になるだけ
        if len(hist) >= 32:
            avg_abs = sum(abs(s) for s in hist) / len(hist)
            mean = sum(hist) / len(hist)
            var = sum((s - mean)**2 for s in hist) / len(hist)
            if avg_abs < 0.05 and var < 0.005:
                self.should_stop = True # 💡 外部からこれを見て判断可

        return

"""
 https://github.com/muooon/EmoSens
 Cats was developed with inspiration from Lion, Tiger, and emolynx, 
 which we deeply respect for their lightweight and intelligent design.  
 Cats also integrates EmoNAVI to enhance its capabilities.
"""