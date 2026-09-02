import math
import torch

"""
EmoPulseScheduler v3.9.3+ (260901)
An emotion-driven dynamic scheduler that feels loss and navigates learning rates.

Recommended Learning Rate(rLR) ／ 学習率推奨値 は以下です
TF(Transformer), Unet(SD, SDXL, etc), FFT(Full-Fine-Tuneing)
rLR |Type:LoRA/FFT| TF:1e-5, Unet:1e-4, etc...
基本的に組み合わせる optimizer の期待するLR値を使ってください
Basically, please use the learning rate (LR) expected by the optimizer you are combining it with.

usage ／ 使い方
--lr_scheduler_type=scheduler.emopulse_scheduler.EmoPulse
--lr_scheduler_args "stopcoef=0.07" "histwarmup=50" "boost=10.0"
"""

# ECC - emo closure capture (Loss-Bypass)
if not hasattr(torch.optim.Optimizer, "_manual_loss"):
    torch.optim.Optimizer._manual_loss = 0.0

    # backward-cap (一度だけ実行されるようにする)
    _old_backward = torch.Tensor.backward
    def _new_backward(self, *args, **kwargs):
        if self.ndim == 0:
            try:
                torch.optim.Optimizer._manual_loss = self.item()
            except:
                pass
        return _old_backward(self, *args, **kwargs)
    torch.Tensor.backward = _new_backward
    print("🚩 emo-optim success ecc system ...")

class EmoPulse:

    def __init__(self, optimizer,
                 histwarmup=30,
                 stopcoef=0.04,
                 boost: float = 1.0,
                 notify: bool = True):
        self.optimizer = optimizer
        # オプティマイザの param_groups から初期学習率を自動取得する
        initial_lr = optimizer.param_groups[0].get('lr', 1.0)
        self._init_lr = initial_lr
        self.notify = notify         # 収束・安定の通知切替
        self.should_stop = False     # 停止フラグの初期化
        self.stopcoef = stopcoef     # 収束目標値(ユーザー指定可)
        self.base_scale = initial_lr # optimizer指定LR
        self.histwarmup = histwarmup # loss-ema 正確化のウォームアップ
        self.boost = boost           # 拡張倍率(LoRA等は10.0倍化可)

        # 基準値と最低値
        self.emoScope, self.min_lim = 1.0, 1e-8        
        # 上限値スライド／小さい場合(例：1e-5以下など)でも上限を適切に設定
        lim_boost = boost * 3.0  # 拡張倍率(LoRA等は10.0倍化可)
        self.max_lim = max(min(initial_lr * lim_boost, 3e-3), 3e-8)
        # 履歴初期化
        self.dNR_hist, self.noise_est, self.d_est, self.c_est = 1.0, 1.0, 0.02, 0.0

        # スケジューラ独自の状態保持用辞書
        self.state = {}

    def state_dict(self):
        return {
            'emo_internal': {
                'base_scale': self.base_scale,
                'dNR_hist': self.dNR_hist,
                'noise_est': self.noise_est,
                'd_est': self.d_est,
                'c_est': self.c_est,
                'boost': self.boost,
                'should_stop': self.should_stop,
                'stopcoef': self.stopcoef,
            },
            'scheduler_state': self.state
        }

    def load_state_dict(self, state_dict):
        emo_internal = state_dict.get('emo_internal', None)
        if emo_internal:
            self.base_scale = emo_internal.get('base_scale', self._init_lr)
            self.dNR_hist = emo_internal.get('dNR_hist', 1.0)
            self.noise_est = emo_internal.get('noise_est', 1.0)
            self.d_est = emo_internal.get('d_est', 0.02)
            self.c_est = emo_internal.get('c_est', 0.0)
            self.boost = emo_internal.get('boost', self.boost)
            self.should_stop = emo_internal.get('should_stop', False)
            self.stopcoef = emo_internal.get('stopcoef', self.stopcoef)
        self.state = state_dict.get('scheduler_state', {})

    def get_last_lr(self):
        '''ログ出力用'''
        return [group['lr'] for group in self.optimizer.param_groups]

    def _update_ema(self, loss_val):
        ema = self.state.setdefault('ema', {})
        ema['short'] = 0.3 * loss_val + 0.7 * ema.get('short', loss_val)
        ema['medium'] = 0.05 * loss_val + 0.95 * ema.get('medium', loss_val)
        ema['long'] = 0.01 * loss_val + 0.99 * ema.get('long', loss_val)
        return ema

    def _compute_scalar(self, ema):
        scale_base_l = max(ema['long'], 1e-5)
        scale_base_m = max(ema['medium'], 1e-5)
        diff_base = ema['long'] - ema['short']
        diff_l = diff_base / scale_base_l
        diff_m = diff_base / scale_base_m

        if abs(diff_l) < 0.05:
            res_scalar = math.tanh(diff_l)
        elif abs(diff_m) * scale_base_m < abs(diff_l) * scale_base_l:
            res_scalar = math.tanh(diff_m)
        else:
            res_scalar = math.tanh(diff_l)

        return res_scalar, scale_base_m

    def step(self, loss_val=None):
        '''
        毎ステップ、生のLossの数値を注入して学習率(emoPulse)を自動更新する
        引数なし (None) で呼ばれても ECC で Loss を自動回収する
        '''

        if loss_val is None:
            loss_val = getattr(torch.optim.Optimizer, '_manual_loss', 0.0)

        # カスタムスケジューラ用ウォームアップ(loss-ema 正確化のウォームアップ)
        histstep = self.state.get('histstep', 0)
        warmup_factor = min(histstep / max(self.histwarmup, 1), 1.0)
        self.state['histstep'] = histstep + 1
        self.emoScope = warmup_factor

        # EMA更新・スカラー生成
        ema = self._update_ema(loss_val)
        scalar, scale_base_m = self._compute_scalar(ema)
        trust = math.copysign((1.0 - abs(scalar)), scalar)

        # --- Start emoPulse 機構 ---
        self.noise_est = 0.97 * self.noise_est + 0.03 * abs(scalar)
        self.d_est = 0.97 * self.d_est + 0.03 * abs(trust)
        self.c_est = 0.7 * self.c_est + 0.3 * scalar
        noise = max(self.noise_est, 1e-10)
        d = self.d_est

        Noise_base = abs(scalar - trust) + 0.1
        d_base = abs(noise - d) + 0.1
        dNR_now_val = (d_base / Noise_base) ** 2

        if dNR_now_val >= self.dNR_hist and trust >= 0.5:
            self.dNR_hist = min(dNR_now_val, self.dNR_hist * 1.50)
        elif -0.5 <= trust <= 0.5:
            self.dNR_hist = dNR_now_val * 0.80

        # 基礎倍率 (100.0 ^ c_est)
        cest_boost = self.boost * 10.0 # 拡張倍率(LoRA等は10.0倍化可)
        emoChain = self.base_scale * max((cest_boost ** self.c_est), 1e-3)

        # 最終学習率の実効値決定
        emoPulse = float(max(min(self.dNR_hist * (emoChain * self.emoScope),
                                 self.max_lim), self.min_lim))
        # --- End emoPulse 機構 ---

        # 紐付けられたオプティマイザの学習率をすべて上書き更新
        for group in self.optimizer.param_groups:
            group['lr'] = emoPulse

        # 早期停止 (収束予兆) 判定
        self.stop_base = self.d_est - self.noise_est
        if self.stop_base >= 0.3 and scale_base_m <= self.stopcoef:
            self.should_stop = True
            if self.notify:
                print(f"✨[READY TO STOP]✨")
        else:
            self.should_stop = False

        return emoPulse