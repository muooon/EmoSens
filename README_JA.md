## emo series Optimizers  

 Loss-Bypass(ECC) closure-unneeded  
- ###### EmoSens / 2ndGen (v3.9 / Standard-ECC)  
- ###### EmoTion / 3rdGen (v3.9 / Moment-Free-ECC)  
- ###### emo closure capture (ECC-System)  

readme：[English](README.md) | [日本語](README_JA.md)  

<img width="800" height="400" alt="Screenshot 2026-05-12 at 17-10-55 TensorBoard" src="https://github.com/user-attachments/assets/b478967a-ff15-4fb5-83f4-3588332a5784" />  

yellow:Void, purple:Tion, skyblue:Cats, orange:Airy, gray:Sens,  
SDXL:LoRA, Resolution:512, Rank:8, Alpha1, LR:1.0,  

<details>

<summary> EmoSens Full-log </summary>

Cosine スケジューラの LR が｢1.0 × 10^-4｣固定だった場合を基準とする  

LR (学習率) の分析結果  
最大 LR: 7.5309 × 10^-4 (基準の 1e-4 と比較して、約 7.5 倍もの高出力を瞬間的に出している)  
平均 LR: 1.9590 × 10^-4 (全期間を通しても、基準の約 2 倍の強さで学習し続けている)  

面積 (学習総量) の比較基準を｢1.0 × 10^-4 の Cosine｣とした場合、結果は以下の通り  
Pulse の総面積: 0.07836  
Cosine (1e-4) の総面積: 0.02000  
比較比率: 391.8%  

平均 LR が 1.95e-4 ということは、後半に｢2.92e-4｣あたりで停滞 (READY TO STOP 以降) していた時間は、基準の 3 倍近い出力  
400 ステップという短期間で、通常の 1500〜2000 ステップ分くらいの｢熟成｣を終えてしまったことになる  

```prompt


----------

steps:   0%|                                                 | 0/400 [00:00<?, ?it/s]
epoch 1/40
2026-05-12 18:12:29 INFO     epoch is incremented. current_epoch: 0, train_util.py:787
                             epoch: 1
Loss: 0.1308 | Pulse: 1.0000e-04
steps:   0%| | 1/400 [00:19<2:12:21, 19.90s/it, Average key norm=0.0022, Keys Scaled=Loss: 0.1306 | Pulse: 1.0000e-04
steps:   0%| | 2/400 [00:21<1:10:43, 10.66s/it, Average key norm=0.00484, Keys ScaledLoss: 0.1344 | Pulse: 1.0000e-04
steps:   1%| | 3/400 [00:22<50:10,  7.58s/it, Average key norm=0.0078, Keys Scaled=0,Loss: 0.1228 | Pulse: 1.0000e-04
steps:   1%| | 4/400 [00:24<39:55,  6.05s/it, Average key norm=0.0112, Keys Scaled=0,Loss: 0.1237 | Pulse: 1.0000e-04
steps:   1%| | 5/400 [00:25<33:44,  5.12s/it, Average key norm=0.0148, Keys Scaled=0,Loss: 0.1222 | Pulse: 1.0000e-04
steps:   2%| | 6/400 [00:27<29:36,  4.51s/it, Average key norm=0.0184, Keys Scaled=0,Loss: 0.1241 | Pulse: 1.0000e-04
steps:   2%| | 7/400 [00:28<26:39,  4.07s/it, Average key norm=0.0221, Keys Scaled=0,Loss: 0.1132 | Pulse: 1.0000e-04
steps:   2%| | 8/400 [00:29<24:26,  3.74s/it, Average key norm=0.0258, Keys Scaled=0,Loss: 0.1108 | Pulse: 1.0000e-04
steps:   2%| | 9/400 [00:31<22:46,  3.49s/it, Average key norm=0.0295, Keys Scaled=0,Loss: 0.1033 | Pulse: 1.0000e-04
steps:   2%| | 10/400 [00:33<21:27,  3.30s/it, Average key norm=0.0329, Keys Scaled=0
epoch 2/40
2026-05-12 18:12:45 INFO     epoch is incremented. current_epoch: 1, train_util.py:787
                             epoch: 2
Loss: 0.0952 | Pulse: 1.0000e-04
steps:   3%| | 11/400 [00:34<20:19,  3.13s/it, Average key norm=0.0363, Keys Scaled=0Loss: 0.0904 | Pulse: 1.0000e-04
steps:   3%| | 12/400 [00:36<19:24,  3.00s/it, Average key norm=0.0399, Keys Scaled=0Loss: 0.0918 | Pulse: 1.0000e-04
steps:   3%| | 13/400 [00:37<18:37,  2.89s/it, Average key norm=0.0434, Keys Scaled=0Loss: 0.0819 | Pulse: 1.0000e-04
steps:   4%| | 14/400 [00:39<17:58,  2.79s/it, Average key norm=0.0469, Keys Scaled=0Loss: 0.0746 | Pulse: 1.0000e-04
steps:   4%| | 15/400 [00:40<17:21,  2.71s/it, Average key norm=0.0503, Keys Scaled=0Loss: 0.0747 | Pulse: 1.0000e-04
steps:   4%| | 16/400 [00:42<16:52,  2.64s/it, Average key norm=0.0536, Keys Scaled=0Loss: 0.0725 | Pulse: 1.0000e-04
steps:   4%| | 17/400 [00:43<16:24,  2.57s/it, Average key norm=0.0568, Keys Scaled=0Loss: 0.0646 | Pulse: 1.2466e-04
steps:   4%| | 18/400 [00:45<16:00,  2.51s/it, Average key norm=0.0607, Keys Scaled=0Loss: 0.0688 | Pulse: 1.3326e-04
steps:   5%| | 19/400 [00:46<15:39,  2.47s/it, Average key norm=0.0647, Keys Scaled=0Loss: 0.0641 | Pulse: 1.5484e-04
steps:   5%| | 20/400 [00:48<15:20,  2.42s/it, Average key norm=0.0694, Keys Scaled=0
epoch 3/40
2026-05-12 18:13:00 INFO     epoch is incremented. current_epoch: 2, train_util.py:787
                             epoch: 3
Loss: 0.0609 | Pulse: 2.3227e-04
steps:   5%| | 21/400 [00:50<15:03,  2.38s/it, Average key norm=0.0764, Keys Scaled=0Loss: 0.0599 | Pulse: 2.3227e-04
steps:   6%| | 22/400 [00:51<14:45,  2.34s/it, Average key norm=0.0833, Keys Scaled=0Loss: 0.0563 | Pulse: 2.8809e-04
steps:   6%| | 23/400 [00:53<14:30,  2.31s/it, Average key norm=0.0914, Keys Scaled=0Loss: 0.0585 | Pulse: 3.0983e-04
steps:   6%| | 24/400 [00:54<14:16,  2.28s/it, Average key norm=0.1, Keys Scaled=0, aLoss: 0.0525 | Pulse: 4.2403e-04
steps:   6%| | 25/400 [00:56<14:03,  2.25s/it, Average key norm=0.112, Keys Scaled=0,Loss: 0.0548 | Pulse: 4.7203e-04
steps:   6%| | 26/400 [00:57<13:50,  2.22s/it, Average key norm=0.126, Keys Scaled=0,Loss: 0.0525 | Pulse: 5.7090e-04
steps:   7%| | 27/400 [00:59<13:38,  2.19s/it, Average key norm=0.143, Keys Scaled=0,Loss: 0.0525 | Pulse: 6.4374e-04
steps:   7%| | 28/400 [01:00<13:26,  2.17s/it, Average key norm=0.161, Keys Scaled=0,Loss: 0.0511 | Pulse: 7.5309e-04
steps:   7%| | 29/400 [01:02<13:15,  2.14s/it, Average key norm=0.182, Keys Scaled=0,Loss: 0.0504 | Pulse: 5.4298e-04
steps:   8%| | 30/400 [01:03<13:05,  2.12s/it, Average key norm=0.197, Keys Scaled=1,
epoch 4/40
2026-05-12 18:13:15 INFO     epoch is incremented. current_epoch: 3, train_util.py:787
                             epoch: 4
Loss: 0.0484 | Pulse: 4.1740e-04
steps:   8%| | 31/400 [01:05<12:55,  2.10s/it, Average key norm=0.208, Keys Scaled=3,Loss: 0.0473 | Pulse: 3.3406e-04
steps:   8%| | 32/400 [01:06<12:46,  2.08s/it, Average key norm=0.216, Keys Scaled=4,Loss: 0.0446 | Pulse: 2.5402e-04
steps:   8%| | 33/400 [01:08<12:38,  2.07s/it, Average key norm=0.222, Keys Scaled=4,Loss: 0.0451 | Pulse: 2.2042e-04
steps:   8%| | 34/400 [01:09<12:29,  2.05s/it, Average key norm=0.227, Keys Scaled=5,Loss: 0.0442 | Pulse: 1.9342e-04
steps:   9%| | 35/400 [01:11<12:21,  2.03s/it, Average key norm=0.23, Keys Scaled=8, Loss: 0.0442 | Pulse: 1.7845e-04
steps:   9%| | 36/400 [01:12<12:14,  2.02s/it, Average key norm=0.233, Keys Scaled=13Loss: 0.0432 | Pulse: 1.6311e-04
steps:   9%| | 37/400 [01:14<12:08,  2.01s/it, Average key norm=0.236, Keys Scaled=14Loss: 0.0426 | Pulse: 1.5070e-04
steps:  10%| | 38/400 [01:15<12:01,  1.99s/it, Average key norm=0.238, Keys Scaled=15Loss: 0.0433 | Pulse: 1.4781e-04
steps:  10%| | 39/400 [01:17<11:54,  1.98s/it, Average key norm=0.24, Keys Scaled=17,Loss: 0.0429 | Pulse: 1.4404e-04
steps:  10%| | 40/400 [01:18<11:47,  1.97s/it, Average key norm=0.242, Keys Scaled=18
epoch 5/40
2026-05-12 18:13:30 INFO     epoch is incremented. current_epoch: 4, train_util.py:787
                             epoch: 5
Loss: 0.0440 | Pulse: 1.4922e-04
steps:  10%| | 41/400 [01:20<11:41,  1.95s/it, Average key norm=0.243, Keys Scaled=20Loss: 0.0418 | Pulse: 1.4067e-04
steps:  10%| | 42/400 [01:21<11:35,  1.94s/it, Average key norm=0.244, Keys Scaled=22Loss: 0.0420 | Pulse: 1.3704e-04
steps:  11%| | 43/400 [01:23<11:29,  1.93s/it, Average key norm=0.246, Keys Scaled=22Loss: 0.0422 | Pulse: 1.3625e-04
steps:  11%| | 44/400 [01:24<11:24,  1.92s/it, Average key norm=0.247, Keys Scaled=23Loss: 0.0422 | Pulse: 1.3673e-04
steps:  11%| | 45/400 [01:26<11:18,  1.91s/it, Average key norm=0.248, Keys Scaled=24Loss: 0.0396 | Pulse: 1.2406e-04
steps:  12%| | 46/400 [01:27<11:13,  1.90s/it, Average key norm=0.249, Keys Scaled=25Loss: 0.0398 | Pulse: 1.1812e-04
steps:  12%| | 47/400 [01:28<11:07,  1.89s/it, Average key norm=0.249, Keys Scaled=25Loss: 0.0396 | Pulse: 1.1408e-04
steps:  12%| | 48/400 [01:30<11:03,  1.88s/it, Average key norm=0.25, Keys Scaled=26,Loss: 0.0394 | Pulse: 1.1134e-04
steps:  12%| | 49/400 [01:31<10:58,  1.88s/it, Average key norm=0.251, Keys Scaled=26Loss: 0.0382 | Pulse: 1.0532e-04
steps:  12%|▏| 50/400 [01:33<10:54,  1.87s/it, Average key norm=0.251, Keys Scaled=26
epoch 6/40
2026-05-12 18:13:45 INFO     epoch is incremented. current_epoch: 5, train_util.py:787
                             epoch: 6
Loss: 0.0398 | Pulse: 1.0853e-04
steps:  13%|▏| 51/400 [01:34<10:49,  1.86s/it, Average key norm=0.252, Keys Scaled=27Loss: 0.0398 | Pulse: 1.1166e-04
steps:  13%|▏| 52/400 [01:36<10:46,  1.86s/it, Average key norm=0.253, Keys Scaled=27Loss: 0.0390 | Pulse: 1.1074e-04
steps:  13%|▏| 53/400 [01:38<10:41,  1.85s/it, Average key norm=0.253, Keys Scaled=28Loss: 0.0380 | Pulse: 1.0688e-04
steps:  14%|▏| 54/400 [01:39<10:37,  1.84s/it, Average key norm=0.254, Keys Scaled=29Loss: 0.0374 | Pulse: 1.0263e-04
steps:  14%|▏| 55/400 [01:40<10:33,  1.84s/it, Average key norm=0.254, Keys Scaled=29Loss: 0.0375 | Pulse: 1.0099e-04
steps:  14%|▏| 56/400 [01:42<10:30,  1.83s/it, Average key norm=0.255, Keys Scaled=29Loss: 0.0376 | Pulse: 1.0090e-04
steps:  14%|▏| 57/400 [01:44<10:25,  1.82s/it, Average key norm=0.255, Keys Scaled=30Loss: 0.0382 | Pulse: 1.0396e-04
steps:  14%|▏| 58/400 [01:45<10:22,  1.82s/it, Average key norm=0.255, Keys Scaled=30Loss: 0.0382 | Pulse: 1.0709e-04
steps:  15%|▏| 59/400 [01:46<10:18,  1.81s/it, Average key norm=0.256, Keys Scaled=32Loss: 0.0372 | Pulse: 1.0577e-04
steps:  15%|▏| 60/400 [01:48<10:14,  1.81s/it, Average key norm=0.256, Keys Scaled=32
epoch 7/40
2026-05-12 18:14:00 INFO     epoch is incremented. current_epoch: 6, train_util.py:787
                             epoch: 7
Loss: 0.0370 | Pulse: 1.0467e-04
steps:  15%|▏| 61/400 [01:49<10:10,  1.80s/it, Average key norm=0.257, Keys Scaled=32Loss: 0.0369 | Pulse: 1.0402e-04
steps:  16%|▏| 62/400 [01:51<10:07,  1.80s/it, Average key norm=0.257, Keys Scaled=32Loss: 0.0372 | Pulse: 7.3844e-06
steps:  16%|▏| 63/400 [01:52<10:03,  1.79s/it, Average key norm=0.257, Keys Scaled=32Loss: 0.0354 | Pulse: 1.1603e-04
steps:  16%|▏| 64/400 [01:54<10:00,  1.79s/it, Average key norm=0.257, Keys Scaled=32Loss: 0.0357 | Pulse: 1.1349e-04
steps:  16%|▏| 65/400 [01:55<09:56,  1.78s/it, Average key norm=0.258, Keys Scaled=32Loss: 0.0361 | Pulse: 1.1460e-04
steps:  16%|▏| 66/400 [01:57<09:53,  1.78s/it, Average key norm=0.258, Keys Scaled=33Loss: 0.0352 | Pulse: 1.1138e-04
steps:  17%|▏| 67/400 [01:58<09:50,  1.77s/it, Average key norm=0.258, Keys Scaled=33Loss: 0.0362 | Pulse: 1.1530e-04
steps:  17%|▏| 68/400 [02:00<09:47,  1.77s/it, Average key norm=0.259, Keys Scaled=33Loss: 0.0349 | Pulse: 1.1222e-04
steps:  17%|▏| 69/400 [02:01<09:44,  1.76s/it, Average key norm=0.259, Keys Scaled=34Loss: 0.0354 | Pulse: 1.1349e-04
steps:  18%|▏| 70/400 [02:03<09:40,  1.76s/it, Average key norm=0.26, Keys Scaled=33,
epoch 8/40
2026-05-12 18:14:15 INFO     epoch is incremented. current_epoch: 7, train_util.py:787
                             epoch: 8
Loss: 0.0352 | Pulse: 1.1392e-04
steps:  18%|▏| 71/400 [02:04<09:37,  1.76s/it, Average key norm=0.26, Keys Scaled=33,Loss: 0.0345 | Pulse: 1.1149e-04
steps:  18%|▏| 72/400 [02:06<09:34,  1.75s/it, Average key norm=0.26, Keys Scaled=34,Loss: 0.0341 | Pulse: 1.0843e-04
steps:  18%|▏| 73/400 [02:07<09:31,  1.75s/it, Average key norm=0.261, Keys Scaled=34Loss: 0.0331 | Pulse: 1.0244e-04
steps:  18%|▏| 74/400 [02:09<09:28,  1.74s/it, Average key norm=0.261, Keys Scaled=34Loss: 0.0337 | Pulse: 6.9032e-06
steps:  19%|▏| 75/400 [02:10<09:25,  1.74s/it, Average key norm=0.261, Keys Scaled=34Loss: 0.0331 | Pulse: 1.1715e-04
steps:  19%|▏| 76/400 [02:11<09:22,  1.74s/it, Average key norm=0.261, Keys Scaled=34Loss: 0.0328 | Pulse: 1.1420e-04
steps:  19%|▏| 77/400 [02:13<09:19,  1.73s/it, Average key norm=0.262, Keys Scaled=34Loss: 0.0331 | Pulse: 1.1460e-04
steps:  20%|▏| 78/400 [02:14<09:16,  1.73s/it, Average key norm=0.262, Keys Scaled=34Loss: 0.0323 | Pulse: 1.1168e-04
steps:  20%|▏| 79/400 [02:16<09:13,  1.73s/it, Average key norm=0.262, Keys Scaled=35Loss: 0.0313 | Pulse: 1.0523e-04
steps:  20%|▏| 80/400 [02:17<09:11,  1.72s/it, Average key norm=0.263, Keys Scaled=35
epoch 9/40
2026-05-12 18:14:30 INFO     epoch is incremented. current_epoch: 8, train_util.py:787
                             epoch: 9
Loss: 0.0325 | Pulse: 7.2875e-06
steps:  20%|▏| 81/400 [02:19<09:08,  1.72s/it, Average key norm=0.263, Keys Scaled=35Loss: 0.0329 | Pulse: 1.3032e-04
steps:  20%|▏| 82/400 [02:20<09:05,  1.72s/it, Average key norm=0.263, Keys Scaled=35Loss: 0.0318 | Pulse: 8.2303e-06
steps:  21%|▏| 83/400 [02:22<09:02,  1.71s/it, Average key norm=0.263, Keys Scaled=35Loss: 0.0326 | Pulse: 1.5203e-04
steps:  21%|▏| 84/400 [02:23<09:00,  1.71s/it, Average key norm=0.263, Keys Scaled=35Loss: 0.0309 | Pulse: 1.4329e-04
steps:  21%|▏| 85/400 [02:25<08:57,  1.71s/it, Average key norm=0.264, Keys Scaled=35Loss: 0.0319 | Pulse: 1.4548e-04
steps:  22%|▏| 86/400 [02:26<08:54,  1.70s/it, Average key norm=0.264, Keys Scaled=34Loss: 0.0316 | Pulse: 1.4570e-04
steps:  22%|▏| 87/400 [02:27<08:52,  1.70s/it, Average key norm=0.264, Keys Scaled=34Loss: 0.0320 | Pulse: 8.7724e-06
steps:  22%|▏| 88/400 [02:29<08:49,  1.70s/it, Average key norm=0.264, Keys Scaled=34Loss: 0.0324 | Pulse: 1.7942e-04
steps:  22%|▏| 89/400 [02:30<08:47,  1.70s/it, Average key norm=0.265, Keys Scaled=34Loss: 0.0306 | Pulse: 9.6754e-06
steps:  22%|▏| 90/400 [02:32<08:45,  1.69s/it, Average key norm=0.265, Keys Scaled=34
epoch 10/40
2026-05-12 18:14:44 INFO     epoch is incremented. current_epoch: 9, train_util.py:787
                             epoch: 10
Loss: 0.0302 | Pulse: 1.8236e-04
steps:  23%|▏| 91/400 [02:34<08:43,  1.69s/it, Average key norm=0.265, Keys Scaled=34Loss: 0.0305 | Pulse: 1.7821e-04
steps:  23%|▏| 92/400 [02:35<08:41,  1.69s/it, Average key norm=0.266, Keys Scaled=35Loss: 0.0316 | Pulse: 1.0379e-05
steps:  23%|▏| 93/400 [02:37<08:38,  1.69s/it, Average key norm=0.266, Keys Scaled=35Loss: 0.0301 | Pulse: 2.0496e-04
steps:  24%|▏| 94/400 [02:38<08:36,  1.69s/it, Average key norm=0.266, Keys Scaled=35Loss: 0.0303 | Pulse: 2.0143e-04
steps:  24%|▏| 95/400 [02:40<08:34,  1.69s/it, Average key norm=0.267, Keys Scaled=35Loss: 0.0302 | Pulse: 1.9880e-04
steps:  24%|▏| 96/400 [02:41<08:32,  1.69s/it, Average key norm=0.267, Keys Scaled=35Loss: 0.0303 | Pulse: 1.9904e-04
steps:  24%|▏| 97/400 [02:43<08:30,  1.68s/it, Average key norm=0.268, Keys Scaled=36Loss: 0.0306 | Pulse: 2.0284e-04
steps:  24%|▏| 98/400 [02:45<08:28,  1.68s/it, Average key norm=0.268, Keys Scaled=36Loss: 0.0309 | Pulse: 2.1101e-04
steps:  25%|▏| 99/400 [02:46<08:26,  1.68s/it, Average key norm=0.268, Keys Scaled=36Loss: 0.0301 | Pulse: 1.0287e-05
steps:  25%|▎| 100/400 [02:48<08:24,  1.68s/it, Average key norm=0.268, Keys Scaled=3
saving checkpoint: E:\SdxlWebUi\Lora\MiXOMRR-R08D-M000-esx-512px-step00000100.safetensors
steps:  25%|▎| 100/400 [02:49<08:28,  1.69s/it, Average key norm=0.268, Keys Scaled=3
epoch 11/40
2026-05-12 18:15:01 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             10, epoch: 11
Loss: 0.0288 | Pulse: 2.1741e-04
steps:  25%|▎| 101/400 [02:49<08:22,  1.68s/it, Average key norm=0.269, Keys Scaled=3Loss: 0.0285 | Pulse: 2.0149e-04
steps:  26%|▎| 102/400 [02:51<08:20,  1.68s/it, Average key norm=0.269, Keys Scaled=3Loss: 0.0292 | Pulse: 2.0041e-04
steps:  26%|▎| 103/400 [02:52<08:18,  1.68s/it, Average key norm=0.27, Keys Scaled=34Loss: 0.0290 | Pulse: 1.9811e-04
steps:  26%|▎| 104/400 [02:54<08:16,  1.68s/it, Average key norm=0.27, Keys Scaled=35Loss: 0.0289 | Pulse: 1.9573e-04
steps:  26%|▎| 105/400 [02:56<08:14,  1.68s/it, Average key norm=0.271, Keys Scaled=3Loss: 0.0292 | Pulse: 1.9845e-04
steps:  26%|▎| 106/400 [02:57<08:12,  1.68s/it, Average key norm=0.271, Keys Scaled=3Loss: 0.0294 | Pulse: 2.0447e-04
steps:  27%|▎| 107/400 [02:59<08:10,  1.68s/it, Average key norm=0.272, Keys Scaled=3Loss: 0.0283 | Pulse: 1.9577e-04
steps:  27%|▎| 108/400 [03:00<08:08,  1.67s/it, Average key norm=0.272, Keys Scaled=3Loss: 0.0282 | Pulse: 1.9004e-04
steps:  27%|▎| 109/400 [03:02<08:06,  1.67s/it, Average key norm=0.272, Keys Scaled=3Loss: 0.0283 | Pulse: 1.8780e-04
steps:  28%|▎| 110/400 [03:03<08:04,  1.67s/it, Average key norm=0.272, Keys Scaled=3
epoch 12/40
2026-05-12 18:15:17 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             11, epoch: 12
Loss: 0.0281 | Pulse: 1.8502e-04
steps:  28%|▎| 111/400 [03:05<08:02,  1.67s/it, Average key norm=0.273, Keys Scaled=3Loss: 0.0278 | Pulse: 1.7988e-04
steps:  28%|▎| 112/400 [03:06<08:00,  1.67s/it, Average key norm=0.273, Keys Scaled=3Loss: 0.0269 | Pulse: 1.6769e-04
steps:  28%|▎| 113/400 [03:08<07:58,  1.67s/it, Average key norm=0.273, Keys Scaled=3Loss: 0.0266 | Pulse: 1.5764e-04
steps:  28%|▎| 114/400 [03:09<07:55,  1.66s/it, Average key norm=0.274, Keys Scaled=3Loss: 0.0271 | Pulse: 1.5719e-04
steps:  29%|▎| 115/400 [03:11<07:53,  1.66s/it, Average key norm=0.274, Keys Scaled=3Loss: 0.0283 | Pulse: 1.6985e-04
steps:  29%|▎| 116/400 [03:12<07:52,  1.66s/it, Average key norm=0.274, Keys Scaled=3Loss: 0.0270 | Pulse: 8.3102e-06
steps:  29%|▎| 117/400 [03:14<07:50,  1.66s/it, Average key norm=0.274, Keys Scaled=3Loss: 0.0277 | Pulse: 2.0052e-04
steps:  30%|▎| 118/400 [03:15<07:48,  1.66s/it, Average key norm=0.275, Keys Scaled=3Loss: 0.0257 | Pulse: 1.7965e-04
steps:  30%|▎| 119/400 [03:17<07:45,  1.66s/it, Average key norm=0.275, Keys Scaled=3Loss: 0.0264 | Pulse: 8.9779e-06
steps:  30%|▎| 120/400 [03:18<07:43,  1.66s/it, Average key norm=0.275, Keys Scaled=3
epoch 13/40
2026-05-12 18:15:32 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             12, epoch: 13
Loss: 0.0261 | Pulse: 1.9589e-04
steps:  30%|▎| 121/400 [03:20<07:42,  1.66s/it, Average key norm=0.275, Keys Scaled=3Loss: 0.0264 | Pulse: 1.9682e-04
steps:  30%|▎| 122/400 [03:21<07:40,  1.66s/it, Average key norm=0.275, Keys Scaled=3Loss: 0.0267 | Pulse: 2.0223e-04
steps:  31%|▎| 123/400 [03:23<07:38,  1.65s/it, Average key norm=0.276, Keys Scaled=3Loss: 0.0257 | Pulse: 1.9259e-04
steps:  31%|▎| 124/400 [03:24<07:36,  1.65s/it, Average key norm=0.276, Keys Scaled=3Loss: 0.0269 | Pulse: 2.0448e-04
steps:  31%|▎| 125/400 [03:26<07:34,  1.65s/it, Average key norm=0.276, Keys Scaled=3Loss: 0.0265 | Pulse: 2.0817e-04
steps:  32%|▎| 126/400 [03:27<07:31,  1.65s/it, Average key norm=0.277, Keys Scaled=3Loss: 0.0264 | Pulse: 2.1103e-04
steps:  32%|▎| 127/400 [03:29<07:30,  1.65s/it, Average key norm=0.277, Keys Scaled=3Loss: 0.0263 | Pulse: 2.1292e-04
steps:  32%|▎| 128/400 [03:30<07:28,  1.65s/it, Average key norm=0.277, Keys Scaled=3Loss: 0.0258 | Pulse: 8.8880e-06
steps:  32%|▎| 129/400 [03:32<07:26,  1.65s/it, Average key norm=0.277, Keys Scaled=3Loss: 0.0256 | Pulse: 2.3207e-04
steps:  32%|▎| 130/400 [03:33<07:24,  1.65s/it, Average key norm=0.278, Keys Scaled=3
epoch 14/40
2026-05-12 18:15:47 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             13, epoch: 14
Loss: 0.0255 | Pulse: 2.2571e-04
steps:  33%|▎| 131/400 [03:35<07:22,  1.64s/it, Average key norm=0.278, Keys Scaled=3Loss: 0.0252 | Pulse: 2.1761e-04
steps:  33%|▎| 132/400 [03:36<07:20,  1.64s/it, Average key norm=0.278, Keys Scaled=3Loss: 0.0257 | Pulse: 2.2192e-04
steps:  33%|▎| 133/400 [03:38<07:18,  1.64s/it, Average key norm=0.278, Keys Scaled=3Loss: 0.0254 | Pulse: 2.2131e-04
steps:  34%|▎| 134/400 [03:39<07:16,  1.64s/it, Average key norm=0.279, Keys Scaled=3Loss: 0.0265 | Pulse: 2.4510e-04
steps:  34%|▎| 135/400 [03:41<07:14,  1.64s/it, Average key norm=0.279, Keys Scaled=3Loss: 0.0253 | Pulse: 2.3626e-04
steps:  34%|▎| 136/400 [03:42<07:12,  1.64s/it, Average key norm=0.279, Keys Scaled=2Loss: 0.0246 | Pulse: 2.1777e-04
steps:  34%|▎| 137/400 [03:44<07:11,  1.64s/it, Average key norm=0.279, Keys Scaled=3Loss: 0.0259 | Pulse: 2.3436e-04
steps:  34%|▎| 138/400 [03:46<07:09,  1.64s/it, Average key norm=0.28, Keys Scaled=34Loss: 0.0244 | Pulse: 2.1617e-04
steps:  35%|▎| 139/400 [03:47<07:07,  1.64s/it, Average key norm=0.28, Keys Scaled=34Loss: 0.0256 | Pulse: 2.2947e-04
steps:  35%|▎| 140/400 [03:49<07:05,  1.64s/it, Average key norm=0.28, Keys Scaled=37
epoch 15/40
2026-05-12 18:16:02 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             14, epoch: 15
Loss: 0.0257 | Pulse: 2.4538e-04
steps:  35%|▎| 141/400 [03:50<07:03,  1.64s/it, Average key norm=0.281, Keys Scaled=3Loss: 0.0252 | Pulse: 2.4652e-04
steps:  36%|▎| 142/400 [03:52<07:01,  1.63s/it, Average key norm=0.281, Keys Scaled=3Loss: 0.0242 | Pulse: 2.2303e-04
steps:  36%|▎| 143/400 [03:53<06:59,  1.63s/it, Average key norm=0.281, Keys Scaled=3Loss: 0.0250 | Pulse: 2.2845e-04
steps:  36%|▎| 144/400 [03:55<06:57,  1.63s/it, Average key norm=0.281, Keys Scaled=3Loss: 0.0252 | Pulse: 2.4006e-04
steps:  36%|▎| 145/400 [03:56<06:56,  1.63s/it, Average key norm=0.282, Keys Scaled=3Loss: 0.0248 | Pulse: 2.3979e-04
steps:  36%|▎| 146/400 [03:58<06:54,  1.63s/it, Average key norm=0.282, Keys Scaled=3Loss: 0.0248 | Pulse: 2.4194e-04
steps:  37%|▎| 147/400 [03:59<06:52,  1.63s/it, Average key norm=0.282, Keys Scaled=3Loss: 0.0255 | Pulse: 2.6838e-04
steps:  37%|▎| 148/400 [04:00<06:50,  1.63s/it, Average key norm=0.283, Keys Scaled=3Loss: 0.0242 | Pulse: 2.4817e-04
steps:  37%|▎| 149/400 [04:02<06:48,  1.63s/it, Average key norm=0.283, Keys Scaled=3Loss: 0.0246 | Pulse: 2.4791e-04
steps:  38%|▍| 150/400 [04:03<06:46,  1.63s/it, Average key norm=0.283, Keys Scaled=3
epoch 16/40
2026-05-12 18:16:17 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             15, epoch: 16
Loss: 0.0242 | Pulse: 2.3854e-04
steps:  38%|▍| 151/400 [04:05<06:44,  1.62s/it, Average key norm=0.284, Keys Scaled=3Loss: 0.0236 | Pulse: 2.1744e-04
steps:  38%|▍| 152/400 [04:06<06:42,  1.62s/it, Average key norm=0.284, Keys Scaled=3Loss: 0.0234 | Pulse: 2.0242e-04
steps:  38%|▍| 153/400 [04:08<06:40,  1.62s/it, Average key norm=0.284, Keys Scaled=3Loss: 0.0232 | Pulse: 1.8985e-04
steps:  38%|▍| 154/400 [04:09<06:38,  1.62s/it, Average key norm=0.284, Keys Scaled=3Loss: 0.0233 | Pulse: 1.8665e-04
steps:  39%|▍| 155/400 [04:11<06:36,  1.62s/it, Average key norm=0.284, Keys Scaled=3Loss: 0.0235 | Pulse: 6.2770e-06
steps:  39%|▍| 156/400 [04:12<06:34,  1.62s/it, Average key norm=0.284, Keys Scaled=3Loss: 0.0240 | Pulse: 2.4857e-04
steps:  39%|▍| 157/400 [04:13<06:33,  1.62s/it, Average key norm=0.285, Keys Scaled=2Loss: 0.0220 | Pulse: 2.0782e-04
steps:  40%|▍| 158/400 [04:15<06:31,  1.62s/it, Average key norm=0.285, Keys Scaled=3Loss: 0.0221 | Pulse: 1.8830e-04
steps:  40%|▍| 159/400 [04:16<06:29,  1.62s/it, Average key norm=0.285, Keys Scaled=3Loss: 0.0220 | Pulse: 1.7584e-04
steps:  40%|▍| 160/400 [04:18<06:27,  1.61s/it, Average key norm=0.285, Keys Scaled=3
epoch 17/40
2026-05-12 18:16:31 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             16, epoch: 17
Loss: 0.0226 | Pulse: 1.8139e-04
steps:  40%|▍| 161/400 [04:19<06:25,  1.61s/it, Average key norm=0.286, Keys Scaled=3Loss: 0.0223 | Pulse: 1.7922e-04
steps:  40%|▍| 162/400 [04:21<06:23,  1.61s/it, Average key norm=0.286, Keys Scaled=3Loss: 0.0227 | Pulse: 1.8971e-04
steps:  41%|▍| 163/400 [04:22<06:21,  1.61s/it, Average key norm=0.286, Keys Scaled=3Loss: 0.0221 | Pulse: 6.3559e-06
steps:  41%|▍| 164/400 [04:24<06:20,  1.61s/it, Average key norm=0.286, Keys Scaled=3Loss: 0.0231 | Pulse: 2.5067e-04
steps:  41%|▍| 165/400 [04:25<06:18,  1.61s/it, Average key norm=0.286, Keys Scaled=3Loss: 0.0216 | Pulse: 2.2528e-04
steps:  42%|▍| 166/400 [04:26<06:16,  1.61s/it, Average key norm=0.287, Keys Scaled=3Loss: 0.0221 | Pulse: 2.2431e-04
steps:  42%|▍| 167/400 [04:28<06:14,  1.61s/it, Average key norm=0.287, Keys Scaled=3Loss: 0.0224 | Pulse: 2.3731e-04
steps:  42%|▍| 168/400 [04:29<06:12,  1.61s/it, Average key norm=0.287, Keys Scaled=3Loss: 0.0227 | Pulse: 2.6001e-04
steps:  42%|▍| 169/400 [04:31<06:11,  1.61s/it, Average key norm=0.288, Keys Scaled=3Loss: 0.0234 | Pulse: 3.1669e-04
steps:  42%|▍| 170/400 [04:33<06:09,  1.61s/it, Average key norm=0.288, Keys Scaled=3
epoch 18/40
2026-05-12 18:16:46 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             17, epoch: 18
Loss: 0.0228 | Pulse: 3.7212e-04
steps:  43%|▍| 171/400 [04:34<06:07,  1.61s/it, Average key norm=0.288, Keys Scaled=3Loss: 0.0230 | Pulse: 3.7212e-04
steps:  43%|▍| 172/400 [04:36<06:05,  1.61s/it, Average key norm=0.289, Keys Scaled=3Loss: 0.0235 | Pulse: 7.0346e-06
steps:  43%|▍| 173/400 [04:37<06:04,  1.60s/it, Average key norm=0.289, Keys Scaled=3Loss: 0.0234 | Pulse: 1.0552e-05
steps:  44%|▍| 174/400 [04:39<06:02,  1.60s/it, Average key norm=0.289, Keys Scaled=3Loss: 0.0225 | Pulse: 1.5828e-05
steps:  44%|▍| 175/400 [04:40<06:00,  1.60s/it, Average key norm=0.289, Keys Scaled=3Loss: 0.0222 | Pulse: 7.6929e-06
steps:  44%|▍| 176/400 [04:42<05:58,  1.60s/it, Average key norm=0.289, Keys Scaled=3Loss: 0.0237 | Pulse: 1.1539e-05
steps:  44%|▍| 177/400 [04:43<05:57,  1.60s/it, Average key norm=0.289, Keys Scaled=3Loss: 0.0225 | Pulse: 1.7309e-05
steps:  44%|▍| 178/400 [04:45<05:55,  1.60s/it, Average key norm=0.289, Keys Scaled=3Loss: 0.0222 | Pulse: 2.5963e-05
steps:  45%|▍| 179/400 [04:46<05:53,  1.60s/it, Average key norm=0.289, Keys Scaled=2Loss: 0.0217 | Pulse: 3.8945e-05
steps:  45%|▍| 180/400 [04:48<05:52,  1.60s/it, Average key norm=0.289, Keys Scaled=2
epoch 19/40
2026-05-12 18:17:01 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             18, epoch: 19
Loss: 0.0228 | Pulse: 8.0577e-06
steps:  45%|▍| 181/400 [04:49<05:50,  1.60s/it, Average key norm=0.289, Keys Scaled=2Loss: 0.0215 | Pulse: 1.2087e-05
steps:  46%|▍| 182/400 [04:51<05:48,  1.60s/it, Average key norm=0.289, Keys Scaled=2Loss: 0.0204 | Pulse: 1.8130e-05
steps:  46%|▍| 183/400 [04:52<05:46,  1.60s/it, Average key norm=0.289, Keys Scaled=2Loss: 0.0207 | Pulse: 2.7195e-05
steps:  46%|▍| 184/400 [04:54<05:45,  1.60s/it, Average key norm=0.289, Keys Scaled=2Loss: 0.0223 | Pulse: 4.0792e-05
steps:  46%|▍| 185/400 [04:55<05:43,  1.60s/it, Average key norm=0.289, Keys Scaled=2Loss: 0.0207 | Pulse: 6.1188e-05
steps:  46%|▍| 186/400 [04:57<05:41,  1.60s/it, Average key norm=0.289, Keys Scaled=3Loss: 0.0204 | Pulse: 9.1782e-05
steps:  47%|▍| 187/400 [04:58<05:40,  1.60s/it, Average key norm=0.289, Keys Scaled=3Loss: 0.0214 | Pulse: 1.3767e-04
steps:  47%|▍| 188/400 [05:00<05:38,  1.60s/it, Average key norm=0.289, Keys Scaled=3Loss: 0.0218 | Pulse: 2.0651e-04
steps:  47%|▍| 189/400 [05:01<05:36,  1.60s/it, Average key norm=0.289, Keys Scaled=3Loss: 0.0223 | Pulse: 2.0651e-04
steps:  48%|▍| 190/400 [05:03<05:35,  1.60s/it, Average key norm=0.289, Keys Scaled=3
epoch 20/40
2026-05-12 18:17:16 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             19, epoch: 20
Loss: 0.0220 | Pulse: 2.0651e-04
steps:  48%|▍| 191/400 [05:04<05:33,  1.60s/it, Average key norm=0.29, Keys Scaled=38Loss: 0.0225 | Pulse: 2.0651e-04
steps:  48%|▍| 192/400 [05:06<05:31,  1.59s/it, Average key norm=0.29, Keys Scaled=37Loss: 0.0237 | Pulse: 2.0651e-04
steps:  48%|▍| 193/400 [05:07<05:29,  1.59s/it, Average key norm=0.29, Keys Scaled=38Loss: 0.0221 | Pulse: 2.0651e-04
steps:  48%|▍| 194/400 [05:09<05:28,  1.59s/it, Average key norm=0.29, Keys Scaled=39Loss: 0.0233 | Pulse: 2.0651e-04
steps:  49%|▍| 195/400 [05:10<05:26,  1.59s/it, Average key norm=0.29, Keys Scaled=39Loss: 0.0249 | Pulse: 2.0651e-04
steps:  49%|▍| 196/400 [05:12<05:24,  1.59s/it, Average key norm=0.29, Keys Scaled=36Loss: 0.0223 | Pulse: 2.0651e-04
steps:  49%|▍| 197/400 [05:13<05:23,  1.59s/it, Average key norm=0.291, Keys Scaled=3Loss: 0.0217 | Pulse: 2.0651e-04
steps:  50%|▍| 198/400 [05:15<05:21,  1.59s/it, Average key norm=0.291, Keys Scaled=3Loss: 0.0236 | Pulse: 2.0651e-04
steps:  50%|▍| 199/400 [05:16<05:19,  1.59s/it, Average key norm=0.291, Keys Scaled=3Loss: 0.0234 | Pulse: 2.0651e-04
steps:  50%|▌| 200/400 [05:18<05:18,  1.59s/it, Average key norm=0.291, Keys Scaled=32026-05-12 18:17:31 INFO                                            train_util.py:6468

saving checkpoint: E:\SdxlWebUi\Lora\MiXOMRR-R08D-M000-esx-512px-step00000200.safetensors
steps:  50%|▌| 200/400 [05:18<05:18,  1.59s/it, Average key norm=0.291, Keys Scaled=3
epoch 21/40
2026-05-12 18:17:44 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             20, epoch: 21
Loss: 0.0225 | Pulse: 2.0651e-04
steps:  50%|▌| 201/400 [05:19<05:16,  1.59s/it, Average key norm=0.291, Keys Scaled=3Loss: 0.0214 | Pulse: 2.0651e-04
steps:  50%|▌| 202/400 [05:21<05:14,  1.59s/it, Average key norm=0.291, Keys Scaled=3Loss: 0.0212 | Pulse: 2.0651e-04
steps:  51%|▌| 203/400 [05:22<05:13,  1.59s/it, Average key norm=0.292, Keys Scaled=3Loss: 0.0211 | Pulse: 2.0651e-04
steps:  51%|▌| 204/400 [05:24<05:11,  1.59s/it, Average key norm=0.292, Keys Scaled=3Loss: 0.0216 | Pulse: 2.0651e-04
steps:  51%|▌| 205/400 [05:25<05:09,  1.59s/it, Average key norm=0.292, Keys Scaled=3Loss: 0.0207 | Pulse: 2.0651e-04
steps:  52%|▌| 206/400 [05:27<05:08,  1.59s/it, Average key norm=0.292, Keys Scaled=3Loss: 0.0211 | Pulse: 2.0651e-04
steps:  52%|▌| 207/400 [05:28<05:06,  1.59s/it, Average key norm=0.292, Keys Scaled=3Loss: 0.0210 | Pulse: 2.0651e-04
steps:  52%|▌| 208/400 [05:30<05:04,  1.59s/it, Average key norm=0.293, Keys Scaled=3Loss: 0.0217 | Pulse: 2.0651e-04
steps:  52%|▌| 209/400 [05:31<05:03,  1.59s/it, Average key norm=0.293, Keys Scaled=3Loss: 0.0211 | Pulse: 2.0651e-04
steps:  52%|▌| 210/400 [05:33<05:01,  1.59s/it, Average key norm=0.293, Keys Scaled=3
epoch 22/40
2026-05-12 18:17:59 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             21, epoch: 22
Loss: 0.0220 | Pulse: 2.0651e-04
steps:  53%|▌| 211/400 [05:34<04:59,  1.59s/it, Average key norm=0.293, Keys Scaled=3Loss: 0.0219 | Pulse: 2.0651e-04
steps:  53%|▌| 212/400 [05:36<04:58,  1.59s/it, Average key norm=0.294, Keys Scaled=3Loss: 0.0208 | Pulse: 2.0651e-04
steps:  53%|▌| 213/400 [05:37<04:56,  1.58s/it, Average key norm=0.294, Keys Scaled=3Loss: 0.0208 | Pulse: 2.0651e-04
steps:  54%|▌| 214/400 [05:38<04:54,  1.58s/it, Average key norm=0.294, Keys Scaled=3Loss: 0.0210 | Pulse: 2.0651e-04
steps:  54%|▌| 215/400 [05:40<04:53,  1.58s/it, Average key norm=0.294, Keys Scaled=3Loss: 0.0212 | Pulse: 2.0651e-04
steps:  54%|▌| 216/400 [05:42<04:51,  1.58s/it, Average key norm=0.295, Keys Scaled=3Loss: 0.0214 | Pulse: 2.0651e-04
steps:  54%|▌| 217/400 [05:43<04:49,  1.58s/it, Average key norm=0.295, Keys Scaled=3Loss: 0.0210 | Pulse: 2.0651e-04
steps:  55%|▌| 218/400 [05:45<04:48,  1.58s/it, Average key norm=0.295, Keys Scaled=3Loss: 0.0205 | Pulse: 2.0651e-04
steps:  55%|▌| 219/400 [05:46<04:46,  1.58s/it, Average key norm=0.295, Keys Scaled=3Loss: 0.0202 | Pulse: 2.0651e-04
steps:  55%|▌| 220/400 [05:47<04:44,  1.58s/it, Average key norm=0.296, Keys Scaled=3
epoch 23/40
2026-05-12 18:18:14 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             22, epoch: 23
Loss: 0.0204 | Pulse: 2.0651e-04
steps:  55%|▌| 221/400 [05:49<04:43,  1.58s/it, Average key norm=0.296, Keys Scaled=4Loss: 0.0205 | Pulse: 2.0651e-04
steps:  56%|▌| 222/400 [05:50<04:41,  1.58s/it, Average key norm=0.296, Keys Scaled=4Loss: 0.0213 | Pulse: 2.0651e-04
steps:  56%|▌| 223/400 [05:52<04:39,  1.58s/it, Average key norm=0.296, Keys Scaled=4Loss: 0.0205 | Pulse: 2.0651e-04
steps:  56%|▌| 224/400 [05:53<04:37,  1.58s/it, Average key norm=0.296, Keys Scaled=4Loss: 0.0201 | Pulse: 2.0651e-04
steps:  56%|▌| 225/400 [05:55<04:36,  1.58s/it, Average key norm=0.297, Keys Scaled=4Loss: 0.0203 | Pulse: 2.0651e-04
steps:  56%|▌| 226/400 [05:56<04:34,  1.58s/it, Average key norm=0.297, Keys Scaled=4Loss: 0.0216 | Pulse: 2.0651e-04
steps:  57%|▌| 227/400 [05:58<04:32,  1.58s/it, Average key norm=0.297, Keys Scaled=4Loss: 0.0213 | Pulse: 2.0651e-04
steps:  57%|▌| 228/400 [05:59<04:31,  1.58s/it, Average key norm=0.297, Keys Scaled=4Loss: 0.0202 | Pulse: 2.0651e-04
steps:  57%|▌| 229/400 [06:01<04:29,  1.58s/it, Average key norm=0.297, Keys Scaled=4Loss: 0.0207 | Pulse: 2.0651e-04
steps:  57%|▌| 230/400 [06:02<04:27,  1.58s/it, Average key norm=0.297, Keys Scaled=3
epoch 24/40
2026-05-12 18:18:28 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             23, epoch: 24
Loss: 0.0191 | Pulse: 2.0651e-04
steps:  58%|▌| 231/400 [06:03<04:26,  1.58s/it, Average key norm=0.298, Keys Scaled=3Loss: 0.0196 | Pulse: 2.0651e-04
steps:  58%|▌| 232/400 [06:05<04:24,  1.57s/it, Average key norm=0.298, Keys Scaled=3Loss: 0.0205 | Pulse: 2.0651e-04
steps:  58%|▌| 233/400 [06:06<04:22,  1.57s/it, Average key norm=0.298, Keys Scaled=3Loss: 0.0203 | Pulse: 2.0651e-04
steps:  58%|▌| 234/400 [06:08<04:21,  1.57s/it, Average key norm=0.298, Keys Scaled=3Loss: 0.0213 | Pulse: 2.0651e-04
steps:  59%|▌| 235/400 [06:09<04:19,  1.57s/it, Average key norm=0.298, Keys Scaled=3Loss: 0.0195 | Pulse: 2.0651e-04
steps:  59%|▌| 236/400 [06:11<04:17,  1.57s/it, Average key norm=0.298, Keys Scaled=3Loss: 0.0193 | Pulse: 2.0651e-04
steps:  59%|▌| 237/400 [06:12<04:16,  1.57s/it, Average key norm=0.298, Keys Scaled=3Loss: 0.0194 | Pulse: 2.0651e-04
steps:  60%|▌| 238/400 [06:13<04:14,  1.57s/it, Average key norm=0.298, Keys Scaled=4Loss: 0.0194 | Pulse: 2.0651e-04
steps:  60%|▌| 239/400 [06:15<04:12,  1.57s/it, Average key norm=0.299, Keys Scaled=4Loss: 0.0188 | Pulse: 2.0651e-04
steps:  60%|▌| 240/400 [06:16<04:11,  1.57s/it, Average key norm=0.299, Keys Scaled=4
epoch 25/40
2026-05-12 18:18:42 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             24, epoch: 25
Loss: 0.0192 | Pulse: 2.0651e-04
steps:  60%|▌| 241/400 [06:18<04:09,  1.57s/it, Average key norm=0.299, Keys Scaled=4Loss: 0.0204 | Pulse: 2.0651e-04
steps:  60%|▌| 242/400 [06:19<04:08,  1.57s/it, Average key norm=0.299, Keys Scaled=4Loss: 0.0196 | Pulse: 1.7786e-05
steps:  61%|▌| 243/400 [06:21<04:06,  1.57s/it, Average key norm=0.299, Keys Scaled=4Loss: 0.0198 | Pulse: 2.6679e-05
steps:  61%|▌| 244/400 [06:22<04:04,  1.57s/it, Average key norm=0.299, Keys Scaled=3Loss: 0.0214 | Pulse: 3.7742e-05
steps:  61%|▌| 245/400 [06:24<04:03,  1.57s/it, Average key norm=0.299, Keys Scaled=3Loss: 0.0195 | Pulse: 4.0589e-05
steps:  62%|▌| 246/400 [06:26<04:01,  1.57s/it, Average key norm=0.299, Keys Scaled=2Loss: 0.0200 | Pulse: 4.0870e-05
steps:  62%|▌| 247/400 [06:27<04:00,  1.57s/it, Average key norm=0.299, Keys Scaled=2Loss: 0.0192 | Pulse: 4.4422e-05
steps:  62%|▌| 248/400 [06:29<03:58,  1.57s/it, Average key norm=0.299, Keys Scaled=2Loss: 0.0189 | Pulse: 4.8433e-05
steps:  62%|▌| 249/400 [06:30<03:56,  1.57s/it, Average key norm=0.299, Keys Scaled=2Loss: 0.0186 | Pulse: 5.3003e-05
steps:  62%|▋| 250/400 [06:32<03:55,  1.57s/it, Average key norm=0.299, Keys Scaled=2
epoch 26/40
2026-05-12 18:18:58 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             25, epoch: 26
Loss: 0.0182 | Pulse: 5.8079e-05
steps:  63%|▋| 251/400 [06:33<03:53,  1.57s/it, Average key norm=0.299, Keys Scaled=2Loss: 0.0197 | Pulse: 5.8079e-05
steps:  63%|▋| 252/400 [06:35<03:52,  1.57s/it, Average key norm=0.299, Keys Scaled=2Loss: 0.0180 | Pulse: 5.9097e-05
steps:  63%|▋| 253/400 [06:36<03:50,  1.57s/it, Average key norm=0.299, Keys Scaled=2Loss: 0.0200 | Pulse: 2.7995e-05
steps:  64%|▋| 254/400 [06:38<03:49,  1.57s/it, Average key norm=0.299, Keys Scaled=2Loss: 0.0195 | Pulse: 4.1993e-05
steps:  64%|▋| 255/400 [06:40<03:47,  1.57s/it, Average key norm=0.299, Keys Scaled=2Loss: 0.0186 | Pulse: 4.7574e-05
steps:  64%|▋| 256/400 [06:41<03:45,  1.57s/it, Average key norm=0.299, Keys Scaled=2Loss: 0.0191 | Pulse: 4.7598e-05
steps:  64%|▋| 257/400 [06:43<03:44,  1.57s/it, Average key norm=0.299, Keys Scaled=2Loss: 0.0186 | Pulse: 4.9615e-05
steps:  64%|▋| 258/400 [06:44<03:42,  1.57s/it, Average key norm=0.3, Keys Scaled=31,Loss: 0.0191 | Pulse: 4.9615e-05
steps:  65%|▋| 259/400 [06:46<03:41,  1.57s/it, Average key norm=0.3, Keys Scaled=32,Loss: 0.0191 | Pulse: 4.9615e-05
steps:  65%|▋| 260/400 [06:47<03:39,  1.57s/it, Average key norm=0.3, Keys Scaled=34,
epoch 27/40
2026-05-12 18:19:13 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             26, epoch: 27
Loss: 0.0190 | Pulse: 4.9615e-05
steps:  65%|▋| 261/400 [06:49<03:37,  1.57s/it, Average key norm=0.3, Keys Scaled=35,Loss: 0.0189 | Pulse: 4.9615e-05
steps:  66%|▋| 262/400 [06:50<03:36,  1.57s/it, Average key norm=0.3, Keys Scaled=35,Loss: 0.0184 | Pulse: 5.1794e-05
steps:  66%|▋| 263/400 [06:52<03:34,  1.57s/it, Average key norm=0.3, Keys Scaled=38,Loss: 0.0196 | Pulse: 5.1794e-05
steps:  66%|▋| 264/400 [06:53<03:33,  1.57s/it, Average key norm=0.3, Keys Scaled=38,Loss: 0.0186 | Pulse: 5.1794e-05
steps:  66%|▋| 265/400 [06:55<03:31,  1.57s/it, Average key norm=0.3, Keys Scaled=42,Loss: 0.0175 | Pulse: 5.7518e-05
steps:  66%|▋| 266/400 [06:56<03:29,  1.57s/it, Average key norm=0.3, Keys Scaled=42,Loss: 0.0176 | Pulse: 6.2256e-05
steps:  67%|▋| 267/400 [06:58<03:28,  1.57s/it, Average key norm=0.3, Keys Scaled=42,Loss: 0.0185 | Pulse: 6.2256e-05
steps:  67%|▋| 268/400 [06:59<03:26,  1.57s/it, Average key norm=0.3, Keys Scaled=42,Loss: 0.0194 | Pulse: 5.8092e-05
steps:  67%|▋| 269/400 [07:00<03:25,  1.56s/it, Average key norm=0.3, Keys Scaled=43,Loss: 0.0191 | Pulse: 5.8092e-05
steps:  68%|▋| 270/400 [07:02<03:23,  1.56s/it, Average key norm=0.3, Keys Scaled=44,
epoch 28/40
2026-05-12 18:19:28 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             27, epoch: 28
Loss: 0.0196 | Pulse: 5.8092e-05
steps:  68%|▋| 271/400 [07:04<03:21,  1.56s/it, Average key norm=0.3, Keys Scaled=44,Loss: 0.0184 | Pulse: 5.8092e-05
steps:  68%|▋| 272/400 [07:05<03:20,  1.56s/it, Average key norm=0.3, Keys Scaled=44,Loss: 0.0181 | Pulse: 5.8092e-05
steps:  68%|▋| 273/400 [07:07<03:18,  1.56s/it, Average key norm=0.3, Keys Scaled=42,Loss: 0.0174 | Pulse: 5.8092e-05
steps:  68%|▋| 274/400 [07:08<03:17,  1.56s/it, Average key norm=0.3, Keys Scaled=42,Loss: 0.0188 | Pulse: 5.8092e-05
steps:  69%|▋| 275/400 [07:09<03:15,  1.56s/it, Average key norm=0.3, Keys Scaled=37,Loss: 0.0181 | Pulse: 5.8092e-05
steps:  69%|▋| 276/400 [07:11<03:13,  1.56s/it, Average key norm=0.3, Keys Scaled=37,Loss: 0.0187 | Pulse: 5.8092e-05
steps:  69%|▋| 277/400 [07:12<03:12,  1.56s/it, Average key norm=0.3, Keys Scaled=36,Loss: 0.0191 | Pulse: 5.8092e-05
steps:  70%|▋| 278/400 [07:14<03:10,  1.56s/it, Average key norm=0.301, Keys Scaled=3Loss: 0.0181 | Pulse: 5.8092e-05
steps:  70%|▋| 279/400 [07:16<03:09,  1.56s/it, Average key norm=0.301, Keys Scaled=3Loss: 0.0186 | Pulse: 5.8092e-05
steps:  70%|▋| 280/400 [07:17<03:07,  1.56s/it, Average key norm=0.301, Keys Scaled=3
epoch 29/40
2026-05-12 18:19:43 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             28, epoch: 29
Loss: 0.0191 | Pulse: 5.8092e-05
steps:  70%|▋| 281/400 [07:18<03:05,  1.56s/it, Average key norm=0.301, Keys Scaled=3Loss: 0.0179 | Pulse: 5.8092e-05
steps:  70%|▋| 282/400 [07:20<03:04,  1.56s/it, Average key norm=0.301, Keys Scaled=3Loss: 0.0195 | Pulse: 5.8092e-05
steps:  71%|▋| 283/400 [07:21<03:02,  1.56s/it, Average key norm=0.301, Keys Scaled=4Loss: 0.0189 | Pulse: 2.9285e-04
steps:  71%|▋| 284/400 [07:23<03:01,  1.56s/it, Average key norm=0.301, Keys Scaled=4Loss: 0.0199 | Pulse: 2.9285e-04
steps:  71%|▋| 285/400 [07:24<02:59,  1.56s/it, Average key norm=0.301, Keys Scaled=4Loss: 0.0196 | Pulse: 2.9285e-04
steps:  72%|▋| 286/400 [07:26<02:57,  1.56s/it, Average key norm=0.302, Keys Scaled=4Loss: 0.0195 | Pulse: 2.9285e-04
steps:  72%|▋| 287/400 [07:27<02:56,  1.56s/it, Average key norm=0.302, Keys Scaled=4Loss: 0.0187 | Pulse: 2.9285e-04
steps:  72%|▋| 288/400 [07:29<02:54,  1.56s/it, Average key norm=0.302, Keys Scaled=4Loss: 0.0192 | Pulse: 2.9285e-04
steps:  72%|▋| 289/400 [07:30<02:53,  1.56s/it, Average key norm=0.302, Keys Scaled=4Loss: 0.0183 | Pulse: 2.9285e-04
steps:  72%|▋| 290/400 [07:32<02:51,  1.56s/it, Average key norm=0.302, Keys Scaled=4
epoch 30/40
2026-05-12 18:19:58 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             29, epoch: 30
Loss: 0.0185 | Pulse: 2.9285e-04
steps:  73%|▋| 291/400 [07:33<02:49,  1.56s/it, Average key norm=0.302, Keys Scaled=4Loss: 0.0195 | Pulse: 2.9285e-04
steps:  73%|▋| 292/400 [07:35<02:48,  1.56s/it, Average key norm=0.302, Keys Scaled=4Loss: 0.0200 | Pulse: 2.9285e-04
steps:  73%|▋| 293/400 [07:36<02:46,  1.56s/it, Average key norm=0.302, Keys Scaled=4Loss: 0.0193 | Pulse: 2.9285e-04
steps:  74%|▋| 294/400 [07:38<02:45,  1.56s/it, Average key norm=0.303, Keys Scaled=3Loss: 0.0188 | Pulse: 2.9285e-04
steps:  74%|▋| 295/400 [07:39<02:43,  1.56s/it, Average key norm=0.303, Keys Scaled=3Loss: 0.0183 | Pulse: 2.9285e-04
steps:  74%|▋| 296/400 [07:41<02:42,  1.56s/it, Average key norm=0.303, Keys Scaled=3Loss: 0.0190 | Pulse: 2.9285e-04
steps:  74%|▋| 297/400 [07:42<02:40,  1.56s/it, Average key norm=0.303, Keys Scaled=4Loss: 0.0193 | Pulse: 2.9285e-04
steps:  74%|▋| 298/400 [07:44<02:38,  1.56s/it, Average key norm=0.303, Keys Scaled=4Loss: 0.0191 | Pulse: 2.9285e-04
steps:  75%|▋| 299/400 [07:45<02:37,  1.56s/it, Average key norm=0.304, Keys Scaled=4Loss: 0.0205 | Pulse: 2.9285e-04
steps:  75%|▊| 300/400 [07:47<02:35,  1.56s/it, Average key norm=0.304, Keys Scaled=4
saving checkpoint: E:\SdxlWebUi\Lora\MiXOMRR-R08D-M000-esx-512px-step00000300.safetensors
steps:  75%|▊| 300/400 [07:47<02:35,  1.56s/it, Average key norm=0.304, Keys Scaled=4
epoch 31/40
2026-05-12 18:20:13 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             30, epoch: 31
Loss: 0.0190 | Pulse: 2.9285e-04
steps:  75%|▊| 301/400 [07:48<02:34,  1.56s/it, Average key norm=0.304, Keys Scaled=4Loss: 0.0192 | Pulse: 2.9285e-04
steps:  76%|▊| 302/400 [07:50<02:32,  1.56s/it, Average key norm=0.304, Keys Scaled=4Loss: 0.0184 | Pulse: 2.9285e-04
steps:  76%|▊| 303/400 [07:51<02:30,  1.56s/it, Average key norm=0.304, Keys Scaled=3Loss: 0.0188 | Pulse: 2.9285e-04
steps:  76%|▊| 304/400 [07:52<02:29,  1.56s/it, Average key norm=0.305, Keys Scaled=3Loss: 0.0204 | Pulse: 2.9285e-04
steps:  76%|▊| 305/400 [07:54<02:27,  1.56s/it, Average key norm=0.305, Keys Scaled=3Loss: 0.0184 | Pulse: 2.9285e-04
steps:  76%|▊| 306/400 [07:55<02:26,  1.56s/it, Average key norm=0.305, Keys Scaled=3Loss: 0.0196 | Pulse: 2.9285e-04
steps:  77%|▊| 307/400 [07:57<02:24,  1.56s/it, Average key norm=0.305, Keys Scaled=3Loss: 0.0198 | Pulse: 2.9285e-04
steps:  77%|▊| 308/400 [07:58<02:23,  1.55s/it, Average key norm=0.305, Keys Scaled=3Loss: 0.0180 | Pulse: 2.9285e-04
steps:  77%|▊| 309/400 [08:00<02:21,  1.55s/it, Average key norm=0.305, Keys Scaled=3Loss: 0.0186 | Pulse: 2.9285e-04
steps:  78%|▊| 310/400 [08:01<02:19,  1.55s/it, Average key norm=0.305, Keys Scaled=3
epoch 32/40
2026-05-12 18:20:28 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             31, epoch: 32
Loss: 0.0181 | Pulse: 2.9285e-04
steps:  78%|▊| 311/400 [08:03<02:18,  1.55s/it, Average key norm=0.306, Keys Scaled=4Loss: 0.0183 | Pulse: 2.9285e-04
steps:  78%|▊| 312/400 [08:04<02:16,  1.55s/it, Average key norm=0.306, Keys Scaled=4Loss: 0.0189 | Pulse: 2.9285e-04
steps:  78%|▊| 313/400 [08:06<02:15,  1.55s/it, Average key norm=0.306, Keys Scaled=4Loss: 0.0183 | Pulse: 2.9285e-04
steps:  78%|▊| 314/400 [08:07<02:13,  1.55s/it, Average key norm=0.306, Keys Scaled=4Loss: 0.0193 | Pulse: 2.9285e-04
steps:  79%|▊| 315/400 [08:09<02:11,  1.55s/it, Average key norm=0.306, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0176 | Pulse: 2.9285e-04
steps:  79%|▊| 316/400 [08:10<02:10,  1.55s/it, Average key norm=0.306, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0188 | Pulse: 2.9285e-04
steps:  79%|▊| 317/400 [08:11<02:08,  1.55s/it, Average key norm=0.306, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0192 | Pulse: 2.9285e-04
steps:  80%|▊| 318/400 [08:13<02:07,  1.55s/it, Average key norm=0.307, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0176 | Pulse: 2.9285e-04
steps:  80%|▊| 319/400 [08:14<02:05,  1.55s/it, Average key norm=0.307, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0183 | Pulse: 2.9285e-04
steps:  80%|▊| 320/400 [08:16<02:04,  1.55s/it, Average key norm=0.307, Keys Scaled=3
epoch 33/40
2026-05-12 18:20:43 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             32, epoch: 33
✨[READY TO STOP]✨
Loss: 0.0193 | Pulse: 2.9285e-04
steps:  80%|▊| 321/400 [08:18<02:02,  1.55s/it, Average key norm=0.307, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0182 | Pulse: 2.9285e-04
steps:  80%|▊| 322/400 [08:19<02:01,  1.55s/it, Average key norm=0.307, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0186 | Pulse: 2.9285e-04
steps:  81%|▊| 323/400 [08:21<01:59,  1.55s/it, Average key norm=0.308, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0185 | Pulse: 2.9285e-04
steps:  81%|▊| 324/400 [08:22<01:57,  1.55s/it, Average key norm=0.308, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0179 | Pulse: 2.9285e-04
steps:  81%|▊| 325/400 [08:24<01:56,  1.55s/it, Average key norm=0.308, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0181 | Pulse: 2.9285e-04
steps:  82%|▊| 326/400 [08:25<01:54,  1.55s/it, Average key norm=0.308, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0188 | Pulse: 2.9285e-04
steps:  82%|▊| 327/400 [08:27<01:53,  1.55s/it, Average key norm=0.308, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0169 | Pulse: 2.9285e-04
steps:  82%|▊| 328/400 [08:28<01:51,  1.55s/it, Average key norm=0.308, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0178 | Pulse: 2.9285e-04
steps:  82%|▊| 329/400 [08:30<01:50,  1.55s/it, Average key norm=0.308, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0184 | Pulse: 2.9285e-04
steps:  82%|▊| 330/400 [08:31<01:48,  1.55s/it, Average key norm=0.309, Keys Scaled=4
epoch 34/40
2026-05-12 18:20:58 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             33, epoch: 34
✨[READY TO STOP]✨
Loss: 0.0207 | Pulse: 2.9285e-04
steps:  83%|▊| 331/400 [08:33<01:47,  1.55s/it, Average key norm=0.309, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0176 | Pulse: 2.9285e-04
steps:  83%|▊| 332/400 [08:35<01:45,  1.55s/it, Average key norm=0.309, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0183 | Pulse: 2.9285e-04
steps:  83%|▊| 333/400 [08:36<01:43,  1.55s/it, Average key norm=0.309, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0190 | Pulse: 2.9285e-04
steps:  84%|▊| 334/400 [08:38<01:42,  1.55s/it, Average key norm=0.309, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0173 | Pulse: 2.9285e-04
steps:  84%|▊| 335/400 [08:39<01:40,  1.55s/it, Average key norm=0.309, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0184 | Pulse: 2.9285e-04
steps:  84%|▊| 336/400 [08:41<01:39,  1.55s/it, Average key norm=0.309, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0172 | Pulse: 2.9285e-04
steps:  84%|▊| 337/400 [08:42<01:37,  1.55s/it, Average key norm=0.31, Keys Scaled=38✨[READY TO STOP]✨
Loss: 0.0185 | Pulse: 2.9285e-04
steps:  84%|▊| 338/400 [08:44<01:36,  1.55s/it, Average key norm=0.31, Keys Scaled=41✨[READY TO STOP]✨
Loss: 0.0190 | Pulse: 2.9285e-04
steps:  85%|▊| 339/400 [08:45<01:34,  1.55s/it, Average key norm=0.31, Keys Scaled=44✨[READY TO STOP]✨
Loss: 0.0187 | Pulse: 2.9285e-04
steps:  85%|▊| 340/400 [08:47<01:33,  1.55s/it, Average key norm=0.31, Keys Scaled=46
epoch 35/40
2026-05-12 18:21:14 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             34, epoch: 35
✨[READY TO STOP]✨
Loss: 0.0188 | Pulse: 2.9285e-04
steps:  85%|▊| 341/400 [08:49<01:31,  1.55s/it, Average key norm=0.31, Keys Scaled=47✨[READY TO STOP]✨
Loss: 0.0188 | Pulse: 2.9285e-04
steps:  86%|▊| 342/400 [08:50<01:29,  1.55s/it, Average key norm=0.311, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0186 | Pulse: 2.9285e-04
steps:  86%|▊| 343/400 [08:52<01:28,  1.55s/it, Average key norm=0.311, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0172 | Pulse: 2.9285e-04
steps:  86%|▊| 344/400 [08:53<01:26,  1.55s/it, Average key norm=0.311, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0167 | Pulse: 2.9285e-04
steps:  86%|▊| 345/400 [08:55<01:25,  1.55s/it, Average key norm=0.311, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0175 | Pulse: 2.9285e-04
steps:  86%|▊| 346/400 [08:56<01:23,  1.55s/it, Average key norm=0.311, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0187 | Pulse: 2.9285e-04
steps:  87%|▊| 347/400 [08:58<01:22,  1.55s/it, Average key norm=0.311, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0186 | Pulse: 2.9285e-04
steps:  87%|▊| 348/400 [08:59<01:20,  1.55s/it, Average key norm=0.311, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0184 | Pulse: 2.9285e-04
steps:  87%|▊| 349/400 [09:01<01:19,  1.55s/it, Average key norm=0.311, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0182 | Pulse: 2.9285e-04
steps:  88%|▉| 350/400 [09:02<01:17,  1.55s/it, Average key norm=0.312, Keys Scaled=4
epoch 36/40
2026-05-12 18:21:29 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             35, epoch: 36
✨[READY TO STOP]✨
Loss: 0.0173 | Pulse: 2.9285e-04
steps:  88%|▉| 351/400 [09:04<01:15,  1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0191 | Pulse: 2.9285e-04
steps:  88%|▉| 352/400 [09:05<01:14,  1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0168 | Pulse: 2.9285e-04
steps:  88%|▉| 353/400 [09:07<01:12,  1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0171 | Pulse: 2.9285e-04
steps:  88%|▉| 354/400 [09:08<01:11,  1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0182 | Pulse: 2.9285e-04
steps:  89%|▉| 355/400 [09:10<01:09,  1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0188 | Pulse: 2.9285e-04
steps:  89%|▉| 356/400 [09:11<01:08,  1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0198 | Pulse: 2.9285e-04
steps:  89%|▉| 357/400 [09:13<01:06,  1.55s/it, Average key norm=0.313, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0187 | Pulse: 2.9285e-04
steps:  90%|▉| 358/400 [09:14<01:05,  1.55s/it, Average key norm=0.313, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0190 | Pulse: 2.9285e-04
steps:  90%|▉| 359/400 [09:16<01:03,  1.55s/it, Average key norm=0.313, Keys Scaled=3✨[READY TO STOP]✨
Loss: 0.0181 | Pulse: 2.9285e-04
steps:  90%|▉| 360/400 [09:17<01:01,  1.55s/it, Average key norm=0.313, Keys Scaled=3
epoch 37/40
2026-05-12 18:21:44 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             36, epoch: 37
✨[READY TO STOP]✨
Loss: 0.0180 | Pulse: 2.9285e-04
steps:  90%|▉| 361/400 [09:19<01:00,  1.55s/it, Average key norm=0.313, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0186 | Pulse: 2.9285e-04
steps:  90%|▉| 362/400 [09:20<00:58,  1.55s/it, Average key norm=0.314, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0171 | Pulse: 2.9285e-04
steps:  91%|▉| 363/400 [09:22<00:57,  1.55s/it, Average key norm=0.314, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0183 | Pulse: 2.9285e-04
steps:  91%|▉| 364/400 [09:23<00:55,  1.55s/it, Average key norm=0.314, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0178 | Pulse: 2.9285e-04
steps:  91%|▉| 365/400 [09:25<00:54,  1.55s/it, Average key norm=0.314, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0182 | Pulse: 2.9285e-04
steps:  92%|▉| 366/400 [09:27<00:52,  1.55s/it, Average key norm=0.314, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0168 | Pulse: 2.9285e-04
steps:  92%|▉| 367/400 [09:28<00:51,  1.55s/it, Average key norm=0.314, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0167 | Pulse: 2.9285e-04
steps:  92%|▉| 368/400 [09:30<00:49,  1.55s/it, Average key norm=0.314, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0176 | Pulse: 2.9285e-04
steps:  92%|▉| 369/400 [09:31<00:48,  1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0168 | Pulse: 2.9285e-04
steps:  92%|▉| 370/400 [09:33<00:46,  1.55s/it, Average key norm=0.315, Keys Scaled=4
epoch 38/40
2026-05-12 18:21:59 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             37, epoch: 38
✨[READY TO STOP]✨
Loss: 0.0167 | Pulse: 2.9285e-04
steps:  93%|▉| 371/400 [09:34<00:44,  1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0162 | Pulse: 2.9285e-04
steps:  93%|▉| 372/400 [09:36<00:43,  1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0184 | Pulse: 2.9285e-04
steps:  93%|▉| 373/400 [09:37<00:41,  1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0168 | Pulse: 2.9285e-04
steps:  94%|▉| 374/400 [09:39<00:40,  1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0163 | Pulse: 2.9285e-04
steps:  94%|▉| 375/400 [09:41<00:38,  1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0166 | Pulse: 2.9285e-04
steps:  94%|▉| 376/400 [09:42<00:37,  1.55s/it, Average key norm=0.316, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0171 | Pulse: 2.9285e-04
steps:  94%|▉| 377/400 [09:44<00:35,  1.55s/it, Average key norm=0.316, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0157 | Pulse: 2.9285e-04
steps:  94%|▉| 378/400 [09:45<00:34,  1.55s/it, Average key norm=0.316, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0176 | Pulse: 2.9285e-04
steps:  95%|▉| 379/400 [09:47<00:32,  1.55s/it, Average key norm=0.316, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0164 | Pulse: 2.9285e-04
steps:  95%|▉| 380/400 [09:48<00:30,  1.55s/it, Average key norm=0.316, Keys Scaled=4
epoch 39/40
2026-05-12 18:22:15 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             38, epoch: 39
✨[READY TO STOP]✨
Loss: 0.0162 | Pulse: 2.9285e-04
steps:  95%|▉| 381/400 [09:49<00:29,  1.55s/it, Average key norm=0.316, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0172 | Pulse: 2.9285e-04
steps:  96%|▉| 382/400 [09:51<00:27,  1.55s/it, Average key norm=0.317, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0184 | Pulse: 2.9285e-04
steps:  96%|▉| 383/400 [09:52<00:26,  1.55s/it, Average key norm=0.317, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0170 | Pulse: 2.9285e-04
steps:  96%|▉| 384/400 [09:54<00:24,  1.55s/it, Average key norm=0.317, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0181 | Pulse: 2.9285e-04
steps:  96%|▉| 385/400 [09:55<00:23,  1.55s/it, Average key norm=0.317, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0174 | Pulse: 2.9285e-04
steps:  96%|▉| 386/400 [09:57<00:21,  1.55s/it, Average key norm=0.317, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0167 | Pulse: 2.9285e-04
steps:  97%|▉| 387/400 [09:58<00:20,  1.55s/it, Average key norm=0.317, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0158 | Pulse: 2.9285e-04
steps:  97%|▉| 388/400 [10:00<00:18,  1.55s/it, Average key norm=0.317, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0181 | Pulse: 2.9285e-04
steps:  97%|▉| 389/400 [10:01<00:17,  1.55s/it, Average key norm=0.317, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0181 | Pulse: 2.9285e-04
steps:  98%|▉| 390/400 [10:02<00:15,  1.55s/it, Average key norm=0.318, Keys Scaled=4
epoch 40/40
2026-05-12 18:22:29 INFO     epoch is incremented. current_epoch:    train_util.py:787
                             39, epoch: 40
✨[READY TO STOP]✨
Loss: 0.0167 | Pulse: 2.9285e-04
steps:  98%|▉| 391/400 [10:04<00:13,  1.55s/it, Average key norm=0.318, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0180 | Pulse: 2.9285e-04
steps:  98%|▉| 392/400 [10:05<00:12,  1.55s/it, Average key norm=0.318, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0164 | Pulse: 2.9285e-04
steps:  98%|▉| 393/400 [10:07<00:10,  1.55s/it, Average key norm=0.318, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0162 | Pulse: 2.9285e-04
steps:  98%|▉| 394/400 [10:08<00:09,  1.55s/it, Average key norm=0.318, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0167 | Pulse: 2.9285e-04
steps:  99%|▉| 395/400 [10:10<00:07,  1.54s/it, Average key norm=0.318, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0178 | Pulse: 2.9285e-04
steps:  99%|▉| 396/400 [10:11<00:06,  1.54s/it, Average key norm=0.318, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0160 | Pulse: 2.9285e-04
steps:  99%|▉| 397/400 [10:13<00:04,  1.54s/it, Average key norm=0.318, Keys Scaled=4✨[READY TO STOP]✨
Loss: 0.0168 | Pulse: 2.9285e-04
steps: 100%|▉| 398/400 [10:14<00:03,  1.54s/it, Average key norm=0.319, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0186 | Pulse: 2.9285e-04
steps: 100%|▉| 399/400 [10:16<00:01,  1.54s/it, Average key norm=0.319, Keys Scaled=5✨[READY TO STOP]✨
Loss: 0.0171 | Pulse: 2.9285e-04
steps: 100%|█| 400/400 [10:17<00:00,  1.54s/it, Average key norm=0.319, Keys Scaled=5
```

</details>

---

以下、v3.8 以前 について  

---

#### ！ 皆さまへのお詫び (重要なお知らせ) ！  
まず最初に、ご覧の皆さまに率直にお詫び申し上げます  
本リポジトリで公開している emo系オプティマイザに関する  
｢実験結果｣｢理論的主張｣について、自分自身の後続検証により  
"成立しない可能性が高い"ことが判明しました  
特に emoPulse に関する主張は closure 前提の設計にも関わらず  
 closure を適用をしていないことが判明し  
これにより理論と実装の整合性が崩れていることを確認しました  
当時の記述は、その時点での理解に基づくものであり  
現在の視点では"未検証／誤りを含む草案"とすべき内容です  
記述を信じてくださった皆さまに対し  
誤解を招く結果となったことを深くお詫び申し上げます  


#### 以下の内容はすべて ｢仮説｣｢未検証｣｢草案｣ です  
※ 理論的な解釈を誤っていた可能性があります  
※ 信頼できるエビデンスとは言えません  

--- 

- ###### 共鳴収縮法(共鳴投影場)をする新世代optimizer群です／勾配降下法ではない   
- ###### EmoSens / 2ndGen (v3.8 / Standard)  
- ###### EmoTion / 3rdGen (v3.8 / Moment-Free)  
readme：[English](README.md) | [日本語](README_JA.md)  

---

#### 共鳴収縮法によるアーキテクチャの進化  
こちらにて Transformer の進化型を紹介しています  
https://github.com/muooon/DRNA  

---

# EmoSens / Tion 最新版 update  

- EmoVoid は"波動散乱逆問題"解析ソルバとして機能する可能性があります  
- アーリーストップ通知機能を正確化、学習引き継ぎ対応、beginners版との統合(260404)  
- EmoSens (v3.8) emoPulse (完全自動学習率) 等の調整をしました  
- EmoTion (v3.8) オリジナル W-Ref-Geometry and Moment-Free の公開  

##### ※ FFT版を統合済み(フルファインチューン) Optionの引数でモード切替可です  

##### ※ FFT-Aware version integrated,"FFT(full fine-tuning)" Mode switching available via Option arguments

### v3.7以降の特徴  
- 完全自動学習率：高速化と精緻化を同時に達成しつつ初期LRに悩まなくていい  
- emoPulse：自律的にLRを増減させ"極低精度･超量子化"も安全安定で進行します  
- 初期LRは1.0で大丈夫です(データセットの工夫にあなたの時間を割いてください)  

### 解説 ･ Explanation  
Mathematical Explanation Here (paper) v3.7 and later  
(非凸関数に対する期待値収束(フローマッチングへの適応なども保証します)  
(論文ではフラットミニマやグロッキングに対しての挙動も考察しています)    

#### [数学的解説はこちら(論文)](https://huggingface.co/muooon/EmoNAVI/raw/main/emo-v386plus-paper(JPN).txt)  

#### [DOI取得版/DOI-Acquired Version](https://huggingface.co/muooon/EmoTion-Optimizer)  

---

<details>

<summary> 共鳴収縮法の基本定理 / resonant contraction method </summary>

共鳴収縮法の基本定理(概要)  

1. 状態の定義：三要素の共鳴  
    パラメータ w の更新は、以下の3つの独立した次元の相乗効果(共鳴)によって決定される  
    時間軸 (ηt)：emoPulse：システム内部の｢信頼度｣(SNR)から自律生成される歩幅  
    空間軸 (Rt)：W-Ref Geometry：現在の重みと勾配の｢直交性｣から計算される新規性ゲイン  
    方向軸 (ut)：Pure Will：勾配の大きさを捨て、時間的に純化された｢符号｣(sign)のみの意志  
※ ηt (時間軸)：高精度の動的スケジューラとして機能するものがあれば代替可能  
※ Rt (空間軸)：高精度の２次モーメントでも代替可能(同等の類するもの可)  
※ ut (方向軸)：高精度の１次モーメントでも代替可能(同等の類するもの可)  

2. 更新の基本方程式  
    勾配を g としたとき、伝統的な Δw=−ηg を破棄し、以下の式を適用する  
    離散時間：  
	Δwt = −ηt ⋅ Rt ⋅ sign(mt)  
	連続時間：  
	\frac{dw}{dt} = - λ ⋅ η(t) ⋅ w(t) - η(t) ⋅ R(t) ⋅ u(t)  
    これにより｢勾配の大きさ｣という外力への依存が完全に消滅し、システムは内部状態に基づいた自律的な移動へと移行する  
※ (mt)：時間的に安定化された方向ベクトル(mt = moment ではない)  
    (mt) は、勾配 gt の大きさを無視し、時間的平滑化を施した方向成分で｢累積的な確信度｣を保持し ut = sign(mt) を通じて Pure Will (方向軸) を形成する (つまり大きさを時間軸に委ねる)  

3. 定理が保証する3つの性質  
a. 自律的収縮(Contraction Property)  
    システムのエネルギー(Loss)が低下するにつれ ηt が｢自律的なブレーキ｣として機能する  
    結果：外部からのスケジュール調整なしに、システムは指数関数的に一点(解の多様体)へ収縮し安定する  
b. 幾何学的最短路(Geodesic Path)  
    Rt が｢既知の方向｣(重みと平行な成分)を抑制し、｢未知の方向｣(直交する成分)を加速させる  
    結果：パラメータ空間という球面(多様体）の上を、無駄な蛇行をせず最短距離で滑るように移動する  
c. 情報の純化(Information Bottleneck)  
    sign 関数による方向の抽出が、勾配に含まれる微細なノイズを遮断するフィルターとして機能する  
    結果：複雑すぎる解(過学習)を避け、最もシンプルで汎用性の高い｢平坦な解｣(Flat Minima)に定着する  

結論：共鳴収縮法とはなにか？  
    emoPulse のような｢動的スケジューラ｣(自律的)は、確率的勾配降下法(受動的)を、システムの内部状態に基づく"共鳴収縮法"(共鳴投影場)へ upgrade する。 こうした自律した機構を持つことはSDE-DDE-ODE縮約近似を果たし、この最適化は頑健性と正確性を得て高度な収縮プロセスへと進化する  

</details>

---

<div align="center">
  <img width="500" alt="emo-system001" src="https://github.com/user-attachments/assets/7e7160a9-046a-4212-bcde-d338c26ed846" />
</div>

---

emo系 v3.8 (Standard / Moment-Free) の特徴等  

| 名称      | 時間的正確性 | メモリ効率 | 備考                                      |  
|-----------|------------|------------|-------------------------------------------|  
| emosens   | ★★★★       | ★★         | 最初に誕生｜正確｜Adam型       |  
| emoairy   | ★★         | ★★★★       | ２番目に誕生｜軽量｜Adafactor型 |  
| emocats   | ★★★☆        | ★★★☆        | 軽量＆正確の両立｜Lion型         |  
|-----------|------------|------------|-------------------------------------------|  
| emotion   | ★★★★       | ★★★☆       | "軽量"で正確｜オリジナル型         |  
| emovoid   | ★★☆        | ★★★★★      | "最軽量･最速"｜オリジナル型         |  

[効率性] 危険抑止更新：過学習や収束の停滞に先回りし無駄な更新を排除します  
[機能性] 軽量で高機能：停止合図や自律した分散学習等でユーザー体験を向上させます  
[信頼性] 安全優先設計：動的制御で不安定な局面でモデルを保護し安定収束を促します  
※ 完全自律型のため、積層、再開、非同期、で、自由な学習を自由に組むことが可能です  
※ EmoTion は、幾何学的直交更新と２次モーメント排除で正確性と効率性を向上します  
※ EmoVoid は、幾何学的直交更新と１次２次モーメント排除でVRAM効率を向上します

---  

#### Loss あるかぎり emoPulse(鼓動) はやまない ――  
##### Grokking を経ずに フラットミニマ へ到達できるかもしれない選択肢 

---  

### 学習の情報、そのすべては Loss値 に集約されている  
##### Loss値はモデルのshadowである、  
##### Loss値にすべてが集約されている、  
##### 学習状況もモデル状況もLoss値が教えてくれる、  
##### Lossを感じろ、 Lossこそオリジン(原点)だ、  

---  

### EmoSens 主な特徴 / Main Features of EmoSens  

---  

<details>

<summary> Main Features </summary>

||| 自律性と信頼性 |||  
過学習や発散を抑制、自己修復的機能をもちます  
学習率やスケジューラも自律調整、モデル自身で判断します  
学習の 再開、追加、積層、等で"引き継ぎ不要"、誰でも簡単です  
分散学習で 他ノード等との"同期不要"、完全自律です  

||| 感情駆動型自律サイクル |||  
emo系 は既存のオプティマイザにはない｢感情駆動型｣です  
調整の複雑なマルチモーダル学習などの新しい分野の課題への対応も期待できます  
emo系は、観察、判断、決定、行動、記憶、反省、という自律サイクルを行います  

||| 最終進化と哲学 |||  
ものすごく単純にいうと｢emo系 emoPulse は高級スケジューラ｣です  
Sharpness-Aware Minimization の最終進化でもあります  
SDEながらODE近似になる―という"正確さ"を実現しています(止観と止揚です)  
RNN進化系の Liquid型(LiquidAI/MIT)、Titans(Google)、Mamba(CMU/Princeton)等々と好相性です  

||| 高効率性と集積度 (近似的構造) |||  
複数の高次moment、履歴補償、量子化補償(Kahan補償と違う制御)、信頼度フィルタ、  
動的スケーリング、分散･継続学習での独立性、自己修復･モデル修復(LoRAによる逆位相マージ)、  
自己停止、ハイパーパラメータの自律調整、構造的耐性、等を内包する自己回帰型の学習をします  
動的学習率、動的スケジューラ、動的Rank/Aplha、SVD、infLoRA、ABBA-LoRA、PiSSA、  
FourierFT、DoRA、PRO-LoRA、DARE、Ties、Tall-Mask-Merge、などを含めた多機能性を、  
追加テンソル不要、計算負荷ほぼなし、ここまですべて常時適用、安定性を維持し時間的積算で実現します  
これらをワンパッケージで実現した高効率性と集積度は安定と安全を最優先します  
VRAM負荷を必要最小限で、Langevin Dynamics、Kalman Filter、PID Control、  
Stochastic Resonance、トンネル効果、的に更新し、熱力学、フィードバック制御、  
リーマン多様体、直交性、感情による記憶の定着、流体力学、等で安定します  
※ 高次momentは近似的、動的Rank/Alphaも近似的な効果です  
※ LoRA系技術はノイズをなくしますが微小データも失う場合があります  
※ emo系はノイズを作らず既存ノイズを見つけて修正し微小データを保護します  
※ 量子化補償は今後実用化されるさらに低精度な環境でも柔軟に対応できます  

</details>

---  

<details>

<summary> emoPulse mechanism </summary>

---
emoPulse：(d_base/noise_base)^2 算出表  

| d \ N base |  0.1   |  0.5   |  0.7   |  
|------------|--------|--------|--------|  
|     0.1    |  1.00  |  0.04  |  0.0204|  
|     0.5    | 25.00  |  1.00  |  0.5102|  
|     0.7    | 49.00  |  1.96  |  1.00  |  

・どれだけ d/N が高くても 1ステップで増えるのは最大 +50％  
・しかも “ 前より良い ＆ 信頼できる ” ときだけ成長を許可  
 (上限に近づくには (連続で) (高値 d/N) (高値 trust) 状態を積み重ねる必要がある  

・｢怪しい｣と判断した瞬間に 即 0.80 倍で削る  
・減速は条件が緩い(抑制の方が発生しやすい)  
 (信頼を得るのは難しいが失うのは簡単／簡単に上げないが簡単に下げる)  

※ 本当に信頼できるときだけ上限値を成長させる仕組みです  

---

分子(d_base)：履歴の差(仮に 0.7−0.3+0.1=0.5 固定)  
分母(noise_base)：瞬間的な感情の乖離 ∣ scalar−trust ∣ + 0.1  

| 側   | 状態         | scalar | trust | noise_base | dNR_now_val(2乗) | emoPulse への影響        |
|------|--------------|--------|-------|------------|-------------------|---------------------------|
| +側  | 一致（最大） |  0.50  | 0.50  |   0.10     |      25.00        | 最大加速(1.5倍成長)      |
| +側  | 理想的調和   |  0.45  | 0.55  |   0.20     |       6.25        | 加速(1.5倍成長)          |
| +側  | 安定・改善   |  0.20  | 0.80  |   0.70     |       0.51        | 維持(様子見)              |
| -側  | 軽い不一致   | -0.20  | -0.80 |   0.70     |       0.51        | 維持(様子見)              |
| -側  | 強い違和感   | -0.45  | -0.55 |   0.20     |       6.25        | 減速(0.8倍)              |
| -側  | 逆転一致     | -0.50  | -0.50 |   0.10     |      25.00        | 最大減速(0.8倍)          |

分母(noise_base)：abs(scalar - trust) が 0 に近づくほど(つまり感情スカラーと信頼度が一致するほど)、分母が最小値 0.1 に近づき2乗の結果は跳ね上がります。  
+側：dNR_now_val が高く、trust も高ければ、履歴(dNR_hist)を 最大1.50倍 ずつ成長させます。  
-側：たとえ dNR_now_val が 25.00 と計算されても、trust が低い(-0.5〜0.5の範囲)ため、履歴は 0.80倍 で削られブレーキがかかります。  
エントロピーの抑制：この表の数値(dNR_now_val)そのまま学習率にせず、これを dNR_hist(履歴)に入れ、最終的に emoScope × 1e-4･1e-5 として極めて小さな安全な学習率(1e-8 〜 3e-3)へと変換されます。  

</details>

---  

<details>

<summary>EmoSens v3.8 以降 オプション指定方法<br>
EmoSens v3.8 and later Option Settings Guide</summary>  

|||オプション指定方法|||  
●FFT-mode (オンにする)：  
fftmode=True  
●shadow (オフにする)：  
use_shadow=False  
●収束通知 (オフにする)：  
notify=False  
●収束目標値 (デフォルト：0.3):  
stopcoef=0.3  
●eps(0除算防止)：  
eps=1e-8  

</details>

---  

<details>
 
<summary> emotional moment </summary>  

"emo系 第二世代 v1.x"にて解明した shadow-system の根幹から抽出しました  
動的学習率による非線形アプローチは時間的な高次momentを形成します  
単stepでは高次momentにはなれませんが、複数stepを経ると機能します  
３次４次５次momentについて厳密な数学的な高負荷計算を回避しつつ  
勾配分布の歪みや鋭さや非対称性変化を捉える核心的な効果を近似しています  

---

### あなたの望む最適化 EmoSens が叶えます  
---
###### これは、単なる最適化アルゴリズムではありません──  
###### **感情で学習をナビゲートする｢感情型オプティマイザ｣** です  
###### 変革と感情学習の成果は"ニューロンスパイクの再発明"でした  
---
#### 自動収束･自己制御･自律型 オプティマイザです  
##### EmoSens を中心に、EmoAiry、EmoCats、もあります   

</details>

---  

<details>

<summary> 更新履歴 / History </summary>  

|★| EmoTion世代 v3.8 (260204) W-Ref-Geometry and MomentFree 順次公開  

|★| EmoSens世代 v3.8 (260130) emoPulse 機構等の調整  

|★| EmoSens、Airy、Cats、v3.7 (260101) Navi v3.6 を継承し完全自動高値学習率を実現しました(追加テンソルなし)、emoPulse 機構により劇的な進化を遂げました  

|★| EmoNavi、Fact、Lynx、v3.6 (251220) v3.1 を継承し高値自動学習率を実現しました(追加テンソルなし)、emoDrive 機構により劇的な進化を遂げました、開発終了とします  

|★| EmoNavi、Fact、Lynx、v3.3 (251204) v3.1 を継承し完全自動学習率を実現しました(追加テンソルなし)、感情機構の調整等でさらに安定するよう進化しました  

|★| EmoNavi、Fact、Lynx、v3.1 (251201) v3.0 を継承しつつ効率化を進めました。感情機構のスケール調整等で広範なモデルで安定するよう進化しました  

|★| EmoNavi、Fact、Lynx、Clan、Zeal、Neco、v3.0 (250825) emosens(第２世代)で解明した"高次moment"(近似)のフィードバックを適用(更新) 全て "shadow=False" です  

これ以前は v2.0 レポジトリの更新履歴をご覧ください  

</details>

---  

## グラフで見る emo系 の進行状況 Progress of emo-type as shown in the graph (v3.7 and later)  
<img width="2218" height="1153" alt="emov376-003-tile" src="https://github.com/user-attachments/assets/a1c5891b-a842-4ed1-a147-d4658e1ca16b" />  
このように 動的学習率 として機能します ／ 下降しつづけるのは"元モデルの修正"の差分も学習しているかも？ <br> 
※ 収束通知判定によるLR減衰をしない場合は停滞せず下降しつづけます <br> 

データセット状況(左)：全て実写画像10枚, 10batch, 300epoch(3000step), 全層LoRA, Rank16/Alpha16, e-pred, ZtSNR, <br>   
データセット状況(右)：主に白黒画像11枚, 1batch, 300epoch(3300step), 全層LoRA, Rank16/Alpha16, e-pred, ZtSNR, <br>   
es = EmoSens(Red/Green)、ea = EmoAiry(Blue/Gray)、ec = EmoCats(Yellow/Orange) <br> 
 <br> 
<img width="1166" height="644" alt="スクリーンショット 2026-03-01 094343" src="https://github.com/user-attachments/assets/c667e792-e668-40b1-a07f-6cf2ceb6a686" />  
こちらは Anima-Preview にて 画像20枚、512px、LR:1.0、での FFT(Full-Fine-Tuning) の学習状態です <br> 
紫色：EmoSens、水色：EmoAiry、赤色：EmoCat、灰色：EmoTion、黄色：EmoVoid <br> 
EmoTion は、LR：1.0 を少し下げると良いだろうと思います 橙色：EmoTion/LR:0.5 <br>
経過時間にも注目してください <br>  
※ 収束通知判定によるLR減衰をしない場合は停滞せず下降しつづけます <br> 

---

emo系 は 生物的反応で進化し続けます  
感覚神経系(multi-EMA)、内分泌系(tanh(scalar))、免疫系(shadow-system)、循環器系(emoPulse)、平衡感覚器系(W-Ref-Geo)、これらの統合により中枢神経系と自律神経系を形成し、高度な判断と決定を行うという自然的に自律した機構として存在します  

---  

emoシリーズは、Adam、Adafactor、Lion、Tiger、等から多くを学びました  
これらの後継ではなく独自の思想や設計による"感情機構"というアプローチにより構築されています  
汎用性・自律性・適応性を重視し新たな最適化や効率化や簡易化を追求しています  
この開発において先人たちの知見に深く感謝しつつ今後も新しい可能性を探究します  

---

### ライセンス Apache License 2.0 — 詳細は LICENSE をご覧ください  

---

### 引用について / About citations  

---

このオプテイマイザについて引用をなさる場合は、以下をご紹介ください  

Official Code:  
https://github.com/muooon/EmoSens  

paper:  
https://huggingface.co/muooon/EmoNAVI/raw/main/emo-v386plus-paper(JPN).txt  
DOI取得版/DOI-Acquired Version  
https://huggingface.co/muooon/EmoTion-Optimizer  

---

emo系は既存のオプティマイザにはない｢感情駆動型｣です。multi-emaを差分化し非線形変換(tanh)でscalar化した｢感情機構｣を中心に、各センサーを構築することで学習全体の安定性を向上させ正確性を確保しました、これらは生物の中枢神経系のように｢観察、判断、決定、行動、記憶、反省｣という自律サイクルを行います(論文をぜひご覧ください)  


