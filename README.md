## emo series Optimizers  

 Loss-Bypass (ECC) closure-unneeded   
- ###### EmoSens / 2ndGen (v3.9 / Standard-ECC)  
- ###### EmoTion / 3rdGen (v3.9 / Moment-Free-ECC)  
- ###### emo closure capture (ECC-System)  

readme：[English](README.md) | [日本語](README_JA.md)  

<img width="800" height="400" alt="Screenshot 2026-05-12 at 17-10-55 TensorBoard" src="https://github.com/user-attachments/assets/b478967a-ff15-4fb5-83f4-3588332a5784" />  

yellow:Void, purple:Tion, skyblue:Cats, orange:Airy, gray:Sens,  
SDXL:LoRA, Resolution:512, Rank:8, Alpha1, LR:1.0,  

---

<details>

<summary> EmoSens Full-log </summary>

Cosine scheduler LR “1e-4” is used as the baseline  

Analysis results of the learning rate (LR)  
Maximum LR: 7.5309 × 1e-4 (compared to the baseline 1e-4, it momentarily outputs about 7.5 times higher)  
Average LR: 2.0752 × 1e-4 (throughout the entire period, it continues learning at about twice the strength of the baseline)  
Minimum LR: 6.2770 × 1e-6 (the number of times it reached 1e-6 during the entire period was only a few, and it did not sustain it (concentrated in the early phase))  

When comparing the area (total learning amount) using “1e-4 Cosine” as the baseline, the results are as follows  
Total area of emoPulse: 0.0830083  
Total area of Cosine (1e-4): 0.0254647  
Comparison ratio: 325.97%  

The period in the latter half where it stagnated around “2.92e-4” (after READY TO STOP) corresponds to nearly three times the baseline output  
Within a short span of 400 steps, it progressed through the equivalent of about 1200–1300 steps of “learning” under normal conditions  

```prompt

----------

steps:  0%|                         | 0/400 [00:00<?, ?it/s] , Loss: 0.1308 | Pulse: 1.0000e-04
steps:  0%| | 1/400 [00:19<2:12:21, 19.90s/it, Average key norm=0.0022, Keys Scaled, Loss: 0.1306 | Pulse: 1.0000e-04
steps:  0%| | 2/400 [00:21<1:10:43, 10.66s/it, Average key norm=0.00484, Keys Scaled, Loss: 0.1344 | Pulse: 1.0000e-04
steps:  1%| | 3/400 [00:22<50:10, 7.58s/it, Average key norm=0.0078, Keys Scaled=0, Loss: 0.1228 | Pulse: 1.0000e-04
steps:  1%| | 4/400 [00:24<39:55, 6.05s/it, Average key norm=0.0112, Keys Scaled=0, Loss: 0.1237 | Pulse: 1.0000e-04
steps:  1%| | 5/400 [00:25<33:44, 5.12s/it, Average key norm=0.0148, Keys Scaled=0, Loss: 0.1222 | Pulse: 1.0000e-04
steps:  2%| | 6/400 [00:27<29:36, 4.51s/it, Average key norm=0.0184, Keys Scaled=0, Loss: 0.1241 | Pulse: 1.0000e-04
steps:  2%| | 7/400 [00:28<26:39, 4.07s/it, Average key norm=0.0221, Keys Scaled=0, Loss: 0.1132 | Pulse: 1.0000e-04
steps:  2%| | 8/400 [00:29<24:26, 3.74s/it, Average key norm=0.0258, Keys Scaled=0, Loss: 0.1108 | Pulse: 1.0000e-04
steps:  2%| | 9/400 [00:31<22:46, 3.49s/it, Average key norm=0.0295, Keys Scaled=0, Loss: 0.1033 | Pulse: 1.0000e-04
steps:  2%| | 10/400 [00:33<21:27, 3.30s/it, Average key norm=0.0329, Keys Scaled=0, Loss: 0.0952 | Pulse: 1.0000e-04

<details>

<summary> Full-log </summary>

steps:  3%| | 11/400 [00:34<20:19, 3.13s/it, Average key norm=0.0363, Keys Scaled=0, Loss: 0.0904 | Pulse: 1.0000e-04
steps:  3%| | 12/400 [00:36<19:24, 3.00s/it, Average key norm=0.0399, Keys Scaled=0, Loss: 0.0918 | Pulse: 1.0000e-04
steps:  3%| | 13/400 [00:37<18:37, 2.89s/it, Average key norm=0.0434, Keys Scaled=0, Loss: 0.0819 | Pulse: 1.0000e-04
steps:  4%| | 14/400 [00:39<17:58, 2.79s/it, Average key norm=0.0469, Keys Scaled=0, Loss: 0.0746 | Pulse: 1.0000e-04
steps:  4%| | 15/400 [00:40<17:21, 2.71s/it, Average key norm=0.0503, Keys Scaled=0, Loss: 0.0747 | Pulse: 1.0000e-04
steps:  4%| | 16/400 [00:42<16:52, 2.64s/it, Average key norm=0.0536, Keys Scaled=0, Loss: 0.0725 | Pulse: 1.0000e-04
steps:  4%| | 17/400 [00:43<16:24, 2.57s/it, Average key norm=0.0568, Keys Scaled=0, Loss: 0.0646 | Pulse: 1.2466e-04
steps:  4%| | 18/400 [00:45<16:00, 2.51s/it, Average key norm=0.0607, Keys Scaled=0, Loss: 0.0688 | Pulse: 1.3326e-04
steps:  5%| | 19/400 [00:46<15:39, 2.47s/it, Average key norm=0.0647, Keys Scaled=0, Loss: 0.0641 | Pulse: 1.5484e-04
steps:  5%| | 20/400 [00:48<15:20, 2.42s/it, Average key norm=0.0694, Keys Scaled=0, Loss: 0.0609 | Pulse: 2.3227e-04

steps:  5%| | 21/400 [00:50<15:03, 2.38s/it, Average key norm=0.0764, Keys Scaled=0, Loss: 0.0599 | Pulse: 2.3227e-04
steps:  6%| | 22/400 [00:51<14:45, 2.34s/it, Average key norm=0.0833, Keys Scaled=0, Loss: 0.0563 | Pulse: 2.8809e-04
steps:  6%| | 23/400 [00:53<14:30, 2.31s/it, Average key norm=0.0914, Keys Scaled=0, Loss: 0.0585 | Pulse: 3.0983e-04
steps:  6%| | 24/400 [00:54<14:16, 2.28s/it, Average key norm=0.1, Keys Scaled=0, Loss: 0.0525 | Pulse: 4.2403e-04
steps:  6%| | 25/400 [00:56<14:03, 2.25s/it, Average key norm=0.112, Keys Scaled=0, Loss: 0.0548 | Pulse: 4.7203e-04
steps:  6%| | 26/400 [00:57<13:50, 2.22s/it, Average key norm=0.126, Keys Scaled=0, Loss: 0.0525 | Pulse: 5.7090e-04
steps:  7%| | 27/400 [00:59<13:38, 2.19s/it, Average key norm=0.143, Keys Scaled=0, Loss: 0.0525 | Pulse: 6.4374e-04
steps:  7%| | 28/400 [01:00<13:26, 2.17s/it, Average key norm=0.161, Keys Scaled=0, Loss: 0.0511 | Pulse: 7.5309e-04
steps:  7%| | 29/400 [01:02<13:15, 2.14s/it, Average key norm=0.182, Keys Scaled=0, Loss: 0.0504 | Pulse: 5.4298e-04
steps:  8%| | 30/400 [01:03<13:05, 2.12s/it, Average key norm=0.197, Keys Scaled=1, Loss: 0.0484 | Pulse: 4.1740e-04
steps:  8%| | 31/400 [01:05<12:55, 2.10s/it, Average key norm=0.208, Keys Scaled=3, Loss: 0.0473 | Pulse: 3.3406e-04
steps:  8%| | 32/400 [01:06<12:46, 2.08s/it, Average key norm=0.216, Keys Scaled=4, Loss: 0.0446 | Pulse: 2.5402e-04
steps:  8%| | 33/400 [01:08<12:38, 2.07s/it, Average key norm=0.222, Keys Scaled=4, Loss: 0.0451 | Pulse: 2.2042e-04
steps:  8%| | 34/400 [01:09<12:29, 2.05s/it, Average key norm=0.227, Keys Scaled=5, Loss: 0.0442 | Pulse: 1.9342e-04
steps:  9%| | 35/400 [01:11<12:21, 2.03s/it, Average key norm=0.23, Keys Scaled=8, Loss: 0.0442 | Pulse: 1.7845e-04
steps:  9%| | 36/400 [01:12<12:14, 2.02s/it, Average key norm=0.233, Keys Scaled=13, Loss: 0.0432 | Pulse: 1.6311e-04
steps:  9%| | 37/400 [01:14<12:08, 2.01s/it, Average key norm=0.236, Keys Scaled=14, Loss: 0.0426 | Pulse: 1.5070e-04
steps: 10%| | 38/400 [01:15<12:01, 1.99s/it, Average key norm=0.238, Keys Scaled=15, Loss: 0.0433 | Pulse: 1.4781e-04
steps: 10%| | 39/400 [01:17<11:54, 1.98s/it, Average key norm=0.24, Keys Scaled=17, Loss: 0.0429 | Pulse: 1.4404e-04
steps: 10%| | 40/400 [01:18<11:47, 1.97s/it, Average key norm=0.242, Keys Scaled=18, Loss: 0.0440 | Pulse: 1.4922e-04

steps: 10%| | 41/400 [01:20<11:41, 1.95s/it, Average key norm=0.243, Keys Scaled=20, Loss: 0.0418 | Pulse: 1.4067e-04
steps: 10%| | 42/400 [01:21<11:35, 1.94s/it, Average key norm=0.244, Keys Scaled=22, Loss: 0.0420 | Pulse: 1.3704e-04
steps: 11%| | 43/400 [01:23<11:29, 1.93s/it, Average key norm=0.246, Keys Scaled=22, Loss: 0.0422 | Pulse: 1.3625e-04
steps: 11%| | 44/400 [01:24<11:24, 1.92s/it, Average key norm=0.247, Keys Scaled=23, Loss: 0.0422 | Pulse: 1.3673e-04
steps: 11%| | 45/400 [01:26<11:18, 1.91s/it, Average key norm=0.248, Keys Scaled=24, Loss: 0.0396 | Pulse: 1.2406e-04
steps: 12%| | 46/400 [01:27<11:13, 1.90s/it, Average key norm=0.249, Keys Scaled=25, Loss: 0.0398 | Pulse: 1.1812e-04
steps: 12%| | 47/400 [01:28<11:07, 1.89s/it, Average key norm=0.249, Keys Scaled=25, Loss: 0.0396 | Pulse: 1.1408e-04
steps: 12%| | 48/400 [01:30<11:03, 1.88s/it, Average key norm=0.25, Keys Scaled=26, Loss: 0.0394 | Pulse: 1.1134e-04
steps: 12%| | 49/400 [01:31<10:58, 1.88s/it, Average key norm=0.251, Keys Scaled=26, Loss: 0.0382 | Pulse: 1.0532e-04
steps: 12%|▏| 50/400 [01:33<10:54, 1.87s/it, Average key norm=0.251, Keys Scaled=26, Loss: 0.0398 | Pulse: 1.0853e-04

steps: 13%|▏| 51/400 [01:34<10:49, 1.86s/it, Average key norm=0.252, Keys Scaled=27, Loss: 0.0398 | Pulse: 1.1166e-04
steps: 13%|▏| 52/400 [01:36<10:46, 1.86s/it, Average key norm=0.253, Keys Scaled=27, Loss: 0.0390 | Pulse: 1.1074e-04
steps: 13%|▏| 53/400 [01:38<10:41, 1.85s/it, Average key norm=0.253, Keys Scaled=28, Loss: 0.0380 | Pulse: 1.0688e-04
steps: 14%|▏| 54/400 [01:39<10:37, 1.84s/it, Average key norm=0.254, Keys Scaled=29, Loss: 0.0374 | Pulse: 1.0263e-04
steps: 14%|▏| 55/400 [01:40<10:33, 1.84s/it, Average key norm=0.254, Keys Scaled=29, Loss: 0.0375 | Pulse: 1.0099e-04
steps: 14%|▏| 56/400 [01:42<10:30, 1.83s/it, Average key norm=0.255, Keys Scaled=29, Loss: 0.0376 | Pulse: 1.0090e-04
steps: 14%|▏| 57/400 [01:44<10:25, 1.82s/it, Average key norm=0.255, Keys Scaled=30, Loss: 0.0382 | Pulse: 1.0396e-04
steps: 14%|▏| 58/400 [01:45<10:22, 1.82s/it, Average key norm=0.255, Keys Scaled=30, Loss: 0.0382 | Pulse: 1.0709e-04
steps: 15%|▏| 59/400 [01:46<10:18, 1.81s/it, Average key norm=0.256, Keys Scaled=32, Loss: 0.0372 | Pulse: 1.0577e-04
steps: 15%|▏| 60/400 [01:48<10:14, 1.81s/it, Average key norm=0.256, Keys Scaled=32, Loss: 0.0370 | Pulse: 1.0467e-04

steps: 15%|▏| 61/400 [01:49<10:10, 1.80s/it, Average key norm=0.257, Keys Scaled=32, Loss: 0.0369 | Pulse: 1.0402e-04
steps: 16%|▏| 62/400 [01:51<10:07, 1.80s/it, Average key norm=0.257, Keys Scaled=32, Loss: 0.0372 | Pulse: 7.3844e-06
steps: 16%|▏| 63/400 [01:52<10:03, 1.79s/it, Average key norm=0.257, Keys Scaled=32, Loss: 0.0354 | Pulse: 1.1603e-04
steps: 16%|▏| 64/400 [01:54<10:00, 1.79s/it, Average key norm=0.257, Keys Scaled=32, Loss: 0.0357 | Pulse: 1.1349e-04
steps: 16%|▏| 65/400 [01:55<09:56, 1.78s/it, Average key norm=0.258, Keys Scaled=32, Loss: 0.0361 | Pulse: 1.1460e-04
steps: 16%|▏| 66/400 [01:57<09:53, 1.78s/it, Average key norm=0.258, Keys Scaled=33, Loss: 0.0352 | Pulse: 1.1138e-04
steps: 17%|▏| 67/400 [01:58<09:50, 1.77s/it, Average key norm=0.258, Keys Scaled=33, Loss: 0.0362 | Pulse: 1.1530e-04
steps: 17%|▏| 68/400 [02:00<09:47, 1.77s/it, Average key norm=0.259, Keys Scaled=33, Loss: 0.0349 | Pulse: 1.1222e-04
steps: 17%|▏| 69/400 [02:01<09:44, 1.76s/it, Average key norm=0.259, Keys Scaled=34, Loss: 0.0354 | Pulse: 1.1349e-04
steps: 18%|▏| 70/400 [02:03<09:40, 1.76s/it, Average key norm=0.26, Keys Scaled=33, Loss: 0.0352 | Pulse: 1.1392e-04

steps: 18%|▏| 71/400 [02:04<09:37, 1.76s/it, Average key norm=0.26, Keys Scaled=33, Loss: 0.0345 | Pulse: 1.1149e-04
steps: 18%|▏| 72/400 [02:06<09:34, 1.75s/it, Average key norm=0.26, Keys Scaled=34, Loss: 0.0341 | Pulse: 1.0843e-04
steps: 18%|▏| 73/400 [02:07<09:31, 1.75s/it, Average key norm=0.261, Keys Scaled=34, Loss: 0.0331 | Pulse: 1.0244e-04
steps: 18%|▏| 74/400 [02:09<09:28, 1.74s/it, Average key norm=0.261, Keys Scaled=34, Loss: 0.0337 | Pulse: 6.9032e-06
steps: 19%|▏| 75/400 [02:10<09:25, 1.74s/it, Average key norm=0.261, Keys Scaled=34, Loss: 0.0331 | Pulse: 1.1715e-04
steps: 19%|▏| 76/400 [02:11<09:22, 1.74s/it, Average key norm=0.261, Keys Scaled=34, Loss: 0.0328 | Pulse: 1.1420e-04
steps: 19%|▏| 77/400 [02:13<09:19, 1.73s/it, Average key norm=0.262, Keys Scaled=34, Loss: 0.0331 | Pulse: 1.1460e-04
steps: 20%|▏| 78/400 [02:14<09:16, 1.73s/it, Average key norm=0.262, Keys Scaled=34, Loss: 0.0323 | Pulse: 1.1168e-04
steps: 20%|▏| 79/400 [02:16<09:13, 1.73s/it, Average key norm=0.262, Keys Scaled=35, Loss: 0.0313 | Pulse: 1.0523e-04
steps: 20%|▏| 80/400 [02:17<09:11, 1.72s/it, Average key norm=0.263, Keys Scaled=35, Loss: 0.0325 | Pulse: 7.2875e-06

steps: 20%|▏| 81/400 [02:19<09:08, 1.72s/it, Average key norm=0.263, Keys Scaled=35, Loss: 0.0329 | Pulse: 1.3032e-04
steps: 20%|▏| 82/400 [02:20<09:05, 1.72s/it, Average key norm=0.263, Keys Scaled=35, Loss: 0.0318 | Pulse: 8.2303e-06
steps: 21%|▏| 83/400 [02:22<09:02, 1.71s/it, Average key norm=0.263, Keys Scaled=35, Loss: 0.0326 | Pulse: 1.5203e-04
steps: 21%|▏| 84/400 [02:23<09:00, 1.71s/it, Average key norm=0.263, Keys Scaled=35, Loss: 0.0309 | Pulse: 1.4329e-04
steps: 21%|▏| 85/400 [02:25<08:57, 1.71s/it, Average key norm=0.264, Keys Scaled=35, Loss: 0.0319 | Pulse: 1.4548e-04
steps: 22%|▏| 86/400 [02:26<08:54, 1.70s/it, Average key norm=0.264, Keys Scaled=34, Loss: 0.0316 | Pulse: 1.4570e-04
steps: 22%|▏| 87/400 [02:27<08:52, 1.70s/it, Average key norm=0.264, Keys Scaled=34, Loss: 0.0320 | Pulse: 8.7724e-06
steps: 22%|▏| 88/400 [02:29<08:49, 1.70s/it, Average key norm=0.264, Keys Scaled=34, Loss: 0.0324 | Pulse: 1.7942e-04
steps: 22%|▏| 89/400 [02:30<08:47, 1.70s/it, Average key norm=0.265, Keys Scaled=34, Loss: 0.0306 | Pulse: 9.6754e-06
steps: 22%|▏| 90/400 [02:32<08:45, 1.69s/it, Average key norm=0.265, Keys Scaled=34, Loss: 0.0302 | Pulse: 1.8236e-04

steps: 23%|▏| 91/400 [02:34<08:43, 1.69s/it, Average key norm=0.265, Keys Scaled=34, Loss: 0.0305 | Pulse: 1.7821e-04
steps: 23%|▏| 92/400 [02:35<08:41, 1.69s/it, Average key norm=0.266, Keys Scaled=35, Loss: 0.0316 | Pulse: 1.0379e-05
steps: 23%|▏| 93/400 [02:37<08:38, 1.69s/it, Average key norm=0.266, Keys Scaled=35, Loss: 0.0301 | Pulse: 2.0496e-04
steps: 24%|▏| 94/400 [02:38<08:36, 1.69s/it, Average key norm=0.266, Keys Scaled=35, Loss: 0.0303 | Pulse: 2.0143e-04
steps: 24%|▏| 95/400 [02:40<08:34, 1.69s/it, Average key norm=0.267, Keys Scaled=35, Loss: 0.0302 | Pulse: 1.9880e-04
steps: 24%|▏| 96/400 [02:41<08:32, 1.69s/it, Average key norm=0.267, Keys Scaled=35, Loss: 0.0303 | Pulse: 1.9904e-04
steps: 24%|▏| 97/400 [02:43<08:30, 1.68s/it, Average key norm=0.268, Keys Scaled=36, Loss: 0.0306 | Pulse: 2.0284e-04
steps: 24%|▏| 98/400 [02:45<08:28, 1.68s/it, Average key norm=0.268, Keys Scaled=36, Loss: 0.0309 | Pulse: 2.1101e-04
steps: 25%|▏| 99/400 [02:46<08:26, 1.68s/it, Average key norm=0.268, Keys Scaled=36, Loss: 0.0301 | Pulse: 1.0287e-05
steps: 25%|▎| 100/400 [02:48<08:24, 1.68s/it, Average key norm=0.268, Keys Scaled=3, Loss: 0.0288 | Pulse: 2.1741e-04

steps: 25%|▎| 101/400 [02:49<08:22, 1.68s/it, Average key norm=0.269, Keys Scaled=3, Loss: 0.0285 | Pulse: 2.0149e-04
steps: 26%|▎| 102/400 [02:51<08:20, 1.68s/it, Average key norm=0.269, Keys Scaled=3, Loss: 0.0292 | Pulse: 2.0041e-04
steps: 26%|▎| 103/400 [02:52<08:18, 1.68s/it, Average key norm=0.27, Keys Scaled=34, Loss: 0.0290 | Pulse: 1.9811e-04
steps: 26%|▎| 104/400 [02:54<08:16, 1.68s/it, Average key norm=0.27, Keys Scaled=35, Loss: 0.0289 | Pulse: 1.9573e-04
steps: 26%|▎| 105/400 [02:56<08:14, 1.68s/it, Average key norm=0.271, Keys Scaled=3, Loss: 0.0292 | Pulse: 1.9845e-04
steps: 26%|▎| 106/400 [02:57<08:12, 1.68s/it, Average key norm=0.271, Keys Scaled=3, Loss: 0.0294 | Pulse: 2.0447e-04
steps: 27%|▎| 107/400 [02:59<08:10, 1.68s/it, Average key norm=0.272, Keys Scaled=3, Loss: 0.0283 | Pulse: 1.9577e-04
steps: 27%|▎| 108/400 [03:00<08:08, 1.67s/it, Average key norm=0.272, Keys Scaled=3, Loss: 0.0282 | Pulse: 1.9004e-04
steps: 27%|▎| 109/400 [03:02<08:06, 1.67s/it, Average key norm=0.272, Keys Scaled=3, Loss: 0.0283 | Pulse: 1.8780e-04
steps: 28%|▎| 110/400 [03:03<08:04, 1.67s/it, Average key norm=0.272, Keys Scaled=3, Loss: 0.0281 | Pulse: 1.8502e-04

steps: 28%|▎| 111/400 [03:05<08:02, 1.67s/it, Average key norm=0.273, Keys Scaled=3, Loss: 0.0278 | Pulse: 1.7988e-04
steps: 28%|▎| 112/400 [03:06<08:00, 1.67s/it, Average key norm=0.273, Keys Scaled=3, Loss: 0.0269 | Pulse: 1.6769e-04
steps: 28%|▎| 113/400 [03:08<07:58, 1.67s/it, Average key norm=0.273, Keys Scaled=3, Loss: 0.0266 | Pulse: 1.5764e-04
steps: 28%|▎| 114/400 [03:09<07:55, 1.66s/it, Average key norm=0.274, Keys Scaled=3, Loss: 0.0271 | Pulse: 1.5719e-04
steps: 29%|▎| 115/400 [03:11<07:53, 1.66s/it, Average key norm=0.274, Keys Scaled=3, Loss: 0.0283 | Pulse: 1.6985e-04
steps: 29%|▎| 116/400 [03:12<07:52, 1.66s/it, Average key norm=0.274, Keys Scaled=3, Loss: 0.0270 | Pulse: 8.3102e-06
steps: 29%|▎| 117/400 [03:14<07:50, 1.66s/it, Average key norm=0.274, Keys Scaled=3, Loss: 0.0277 | Pulse: 2.0052e-04
steps: 30%|▎| 118/400 [03:15<07:48, 1.66s/it, Average key norm=0.275, Keys Scaled=3, Loss: 0.0257 | Pulse: 1.7965e-04
steps: 30%|▎| 119/400 [03:17<07:45, 1.66s/it, Average key norm=0.275, Keys Scaled=3, Loss: 0.0264 | Pulse: 8.9779e-06
steps: 30%|▎| 120/400 [03:18<07:43, 1.66s/it, Average key norm=0.275, Keys Scaled=3, Loss: 0.0261 | Pulse: 1.9589e-04

steps: 30%|▎| 121/400 [03:20<07:42, 1.66s/it, Average key norm=0.275, Keys Scaled=3, Loss: 0.0264 | Pulse: 1.9682e-04
steps: 30%|▎| 122/400 [03:21<07:40, 1.66s/it, Average key norm=0.275, Keys Scaled=3, Loss: 0.0267 | Pulse: 2.0223e-04
steps: 31%|▎| 123/400 [03:23<07:38, 1.65s/it, Average key norm=0.276, Keys Scaled=3, Loss: 0.0257 | Pulse: 1.9259e-04
steps: 31%|▎| 124/400 [03:24<07:36, 1.65s/it, Average key norm=0.276, Keys Scaled=3, Loss: 0.0269 | Pulse: 2.0448e-04
steps: 31%|▎| 125/400 [03:26<07:34, 1.65s/it, Average key norm=0.276, Keys Scaled=3, Loss: 0.0265 | Pulse: 2.0817e-04
steps: 32%|▎| 126/400 [03:27<07:31, 1.65s/it, Average key norm=0.277, Keys Scaled=3, Loss: 0.0264 | Pulse: 2.1103e-04
steps: 32%|▎| 127/400 [03:29<07:30, 1.65s/it, Average key norm=0.277, Keys Scaled=3, Loss: 0.0263 | Pulse: 2.1292e-04
steps: 32%|▎| 128/400 [03:30<07:28, 1.65s/it, Average key norm=0.277, Keys Scaled=3, Loss: 0.0258 | Pulse: 8.8880e-06
steps: 32%|▎| 129/400 [03:32<07:26, 1.65s/it, Average key norm=0.277, Keys Scaled=3, Loss: 0.0256 | Pulse: 2.3207e-04
steps: 32%|▎| 130/400 [03:33<07:24, 1.65s/it, Average key norm=0.278, Keys Scaled=3, Loss: 0.0255 | Pulse: 2.2571e-04

steps: 33%|▎| 131/400 [03:35<07:22, 1.64s/it, Average key norm=0.278, Keys Scaled=3, Loss: 0.0252 | Pulse: 2.1761e-04
steps: 33%|▎| 132/400 [03:36<07:20, 1.64s/it, Average key norm=0.278, Keys Scaled=3, Loss: 0.0257 | Pulse: 2.2192e-04
steps: 33%|▎| 133/400 [03:38<07:18, 1.64s/it, Average key norm=0.278, Keys Scaled=3, Loss: 0.0254 | Pulse: 2.2131e-04
steps: 34%|▎| 134/400 [03:39<07:16, 1.64s/it, Average key norm=0.279, Keys Scaled=3, Loss: 0.0265 | Pulse: 2.4510e-04
steps: 34%|▎| 135/400 [03:41<07:14, 1.64s/it, Average key norm=0.279, Keys Scaled=3, Loss: 0.0253 | Pulse: 2.3626e-04
steps: 34%|▎| 136/400 [03:42<07:12, 1.64s/it, Average key norm=0.279, Keys Scaled=2, Loss: 0.0246 | Pulse: 2.1777e-04
steps: 34%|▎| 137/400 [03:44<07:11, 1.64s/it, Average key norm=0.279, Keys Scaled=3, Loss: 0.0259 | Pulse: 2.3436e-04
steps: 34%|▎| 138/400 [03:46<07:09, 1.64s/it, Average key norm=0.28, Keys Scaled=34, Loss: 0.0244 | Pulse: 2.1617e-04
steps: 35%|▎| 139/400 [03:47<07:07, 1.64s/it, Average key norm=0.28, Keys Scaled=34, Loss: 0.0256 | Pulse: 2.2947e-04
steps: 35%|▎| 140/400 [03:49<07:05, 1.64s/it, Average key norm=0.28, Keys Scaled=3, Loss: 0.0257 | Pulse: 2.4538e-04

steps: 35%|▎| 141/400 [03:50<07:03, 1.64s/it, Average key norm=0.281, Keys Scaled=3, Loss: 0.0252 | Pulse: 2.4652e-04
steps: 36%|▎| 142/400 [03:52<07:01, 1.63s/it, Average key norm=0.281, Keys Scaled=3, Loss: 0.0242 | Pulse: 2.2303e-04
steps: 36%|▎| 143/400 [03:53<06:59, 1.63s/it, Average key norm=0.281, Keys Scaled=3, Loss: 0.0250 | Pulse: 2.2845e-04
steps: 36%|▎| 144/400 [03:55<06:57, 1.63s/it, Average key norm=0.281, Keys Scaled=3, Loss: 0.0252 | Pulse: 2.4006e-04
steps: 36%|▎| 145/400 [03:56<06:56, 1.63s/it, Average key norm=0.282, Keys Scaled=3, Loss: 0.0248 | Pulse: 2.3979e-04
steps: 36%|▎| 146/400 [03:58<06:54, 1.63s/it, Average key norm=0.282, Keys Scaled=3, Loss: 0.0248 | Pulse: 2.4194e-04
steps: 37%|▎| 147/400 [03:59<06:52, 1.63s/it, Average key norm=0.282, Keys Scaled=3, Loss: 0.0255 | Pulse: 2.6838e-04
steps: 37%|▎| 148/400 [04:00<06:50, 1.63s/it, Average key norm=0.283, Keys Scaled=3, Loss: 0.0242 | Pulse: 2.4817e-04
steps: 37%|▎| 149/400 [04:02<06:48, 1.63s/it, Average key norm=0.283, Keys Scaled=3, Loss: 0.0246 | Pulse: 2.4791e-04
steps: 38%|▍| 150/400 [04:03<06:46, 1.63s/it, Average key norm=0.283, Keys Scaled=3, Loss: 0.0242 | Pulse: 2.3854e-04

steps: 38%|▍| 151/400 [04:05<06:44, 1.62s/it, Average key norm=0.284, Keys Scaled=3, Loss: 0.0236 | Pulse: 2.1744e-04
steps: 38%|▍| 152/400 [04:06<06:42, 1.62s/it, Average key norm=0.284, Keys Scaled=3, Loss: 0.0234 | Pulse: 2.0242e-04
steps: 38%|▍| 153/400 [04:08<06:40, 1.62s/it, Average key norm=0.284, Keys Scaled=3, Loss: 0.0232 | Pulse: 1.8985e-04
steps: 38%|▍| 154/400 [04:09<06:38, 1.62s/it, Average key norm=0.284, Keys Scaled=3, Loss: 0.0233 | Pulse: 1.8665e-04
steps: 39%|▍| 155/400 [04:11<06:36, 1.62s/it, Average key norm=0.284, Keys Scaled=3, Loss: 0.0235 | Pulse: 6.2770e-06
steps: 39%|▍| 156/400 [04:12<06:34, 1.62s/it, Average key norm=0.284, Keys Scaled=3, Loss: 0.0240 | Pulse: 2.4857e-04
steps: 39%|▍| 157/400 [04:13<06:33, 1.62s/it, Average key norm=0.285, Keys Scaled=2, Loss: 0.0220 | Pulse: 2.0782e-04
steps: 40%|▍| 158/400 [04:15<06:31, 1.62s/it, Average key norm=0.285, Keys Scaled=3, Loss: 0.0221 | Pulse: 1.8830e-04
steps: 40%|▍| 159/400 [04:16<06:29, 1.62s/it, Average key norm=0.285, Keys Scaled=3, Loss: 0.0220 | Pulse: 1.7584e-04
steps: 40%|▍| 160/400 [04:18<06:27, 1.61s/it, Average key norm=0.285, Keys Scaled=3, Loss: 0.0226 | Pulse: 1.8139e-04

steps: 40%|▍| 161/400 [04:19<06:25, 1.61s/it, Average key norm=0.286, Keys Scaled=3, Loss: 0.0223 | Pulse: 1.7922e-04
steps: 40%|▍| 162/400 [04:21<06:23, 1.61s/it, Average key norm=0.286, Keys Scaled=3, Loss: 0.0227 | Pulse: 1.8971e-04
steps: 41%|▍| 163/400 [04:22<06:21, 1.61s/it, Average key norm=0.286, Keys Scaled=3, Loss: 0.0221 | Pulse: 6.3559e-06
steps: 41%|▍| 164/400 [04:24<06:20, 1.61s/it, Average key norm=0.286, Keys Scaled=3, Loss: 0.0231 | Pulse: 2.5067e-04
steps: 41%|▍| 165/400 [04:25<06:18, 1.61s/it, Average key norm=0.286, Keys Scaled=3, Loss: 0.0216 | Pulse: 2.2528e-04
steps: 42%|▍| 166/400 [04:26<06:16, 1.61s/it, Average key norm=0.287, Keys Scaled=3, Loss: 0.0221 | Pulse: 2.2431e-04
steps: 42%|▍| 167/400 [04:28<06:14, 1.61s/it, Average key norm=0.287, Keys Scaled=3, Loss: 0.0224 | Pulse: 2.3731e-04
steps: 42%|▍| 168/400 [04:29<06:12, 1.61s/it, Average key norm=0.287, Keys Scaled=3, Loss: 0.0227 | Pulse: 2.6001e-04
steps: 42%|▍| 169/400 [04:31<06:11, 1.61s/it, Average key norm=0.288, Keys Scaled=3, Loss: 0.0234 | Pulse: 3.1669e-04
steps: 42%|▍| 170/400 [04:33<06:09, 1.61s/it, Average key norm=0.288, Keys Scaled=3, Loss: 0.0228 | Pulse: 3.7212e-04

steps: 43%|▍| 171/400 [04:34<06:07, 1.61s/it, Average key norm=0.288, Keys Scaled=3, Loss: 0.0230 | Pulse: 3.7212e-04
steps: 43%|▍| 172/400 [04:36<06:05, 1.61s/it, Average key norm=0.289, Keys Scaled=3, Loss: 0.0235 | Pulse: 7.0346e-06
steps: 43%|▍| 173/400 [04:37<06:04, 1.60s/it, Average key norm=0.289, Keys Scaled=3, Loss: 0.0234 | Pulse: 1.0552e-05
steps: 44%|▍| 174/400 [04:39<06:02, 1.60s/it, Average key norm=0.289, Keys Scaled=3, Loss: 0.0225 | Pulse: 1.5828e-05
steps: 44%|▍| 175/400 [04:40<06:00, 1.60s/it, Average key norm=0.289, Keys Scaled=3, Loss: 0.0222 | Pulse: 7.6929e-06
steps: 44%|▍| 176/400 [04:42<05:58, 1.60s/it, Average key norm=0.289, Keys Scaled=3, Loss: 0.0237 | Pulse: 1.1539e-05
steps: 44%|▍| 177/400 [04:43<05:57, 1.60s/it, Average key norm=0.289, Keys Scaled=3, Loss: 0.0225 | Pulse: 1.7309e-05
steps: 44%|▍| 178/400 [04:45<05:55, 1.60s/it, Average key norm=0.289, Keys Scaled=3, Loss: 0.0222 | Pulse: 2.5963e-05
steps: 45%|▍| 179/400 [04:46<05:53, 1.60s/it, Average key norm=0.289, Keys Scaled=2, Loss: 0.0217 | Pulse: 3.8945e-05
steps: 45%|▍| 180/400 [04:48<05:52, 1.60s/it, Average key norm=0.289, Keys Scaled=2, Loss: 0.0228 | Pulse: 8.0577e-06

steps: 45%|▍| 181/400 [04:49<05:50, 1.60s/it, Average key norm=0.289, Keys Scaled=2, Loss: 0.0215 | Pulse: 1.2087e-05
steps: 46%|▍| 182/400 [04:51<05:48, 1.60s/it, Average key norm=0.289, Keys Scaled=2, Loss: 0.0204 | Pulse: 1.8130e-05
steps: 46%|▍| 183/400 [04:52<05:46, 1.60s/it, Average key norm=0.289, Keys Scaled=2, Loss: 0.0207 | Pulse: 2.7195e-05
steps: 46%|▍| 184/400 [04:54<05:45, 1.60s/it, Average key norm=0.289, Keys Scaled=2, Loss: 0.0223 | Pulse: 4.0792e-05
steps: 46%|▍| 185/400 [04:55<05:43, 1.60s/it, Average key norm=0.289, Keys Scaled=2, Loss: 0.0207 | Pulse: 6.1188e-05
steps: 46%|▍| 186/400 [04:57<05:41, 1.60s/it, Average key norm=0.289, Keys Scaled=3, Loss: 0.0204 | Pulse: 9.1782e-05
steps: 47%|▍| 187/400 [04:58<05:40, 1.60s/it, Average key norm=0.289, Keys Scaled=3, Loss: 0.0214 | Pulse: 1.3767e-04
steps: 47%|▍| 188/400 [05:00<05:38, 1.60s/it, Average key norm=0.289, Keys Scaled=3, Loss: 0.0218 | Pulse: 2.0651e-04
steps: 47%|▍| 189/400 [05:01<05:36, 1.60s/it, Average key norm=0.289, Keys Scaled=3, Loss: 0.0223 | Pulse: 2.0651e-04
steps: 48%|▍| 190/400 [05:03<05:35, 1.60s/it, Average key norm=0.289, Keys Scaled=3, Loss: 0.0220 | Pulse: 2.0651e-04

steps: 48%|▍| 191/400 [05:04<05:33, 1.60s/it, Average key norm=0.29, Keys Scaled=38, Loss: 0.0225 | Pulse: 2.0651e-04
steps: 48%|▍| 192/400 [05:06<05:31, 1.59s/it, Average key norm=0.29, Keys Scaled=37, Loss: 0.0237 | Pulse: 2.0651e-04
steps: 48%|▍| 193/400 [05:07<05:29, 1.59s/it, Average key norm=0.29, Keys Scaled=38, Loss: 0.0221 | Pulse: 2.0651e-04
steps: 48%|▍| 194/400 [05:09<05:28, 1.59s/it, Average key norm=0.29, Keys Scaled=39, Loss: 0.0233 | Pulse: 2.0651e-04
steps: 49%|▍| 195/400 [05:10<05:26, 1.59s/it, Average key norm=0.29, Keys Scaled=39, Loss: 0.0249 | Pulse: 2.0651e-04
steps: 49%|▍| 196/400 [05:12<05:24, 1.59s/it, Average key norm=0.29, Keys Scaled=36, Loss: 0.0223 | Pulse: 2.0651e-04
steps: 49%|▍| 197/400 [05:13<05:23, 1.59s/it, Average key norm=0.291, Keys Scaled=3, Loss: 0.0217 | Pulse: 2.0651e-04
steps: 50%|▍| 198/400 [05:15<05:21, 1.59s/it, Average key norm=0.291, Keys Scaled=3, Loss: 0.0236 | Pulse: 2.0651e-04
steps: 50%|▍| 199/400 [05:16<05:19, 1.59s/it, Average key norm=0.291, Keys Scaled=3, Loss: 0.0234 | Pulse: 2.0651e-04
steps: 50%|▌| 200/400 [05:18<05:18, 1.59s/it, Average key norm=0.291, Keys Scaled=3, Loss: 0.0225 | Pulse: 2.0651e-04

steps: 50%|▌| 201/400 [05:19<05:16, 1.59s/it, Average key norm=0.291, Keys Scaled=3, Loss: 0.0214 | Pulse: 2.0651e-04
steps: 50%|▌| 202/400 [05:21<05:14, 1.59s/it, Average key norm=0.291, Keys Scaled=3, Loss: 0.0212 | Pulse: 2.0651e-04
steps: 51%|▌| 203/400 [05:22<05:13, 1.59s/it, Average key norm=0.292, Keys Scaled=3, Loss: 0.0211 | Pulse: 2.0651e-04
steps: 51%|▌| 204/400 [05:24<05:11, 1.59s/it, Average key norm=0.292, Keys Scaled=3, Loss: 0.0216 | Pulse: 2.0651e-04
steps: 51%|▌| 205/400 [05:25<05:09, 1.59s/it, Average key norm=0.292, Keys Scaled=3, Loss: 0.0207 | Pulse: 2.0651e-04
steps: 52%|▌| 206/400 [05:27<05:08, 1.59s/it, Average key norm=0.292, Keys Scaled=3, Loss: 0.0211 | Pulse: 2.0651e-04
steps: 52%|▌| 207/400 [05:28<05:06, 1.59s/it, Average key norm=0.292, Keys Scaled=3, Loss: 0.0210 | Pulse: 2.0651e-04
steps: 52%|▌| 208/400 [05:30<05:04, 1.59s/it, Average key norm=0.293, Keys Scaled=3, Loss: 0.0217 | Pulse: 2.0651e-04
steps: 52%|▌| 209/400 [05:31<05:03, 1.59s/it, Average key norm=0.293, Keys Scaled=3, Loss: 0.0211 | Pulse: 2.0651e-04
steps: 52%|▌| 210/400 [05:33<05:01, 1.59s/it, Average key norm=0.293, Keys Scaled=3, Loss: 0.0220 | Pulse: 2.0651e-04

steps: 53%|▌| 211/400 [05:34<04:59, 1.59s/it, Average key norm=0.293, Keys Scaled=3, Loss: 0.0219 | Pulse: 2.0651e-04
steps: 53%|▌| 212/400 [05:36<04:58, 1.59s/it, Average key norm=0.294, Keys Scaled=3, Loss: 0.0208 | Pulse: 2.0651e-04
steps: 53%|▌| 213/400 [05:37<04:56, 1.58s/it, Average key norm=0.294, Keys Scaled=3, Loss: 0.0208 | Pulse: 2.0651e-04
steps: 54%|▌| 214/400 [05:38<04:54, 1.58s/it, Average key norm=0.294, Keys Scaled=3, Loss: 0.0210 | Pulse: 2.0651e-04
steps: 54%|▌| 215/400 [05:40<04:53, 1.58s/it, Average key norm=0.294, Keys Scaled=3, Loss: 0.0212 | Pulse: 2.0651e-04
steps: 54%|▌| 216/400 [05:42<04:51, 1.58s/it, Average key norm=0.295, Keys Scaled=3, Loss: 0.0214 | Pulse: 2.0651e-04
steps: 54%|▌| 217/400 [05:43<04:49, 1.58s/it, Average key norm=0.295, Keys Scaled=3, Loss: 0.0210 | Pulse: 2.0651e-04
steps: 55%|▌| 218/400 [05:45<04:48, 1.58s/it, Average key norm=0.295, Keys Scaled=3, Loss: 0.0205 | Pulse: 2.0651e-04
steps: 55%|▌| 219/400 [05:46<04:46, 1.58s/it, Average key norm=0.295, Keys Scaled=3, Loss: 0.0202 | Pulse: 2.0651e-04
steps: 55%|▌| 220/400 [05:47<04:44, 1.58s/it, Average key norm=0.296, Keys Scaled=3, Loss: 0.0204 | Pulse: 2.0651e-04

steps: 55%|▌| 221/400 [05:49<04:43, 1.58s/it, Average key norm=0.296, Keys Scaled=4, Loss: 0.0205 | Pulse: 2.0651e-04
steps: 56%|▌| 222/400 [05:50<04:41, 1.58s/it, Average key norm=0.296, Keys Scaled=4, Loss: 0.0213 | Pulse: 2.0651e-04
steps: 56%|▌| 223/400 [05:52<04:39, 1.58s/it, Average key norm=0.296, Keys Scaled=4, Loss: 0.0205 | Pulse: 2.0651e-04
steps: 56%|▌| 224/400 [05:53<04:37, 1.58s/it, Average key norm=0.296, Keys Scaled=4, Loss: 0.0201 | Pulse: 2.0651e-04
steps: 56%|▌| 225/400 [05:55<04:36, 1.58s/it, Average key norm=0.297, Keys Scaled=4, Loss: 0.0203 | Pulse: 2.0651e-04
steps: 56%|▌| 226/400 [05:56<04:34, 1.58s/it, Average key norm=0.297, Keys Scaled=4, Loss: 0.0216 | Pulse: 2.0651e-04
steps: 57%|▌| 227/400 [05:58<04:32, 1.58s/it, Average key norm=0.297, Keys Scaled=4, Loss: 0.0213 | Pulse: 2.0651e-04
steps: 57%|▌| 228/400 [05:59<04:31, 1.58s/it, Average key norm=0.297, Keys Scaled=4, Loss: 0.0202 | Pulse: 2.0651e-04
steps: 57%|▌| 229/400 [06:01<04:29, 1.58s/it, Average key norm=0.297, Keys Scaled=4, Loss: 0.0207 | Pulse: 2.0651e-04
steps: 57%|▌| 230/400 [06:02<04:27, 1.58s/it, Average key norm=0.297, Keys Scaled=3, Loss: 0.0191 | Pulse: 2.0651e-04

steps: 58%|▌| 231/400 [06:03<04:26, 1.58s/it, Average key norm=0.298, Keys Scaled=3, Loss: 0.0196 | Pulse: 2.0651e-04
steps: 58%|▌| 232/400 [06:05<04:24, 1.57s/it, Average key norm=0.298, Keys Scaled=3, Loss: 0.0205 | Pulse: 2.0651e-04
steps: 58%|▌| 233/400 [06:06<04:22, 1.57s/it, Average key norm=0.298, Keys Scaled=3, Loss: 0.0203 | Pulse: 2.0651e-04
steps: 58%|▌| 234/400 [06:08<04:21, 1.57s/it, Average key norm=0.298, Keys Scaled=3, Loss: 0.0213 | Pulse: 2.0651e-04
steps: 59%|▌| 235/400 [06:09<04:19, 1.57s/it, Average key norm=0.298, Keys Scaled=3, Loss: 0.0195 | Pulse: 2.0651e-04
steps: 59%|▌| 236/400 [06:11<04:17, 1.57s/it, Average key norm=0.298, Keys Scaled=3, Loss: 0.0193 | Pulse: 2.0651e-04
steps: 59%|▌| 237/400 [06:12<04:16, 1.57s/it, Average key norm=0.298, Keys Scaled=3, Loss: 0.0194 | Pulse: 2.0651e-04
steps: 60%|▌| 238/400 [06:13<04:14, 1.57s/it, Average key norm=0.298, Keys Scaled=4, Loss: 0.0194 | Pulse: 2.0651e-04
steps: 60%|▌| 239/400 [06:15<04:12, 1.57s/it, Average key norm=0.299, Keys Scaled=4, Loss: 0.0188 | Pulse: 2.0651e-04
steps: 60%|▌| 240/400 [06:16<04:11, 1.57s/it, Average key norm=0.299, Keys Scaled=4, Loss: 0.0192 | Pulse: 2.0651e-04

steps: 60%|▌| 241/400 [06:18<04:09, 1.57s/it, Average key norm=0.299, Keys Scaled=4, Loss: 0.0204 | Pulse: 2.0651e-04
steps: 60%|▌| 242/400 [06:19<04:08, 1.57s/it, Average key norm=0.299, Keys Scaled=4, Loss: 0.0196 | Pulse: 1.7786e-05
steps: 61%|▌| 243/400 [06:21<04:06, 1.57s/it, Average key norm=0.299, Keys Scaled=4, Loss: 0.0198 | Pulse: 2.6679e-05
steps: 61%|▌| 244/400 [06:22<04:04, 1.57s/it, Average key norm=0.299, Keys Scaled=3, Loss: 0.0214 | Pulse: 3.7742e-05
steps: 61%|▌| 245/400 [06:24<04:03, 1.57s/it, Average key norm=0.299, Keys Scaled=3, Loss: 0.0195 | Pulse: 4.0589e-05
steps: 62%|▌| 246/400 [06:26<04:01, 1.57s/it, Average key norm=0.299, Keys Scaled=2, Loss: 0.0200 | Pulse: 4.0870e-05
steps: 62%|▌| 247/400 [06:27<04:00, 1.57s/it, Average key norm=0.299, Keys Scaled=2, Loss: 0.0192 | Pulse: 4.4422e-05
steps: 62%|▌| 248/400 [06:29<03:58, 1.57s/it, Average key norm=0.299, Keys Scaled=2, Loss: 0.0189 | Pulse: 4.8433e-05
steps: 62%|▌| 249/400 [06:30<03:56, 1.57s/it, Average key norm=0.299, Keys Scaled=2, Loss: 0.0186 | Pulse: 5.3003e-05
steps: 62%|▋| 250/400 [06:32<03:55, 1.57s/it, Average key norm=0.299, Keys Scaled=2, Loss: 0.0182 | Pulse: 5.8079e-05

steps: 63%|▋| 251/400 [06:33<03:53, 1.57s/it, Average key norm=0.299, Keys Scaled=2, Loss: 0.0197 | Pulse: 5.8079e-05
steps: 63%|▋| 252/400 [06:35<03:52, 1.57s/it, Average key norm=0.299, Keys Scaled=2, Loss: 0.0180 | Pulse: 5.9097e-05
steps: 63%|▋| 253/400 [06:36<03:50, 1.57s/it, Average key norm=0.299, Keys Scaled=2, Loss: 0.0200 | Pulse: 2.7995e-05
steps: 64%|▋| 254/400 [06:38<03:49, 1.57s/it, Average key norm=0.299, Keys Scaled=2, Loss: 0.0195 | Pulse: 4.1993e-05
steps: 64%|▋| 255/400 [06:40<03:47, 1.57s/it, Average key norm=0.299, Keys Scaled=2, Loss: 0.0186 | Pulse: 4.7574e-05
steps: 64%|▋| 256/400 [06:41<03:45, 1.57s/it, Average key norm=0.299, Keys Scaled=2, Loss: 0.0191 | Pulse: 4.7598e-05
steps: 64%|▋| 257/400 [06:43<03:44, 1.57s/it, Average key norm=0.299, Keys Scaled=2, Loss: 0.0186 | Pulse: 4.9615e-05
steps: 64%|▋| 258/400 [06:44<03:42, 1.57s/it, Average key norm=0.3, Keys Scaled=31, Loss: 0.0191 | Pulse: 4.9615e-05
steps: 65%|▋| 259/400 [06:46<03:41, 1.57s/it, Average key norm=0.3, Keys Scaled=32, Loss: 0.0191 | Pulse: 4.9615e-05
steps: 65%|▋| 260/400 [06:47<03:39, 1.57s/it, Average key norm=0.3, Keys Scaled=34, Loss: 0.0190 | Pulse: 4.9615e-05

steps: 65%|▋| 261/400 [06:49<03:37, 1.57s/it, Average key norm=0.3, Keys Scaled=35, Loss: 0.0189 | Pulse: 4.9615e-05
steps: 66%|▋| 262/400 [06:50<03:36, 1.57s/it, Average key norm=0.3, Keys Scaled=35, Loss: 0.0184 | Pulse: 5.1794e-05
steps: 66%|▋| 263/400 [06:52<03:34, 1.57s/it, Average key norm=0.3, Keys Scaled=38, Loss: 0.0196 | Pulse: 5.1794e-05
steps: 66%|▋| 264/400 [06:53<03:33, 1.57s/it, Average key norm=0.3, Keys Scaled=38, Loss: 0.0186 | Pulse: 5.1794e-05
steps: 66%|▋| 265/400 [06:55<03:31, 1.57s/it, Average key norm=0.3, Keys Scaled=42, Loss: 0.0175 | Pulse: 5.7518e-05
steps: 66%|▋| 266/400 [06:56<03:29, 1.57s/it, Average key norm=0.3, Keys Scaled=42, Loss: 0.0176 | Pulse: 6.2256e-05
steps: 67%|▋| 267/400 [06:58<03:28, 1.57s/it, Average key norm=0.3, Keys Scaled=42, Loss: 0.0185 | Pulse: 6.2256e-05
steps: 67%|▋| 268/400 [06:59<03:26, 1.57s/it, Average key norm=0.3, Keys Scaled=42, Loss: 0.0194 | Pulse: 5.8092e-05
steps: 67%|▋| 269/400 [07:00<03:25, 1.56s/it, Average key norm=0.3, Keys Scaled=43, Loss: 0.0191 | Pulse: 5.8092e-05
steps: 68%|▋| 270/400 [07:02<03:23, 1.56s/it, Average key norm=0.3, Keys Scaled=44, Loss: 0.0196 | Pulse: 5.8092e-05

steps: 68%|▋| 271/400 [07:04<03:21, 1.56s/it, Average key norm=0.3, Keys Scaled=44, Loss: 0.0184 | Pulse: 5.8092e-05
steps: 68%|▋| 272/400 [07:05<03:20, 1.56s/it, Average key norm=0.3, Keys Scaled=44, Loss: 0.0181 | Pulse: 5.8092e-05
steps: 68%|▋| 273/400 [07:07<03:18, 1.56s/it, Average key norm=0.3, Keys Scaled=42, Loss: 0.0174 | Pulse: 5.8092e-05
steps: 68%|▋| 274/400 [07:08<03:17, 1.56s/it, Average key norm=0.3, Keys Scaled=42, Loss: 0.0188 | Pulse: 5.8092e-05
steps: 69%|▋| 275/400 [07:09<03:15, 1.56s/it, Average key norm=0.3, Keys Scaled=37, Loss: 0.0181 | Pulse: 5.8092e-05
steps: 69%|▋| 276/400 [07:11<03:13, 1.56s/it, Average key norm=0.3, Keys Scaled=37, Loss: 0.0187 | Pulse: 5.8092e-05
steps: 69%|▋| 277/400 [07:12<03:12, 1.56s/it, Average key norm=0.3, Keys Scaled=36, Loss: 0.0191 | Pulse: 5.8092e-05
steps: 70%|▋| 278/400 [07:14<03:10, 1.56s/it, Average key norm=0.301, Keys Scaled=3, Loss: 0.0181 | Pulse: 5.8092e-05
steps: 70%|▋| 279/400 [07:16<03:09, 1.56s/it, Average key norm=0.301, Keys Scaled=3, Loss: 0.0186 | Pulse: 5.8092e-05
steps: 70%|▋| 280/400 [07:17<03:07, 1.56s/it, Average key norm=0.301, Keys Scaled=3, Loss: 0.0191 | Pulse: 5.8092e-05

steps: 70%|▋| 281/400 [07:18<03:05, 1.56s/it, Average key norm=0.301, Keys Scaled=3, Loss: 0.0179 | Pulse: 5.8092e-05
steps: 70%|▋| 282/400 [07:20<03:04, 1.56s/it, Average key norm=0.301, Keys Scaled=3, Loss: 0.0195 | Pulse: 5.8092e-05
steps: 71%|▋| 283/400 [07:21<03:02, 1.56s/it, Average key norm=0.301, Keys Scaled=4, Loss: 0.0189 | Pulse: 2.9285e-04
steps: 71%|▋| 284/400 [07:23<03:01, 1.56s/it, Average key norm=0.301, Keys Scaled=4, Loss: 0.0199 | Pulse: 2.9285e-04
steps: 71%|▋| 285/400 [07:24<02:59, 1.56s/it, Average key norm=0.301, Keys Scaled=4, Loss: 0.0196 | Pulse: 2.9285e-04
steps: 72%|▋| 286/400 [07:26<02:57, 1.56s/it, Average key norm=0.302, Keys Scaled=4, Loss: 0.0195 | Pulse: 2.9285e-04
steps: 72%|▋| 287/400 [07:27<02:56, 1.56s/it, Average key norm=0.302, Keys Scaled=4, Loss: 0.0187 | Pulse: 2.9285e-04
steps: 72%|▋| 288/400 [07:29<02:54, 1.56s/it, Average key norm=0.302, Keys Scaled=4, Loss: 0.0192 | Pulse: 2.9285e-04
steps: 72%|▋| 289/400 [07:30<02:53, 1.56s/it, Average key norm=0.302, Keys Scaled=4, Loss: 0.0183 | Pulse: 2.9285e-04
steps: 72%|▋| 290/400 [07:32<02:51, 1.56s/it, Average key norm=0.302, Keys Scaled=4, Loss: 0.0185 | Pulse: 2.9285e-04

steps: 73%|▋| 291/400 [07:33<02:49, 1.56s/it, Average key norm=0.302, Keys Scaled=4, Loss: 0.0195 | Pulse: 2.9285e-04
steps: 73%|▋| 292/400 [07:35<02:48, 1.56s/it, Average key norm=0.302, Keys Scaled=4, Loss: 0.0200 | Pulse: 2.9285e-04
steps: 73%|▋| 293/400 [07:36<02:46, 1.56s/it, Average key norm=0.302, Keys Scaled=4, Loss: 0.0193 | Pulse: 2.9285e-04
steps: 74%|▋| 294/400 [07:38<02:45, 1.56s/it, Average key norm=0.303, Keys Scaled=3, Loss: 0.0188 | Pulse: 2.9285e-04
steps: 74%|▋| 295/400 [07:39<02:43, 1.56s/it, Average key norm=0.303, Keys Scaled=3, Loss: 0.0183 | Pulse: 2.9285e-04
steps: 74%|▋| 296/400 [07:41<02:42, 1.56s/it, Average key norm=0.303, Keys Scaled=3, Loss: 0.0190 | Pulse: 2.9285e-04
steps: 74%|▋| 297/400 [07:42<02:40, 1.56s/it, Average key norm=0.303, Keys Scaled=4, Loss: 0.0193 | Pulse: 2.9285e-04
steps: 74%|▋| 298/400 [07:44<02:38, 1.56s/it, Average key norm=0.303, Keys Scaled=4, Loss: 0.0191 | Pulse: 2.9285e-04
steps: 75%|▋| 299/400 [07:45<02:37, 1.56s/it, Average key norm=0.304, Keys Scaled=4, Loss: 0.0205 | Pulse: 2.9285e-04
steps: 75%|▊| 300/400 [07:47<02:35, 1.56s/it, Average key norm=0.304, Keys Scaled=4, Loss: 0.0190 | Pulse: 2.9285e-04

steps: 75%|▊| 301/400 [07:48<02:34, 1.56s/it, Average key norm=0.304, Keys Scaled=4, Loss: 0.0192 | Pulse: 2.9285e-04
steps: 76%|▊| 302/400 [07:50<02:32, 1.56s/it, Average key norm=0.304, Keys Scaled=4, Loss: 0.0184 | Pulse: 2.9285e-04
steps: 76%|▊| 303/400 [07:51<02:30, 1.56s/it, Average key norm=0.304, Keys Scaled=3, Loss: 0.0188 | Pulse: 2.9285e-04
steps: 76%|▊| 304/400 [07:52<02:29, 1.56s/it, Average key norm=0.305, Keys Scaled=3, Loss: 0.0204 | Pulse: 2.9285e-04
steps: 76%|▊| 305/400 [07:54<02:27, 1.56s/it, Average key norm=0.305, Keys Scaled=3, Loss: 0.0184 | Pulse: 2.9285e-04
steps: 76%|▊| 306/400 [07:55<02:26, 1.56s/it, Average key norm=0.305, Keys Scaled=3, Loss: 0.0196 | Pulse: 2.9285e-04
steps: 77%|▊| 307/400 [07:57<02:24, 1.56s/it, Average key norm=0.305, Keys Scaled=3, Loss: 0.0198 | Pulse: 2.9285e-04
steps: 77%|▊| 308/400 [07:58<02:23, 1.55s/it, Average key norm=0.305, Keys Scaled=3, Loss: 0.0180 | Pulse: 2.9285e-04
steps: 77%|▊| 309/400 [08:00<02:21, 1.55s/it, Average key norm=0.305, Keys Scaled=3, Loss: 0.0186 | Pulse: 2.9285e-04
steps: 78%|▊| 310/400 [08:01<02:19, 1.55s/it, Average key norm=0.305, Keys Scaled=3, Loss: 0.0181 | Pulse: 2.9285e-04

steps: 78%|▊| 311/400 [08:03<02:18, 1.55s/it, Average key norm=0.306, Keys Scaled=4, Loss: 0.0183 | Pulse: 2.9285e-04
steps: 78%|▊| 312/400 [08:04<02:16, 1.55s/it, Average key norm=0.306, Keys Scaled=4, Loss: 0.0189 | Pulse: 2.9285e-04
steps: 78%|▊| 313/400 [08:06<02:15, 1.55s/it, Average key norm=0.306, Keys Scaled=4, Loss: 0.0183 | Pulse: 2.9285e-04
steps: 78%|▊| 314/400 [08:07<02:13, 1.55s/it, Average key norm=0.306, Keys Scaled=4, Loss: 0.0193 | Pulse: 2.9285e-04
steps: 79%|▊| 315/400 [08:09<02:11, 1.55s/it, Average key norm=0.306, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0176 | Pulse: 2.9285e-04
steps: 79%|▊| 316/400 [08:10<02:10, 1.55s/it, Average key norm=0.306, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0188 | Pulse: 2.9285e-04
steps: 79%|▊| 317/400 [08:11<02:08, 1.55s/it, Average key norm=0.306, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0192 | Pulse: 2.9285e-04
steps: 80%|▊| 318/400 [08:13<02:07, 1.55s/it, Average key norm=0.307, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0176 | Pulse: 2.9285e-04
steps: 80%|▊| 319/400 [08:14<02:05, 1.55s/it, Average key norm=0.307, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0183 | Pulse: 2.9285e-04
steps: 80%|▊| 320/400 [08:16<02:04, 1.55s/it, Average key norm=0.307, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0193 | Pulse: 2.9285e-04

steps: 80%|▊| 321/400 [08:18<02:02, 1.55s/it, Average key norm=0.307, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0182 | Pulse: 2.9285e-04
steps: 80%|▊| 322/400 [08:19<02:01, 1.55s/it, Average key norm=0.307, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0186 | Pulse: 2.9285e-04
steps: 81%|▊| 323/400 [08:21<01:59, 1.55s/it, Average key norm=0.308, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0185 | Pulse: 2.9285e-04
steps: 81%|▊| 324/400 [08:22<01:57, 1.55s/it, Average key norm=0.308, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0179 | Pulse: 2.9285e-04
steps: 81%|▊| 325/400 [08:24<01:56, 1.55s/it, Average key norm=0.308, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0181 | Pulse: 2.9285e-04
steps: 82%|▊| 326/400 [08:25<01:54, 1.55s/it, Average key norm=0.308, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0188 | Pulse: 2.9285e-04
steps: 82%|▊| 327/400 [08:27<01:53, 1.55s/it, Average key norm=0.308, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0169 | Pulse: 2.9285e-04
steps: 82%|▊| 328/400 [08:28<01:51, 1.55s/it, Average key norm=0.308, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0178 | Pulse: 2.9285e-04
steps: 82%|▊| 329/400 [08:30<01:50, 1.55s/it, Average key norm=0.308, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0184 | Pulse: 2.9285e-04
steps: 82%|▊| 330/400 [08:31<01:48, 1.55s/it, Average key norm=0.309, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0207 | Pulse: 2.9285e-04

steps: 83%|▊| 331/400 [08:33<01:47, 1.55s/it, Average key norm=0.309, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0176 | Pulse: 2.9285e-04
steps: 83%|▊| 332/400 [08:35<01:45, 1.55s/it, Average key norm=0.309, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0183 | Pulse: 2.9285e-04
steps: 83%|▊| 333/400 [08:36<01:43, 1.55s/it, Average key norm=0.309, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0190 | Pulse: 2.9285e-04
steps: 84%|▊| 334/400 [08:38<01:42, 1.55s/it, Average key norm=0.309, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0173 | Pulse: 2.9285e-04
steps: 84%|▊| 335/400 [08:39<01:40, 1.55s/it, Average key norm=0.309, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0184 | Pulse: 2.9285e-04
steps: 84%|▊| 336/400 [08:41<01:39, 1.55s/it, Average key norm=0.309, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0172 | Pulse: 2.9285e-04
steps: 84%|▊| 337/400 [08:42<01:37, 1.55s/it, Average key norm=0.31, Keys Scaled=38✨[READY TO STOP]✨
, Loss: 0.0185 | Pulse: 2.9285e-04
steps: 84%|▊| 338/400 [08:44<01:36, 1.55s/it, Average key norm=0.31, Keys Scaled=41✨[READY TO STOP]✨
, Loss: 0.0190 | Pulse: 2.9285e-04
steps: 85%|▊| 339/400 [08:45<01:34, 1.55s/it, Average key norm=0.31, Keys Scaled=44✨[READY TO STOP]✨
, Loss: 0.0187 | Pulse: 2.9285e-04
steps: 85%|▊| 340/400 [08:47<01:33, 1.55s/it, Average key norm=0.31, Keys Scaled=46✨[READY TO STOP]✨
, Loss: 0.0188 | Pulse: 2.9285e-04

steps: 85%|▊| 341/400 [08:49<01:31, 1.55s/it, Average key norm=0.31, Keys Scaled=47✨[READY TO STOP]✨
, Loss: 0.0188 | Pulse: 2.9285e-04
steps: 86%|▊| 342/400 [08:50<01:29, 1.55s/it, Average key norm=0.311, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0186 | Pulse: 2.9285e-04
steps: 86%|▊| 343/400 [08:52<01:28, 1.55s/it, Average key norm=0.311, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0172 | Pulse: 2.9285e-04
steps: 86%|▊| 344/400 [08:53<01:26, 1.55s/it, Average key norm=0.311, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0167 | Pulse: 2.9285e-04
steps: 86%|▊| 345/400 [08:55<01:25, 1.55s/it, Average key norm=0.311, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0175 | Pulse: 2.9285e-04
steps: 86%|▊| 346/400 [08:56<01:23, 1.55s/it, Average key norm=0.311, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0187 | Pulse: 2.9285e-04
steps: 87%|▊| 347/400 [08:58<01:22, 1.55s/it, Average key norm=0.311, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0186 | Pulse: 2.9285e-04
steps: 87%|▊| 348/400 [08:59<01:20, 1.55s/it, Average key norm=0.311, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0184 | Pulse: 2.9285e-04
steps: 87%|▊| 349/400 [09:01<01:19, 1.55s/it, Average key norm=0.311, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0182 | Pulse: 2.9285e-04
steps: 88%|▉| 350/400 [09:02<01:17, 1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0173 | Pulse: 2.9285e-04

steps: 88%|▉| 351/400 [09:04<01:15, 1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0191 | Pulse: 2.9285e-04
steps: 88%|▉| 352/400 [09:05<01:14, 1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0168 | Pulse: 2.9285e-04
steps: 88%|▉| 353/400 [09:07<01:12, 1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0171 | Pulse: 2.9285e-04
steps: 88%|▉| 354/400 [09:08<01:11, 1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0182 | Pulse: 2.9285e-04
steps: 89%|▉| 355/400 [09:10<01:09, 1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0188 | Pulse: 2.9285e-04
steps: 89%|▉| 356/400 [09:11<01:08, 1.55s/it, Average key norm=0.312, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0198 | Pulse: 2.9285e-04
steps: 89%|▉| 357/400 [09:13<01:06, 1.55s/it, Average key norm=0.313, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0187 | Pulse: 2.9285e-04
steps: 90%|▉| 358/400 [09:14<01:05, 1.55s/it, Average key norm=0.313, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0190 | Pulse: 2.9285e-04
steps: 90%|▉| 359/400 [09:16<01:03, 1.55s/it, Average key norm=0.313, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0181 | Pulse: 2.9285e-04
steps: 90%|▉| 360/400 [09:17<01:01, 1.55s/it, Average key norm=0.313, Keys Scaled=3✨[READY TO STOP]✨
, Loss: 0.0180 | Pulse: 2.9285e-04

steps: 90%|▉| 361/400 [09:19<01:00, 1.55s/it, Average key norm=0.313, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0186 | Pulse: 2.9285e-04
steps: 90%|▉| 362/400 [09:20<00:58, 1.55s/it, Average key norm=0.314, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0171 | Pulse: 2.9285e-04
steps: 91%|▉| 363/400 [09:22<00:57, 1.55s/it, Average key norm=0.314, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0183 | Pulse: 2.9285e-04
steps: 91%|▉| 364/400 [09:23<00:55, 1.55s/it, Average key norm=0.314, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0178 | Pulse: 2.9285e-04
steps: 91%|▉| 365/400 [09:25<00:54, 1.55s/it, Average key norm=0.314, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0182 | Pulse: 2.9285e-04
steps: 92%|▉| 366/400 [09:27<00:52, 1.55s/it, Average key norm=0.314, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0168 | Pulse: 2.9285e-04
steps: 92%|▉| 367/400 [09:28<00:51, 1.55s/it, Average key norm=0.314, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0167 | Pulse: 2.9285e-04
steps: 92%|▉| 368/400 [09:30<00:49, 1.55s/it, Average key norm=0.314, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0176 | Pulse: 2.9285e-04
steps: 92%|▉| 369/400 [09:31<00:48, 1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0168 | Pulse: 2.9285e-04
steps: 92%|▉| 370/400 [09:33<00:46, 1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0167 | Pulse: 2.9285e-04

steps: 93%|▉| 371/400 [09:34<00:44, 1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0162 | Pulse: 2.9285e-04
steps: 93%|▉| 372/400 [09:36<00:43, 1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0184 | Pulse: 2.9285e-04
steps: 93%|▉| 373/400 [09:37<00:41, 1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0168 | Pulse: 2.9285e-04
steps: 94%|▉| 374/400 [09:39<00:40, 1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0163 | Pulse: 2.9285e-04
steps: 94%|▉| 375/400 [09:41<00:38, 1.55s/it, Average key norm=0.315, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0166 | Pulse: 2.9285e-04
steps: 94%|▉| 376/400 [09:42<00:37, 1.55s/it, Average key norm=0.316, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0171 | Pulse: 2.9285e-04
steps: 94%|▉| 377/400 [09:44<00:35, 1.55s/it, Average key norm=0.316, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0157 | Pulse: 2.9285e-04
steps: 94%|▉| 378/400 [09:45<00:34, 1.55s/it, Average key norm=0.316, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0176 | Pulse: 2.9285e-04
steps: 95%|▉| 379/400 [09:47<00:32, 1.55s/it, Average key norm=0.316, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0164 | Pulse: 2.9285e-04
steps: 95%|▉| 380/400 [09:48<00:30, 1.55s/it, Average key norm=0.316, Keys Scaled=4✨[READY TO STOP]✨

, Loss: 0.0162 | Pulse: 2.9285e-04
steps: 95%|▉| 381/400 [09:49<00:29, 1.55s/it, Average key norm=0.316, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0172 | Pulse: 2.9285e-04
steps: 96%|▉| 382/400 [09:51<00:27, 1.55s/it, Average key norm=0.317, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0184 | Pulse: 2.9285e-04
steps: 96%|▉| 383/400 [09:52<00:26, 1.55s/it, Average key norm=0.317, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0170 | Pulse: 2.9285e-04
steps: 96%|▉| 384/400 [09:54<00:24, 1.55s/it, Average key norm=0.317, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0181 | Pulse: 2.9285e-04
steps: 96%|▉| 385/400 [09:55<00:23, 1.55s/it, Average key norm=0.317, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0174 | Pulse: 2.9285e-04
steps: 96%|▉| 386/400 [09:57<00:21, 1.55s/it, Average key norm=0.317, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0167 | Pulse: 2.9285e-04
steps: 97%|▉| 387/400 [09:58<00:20, 1.55s/it, Average key norm=0.317, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0158 | Pulse: 2.9285e-04
steps: 97%|▉| 388/400 [10:00<00:18, 1.55s/it, Average key norm=0.317, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0181 | Pulse: 2.9285e-04
steps: 97%|▉| 389/400 [10:01<00:17, 1.55s/it, Average key norm=0.317, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0181 | Pulse: 2.9285e-04
steps: 98%|▉| 390/400 [10:02<00:15, 1.55s/it, Average key norm=0.318, Keys Scaled=4✨[READY TO STOP]✨

</details>

, Loss: 0.0167 | Pulse: 2.9285e-04
steps: 98%|▉| 391/400 [10:04<00:13, 1.55s/it, Average key norm=0.318, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0180 | Pulse: 2.9285e-04
steps: 98%|▉| 392/400 [10:05<00:12, 1.55s/it, Average key norm=0.318, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0164 | Pulse: 2.9285e-04
steps: 98%|▉| 393/400 [10:07<00:10, 1.55s/it, Average key norm=0.318, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0162 | Pulse: 2.9285e-04
steps: 98%|▉| 394/400 [10:08<00:09, 1.55s/it, Average key norm=0.318, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0167 | Pulse: 2.9285e-04
steps: 99%|▉| 395/400 [10:10<00:07, 1.54s/it, Average key norm=0.318, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0178 | Pulse: 2.9285e-04
steps: 99%|▉| 396/400 [10:11<00:06, 1.54s/it, Average key norm=0.318, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0160 | Pulse: 2.9285e-04
steps: 99%|▉| 397/400 [10:13<00:04, 1.54s/it, Average key norm=0.318, Keys Scaled=4✨[READY TO STOP]✨
, Loss: 0.0168 | Pulse: 2.9285e-04
steps: 100%|▉| 398/400 [10:14<00:03, 1.54s/it, Average key norm=0.319, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0186 | Pulse: 2.9285e-04
steps: 100%|▉| 399/400 [10:16<00:01, 1.54s/it, Average key norm=0.319, Keys Scaled=5✨[READY TO STOP]✨
, Loss: 0.0171 | Pulse: 2.9285e-04
steps: 100%|█| 400/400 [10:17<00:00, 1.54s/it, Average key norm=0.319, Keys Scaled=5,

```

</details>

---

Regarding versions prior to v3.8  

---


##### ！ Apology to All (Important Notice) ！  

First of all, I would like to offer my sincere apologies to everyone who has viewed this repository.  
Regarding the "experimental results" and "theoretical claims" related to the emo‑series optimizers published in this repository,  
my own subsequent verification has revealed that they are highly likely not to hold.  
In particular, it was found that the claims regarding emoPulse were based on a design that assumes the use of a closure,  
yet the implementation did not apply a closure.  
As a result, I have confirmed that the consistency between the theory and the implementation has collapsed.  
The descriptions written at that time were based on my understanding at that moment,  
and from the current perspective they should be regarded as "unverified / draft content containing errors."  
To everyone who trusted those descriptions,  
I deeply apologize for having caused misunderstandings.  


##### The following content is entirely "hypothesis", "unverified", and "draft"  
※ The theoretical interpretation may have been incorrect  
※ It cannot be considered reliable evidence  

---  

- ###### This is a new generation of optimizers that use the Resonant Contraction Method (Resonant Projection Field) / It is not a Gradient Descent Method  
- ###### EmoSens / 2ndGen (v3.8 / Standard)  
- ###### EmoTion / 3rdGen (v3.8 / Moment-Free)  
readme：[English](README.md) | [日本語](README_JA.md)  

---

#### Architectural Evolution via Resonant Contraction  
We introduce an evolved version of the Transformer here  
https://github.com/muooon/DRNA  

---

# EmoSens / Tion update  

- EmoVoid has the potential to function as an analytical solver for “wave scattering inverse problems”  
- Improved accuracy of the early stop notification feature, support for learning transfer, and integration with the Beginners Edition (260404)  
- EmoSens (v3.8) emoPulse (Fully Automatic Learning Rate) Adjustment  
- EmoTion (v3.8) Release of W-Ref-Geometry and Moment-Free  

##### ※ FFT-Aware version integrated,"FFT(full fine-tuning)" Mode switching available via Option arguments

Features in v3.7 and later  
- Fully Automatic Value Learning Rate: Achieves both acceleration and refinement while eliminating the need to worry about the initial learning rate.  
- emoPulse： Autonomously adjusts LR levels to safely and stably proceed with “ultra-low precision, ultra-quantization.”  
- The initial LR can be set to 1.0 (please focus your time on refining the dataset).   

### Explanation  
Expected value convergence for non-convex functions  
(also guarantees adaptability to flow matching)  
(Providing a direct path to Flat Minima without the necessity of Grokking.)  

#### [emo-paper(article)](https://huggingface.co/muooon/EmoNAVI/raw/main/emo-v386plus-paper(ENG).txt)  

#### [DOI-Acquired Version](https://huggingface.co/muooon/EmoTion-Optimizer)  

---

<details>

<summary> resonant contraction method </summary>

Fundamental Theorem of the Resonant Contraction Method (Overview)  

1. Definition of the State: Resonante of the Three Elements  
    The update of parameter w is determined by the synergistic effects (resonant) of the following three independent dimensions.  
    Time axis (ηt: emoPulse): Step size autonomously generated from the system's internal “reliability” (SNR).  
    Spatial axis (Rt: W-Ref Geometry): Novelty gain calculated based on the “orthogonality” of the current weights and gradients.  
    Directional axis (ut: Pure Will): Will consisting solely of a “sign” purified over time, with the magnitude of the gradient discarded.  
※ ηt(Temporal axis): Can be substituted by any component functioning as a high-precision dynamic scheduler.  
※ Rt(Spatial axis): Can be substituted by high-precision 2nd-order moments or equivalent metrics.  
※ ut(Directional axis): Can be substituted by high-precision 1st-order moments or equivalent metrics.  

2. The Basic Equation for Updates  
    When the gradient is denoted by g, we abandon the traditional Δw = −ηg and apply the following equation:  
	Discrete-time representation:  
    Δwt = −ηt ⋅ Rt ⋅ sign(mt)  
	Continuous-time representation:  
	\frac{dw}{dt} = - λ ⋅ η(t) ⋅ w(t) - η(t) ⋅ R(t) ⋅ u(t)  
    As a result, the system’s dependence on external forces—specifically the “magnitude of the gradient”—is completely eliminated, and it transitions to autonomous movement based on its internal state.  
※ (mt): A temporally stabilized directional potential (not a "moment" in the traditional sense).  
    (mt) ignores the magnitude of the gradient gt and maintains "cumulative confidence" through temporal smoothing. It forms the "Pure Will" via ut = sign(mt), delegating the control of amplitude entirely to the temporal axis (ηt).  

3. The Three Properties Guaranteed by the Theorem  
a. Autonomous contraction (Contraction Property)  
    As the system's energy (loss) decreases, ηt functions as an “autonomous brake”.  
    Result: Without any external schedule adjustments, the system contracts exponentially toward a single point (the solution manifold) and stabilizes.  
b. Geodesic Path  
    Rt suppresses the “known direction” (the component parallel to the weight) and accelerates the “unknown direction” (the orthogonal component).  
    Result: Moving along the spherical surface (manifold) known as the parameter space in the shortest possible path, without any unnecessary detours.  
c. Information Bottleneck  
    Extracting direction using the sign function acts as a filter that blocks out the fine noise contained in the gradient.  
    Result: The algorithm avoids overly complex solutions (overfitting) and converges to the simplest and most general-purpose “flat minima”.  

Conclusion: What is the Resonance Contraction Method ?  
    An autonomous "Dynamic Scheduler" like emoPulse upgrades passive Stochastic Gradient Descent (SGD) into the autonomous "Resonance Contraction Method" (Resonance Projection Field) based on the system's internal state. By achieving SDE-DDE-ODE reduction approximation, this optimization evolves into a highly sophisticated contraction process, attaining unparalleled robustness and precision.  

</details>

---

<div align="center">
  <img width="500" alt="emo-system001" src="https://github.com/user-attachments/assets/7e7160a9-046a-4212-bcde-d338c26ed846" />
</div>

---

emo-series v3.8 (Standard / Moment-Free) Features  

| Name    | Time-Accurate | MemoryLoad | Notes                               |  
|---------|---------------|------------|--------------------------------------|  
| emosens | ★★★★          | ★★         | 1st born｜Accurate｜Adam-type         |  
| emoairy | ★★            | ★★★★       | 2nd born｜Lightest｜Adafactor-type    |  
| emocats | ★★★☆          | ★★★☆        | Light & Accurate｜Lion-type          |  
|---------|---------------|------------|--------------------------------------|  
| emotion | ★★★★          | ★★★☆        | “Light” & Accurate｜Original-type    |  
| emovoid | ★★☆           | ★★★★★      | “Lightest & Fastest”｜Original-type |  

[Efficiency] Risk-Aware Updates: Proactively prevents overfitting and convergence stagnation while eliminating redundant updates.  
[Functionality] Lightweight and High-Performance: Enhances user experience through automatic stop signals and support for fully autonomous distributed learning.  
[Reliability] Safety-First Design: Protects the model during unstable learning phases with dynamic control, promoting stable convergence.  
※ Fully autonomous, enabling flexible learning configurations through layering, resumption, and asynchronous processing  
※ EmoTion enhances accuracy and efficiency through geometric orthogonal updating and elimination of second moments.  
※ EmoVoid improves VRAM efficiency by using geometric orthogonal updates and  eliminating first and second moments.  

---  

##### “As long as there is loss, emoPulse(Heartbeat) will never stop —”  
###### An option that might allow reaching Flat Minima without Grokking  

---  

#### Learning Information, Everything is consolidated into the Loss value.  
###### The loss value is the model's shadow.  
###### The loss value embodies everything.  
###### The loss value tells you both the learning status and the model's condition.  
###### Feel the loss. Loss is the origin.  

---  

### Main Features of EmoSens  

---  

<details>

<summary> Main Features </summary>

||| Autonomy and Reliability |||  
Self-repairing, with no over-learning or divergence.  
Autonomously adjusts learning rate and scheduler, so models make their own decisions.  
Resuming, adding, stacking, etc. learning is synchronization-free" and easy for everyone.  
Distributed learning enables “no synchronization required” with other nodes, achieving full autonomy.  

||| emotion-Driven Cycle |||  
The “emo-series” is an “emotion-driven” optimizer, distinct from existing methods.  
It is expected to overcome current challenges and address new frontiers, such as multimodal learning requiring complex coordination.  
The emo-series follows an autonomous cycle of: observation, judgment, decision, action, memory, and reflection.  

||| The Ultimate Evolution / |||  
To put it very simply, “emo-series” and “emoPulse” is a “high-end scheduler”.  
It is also the Ultimate evolution of Sharpness-Aware Minimization.  
It achieves a level of “precision” where SDE-based dynamics approximate ODE-level accuracy—a synthesis of Shikan (tranquility/contemplation) and Aufheben (sublation).  
It is Highly compatible with advanced RNN variants such as Liquid (LiquidAI/MIT), Titans (Google), and Mamba (CMU/Princeton).  

||| High Efficiency and Integration Density (Approximate Structure) |||  
Multiple higher-order moments, history compensation, quantization compensation (a control method different from Kahan compensation), confidence filters,  
Dynamic scaling, independence in distributed and continuous learning, self-repair and model repair (reverse phase merging using LoRA),  
We will perform self-supervised learning, which incorporates self-stopping, autonomous hyperparameter tuning, structural robustness, and other features.  
Dynamic learning rate, dynamic scheduler, dynamic Rank/Alpha, SVD, infLoRA, ABBA-LoRA, PiSSA,  
A wide range of features, including FourierFT, DoRA, PRO-LoRA, DARE, Ties, and Tall-Mask-Merge,  
No additional tensors required, virtually no computational overhead, all of the above applied at all times, achieved through temporal integration while maintaining stability.  
By integrating these features into a single package, we prioritize stability and safety above all else.  
With minimal VRAM usage, Langevin Dynamics, Kalman Filter, PID Control,  
Stochastic resonance, tunneling effect, target updating, thermodynamics, feedback control,  
It is stable in Riemannian manifolds, orthogonality, emotional memory consolidation, fluid dynamics, and other areas.  
※ Higher-order moments are approximate, and dynamic rank/alpha also has an approximate effect.  
※ LoRA-based techniques eliminate noise, but they may also lose some fine-grained details.  
※ The emo-series approach does not generate noise; instead, it identifies and corrects existing noise to protect microdata.  
※ Quantization compensation can flexibly adapt to even lower-precision environments that will become practical in the future.  

</details>

---  

<details>

<summary> emoPulse mechanism </summary>

---
emoPulse：(d_base/noise_base)^2 Calculation   

| d \ N base |  0.1   |  0.5   |  0.7   |  
|------------|--------|--------|--------|  
|     0.1    |  1.00  |  0.04  |  0.0204|  
|     0.5    | 25.00  |  1.00  |  0.5102|  
|     0.7    | 49.00  |  1.96  |  1.00  |  

・No matter how high the d/N ratio is, the maximum increase in a single step is +50%.  
・And growth is only allowed when it’s “better than before and reliable”  
  To approach the upper limit, you need to accumulate (consecutive) instances of the (high d/N) and (high trust) states.  

・The moment you judge it to be “suspicious,” immediately reduce it by 0.80x  
・Deceleration occurs under less stringent conditions (braking is more likely to occur)  
 (Trust is hard to earn but easy to lose / It’s hard to raise but easy to lower)  

※ This system only increases the upper limit when it is truly trustworthy.  

---

Numerator(d_base)：Difference in History (Assuming 0.7 − 0.3 + 0.1 = 0.5)  
denominator(noise_base)：Momentary Discrepancy in Emotions ∣ scalar−trust ∣ + 0.1  

| side   | status         | scalar | trust | noise_base | dNR_now_val(^2) | Impact on emoPulse       |
|------|--------------|--------|-------|------------|-------------------|---------------------------|
| +side  | Match (Maximum) |  0.50  | 0.50  |   0.10     |      25.00        | Maximum Acceleration (1.5x)     |
| +side  | Ideal Harmony |  0.45  | 0.55  |   0.20     |       6.25        | Acceleration (1.5x)         |
| +side  | Improvement |  0.20  | 0.80  |   0.70     |       0.51        | Maintain (Wait and See)              |
| -side  | ++ Discrepancy | -0.20  | -0.80 |   0.70     |       0.51        | Maintain (Wait and See)              |
| -side  | +++ Discomfort | -0.45  | -0.55 |   0.20     |       6.25        | Deceleration (0.8x)         |
| -side  | Reverse Match | -0.50  | -0.50 |   0.10     |      25.00        | Maximum Deceleration (0.8x)     |

denominator(noise_base): As abs(scalar - trust) approaches 0 (i.e., as the emotion scalar and the confidence level align), the denominator approaches its minimum value of 0.1, causing the squared result to spike.  
+side: If dNR_now_val is high and trust is also high, the history (dNR_hist) is increased by up to 1.50 times.  
-Side: Even if dNR_now_val is calculated to be 25.00, because the trust value is low (within the range of -0.5 to 0.5), the history is reduced by a factor of 0.80, causing the system to apply the brakes.  
Entropy Suppression: The values in this table (dNR_now_val) are not used directly as the learning rate; instead, they are incorporated into dNR_hist (history) and ultimately converted to an extremely small, safe learning rate (1e-8 to 3e-3) using the formula emoScope × 1e-4·1e-5.    

</details>

---  

<details>

<summary>EmoSens v3.8 and later Option Settings Guide</summary>  

|||Usage examples|||  
●FFT-mode on：  
fftmode=True  
●Shadow off:  
use_shadow=False  
●notify off:  
notify=False  
●stopcoef (default：0.3):  
stopcoef=0.3  
●eps(Division by zero prevention)：  
eps=1e-8  


</details>

---  

<details>
 
<summary> emotional moment </summary>  

I invented the emotional moment.  
I extracted it from the core of the shadow-system, which was elucidated in the "emo-style second generation v1.x."  
The nonlinear approach with a dynamic learning rate forms a temporal higher-order moment.  
A single step cannot become a higher-order moment, but it functions after multiple steps.  
It approximates the core effect of capturing changes in gradient distribution's skewness, kurtosis, and asymmetry, while avoiding strict and computationally intensive mathematical calculations for the third, fourth, and fifth moments.  

---

#### The optimization you seek — EmoSens makes it possible  
---
###### This is not just another optimizer —  
###### **It’s an “Emotional Optimizer” that navigates learning through feeling.**  
###### A result of transformative emotional learning: the reinvention of the neural spike.  
--- 
#### Auto-convergence, self-control, autonomous optimizer  
###### It primarily features EmoSens, along with EmoAiry and EmoCats.  

</details>

---  

<details>

<summary> History </summary>  

|★| EmoTion Generation v3.8 (260204) Release of W-Ref-Geometry and MomentFree, etc.  

|★| EmoSens Generation v3.8 (260130) Adjustments to emoPulse Mechanism, etc.   

|★| EmoSens, Airy, Cats, v3.7 (260101) Building upon Navi v3.6, we have achieved fully automatic high-value learning rate optimization (without additional tensors), and through the emoPulse mechanism, we have achieved dramatic evolution.  

|★| EmoNavi, Fact, Lynx, v3.6 (251220) Inherits v3.1 and achieves high-value automatic learning rate (no additional tensors), has undergone dramatic evolution through the emoDrive mechanism, development is now complete.  

|★| EmoNavi, Fact, Lynx, v3.3 (251204) Inherits v3.1 and achieves fully automatic learning rate adjustment (without additional tensors), further evolving for greater stability through adjustments to the sentiment mechanism and other enhancements.  

|★| EmoNavi, Fact, Lynx, v3.1 (251201) We built upon v3.0 while enhancing efficiency. Through adjustments like scaling the emotion mechanism, we evolved the model for broader stability across diverse models.  

|★| EmoNavi, Fact, Lynx, Clan, Zeal, Neco, updated to v3.0 (250825), Incorporates (updates) feedback on “higher moments” (approximations) clarified by emosens (2nd generation). All are “shadow=False”  

For updates prior to this, please refer to the v2.0 repository update history.  

</details>

---  

## Progress of emo-type as shown in the graph (v3.7 and later)  
<img width="2218" height="1153" alt="emov376-003-tile" src="https://github.com/user-attachments/assets/a1c5891b-a842-4ed1-a147-d4658e1ca16b" />  
In this way, it functions as a dynamic learning rate. / Could the fact that it continues to decline mean that it is also learning the differences in the “modifications to the original model”? <br> 
※ If LR decay based on convergence detection is not applied, the curve will continue to decline without plateauing. <br> 

It functions as a dynamic learning rate. ／ Could the continuous decline be due to also learning the differences in “original model corrections”? <br> 
Dataset Status LEFT: Primarily 10 Photo images, 10 batch, 300 epochs (3000 steps), full-layer LoRA, Rank16/Alpha16, e-pred, ZtSNR,  <br>  
Dataset Status RIGHT: Primarily 11 black-and-white images, 1 batch, 300 epochs (3300 steps), full-layer LoRA, Rank16/Alpha16, e-pred, ZtSNR,  <br>  
es = EmoSens(Red/Green)、ea = EmoAiry(Blue/Gray)、ec = EmoCats(Yellow/Orange) <br> 
 <br> 
<img width="1166" height="644" alt="スクリーンショット 2026-03-01 094343" src="https://github.com/user-attachments/assets/c667e792-e668-40b1-a07f-6cf2ceb6a686" />  
This shows the training status of the FFT (Full-Fine-Tuning) model on Anima-Preview, using 20 images at 512px with an LR of 1.0. <br> 
Purple: EmoSens, Light Blue: EmoAiry, Red: EmoCat, Gray: EmoTion, Yellow: EmoVoid <br> 
I think it would be best to lower the LR value for EmoTion slightly. Orange:EmoTion/LR:0.5 <br>
Please also note the elapsed time <br>  
※ If LR decay based on convergence detection is not applied, the curve will continue to decline without plateauing. <br> 

---

The emo series continues to evolve through biological reactions.  
The sensory nervous system (multi-EMA), endocrine system (tanh(scalar)), immune system (shadow-system), circulatory system (emoPulse), and vestibular system (W-Ref-Geo) integrate to form the central nervous system and the autonomic nervous system, functioning as a naturally self-regulating mechanism capable of advanced judgment and decision-making.  

---  

The emo series has learned much from Adam, Adafactor, Lion, and Tiger.  
Rather than being their successors, it is built upon a unique philosophy and design approach centered on "emotional mechanisms".  
It prioritizes generality, autonomy, and adaptability in pursuit of new paths for optimization, efficiency, and simplicity.  
In its development, we deeply appreciate the insights of those who came before us—and continue to explore new possibilities beyond them. 

---

### License Apache License 2.0 — see LICENSE for details.  

---

### About citations  

---

When citing this optimizer, please refer to the following sources:  

Official Code:  
https://github.com/muooon/EmoSens  

paper:  
https://huggingface.co/muooon/EmoNAVI/raw/main/emo-v386plus-paper(ENG).txt  
DOI-Acquired Version:  
https://huggingface.co/muooon/EmoTion-Optimizer  

---

emo-based is an “emotion-driven” approach not found in existing optimizers. By building each sensor around an “emotion mechanism” that differentiates multi-EMA and scalarizes it via nonlinear transformation (tanh), we enhanced overall learning stability and ensured accuracy. This performs an autonomous cycle of “observation, judgment, decision, action, memory, and reflection,” akin to a biological central nervous system. (Please take a look at the paper.)  


