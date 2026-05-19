## emo series Optimizers  

 Loss-Bypass (ECC) closure-unneeded   
- ###### EmoSens / 2ndGen (v3.9 / Standard-ECC)  
- ###### EmoTion / 3rdGen (v3.9 / Moment-Free-ECC)  
- ###### emo closure capture (ECC-System)  

readme：[English](README.md) | [日本語](README_JA.md)  

<img width="800" height="600" alt="Screenshot 2026-05-19 at 18-57-11 TensorBoard" src="https://github.com/user-attachments/assets/8f3d16b5-8597-49d4-a529-4065435960bb" />  

gray:Void, skyblue:Tion, red:Cats, orange:Airy, violet:Sens,  
SDXL:LoRA, Resolution:512, Rank:8, Alpha4, LR:1.0,  

---

<details>

<summary> EmoSens Full-log </summary>

Cosine scheduler LR “1e-4” is used as the baseline  

Analysis results of the learning rate (LR)  
Maximum LR: 3.0000 × 1e-3 (This represents an instantaneous output approximately 30.0 times higher than the standard value of 1e-4.)  
Average LR: 9.2418 × 1e-4 (It continues to learn at a rate approximately 9.0 times stronger than the baseline throughout the entire training period)  
Minimum LR: 6.1434 × 1e-5 (The value reached 1e-5 only a few times over the entire period, and these instances were not sustained (they were concentrated in the first half))  

When comparing the area (total learning amount) using “1e-4 Cosine” as the baseline, the results are as follows  
Total area of emoPulse: 0.222728  
Total area of Cosine (1e-4): 0.012050  
Efficiency Comparison: 18.48 times  

A short training period of 240 steps is equivalent to approximately 4,400 steps when converted to a fixed cosine pace.   

```prompt

----------

🚩 emo-optim success ecc system ...
override steps. steps for 80 epochs is / 指定エポックまでのステップ数: 240
enable fp8 training for U-Net.
enable fp8 training for Text Encoder.
  total optimization steps / 学習ステップ数: 240
steps:   0%|                                                 | 0/240 [00:00<?, ?it/s] | Loss: 0.1657 | Pulse: 1.0000e-04
steps:   0%| | 1/240 [00:24<1:36:20, 24.18s/it, Average key norm=0.00195, Keys Scaled | Loss: 0.1785 | Pulse: 9.6954e-05
steps:   1%| | 2/240 [00:27<53:34, 13.51s/it, Average key norm=0.00393, Keys Scaled=0 | Loss: 0.1859 | Pulse: 9.1232e-05
steps:   1%| | 3/240 [00:28<38:08,  9.65s/it, Average key norm=0.00606, Keys Scaled=0 | Loss: 0.1791 | Pulse: 8.6532e-05
steps:   2%| | 4/240 [00:31<30:40,  7.80s/it, Average key norm=0.0084, Keys Scaled=0 | Loss: 0.1907 | Pulse: 8.0549e-05
steps:   2%| | 5/240 [00:32<25:47,  6.59s/it, Average key norm=0.0107, Keys Scaled=0 | Loss: 0.1740 | Pulse: 7.7869e-05
steps:   2%| | 6/240 [00:35<22:51,  5.86s/it, Average key norm=0.013, Keys Scaled=0 | Loss: 0.1825 | Pulse: 7.5540e-05
steps:   3%| | 7/240 [00:37<20:37,  5.31s/it, Average key norm=0.0154, Keys Scaled=0 | Loss: 0.1614 | Pulse: 7.7179e-05
steps:   3%| | 8/240 [00:39<18:55,  4.89s/it, Average key norm=0.0179, Keys Scaled=0 | Loss: 0.1674 | Pulse: 7.9668e-05
steps:   4%| | 9/240 [00:41<17:39,  4.59s/it, Average key norm=0.0206, Keys Scaled=0 | Loss: 0.1685 | Pulse: 8.2202e-05
steps:   4%| | 10/240 [00:43<16:37,  4.34s/it, Average key norm=0.0234, Keys Scaled=0 | Loss: 0.1559 | Pulse: 8.7181e-05
steps:   5%| | 11/240 [00:45<15:44,  4.13s/it, Average key norm=0.0265, Keys Scaled=0 | Loss: 0.1775 | Pulse: 8.8466e-05
steps:   5%| | 12/240 [00:47<14:57,  3.94s/it, Average key norm=0.0296, Keys Scaled=0 | Loss: 0.1569 | Pulse: 9.2219e-05
steps:   5%| | 13/240 [00:49<14:20,  3.79s/it, Average key norm=0.0329, Keys Scaled=0 | Loss: 0.1378 | Pulse: 1.0160e-04
steps:   6%| | 14/240 [00:51<13:43,  3.64s/it, Average key norm=0.0364, Keys Scaled=0 | Loss: 0.1479 | Pulse: 1.1120e-04
steps:   6%| | 15/240 [00:53<13:15,  3.54s/it, Average key norm=0.0402, Keys Scaled=0 | Loss: 0.1283 | Pulse: 1.2608e-04
steps:   7%| | 16/240 [00:55<12:52,  3.45s/it, Average key norm=0.0447, Keys Scaled=0 | Loss: 0.1339 | Pulse: 1.4176e-04
steps:   7%| | 17/240 [00:57<12:30,  3.36s/it, Average key norm=0.0496, Keys Scaled=0 | Loss: 0.1316 | Pulse: 1.5863e-04
steps:   8%| | 18/240 [00:59<12:08,  3.28s/it, Average key norm=0.0551, Keys Scaled=0 | Loss: 0.1259 | Pulse: 1.7743e-04
steps:   8%| | 19/240 [01:01<11:50,  3.21s/it, Average key norm=0.0611, Keys Scaled=0 | Loss: 0.1234 | Pulse: 1.9549e-04
steps:   8%| | 20/240 [01:02<11:31,  3.14s/it, Average key norm=0.0675, Keys Scaled=0 | Loss: 0.0972 | Pulse: 2.2643e-04


```

<details>

<summary> Full-log </summary>

```prompt
steps:   8%| | 20/240 [01:02<11:31,  3.14s/it, Average key norm=0.0675, Keys Scaled=0 | Loss: 0.0972 | Pulse: 2.2643e-04
steps:   9%| | 21/240 [01:04<11:15,  3.08s/it, Average key norm=0.0749, Keys Scaled=0 | Loss: 0.0884 | Pulse: 2.7002e-04
steps:   9%| | 22/240 [01:06<11:00,  3.03s/it, Average key norm=0.084, Keys Scaled=0 | Loss: 0.0898 | Pulse: 3.1985e-04
steps:  10%| | 23/240 [01:08<10:47,  2.98s/it, Average key norm=0.0951, Keys Scaled=0 | Loss: 0.1052 | Pulse: 3.5894e-04
steps:  10%| | 24/240 [01:10<10:32,  2.93s/it, Average key norm=0.108, Keys Scaled=0 | Loss: 0.0984 | Pulse: 3.9382e-04
steps:  10%| | 25/240 [01:11<10:17,  2.87s/it, Average key norm=0.121, Keys Scaled=0 | Loss: 0.0737 | Pulse: 4.4690e-04
steps:  11%| | 26/240 [01:13<10:07,  2.84s/it, Average key norm=0.136, Keys Scaled=0 | Loss: 0.0852 | Pulse: 4.9650e-04
steps:  11%| | 27/240 [01:15<09:57,  2.80s/it, Average key norm=0.152, Keys Scaled=0 | Loss: 0.0869 | Pulse: 5.3825e-04
steps:  12%| | 28/240 [01:17<09:45,  2.76s/it, Average key norm=0.169, Keys Scaled=0 | Loss: 0.0641 | Pulse: 6.0010e-04
steps:  12%| | 29/240 [01:19<09:36,  2.73s/it, Average key norm=0.188, Keys Scaled=0 | Loss: 0.0788 | Pulse: 6.5048e-04
steps:  12%|▏| 30/240 [01:21<09:28,  2.71s/it, Average key norm=0.208, Keys Scaled=0 | Loss: 0.0691 | Pulse: 7.0382e-04
steps:  13%|▏| 31/240 [01:23<09:20,  2.68s/it, Average key norm=0.23, Keys Scaled=6 | Loss: 0.0794 | Pulse: 7.3908e-04
steps:  13%|▏| 32/240 [01:24<09:11,  2.65s/it, Average key norm=0.252, Keys Scaled=13 | Loss: 0.0673 | Pulse: 7.8005e-04
steps:  14%|▏| 33/240 [01:26<09:04,  2.63s/it, Average key norm=0.273, Keys Scaled=17 | Loss: 0.0715 | Pulse: 8.1364e-04
steps:  14%|▏| 34/240 [01:28<08:57,  2.61s/it, Average key norm=0.295, Keys Scaled=23 | Loss: 0.0555 | Pulse: 1.2457e-04
steps:  15%|▏| 35/240 [01:30<08:48,  2.58s/it, Average key norm=0.298, Keys Scaled=25 | Loss: 0.0681 | Pulse: 1.4563e-03
steps:  15%|▏| 36/240 [01:32<08:41,  2.56s/it, Average key norm=0.331, Keys Scaled=31 | Loss: 0.0597 | Pulse: 9.9652e-04
steps:  15%|▏| 37/240 [01:34<08:36,  2.54s/it, Average key norm=0.354, Keys Scaled=39 | Loss: 0.0687 | Pulse: 1.2194e-03
steps:  16%|▏| 38/240 [01:35<08:30,  2.52s/it, Average key norm=0.382, Keys Scaled=54 | Loss: 0.0732 | Pulse: 1.3501e-03
steps:  16%|▏| 39/240 [01:37<08:22,  2.50s/it, Average key norm=0.409, Keys Scaled=74 | Loss: 0.0588 | Pulse: 1.1192e-03
steps:  17%|▏| 40/240 [01:39<08:17,  2.49s/it, Average key norm=0.431, Keys Scaled=90 | Loss: 0.0558 | Pulse: 7.6133e-04
steps:  17%|▏| 41/240 [01:40<08:10,  2.46s/it, Average key norm=0.445, Keys Scaled=98 | Loss: 0.0703 | Pulse: 1.1524e-03
steps:  18%|▏| 42/240 [01:43<08:05,  2.45s/it, Average key norm=0.464, Keys Scaled=10 | Loss: 0.0617 | Pulse: 1.0286e-03
steps:  18%|▏| 43/240 [01:45<08:01,  2.44s/it, Average key norm=0.48, Keys Scaled=113 | Loss: 0.0608 | Pulse: 9.3303e-04
steps:  18%|▏| 44/240 [01:46<07:55,  2.43s/it, Average key norm=0.494, Keys Scaled=11 | Loss: 0.0584 | Pulse: 8.0347e-04
steps:  19%|▏| 45/240 [01:48<07:52,  2.42s/it, Average key norm=0.506, Keys Scaled=12 | Loss: 0.0628 | Pulse: 8.9808e-04
steps:  19%|▏| 46/240 [01:50<07:47,  2.41s/it, Average key norm=0.518, Keys Scaled=13 | Loss: 0.0511 | Pulse: 6.0983e-04
steps:  20%|▏| 47/240 [01:52<07:44,  2.40s/it, Average key norm=0.526, Keys Scaled=13 | Loss: 0.0666 | Pulse: 8.9614e-04
steps:  20%|▏| 48/240 [01:55<07:40,  2.40s/it, Average key norm=0.536, Keys Scaled=14 | Loss: 0.0481 | Pulse: 5.6743e-04
steps:  20%|▏| 49/240 [01:57<07:37,  2.39s/it, Average key norm=0.543, Keys Scaled=15 | Loss: 0.0584 | Pulse: 7.3689e-05
steps:  21%|▏| 50/240 [01:59<07:32,  2.38s/it, Average key norm=0.543, Keys Scaled=15 | Loss: 0.0668 | Pulse: 1.4502e-03
steps:  21%|▏| 51/240 [02:01<07:28,  2.38s/it, Average key norm=0.557, Keys Scaled=16 | Loss: 0.0554 | Pulse: 1.1314e-03
steps:  22%|▏| 52/240 [02:03<07:25,  2.37s/it, Average key norm=0.568, Keys Scaled=17 | Loss: 0.0593 | Pulse: 1.1563e-03
steps:  22%|▏| 53/240 [02:05<07:22,  2.36s/it, Average key norm=0.578, Keys Scaled=18 | Loss: 0.0585 | Pulse: 1.1434e-03
steps:  22%|▏| 54/240 [02:07<07:17,  2.35s/it, Average key norm=0.589, Keys Scaled=19 | Loss: 0.0518 | Pulse: 8.6399e-04
steps:  23%|▏| 55/240 [02:09<07:14,  2.35s/it, Average key norm=0.596, Keys Scaled=20 | Loss: 0.0612 | Pulse: 1.0883e-03
steps:  23%|▏| 56/240 [02:11<07:11,  2.34s/it, Average key norm=0.605, Keys Scaled=21 | Loss: 0.0576 | Pulse: 1.0984e-03
steps:  24%|▏| 57/240 [02:13<07:08,  2.34s/it, Average key norm=0.613, Keys Scaled=23 | Loss: 0.0509 | Pulse: 8.3527e-04
steps:  24%|▏| 58/240 [02:15<07:06,  2.34s/it, Average key norm=0.619, Keys Scaled=24 | Loss: 0.0608 | Pulse: 1.0921e-03
steps:  25%|▏| 59/240 [02:18<07:03,  2.34s/it, Average key norm=0.627, Keys Scaled=24 | Loss: 0.0585 | Pulse: 1.2058e-03
steps:  25%|▎| 60/240 [02:20<07:01,  2.34s/it, Average key norm=0.635, Keys Scaled=26 | Loss: 0.0452 | Pulse: 7.1841e-04
steps:  25%|▎| 61/240 [02:22<06:58,  2.34s/it, Average key norm=0.64, Keys Scaled=268 | Loss: 0.0555 | Pulse: 7.1676e-05
steps:  26%|▎| 62/240 [02:24<06:55,  2.33s/it, Average key norm=0.64, Keys Scaled=268 | Loss: 0.0563 | Pulse: 1.4515e-03
steps:  26%|▎| 63/240 [02:26<06:51,  2.33s/it, Average key norm=0.649, Keys Scaled=28 | Loss: 0.0536 | Pulse: 1.3169e-03
steps:  27%|▎| 64/240 [02:28<06:49,  2.33s/it, Average key norm=0.656, Keys Scaled=29 | Loss: 0.0417 | Pulse: 7.8884e-04
steps:  27%|▎| 65/240 [02:30<06:45,  2.32s/it, Average key norm=0.661, Keys Scaled=30 | Loss: 0.0554 | Pulse: 9.5848e-04
steps:  28%|▎| 66/240 [02:33<06:43,  2.32s/it, Average key norm=0.666, Keys Scaled=30 | Loss: 0.0488 | Pulse: 8.6210e-04
steps:  28%|▎| 67/240 [02:35<06:40,  2.32s/it, Average key norm=0.671, Keys Scaled=31 | Loss: 0.0495 | Pulse: 8.3668e-04
steps:  28%|▎| 68/240 [02:37<06:38,  2.32s/it, Average key norm=0.675, Keys Scaled=31 | Loss: 0.0645 | Pulse: 1.6251e-03
steps:  29%|▎| 69/240 [02:39<06:35,  2.31s/it, Average key norm=0.682, Keys Scaled=32 | Loss: 0.0574 | Pulse: 1.9358e-03
steps:  29%|▎| 70/240 [02:41<06:32,  2.31s/it, Average key norm=0.692, Keys Scaled=33 | Loss: 0.0585 | Pulse: 2.1356e-03
steps:  30%|▎| 71/240 [02:43<06:29,  2.30s/it, Average key norm=0.703, Keys Scaled=36 | Loss: 0.0456 | Pulse: 1.2472e-03
steps:  30%|▎| 72/240 [02:45<06:26,  2.30s/it, Average key norm=0.71, Keys Scaled=383 | Loss: 0.0511 | Pulse: 1.1678e-03
steps:  30%|▎| 73/240 [02:47<06:23,  2.30s/it, Average key norm=0.716, Keys Scaled=39 | Loss: 0.0484 | Pulse: 9.9377e-04
steps:  31%|▎| 74/240 [02:49<06:20,  2.29s/it, Average key norm=0.721, Keys Scaled=40 | Loss: 0.0638 | Pulse: 1.4500e-03
steps:  31%|▎| 75/240 [02:51<06:16,  2.28s/it, Average key norm=0.728, Keys Scaled=41 | Loss: 0.0521 | Pulse: 2.1442e-03
steps:  32%|▎| 76/240 [02:53<06:14,  2.28s/it, Average key norm=0.739, Keys Scaled=42 | Loss: 0.0492 | Pulse: 1.3759e-03
steps:  32%|▎| 77/240 [02:55<06:11,  2.28s/it, Average key norm=0.746, Keys Scaled=43 | Loss: 0.0555 | Pulse: 1.7404e-03
steps:  32%|▎| 78/240 [02:56<06:07,  2.27s/it, Average key norm=0.755, Keys Scaled=45 | Loss: 0.0386 | Pulse: 7.8443e-04
steps:  33%|▎| 79/240 [02:58<06:03,  2.26s/it, Average key norm=0.758, Keys Scaled=47 | Loss: 0.0537 | Pulse: 1.0494e-03
steps:  33%|▎| 80/240 [03:00<06:00,  2.26s/it, Average key norm=0.763, Keys Scaled=48 | Loss: 0.0521 | Pulse: 1.2134e-03
steps:  34%|▎| 81/240 [03:02<05:58,  2.25s/it, Average key norm=0.767, Keys Scaled=49 | Loss: 0.0403 | Pulse: 7.4133e-04
steps:  34%|▎| 82/240 [03:04<05:55,  2.25s/it, Average key norm=0.77, Keys Scaled=502 | Loss: 0.0539 | Pulse: 1.0614e-03
steps:  35%|▎| 83/240 [03:05<05:51,  2.24s/it, Average key norm=0.774, Keys Scaled=51 | Loss: 0.0562 | Pulse: 1.6833e-03
steps:  35%|▎| 84/240 [03:07<05:48,  2.24s/it, Average key norm=0.779, Keys Scaled=52 | Loss: 0.0469 | Pulse: 1.2679e-03
steps:  35%|▎| 85/240 [03:09<05:46,  2.23s/it, Average key norm=0.783, Keys Scaled=54 | Loss: 0.0432 | Pulse: 8.9502e-04
steps:  36%|▎| 86/240 [03:11<05:42,  2.22s/it, Average key norm=0.785, Keys Scaled=53 | Loss: 0.0544 | Pulse: 1.3678e-03
steps:  36%|▎| 87/240 [03:13<05:39,  2.22s/it, Average key norm=0.789, Keys Scaled=54 | Loss: 0.0489 | Pulse: 1.3311e-03
steps:  37%|▎| 88/240 [03:15<05:37,  2.22s/it, Average key norm=0.792, Keys Scaled=56 | Loss: 0.0469 | Pulse: 1.1627e-03
steps:  37%|▎| 89/240 [03:17<05:34,  2.22s/it, Average key norm=0.795, Keys Scaled=57 | Loss: 0.0481 | Pulse: 6.1434e-05
steps:  38%|▍| 90/240 [03:18<05:31,  2.21s/it, Average key norm=0.795, Keys Scaled=56 | Loss: 0.0425 | Pulse: 1.4469e-03
steps:  38%|▍| 91/240 [03:20<05:28,  2.21s/it, Average key norm=0.798, Keys Scaled=57 | Loss: 0.0542 | Pulse: 9.4021e-05
steps:  38%|▍| 92/240 [03:22<05:26,  2.21s/it, Average key norm=0.798, Keys Scaled=57 | Loss: 0.0420 | Pulse: 1.3279e-04
steps:  39%|▍| 93/240 [03:24<05:23,  2.20s/it, Average key norm=0.798, Keys Scaled=57 | Loss: 0.0472 | Pulse: 3.0000e-03
steps:  39%|▍| 94/240 [03:26<05:21,  2.20s/it, Average key norm=0.804, Keys Scaled=59 | Loss: 0.0475 | Pulse: 3.0000e-03
steps:  40%|▍| 95/240 [03:29<05:19,  2.20s/it, Average key norm=0.809, Keys Scaled=59 | Loss: 0.0500 | Pulse: 3.0000e-03
steps:  40%|▍| 96/240 [03:30<05:16,  2.20s/it, Average key norm=0.814, Keys Scaled=62 | Loss: 0.0478 | Pulse: 3.0000e-03
steps:  40%|▍| 97/240 [03:33<05:14,  2.20s/it, Average key norm=0.82, Keys Scaled=636 | Loss: 0.0549 | Pulse: 2.9594e-03
steps:  41%|▍| 98/240 [03:35<05:11,  2.20s/it, Average key norm=0.825, Keys Scaled=65 | Loss: 0.0403 | Pulse: 2.4600e-03
steps:  41%|▍| 99/240 [03:37<05:09,  2.19s/it, Average key norm=0.83, Keys Scaled=669 | Loss: 0.0386 | Pulse: 1.4198e-03
steps:  42%|▍| 100/240 [03:39<05:07,  2.20s/it, Average key norm=0.832, Keys Scaled=6 | Loss: 0.0521 | Pulse: 2.4461e-03
steps:  42%|▍| 101/240 [03:41<05:05,  2.20s/it, Average key norm=0.836, Keys Scaled=6 | Loss: 0.0523 | Pulse: 2.3548e-03
steps:  42%|▍| 102/240 [03:43<05:02,  2.20s/it, Average key norm=0.84, Keys Scaled=69 | Loss: 0.0461 | Pulse: 2.3031e-03
steps:  43%|▍| 103/240 [03:46<05:00,  2.20s/it, Average key norm=0.843, Keys Scaled=7 | Loss: 0.0463 | Pulse: 2.2717e-03
steps:  43%|▍| 104/240 [03:48<04:58,  2.20s/it, Average key norm=0.846, Keys Scaled=7 | Loss: 0.0621 | Pulse: 2.1432e-03
steps:  44%|▍| 105/240 [03:50<04:56,  2.19s/it, Average key norm=0.849, Keys Scaled=7 | Loss: 0.0553 | Pulse: 2.0287e-03
steps:  44%|▍| 106/240 [03:52<04:53,  2.19s/it, Average key norm=0.852, Keys Scaled=7 | Loss: 0.0391 | Pulse: 2.0335e-03
steps:  45%|▍| 107/240 [03:54<04:51,  2.19s/it, Average key norm=0.854, Keys Scaled=7 | Loss: 0.0506 | Pulse: 2.0196e-03
steps:  45%|▍| 108/240 [03:56<04:49,  2.19s/it, Average key norm=0.856, Keys Scaled=7 | Loss: 0.0500 | Pulse: 1.9995e-03
steps:  45%|▍| 109/240 [03:58<04:46,  2.19s/it, Average key norm=0.859, Keys Scaled=7 | Loss: 0.0495 | Pulse: 1.9803e-03
steps:  46%|▍| 110/240 [04:00<04:44,  2.19s/it, Average key norm=0.861, Keys Scaled=7 | Loss: 0.0450 | Pulse: 1.9902e-03
steps:  46%|▍| 111/240 [04:02<04:41,  2.19s/it, Average key norm=0.862, Keys Scaled=7 | Loss: 0.0476 | Pulse: 1.9955e-03
steps:  47%|▍| 112/240 [04:04<04:39,  2.19s/it, Average key norm=0.865, Keys Scaled=7 | Loss: 0.0480 | Pulse: 6.2102e-05
steps:  47%|▍| 113/240 [04:06<04:37,  2.18s/it, Average key norm=0.865, Keys Scaled=7 | Loss: 0.0444 | Pulse: 8.3431e-05
steps:  48%|▍| 114/240 [04:08<04:34,  2.18s/it, Average key norm=0.865, Keys Scaled=7 | Loss: 0.0452 | Pulse: 8.3963e-05
steps:  48%|▍| 115/240 [04:10<04:32,  2.18s/it, Average key norm=0.865, Keys Scaled=7 | Loss: 0.0481 | Pulse: 1.0528e-04
steps:  48%|▍| 116/240 [04:12<04:30,  2.18s/it, Average key norm=0.865, Keys Scaled=6 | Loss: 0.0464 | Pulse: 1.3919e-04
steps:  49%|▍| 117/240 [04:14<04:27,  2.18s/it, Average key norm=0.865, Keys Scaled=6 | Loss: 0.0424 | Pulse: 1.9338e-04
steps:  49%|▍| 118/240 [04:17<04:25,  2.18s/it, Average key norm=0.865, Keys Scaled=6 | Loss: 0.0424 | Pulse: 2.7696e-04
steps:  50%|▍| 119/240 [04:19<04:23,  2.18s/it, Average key norm=0.865, Keys Scaled=6 | Loss: 0.0549 | Pulse: 3.8734e-04
steps:  50%|▌| 120/240 [04:21<04:21,  2.18s/it, Average key norm=0.865, Keys Scaled=6 | Loss: 0.0491 | Pulse: 4.8350e-04
steps:  50%|▌| 121/240 [04:23<04:18,  2.18s/it, Average key norm=0.865, Keys Scaled=6 | Loss: 0.0425 | Pulse: 5.8657e-04
steps:  51%|▌| 122/240 [04:25<04:16,  2.17s/it, Average key norm=0.865, Keys Scaled=6 | Loss: 0.0337 | Pulse: 9.0114e-04
steps:  51%|▌| 123/240 [04:26<04:13,  2.17s/it, Average key norm=0.865, Keys Scaled=6 | Loss: 0.0422 | Pulse: 1.3016e-03
steps:  52%|▌| 124/240 [04:28<04:11,  2.17s/it, Average key norm=0.866, Keys Scaled=7 | Loss: 0.0348 | Pulse: 9.8096e-04
steps:  52%|▌| 125/240 [04:30<04:08,  2.16s/it, Average key norm=0.867, Keys Scaled=7 | Loss: 0.0486 | Pulse: 1.0672e-03
steps:  52%|▌| 126/240 [04:32<04:06,  2.16s/it, Average key norm=0.867, Keys Scaled=7 | Loss: 0.0409 | Pulse: 1.1356e-03
steps:  53%|▌| 127/240 [04:34<04:04,  2.16s/it, Average key norm=0.868, Keys Scaled=7 | Loss: 0.0352 | Pulse: 1.0915e-03
steps:  53%|▌| 128/240 [04:36<04:01,  2.16s/it, Average key norm=0.869, Keys Scaled=7 | Loss: 0.0479 | Pulse: 1.0749e-03
steps:  54%|▌| 129/240 [04:38<03:59,  2.16s/it, Average key norm=0.869, Keys Scaled=7 | Loss: 0.0389 | Pulse: 1.1047e-03
steps:  54%|▌| 130/240 [04:40<03:56,  2.15s/it, Average key norm=0.87, Keys Scaled=71 | Loss: 0.0388 | Pulse: 1.2829e-03
steps:  55%|▌| 131/240 [04:42<03:54,  2.15s/it, Average key norm=0.871, Keys Scaled=7 | Loss: 0.0533 | Pulse: 1.2275e-03
steps:  55%|▌| 132/240 [04:43<03:52,  2.15s/it, Average key norm=0.871, Keys Scaled=7 | Loss: 0.0386 | Pulse: 1.2121e-03
steps:  55%|▌| 133/240 [04:45<03:49,  2.15s/it, Average key norm=0.872, Keys Scaled=7 | Loss: 0.0431 | Pulse: 1.1963e-03
steps:  56%|▌| 134/240 [04:47<03:47,  2.14s/it, Average key norm=0.873, Keys Scaled=7 | Loss: 0.0440 | Pulse: 1.1770e-03
steps:  56%|▌| 135/240 [04:49<03:45,  2.14s/it, Average key norm=0.873, Keys Scaled=7 | Loss: 0.0435 | Pulse: 1.1590e-03
steps:  57%|▌| 136/240 [04:51<03:42,  2.14s/it, Average key norm=0.874, Keys Scaled=7 | Loss: 0.0425 | Pulse: 1.1466e-03
steps:  57%|▌| 137/240 [04:52<03:40,  2.14s/it, Average key norm=0.875, Keys Scaled=7 | Loss: 0.0379 | Pulse: 1.1566e-03
steps:  57%|▌| 138/240 [04:55<03:38,  2.14s/it, Average key norm=0.875, Keys Scaled=7 | Loss: 0.0379 | Pulse: 1.1761e-03
steps:  58%|▌| 139/240 [04:57<03:35,  2.14s/it, Average key norm=0.876, Keys Scaled=7 | Loss: 0.0419 | Pulse: 1.1804e-03
steps:  58%|▌| 140/240 [04:59<03:33,  2.14s/it, Average key norm=0.877, Keys Scaled=7 | Loss: 0.0426 | Pulse: 1.1726e-03
steps:  59%|▌| 141/240 [05:00<03:31,  2.13s/it, Average key norm=0.877, Keys Scaled=7 | Loss: 0.0360 | Pulse: 1.1876e-03
steps:  59%|▌| 142/240 [05:02<03:28,  2.13s/it, Average key norm=0.877, Keys Scaled=7 | Loss: 0.0361 | Pulse: 1.2116e-03
steps:  60%|▌| 143/240 [05:04<03:26,  2.13s/it, Average key norm=0.878, Keys Scaled=7 | Loss: 0.0473 | Pulse: 1.1861e-03
steps:  60%|▌| 144/240 [05:06<03:24,  2.13s/it, Average key norm=0.878, Keys Scaled=7 | Loss: 0.0388 | Pulse: 1.1765e-03
steps:  60%|▌| 145/240 [05:08<03:22,  2.13s/it, Average key norm=0.878, Keys Scaled=7 | Loss: 0.0379 | Pulse: 1.1784e-03
steps:  61%|▌| 146/240 [05:10<03:19,  2.13s/it, Average key norm=0.879, Keys Scaled=7 | Loss: 0.0513 | Pulse: 1.1244e-03
steps:  61%|▌| 147/240 [05:11<03:17,  2.12s/it, Average key norm=0.879, Keys Scaled=7 | Loss: 0.0340 | Pulse: 1.1257e-03
steps:  62%|▌| 148/240 [05:13<03:15,  2.12s/it, Average key norm=0.879, Keys Scaled=7 | Loss: 0.0413 | Pulse: 1.1196e-03
steps:  62%|▌| 149/240 [05:15<03:12,  2.12s/it, Average key norm=0.88, Keys Scaled=69 | Loss: 0.0450 | Pulse: 1.0932e-03
steps:  62%|▋| 150/240 [05:17<03:10,  2.12s/it, Average key norm=0.88, Keys Scaled=71 | Loss: 0.0339 | Pulse: 1.1074e-03
steps:  63%|▋| 151/240 [05:18<03:07,  2.11s/it, Average key norm=0.881, Keys Scaled=7 | Loss: 0.0356 | Pulse: 1.1320e-03
steps:  63%|▋| 152/240 [05:20<03:05,  2.11s/it, Average key norm=0.881, Keys Scaled=7 | Loss: 0.0471 | Pulse: 1.1069e-03
steps:  64%|▋| 153/240 [05:22<03:03,  2.11s/it, Average key norm=0.881, Keys Scaled=7 | Loss: 0.0396 | Pulse: 1.0929e-03
steps:  64%|▋| 154/240 [05:24<03:01,  2.11s/it, Average key norm=0.882, Keys Scaled=7 | Loss: 0.0383 | Pulse: 1.0902e-03
steps:  65%|▋| 155/240 [05:26<02:59,  2.11s/it, Average key norm=0.882, Keys Scaled=7 | Loss: 0.0393 | Pulse: 1.0880e-03
steps:  65%|▋| 156/240 [05:28<02:56,  2.10s/it, Average key norm=0.882, Keys Scaled=7 | Loss: 0.0366 | Pulse: 1.0977e-03
steps:  65%|▋| 157/240 [05:29<02:54,  2.10s/it, Average key norm=0.882, Keys Scaled=7 | Loss: 0.0489 | Pulse: 1.0558e-03
steps:  66%|▋| 158/240 [05:31<02:52,  2.10s/it, Average key norm=0.883, Keys Scaled=7 | Loss: 0.0361 | Pulse: 1.0500e-03
steps:  66%|▋| 159/240 [05:33<02:49,  2.10s/it, Average key norm=0.883, Keys Scaled=7 | Loss: 0.0384 | Pulse: 1.0508e-03
steps:  67%|▋| 160/240 [05:35<02:47,  2.10s/it, Average key norm=0.883, Keys Scaled=7 | Loss: 0.0323 | Pulse: 1.0808e-03
steps:  67%|▋| 161/240 [05:36<02:45,  2.09s/it, Average key norm=0.884, Keys Scaled=7 | Loss: 0.0402 | Pulse: 1.0863e-03
steps:  68%|▋| 162/240 [05:38<02:43,  2.09s/it, Average key norm=0.884, Keys Scaled=7 | Loss: 0.0374 | Pulse: 1.0913e-03
steps:  68%|▋| 163/240 [05:40<02:40,  2.09s/it, Average key norm=0.884, Keys Scaled=7 | Loss: 0.0304 | Pulse: 1.1273e-03
steps:  68%|▋| 164/240 [05:42<02:38,  2.09s/it, Average key norm=0.885, Keys Scaled=7 | Loss: 0.0454 | Pulse: 1.1035e-03
steps:  69%|▋| 165/240 [05:44<02:36,  2.09s/it, Average key norm=0.885, Keys Scaled=7 | Loss: 0.0316 | Pulse: 1.1178e-03
steps:  69%|▋| 166/240 [05:45<02:34,  2.08s/it, Average key norm=0.885, Keys Scaled=7 | Loss: 0.0355 | Pulse: 1.1295e-03
steps:  70%|▋| 167/240 [05:47<02:31,  2.08s/it, Average key norm=0.886, Keys Scaled=7 | Loss: 0.0438 | Pulse: 1.0977e-03
steps:  70%|▋| 168/240 [05:49<02:29,  2.08s/it, Average key norm=0.886, Keys Scaled=7 | Loss: 0.0372 | Pulse: 1.0793e-03
steps:  70%|▋| 169/240 [05:51<02:27,  2.08s/it, Average key norm=0.886, Keys Scaled=6 | Loss: 0.0347 | Pulse: 1.0800e-03
steps:  71%|▋| 170/240 [05:53<02:25,  2.08s/it, Average key norm=0.886, Keys Scaled=6 | Loss: 0.0386 | Pulse: 1.0700e-03
steps:  71%|▋| 171/240 [05:55<02:23,  2.08s/it, Average key norm=0.887, Keys Scaled=7 | Loss: 0.0468 | Pulse: 1.0159e-03
steps:  72%|▋| 172/240 [05:56<02:21,  2.07s/it, Average key norm=0.887, Keys Scaled=6 | Loss: 0.0347 | Pulse: 1.0027e-03
steps:  72%|▋| 173/240 [05:58<02:18,  2.07s/it, Average key norm=0.887, Keys Scaled=7 | Loss: 0.0345 | Pulse: 1.0095e-03
steps:  72%|▋| 174/240 [06:00<02:16,  2.07s/it, Average key norm=0.887, Keys Scaled=6 | Loss: 0.0478 | Pulse: 9.6331e-04
steps:  73%|▋| 175/240 [06:02<02:14,  2.07s/it, Average key norm=0.887, Keys Scaled=7 | Loss: 0.0373 | Pulse: 9.4412e-04
steps:  73%|▋| 176/240 [06:04<02:12,  2.07s/it, Average key norm=0.887, Keys Scaled=7 | Loss: 0.0305 | Pulse: 9.6863e-04
steps:  74%|▋| 177/240 [06:06<02:10,  2.07s/it, Average key norm=0.887, Keys Scaled=7 | Loss: 0.0352 | Pulse: 9.9101e-04
steps:  74%|▋| 178/240 [06:08<02:08,  2.07s/it, Average key norm=0.887, Keys Scaled=6 | Loss: 0.0296 | Pulse: 1.0358e-03
steps:  75%|▋| 179/240 [06:10<02:06,  2.07s/it, Average key norm=0.887, Keys Scaled=6 | Loss: 0.0431 | Pulse: 1.0229e-03
steps:  75%|▊| 180/240 [06:13<02:04,  2.07s/it, Average key norm=0.887, Keys Scaled=6 | Loss: 0.0295 | Pulse: 1.0474e-03
steps:  75%|▊| 181/240 [06:15<02:02,  2.07s/it, Average key norm=0.888, Keys Scaled=6 | Loss: 0.0353 | Pulse: 1.0589e-03
steps:  76%|▊| 182/240 [06:17<02:00,  2.07s/it, Average key norm=0.888, Keys Scaled=6 | Loss: 0.0448 | Pulse: 1.0148e-03
steps:  76%|▊| 183/240 [06:19<01:58,  2.07s/it, Average key norm=0.888, Keys Scaled=6 | Loss: 0.0313 | Pulse: 1.0140e-03
steps:  77%|▊| 184/240 [06:20<01:55,  2.07s/it, Average key norm=0.888, Keys Scaled=6 | Loss: 0.0395 | Pulse: 9.9302e-04
steps:  77%|▊| 185/240 [06:22<01:53,  2.07s/it, Average key norm=0.888, Keys Scaled=6 | Loss: 0.0323 | Pulse: 9.9848e-04
steps:  78%|▊| 186/240 [06:25<01:51,  2.07s/it, Average key norm=0.888, Keys Scaled=6 | Loss: 0.0324 | Pulse: 1.0148e-03
steps:  78%|▊| 187/240 [06:27<01:49,  2.07s/it, Average key norm=0.888, Keys Scaled=6 | Loss: 0.0331 | Pulse: 1.0310e-03
steps:  78%|▊| 188/240 [06:29<01:47,  2.07s/it, Average key norm=0.888, Keys Scaled=6 | Loss: 0.0376 | Pulse: 1.0226e-03
steps:  79%|▊| 189/240 [06:31<01:45,  2.07s/it, Average key norm=0.889, Keys Scaled=6 | Loss: 0.0472 | Pulse: 9.5409e-04
steps:  79%|▊| 190/240 [06:33<01:43,  2.07s/it, Average key norm=0.889, Keys Scaled=6 | Loss: 0.0338 | Pulse: 9.3014e-04
steps:  80%|▊| 191/240 [06:35<01:41,  2.07s/it, Average key norm=0.889, Keys Scaled=6 | Loss: 0.0323 | Pulse: 9.3473e-04
steps:  80%|▊| 192/240 [06:37<01:39,  2.07s/it, Average key norm=0.889, Keys Scaled=6 | Loss: 0.0346 | Pulse: 9.4072e-04
steps:  80%|▊| 193/240 [06:40<01:37,  2.07s/it, Average key norm=0.889, Keys Scaled=6 | Loss: 0.0288 | Pulse: 9.7428e-04
steps:  81%|▊| 194/240 [06:42<01:35,  2.07s/it, Average key norm=0.889, Keys Scaled=6 | Loss: 0.0447 | Pulse: 1.8021e-04
steps:  81%|▊| 195/240 [06:44<01:33,  2.07s/it, Average key norm=0.889, Keys Scaled=6 | Loss: 0.0302 | Pulse: 2.4756e-04
steps:  82%|▊| 196/240 [06:46<01:31,  2.07s/it, Average key norm=0.889, Keys Scaled=6 | Loss: 0.0334 | Pulse: 3.5074e-04
steps:  82%|▊| 197/240 [06:48<01:29,  2.07s/it, Average key norm=0.889, Keys Scaled=6 | Loss: 0.0366 | Pulse: 4.1741e-04
steps:  82%|▊| 198/240 [06:50<01:27,  2.07s/it, Average key norm=0.889, Keys Scaled=6 | Loss: 0.0350 | Pulse: 4.0088e-04
steps:  83%|▊| 199/240 [06:53<01:25,  2.08s/it, Average key norm=0.889, Keys Scaled=6 | Loss: 0.0291 | Pulse: 1.2180e-04
steps:  83%|▊| 200/240 [06:55<01:23,  2.08s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0334 | Pulse: 1.6500e-04
steps:  84%|▊| 201/240 [06:57<01:21,  2.08s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0342 | Pulse: 1.3615e-04
steps:  84%|▊| 202/240 [06:59<01:18,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0269 | Pulse: 1.8108e-04
steps:  85%|▊| 203/240 [07:01<01:16,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0382 | Pulse: 2.4045e-04
steps:  85%|▊| 204/240 [07:03<01:14,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0343 | Pulse: 3.2899e-04
steps:  85%|▊| 205/240 [07:04<01:12,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0276 | Pulse: 4.7702e-04
steps:  86%|▊| 206/240 [07:06<01:10,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0367 | Pulse: 4.5283e-04
steps:  86%|▊| 207/240 [07:08<01:08,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0369 | Pulse: 4.2706e-04
steps:  87%|▊| 208/240 [07:10<01:06,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0264 | Pulse: 4.8553e-04
steps:  87%|▊| 209/240 [07:12<01:04,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0294 | Pulse: 5.7935e-04
steps:  88%|▉| 210/240 [07:14<01:02,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0295 | Pulse: 6.4994e-04
steps:  88%|▉| 211/240 [07:16<00:59,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0302 | Pulse: 6.5958e-04
steps:  88%|▉| 212/240 [07:18<00:57,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0396 | Pulse: 6.3001e-04
steps:  89%|▉| 213/240 [07:20<00:55,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0377 | Pulse: 5.9290e-04
steps:  89%|▉| 214/240 [07:21<00:53,  2.07s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0299 | Pulse: 5.8227e-04
steps:  90%|▉| 215/240 [07:23<00:51,  2.06s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0296 | Pulse: 5.8522e-04
steps:  90%|▉| 216/240 [07:25<00:49,  2.06s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0296 | Pulse: 5.9420e-04
steps:  90%|▉| 217/240 [07:27<00:47,  2.06s/it, Average key norm=0.889, Keys Scaled=5 | Loss: 0.0338 | Pulse: 5.9061e-04
steps:  91%|▉| 218/240 [07:29<00:45,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0310 | Pulse: 5.8996e-04
steps:  91%|▉| 219/240 [07:31<00:43,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0360 | Pulse: 5.7338e-04

```

</details>

```prompt
steps:  92%|▉| 220/240 [07:33<00:41,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0282 | Pulse: 5.7678e-04
steps:  92%|▉| 221/240 [07:35<00:39,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0257 | Pulse: 6.3375e-04
steps:  92%|▉| 222/240 [07:36<00:37,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0293 | Pulse: 6.5310e-04
steps:  93%|▉| 223/240 [07:39<00:34,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0270 | Pulse: 8.0220e-04
steps:  93%|▉| 224/240 [07:40<00:32,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0358 | Pulse: 7.8447e-04
steps:  94%|▉| 225/240 [07:43<00:30,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0265 | Pulse: 7.9048e-04
steps:  94%|▉| 226/240 [07:45<00:28,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0311 | Pulse: 7.8554e-04
steps:  95%|▉| 227/240 [07:47<00:26,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0337 | Pulse: 7.6315e-04
steps:  95%|▉| 228/240 [07:49<00:24,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0355 | Pulse: 7.2582e-04
steps:  95%|▉| 229/240 [07:51<00:22,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0290 | Pulse: 7.1396e-04
steps:  96%|▉| 230/240 [07:53<00:20,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0270 | Pulse: 7.2321e-04
steps:  96%|▉| 231/240 [07:55<00:18,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0346 | Pulse: 7.0759e-04
steps:  97%|▉| 232/240 [07:57<00:16,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0269 | Pulse: 7.1505e-04
steps:  97%|▉| 233/240 [07:59<00:14,  2.06s/it, Average key norm=0.888, Keys Scaled=4 | Loss: 0.0299 | Pulse: 7.1921e-04
steps:  98%|▉| 234/240 [08:01<00:12,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0289 | Pulse: 7.2497e-04
steps:  98%|▉| 235/240 [08:03<00:10,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0300 | Pulse: 7.2554e-04
steps:  98%|▉| 236/240 [08:05<00:08,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0330 | Pulse: 7.0941e-04
steps:  99%|▉| 237/240 [08:07<00:06,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0255 | Pulse: 7.1996e-04
steps:  99%|▉| 238/240 [08:09<00:04,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0315 | Pulse: 7.1460e-04
steps: 100%|▉| 239/240 [08:11<00:02,  2.06s/it, Average key norm=0.888, Keys Scaled=5 | Loss: 0.0336 | Pulse: 6.9194e-04
steps: 100%|█| 240/240 [08:13<00:00,  2.06s/it, Average key norm=0.888, Keys Scaled=5

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


