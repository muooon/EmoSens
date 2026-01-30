# EmoSENS / Emo-Family (2ndGen-v3.8)  

### EmoSens 最新版 (v3.8) update  


#### Loss あるかぎり emoPulse(鼓動) はやまない ――  
##### “As long as there is loss, emoPulse(Heartbeat) will never stop —”  

##### Grokking を経ずに フラットミニマ へ到達できるかもしれない選択肢  
###### An option that might allow reaching Flat Minima without Grokking  


EmoSens (v3.8) emoPulse (完全自動学習率) 等の調整をしました  
EmoSens (v3.8) emoPulse (Fully Automatic Learning Rate) Adjustment  

v3.7以降の特徴  
- 完全自動学習率：高速化と精緻化を同時に達成しつつ初期LRに悩まなくていい  
- emoPulse：自律的にLRを増減させ"極低精度･超量子化"も安全安定で進行します  

Features in v3.7 and later  
- Fully Automatic Value Learning Rate: Achieves both acceleration and refinement while eliminating the need to worry about the initial learning rate.  
- emoPulse： Autonomously adjusts LR levels to safely and stably proceed with “ultra-low precision, ultra-quantization.”  

<div align="center">
  <img width="500" alt="emo-system001" src="https://github.com/user-attachments/assets/7e7160a9-046a-4212-bcde-d338c26ed846" />
</div>

初期LRは1.0で大丈夫です(データセットの工夫にあなたの時間を割いてください)  
The initial LR can be set to 1.0 (please focus your time on refining the dataset).  

Mathematical Explanation Here (paper) v3.7  
(非凸関数に対する期待値収束(フローマッチングへの適応なども保証します)  
(論文ではフラットミニマやグロッキングに対しての挙動も考察しています)  
Expected value convergence for non-convex functions  
(also guarantees adaptability to flow matching)  
(Providing a direct path to Flat Minima without the necessity of Grokking.)  
#### [emo-paper(article)](https://huggingface.co/muooon/EmoNAVI/raw/main/emo-v37-paper(ENG).txt)  
#### [数学的解説はこちら(論文)](https://huggingface.co/muooon/EmoNAVI/raw/main/emo-v37-paper(JPN).txt)  

---

emo系 v3.8 (スタンダードモデル) の特徴等  

| 名称      | 正確性 | メモリ負荷 | 備考                                      |  
|-----------|--------|------------|-------------------------------------------|  
| emosens   | ★★★★   | ★★         | 最初に誕生｜正確｜Adam型       |  
| emoairy   | ★★     | ★★★★       | ２番目に誕生｜軽量｜Adafactor型 |  
| emocats   | ★★★    | ★★★        | 軽量＆正確の両立｜Lion型         |  
| emotion   | ★★★    | ★★★★★      | "最軽量"で正確｜オリジナル型         |  

[効率性] 危険抑止更新：過学習や収束の停滞に先回りし無駄な更新を排除しながら進行します  
[機能性] 軽量で高機能：自動停止合図や完全自律型の分散学習への対応でユーザー体験を向上させます  
[信頼性] 安全優先設計：動的制御で学習の不安定な局面でモデルを保護し安定した収束を促します  
完全自律型のため、積層、再開、非同期、で、自由な学習を自由に組むことが可能です  

emo-series v3.8 (Standard-models) Features  

| Name    | Accurate | MemoryLoad | Notes                                           |  
|---------|----------|------------|--------------------------------------------------|  
| emosens | ★★★★     | ★★         | 1st born｜accurate｜Adam-type         |  
| emoairy | ★★       | ★★★★       | 2nd born｜Lightweight｜Adafactor-type |  
| emocats | ★★★      | ★★★        | Accurate and Lightweight｜Lion-type |  
| emocats | ★★★      | ★★★★★      | Lightest ＆ "most accurate"｜Origenal-type |

[Efficiency] Risk-Aware Updates: Proactively prevents overfitting and convergence stagnation while eliminating redundant updates.  
[Functionality] Lightweight and High-Performance: Enhances user experience through automatic stop signals and support for fully autonomous distributed learning.  
[Reliability] Safety-First Design: Protects the model during unstable learning phases with dynamic control, promoting stable convergence.  
Fully autonomous, enabling flexible learning configurations through layering, resumption, and asynchronous processing    

---

### 学習の情報、そのすべては Loss値 に集約されている  
#### Learning Information, Everything is consolidated into the Loss value.  

##### Loss値はモデルのshadowである、  
##### Loss値にすべてが集約されている、  
##### 学習状況もモデル状況もLoss値が教えてくれる、  
##### Lossを感じろ、 Lossこそオリジン(原点)だ、  
###### The loss value is the model's shadow.  
###### The loss value embodies everything.  
###### The loss value tells you both the learning status and the model's condition.  
###### Feel the loss. Loss is the origin.  

<details>
 
<summary> emotional moment </summary>  

"emo系 第二世代 v1.x"にて解明した shadow-system の根幹から抽出しました  
動的学習率による非線形アプローチは時間的な高次momentを形成します  
単stepでは高次momentにはなれませんが、複数stepを経ると機能します  
３次４次５次momentについて厳密な数学的な高負荷計算を回避しつつ  
勾配分布の歪みや鋭さや非対称性変化を捉える核心的な効果を近似しています  
I invented the emotional moment.  
I extracted it from the core of the shadow-system, which was elucidated in the "emo-style second generation v1.x."  
The nonlinear approach with a dynamic learning rate forms a temporal higher-order moment.  
A single step cannot become a higher-order moment, but it functions after multiple steps.  
It approximates the core effect of capturing changes in gradient distribution's skewness, kurtosis, and asymmetry, while avoiding strict and computationally intensive mathematical calculations for the third, fourth, and fifth moments.  

---

### あなたの望む最適化 EmoSens が叶えます  
#### The optimization you seek — EmoSens makes it possible  
---
###### これは、単なる最適化アルゴリズムではありません──  
###### **感情で学習をナビゲートする｢感情型オプティマイザ｣** です  
###### 変革と感情学習の成果は"ニューロンスパイクの再発明"でした  
---
###### This is not just another optimizer —  
###### **It’s an “Emotional Optimizer” that navigates learning through feeling.**  
###### A result of transformative emotional learning: the reinvention of the neural spike.  

---
#### 自動収束･自己制御･自律型 オプティマイザです  
##### EmoSens を中心に、EmoAiry、EmoCats、もあります   
#### Auto-convergence, self-control, autonomous optimizer  
###### It primarily features EmoSens, along with EmoAiry and EmoCats.  

</details>

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
| +側  | 安定・改善   |  0.20  | 0.80  |   0.70     |       0.51        | 維持(様子見)              |
| +側  | 理想的調和   |  0.45  | 0.55  |   0.20     |       6.25        | 加速(1.5倍成長)          |
| +側  | 一致（最大） |  0.50  | 0.50  |   0.10     |      25.00        | 最大加速(1.5倍成長)      |
| -側  | 軽い不一致   | -0.20  | -0.80 |   0.70     |       0.51        | 維持(様子見)              |
| -側  | 強い違和感   | -0.45  | -0.55 |   0.20     |       6.25        | 減速(0.8倍)              |
| -側  | 逆転一致     | -0.50  | -0.50 |   0.10     |      25.00        | 最大減速(0.8倍)          |

分母(noise_base)：abs(scalar - trust) が 0 に近づくほど(つまり感情スカラーと信頼度が一致するほど)、分母が最小値 0.1 に近づき2乗の結果は跳ね上がります。  
+側：dNR_now_val が高く、trust も高ければ、履歴(dNR_hist)を 最大1.50倍 ずつ成長させます。  
-側：たとえ dNR_now_val が 25.00 と計算されても、trust が低い(-0.5〜0.5の範囲)ため、履歴は 0.80倍 で削られブレーキがかかります。  
エントロピーの抑制：この表の数値(dNR_now_val)そのまま学習率にせず、これを dNR_hist(履歴)に入れ、最終的に emoScope × 1e-4 として極めて小さな安全な学習率(1e-6 〜 3e-3)へと変換されます。  

---  

### EmoSens 主な特徴 / Main Features of EmoSens  

---

<details>

過学習や発散を抑制、自己修復的機能をもちます  
学習率やスケジューラも自律調整、モデル自身で判断します  
学習の 再開、追加、積層、等で"引き継ぎ不要"、誰でも簡単です  
分散学習で 他ノード等との"同期不要"、完全自律です  
Self-repairing, with no over-learning or divergence  
Autonomously adjusts learning rate and scheduler, so models make their own decisions  
Resuming, adding, stacking, etc. learning is synchronization-free" and easy for everyone  
Distributed learning enables “no synchronization required” with other nodes, achieving full autonomy.  

emo系 は既存のオプティマイザにはない｢感情駆動型｣です、  
調整の複雑なマルチモーダル学習などの新しい分野の課題への対応も期待できます  
emo-based is “emotion-driven,” which is not the case with existing optimizers,  
We expect it to overcome the challenges we currently face,  
while also addressing challenges in new areas such as multimodal learning with complex coordination  

emo系は、観察、判断、決定、行動、記憶、反省、という自律サイクルを行います  
emo-based follows an autonomous cycle of   
observation, judgment, decision, action, memory, and reflection.  

高効率性と集積度  
高次moment、量子化補償(Kahan補償と違う制御)、分散･継続学習での独立性、自己修復･モデル修復、  
ハイパーパラメータの自律調整、信頼度フィルタ、更新ステップの有界性、構造的耐性、自己停止、  
動的学習率、動的スケジューラ、動的Rank/Aplha、履歴補償、などを含めた多機能性を、  
追加テンソル不要、計算負荷ほぼなし、step毎に完全適用、時間的積算で実現します  
これらをワンパッケージで実現した高効率性と集積度は安定と安全を最優先します  
※ 高次momentは近似的、動的Rank/Alphaも近似的な効果です  
※ LoRA系技術はノイズをなくしますが微小データも失う場合があります  
※ emo系はノイズを作らず既存ノイズを見つけて修正し微小データを保護します  
※ 量子化補償は今後実用化されるさらに低精度な環境でも柔軟に対応できます  
High Efficiency and Integration  
Multifunctionality, including higher-order moments, Quantization Compensation (Control Different from Kahan Compensation), independence in distributed and continual learning, self-healing and model repair,  
Autonomous hyperparameter tuning, confidence filtering, bounded update steps, structural robustness (or resilience), self-termination,  
dynamic learning rates, dynamic schedulers, dynamic Rank/Alpha, and historical compensation,  
is achieved without additional tensors, with negligible computational overhead, fully applied at every step, and through temporal accumulation.  
The high efficiency and integration realized in this single package prioritize stability and safety above all else.  
※ Higher-order moments are approximative, and the effects of dynamic Rank/Alpha are also approximative.  
※ LoRA-based techniques eliminate noise but may sometimes lose fine-grained data (or subtle details).  
※ Emo-based techniques detect and correct existing noise without generating new noise, thereby preserving fine-grained data.  
※ Quantization compensation offers flexible adaptability even in lower-precision environments expected to be commercialized (or practical) in the future.  

</details>

---

## グラフで見る emo系 の進行状況 Progress of emo-type as shown in the graph (v3.7.6)  
<img width="2218" height="1153" alt="emov376-003-tile" src="https://github.com/user-attachments/assets/a1c5891b-a842-4ed1-a147-d4658e1ca16b" />  
このように 動的学習率 として機能します ／ 下降しつづけるのは"元モデルの修正"の差分も学習しているかも？ <br>   
It functions as a dynamic learning rate. ／ Could the continuous decline be due to also learning the differences in “original model corrections”? <br> 
データセット状況(左)：全て実写画像10枚, 10batch, 300epoch(3000step), 全層LoRA, Rank16/Alpha16, e-pred, ZtSNR, <br>   
Dataset Status LEFT: Primarily 10 Photo images, 10 batch, 300 epochs (3000 steps), full-layer LoRA, Rank16/Alpha16, e-pred, ZtSNR,  <br>  
データセット状況(右)：主に白黒画像11枚, 1batch, 300epoch(3300step), 全層LoRA, Rank16/Alpha16, e-pred, ZtSNR, <br>   
Dataset Status RIGHT: Primarily 11 black-and-white images, 1 batch, 300 epochs (3300 steps), full-layer LoRA, Rank16/Alpha16, e-pred, ZtSNR,  <br>  
es = EmoSens(Red/Green)、ea = EmoAiry(Blue/Gray)、ec = EmoCats(Yellow/Orange)  

---

<details>

<summary> 更新履歴 / History </summary>  

|★| EmoSens世代 v3.8 (260130) emoPulse 機構等の調整  
|★| EmoSens Generation v3.8 (260130) Adjustments to emoPulse Mechanism, etc.   

|★| EmoSens、Airy、Cats、v3.7 (260101) Navi v3.6 を継承し完全自動高値学習率を実現しました(追加テンソルなし)、emoPulse 機構により劇的な進化を遂げました  
|★| EmoSens, Airy, Cats, v3.7 (260101) Building upon Navi v3.6, we have achieved fully automatic high-value learning rate optimization (without additional tensors), and through the emoPulse mechanism, we have achieved dramatic evolution.  

|★| EmoNavi、Fact、Lynx、v3.6 (251220) v3.1 を継承し高値自動学習率を実現しました(追加テンソルなし)、emoDrive 機構により劇的な進化を遂げました、開発終了とします  
|★| EmoNavi, Fact, Lynx, v3.6 (251220) Inherits v3.1 and achieves high-value automatic learning rate (no additional tensors), has undergone dramatic evolution through the emoDrive mechanism, development is now complete.  

|★| EmoNavi、Fact、Lynx、v3.3 (251204) v3.1 を継承し完全自動学習率を実現しました(追加テンソルなし)、感情機構の調整等でさらに安定するよう進化しました  
|★| EmoNavi, Fact, Lynx, v3.3 (251204) Inherits v3.1 and achieves fully automatic learning rate adjustment (without additional tensors), further evolving for greater stability through adjustments to the sentiment mechanism and other enhancements.  

|★| EmoNavi、Fact、Lynx、v3.1 (251201) v3.0 を継承しつつ効率化を進めました。感情機構のスケール調整等で広範なモデルで安定するよう進化しました  
|★| EmoNavi, Fact, Lynx, v3.1 (251201) We built upon v3.0 while enhancing efficiency. Through adjustments like scaling the emotion mechanism, we evolved the model for broader stability across diverse models.  

|★| EmoNavi、Fact、Lynx、Clan、Zeal、Neco、v3.0 (250825) emosens(第２世代)で解明した"高次moment"(近似)のフィードバックを適用(更新) 全て "shadow=False" です  
|★| EmoNavi, Fact, Lynx, Clan, Zeal, Neco, updated to v3.0 (250825), Incorporates (updates) feedback on “higher moments” (approximations) clarified by emosens (2nd generation). All are “shadow=False”  

これ以前は v2.0 レポジトリの更新履歴をご覧ください  
For updates prior to this, please refer to the v2.0 repository update history.  

</details>

---  

emo系 は 生物的反応で進化し続けます  
感覚神経系(multi-EMA)、内分泌系(tanh(scalar))、免疫系(shadow-system)、循環器系(emoPulse)、これらの統合により中枢神経系と自律神経系を形成し、高度な判断と決定を行うという自然的に自律した機構として存在します  

---  

<details>

<summary>EmoSens v3.7 以降 オプション指定方法<br>
EmoSens v3.7 and later Option Settings Guide</summary>  

|||オプション指定方法|||  
●shadow オフ(False にする)：  
use_shadow=False  
●eps(0除算防止)：  
eps=1e-8  

|||Usage examples|||  
●Shadow off:  
use_shadow=False  
●eps(Division by zero prevention)：  
eps=1e-8  


</details>

---

<details>


</details>

---

emoシリーズは、Adam、Adafactor、Lion、Tiger、等から多くを学びました  
これらの後継ではなく独自の思想や設計による"感情機構"というアプローチにより構築されています  
汎用性・自律性・適応性を重視し新たな最適化や効率化や簡易化を追求しています  
この開発において先人たちの知見に深く感謝しつつ今後も新しい可能性を探究します  
The emo series has learned much from Adam, Adafactor, Lion, and Tiger.  
Rather than being their successors, it is built upon a unique philosophy and design approach centered on "emotional mechanisms".  
It prioritizes generality, autonomy, and adaptability in pursuit of new paths for optimization, efficiency, and simplicity.  
In its development, we deeply appreciate the insights of those who came before us—and continue to explore new possibilities beyond them. 


### License Apache License 2.0 — see LICENSE for details.  
### ライセンス Apache License 2.0 — 詳細は LICENSE をご覧ください  

---

### 引用について / About citations  

---

このオプテイマイザについて引用をなさる場合は、以下をご紹介ください  
When citing this optimizer, please refer to the following sources:  

Official Code:  
https://github.com/muooon/EmoSens 

paper:  
https://huggingface.co/muooon/EmoNAVI/raw/main/emo-v37-paper(ENG).txt

---

emo-based is an “emotion-driven” approach not found in existing optimizers. By building each sensor around an “emotion mechanism” that differentiates multi-EMA and scalarizes it via nonlinear transformation (tanh), we enhanced overall learning stability and ensured accuracy. This performs an autonomous cycle of “observation, judgment, decision, action, memory, and reflection,” akin to a biological central nervous system. (Please take a look at the paper.)  

---

emo系は既存のオプティマイザにはない｢感情駆動型｣です。multi-emaを差分化し非線形変換(tanh)でscalar化した｢感情機構｣を中心に、各センサーを構築することで学習全体の安定性を向上させ正確性を確保しました、これらは生物の中枢神経系のように｢観察、判断、決定、行動、記憶、反省｣という自律サイクルを行います(論文をぜひご覧ください)  

