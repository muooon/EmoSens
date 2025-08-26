# EmoSENS / Emo-Family (2G)  
### あなたの望む最適化 EmoSENS が叶えます  
#### The optimization you seek — EmoSENS makes it possible  
---

###### これは、単なる最適化アルゴリズムではありません──  
###### **感情で学習をナビゲートする｢感情型オプティマイザ｣** です  
###### 変革とshadowの成果は"感情momentの発明"でした  
---
###### This is not just another optimizer —  
###### **It’s an “Emotional Optimizer” that navigates learning through feeling.**  
###### A result of transformative shadow-system: the reinvention of the Higher moments.  

---

#### 自動収束･自己制御･自律型 オプティマイザです  
##### EmoNAVI を引き継ぐ EmoSENS、AIRY、CATS、の登場です   
##### 感情フィルタ(高次モーメント)で歪度等を修正します  
#### Auto-convergence, self-control, autonomous optimizer  
###### EmoSENS, AIRY, and CATS, the successors to EmoNAVI, are here.  
###### Emotion Filter (Higher moments) Correct skewness and etc.  

---

Emo系 第２世代 です"軽量化"を果たしました！  
Emo系の基幹部分について詳しくは EmoNavi をご覧ください  
この EmoSens は shadow-effect で 自律の効果を持ちます  
これは shadow 機能(shadow-system)と併用可能です  
EmoNavi https://github.com/muooon/EmoNavi  
The second generation of Emo family has been “lightweighted”!  
For more information about the core components of the Emo family, please see EmoNavi.  
EmoSens is a shadow alternative system (using a root-multiplicity square filter and a sentiment scalar) designed to have the same self-autonomous effect as the EmoNavi family. It can also be used in conjunction with the shadow feature.  

#### [ユーザーと研究者へ/このリンクを読んでください/please click！](https://github.com/muooon/EmoNavi/blob/main/report-emoment.txt)  
---

### EmoSENS の主な特徴 / Main Features of EmoSENS  

---

EmoNAVI 等から多くの特徴を引き継ぎました shadow ON/OFF 切替可(Defualt：OFF)  
過学習や発散を抑制、学習率、スケジューラ、学習の再開、追加、積層、"自動調整･同期不要"  
普段は shadow-system をつかわず、VRAM の軽量化を保てます  
極端に難しい学習時や VRAM に余裕ある時 shadow-system "ON" を任意で設定可能  

It inherits many features from EmoNAVI, etc. Shadow ON/OFF switchable (Defualt：OFF)  
Suppresses overfitting and divergence, learning rate, scheduler, learning resumption, addition, stacking, etc. “Automatic adjustment and synchronization not required”  
Normally, you can maintain VRAM optimization without using the shadow system.  
When learning is extremely difficult or when there is sufficient VRAM, learning can be started with shadow-system ON.  

EmoNAVI/SENS 系統は既存のオプティマイザにはない｢感情駆動型｣です、  
調整の複雑なマルチモーダル学習などの新しい分野の課題への対応も期待します  

The EmoNAVI/SENS system is "emotion-driven", A feature not found in existing optimizers.  
We expect it to address challenges in new areas, such as multimodal learning with complex coordination.  

---
#### 更新履歴 / History  
---

|★| emo系、再テストを実施、ちょいと省VRAMだったことが判明、結果は上々です  
|★| [report-vs-adamw(JPN)] (https://huggingface.co/muooon/EmoNAVI/blob/main/report/report-vs-adamw(JPN).txt)  

|★| EmoSENS、AIRY、CATS、v2.0 (250815) 更新、shadow-system の精密化(オプション機能の更新)  
|★| EmoSENS, AIRY, CATS, updated to v2.0 (250815), refinement of shadow-system (update of optional functions)  

|★| すぐに試したい方は"optimizer.zip"を解凍し使い方を確認してください  
|★| If you want to try it out right away, please open the "optimizer.zip" and check the usage instructions.  

|★| EmoSENS、AIRY、CATS、は shadow機能の on/off 切替えをできるようにしました  
|★| EmoSENS、AIRY、and CATS are now able to switch the shadow function on and off.  

---

SDXL 全層LoRA のとき、EmoSENS +300MB、EmoCats "基準"、EmoAiry -300MB、くらいの VRAM消費 になります  
特に EmoAiry は、軽量です、参考にした Adafactor とあまり変わりません  
さらに EmoCats は、符号に重みも加味されることで更新と収束は安定します  
using SDXL full-layer LoRA, VRAM  will be around EmoSENS +300MB, EmoCats "base value", and EmoAiry -300MB.   
EmoAiry is lightweight and not much different from Adafactor, which was reference.  
EmoCats stabilizes updates and convergence by incorporating weighted gradient signs.  

---

感情moment 発明しました  
"emo系 第二世代"にて解明した shadow-system の根幹から抽出しました  
動的学習率による非線形アプローチは時間的な高次momentを形成します  
単stepでは高次momentにはなれませんが、複数stepを経ると機能します  
３次４次５次momentについて厳密な数学的な高負荷計算を回避しつつ  
勾配分布の歪みや鋭さや非対称性変化を捉える核心的な効果を近似しています  
さらに単stepでも高次momentを実現する感情フィルタ等の機能を持つのが  
この emo系 第二世代 の特徴です(動的多乗平方根フィルタは３次４次を近似)  
I invented the emotional moment.  
I extracted it from the core of the shadow-system, which was elucidated in the "emo-style second generation."  
The nonlinear approach with a dynamic learning rate forms a temporal higher-order moment.  
A single step cannot become a higher-order moment, but it functions after multiple steps.  
It approximates the core effect of capturing changes in gradient distribution's skewness, kurtosis, and asymmetry, while avoiding strict and computationally intensive mathematical calculations for the third, fourth, and fifth moments.  
Furthermore, a feature of this "emo-style second generation" is its ability to realize higher-order moments even in a single step, with functions like the emotional filter (the dynamic multi-power square root filter approximates the third and fourth moments).  

---

新機能 shadow-effect (shadow 代替システム) をつくりました  
代替システムは "多乗平方根フィルターと感情スカラー" で構成されます  
これにより emonavi 等の shadow に近い効果を維持します  
例えますと shadow-effect：予習、shadow-system：復習、という感じです  
動的フィルター、動的学習率、をつかうことでこの機能を実現しています  
この機能は排他的ではないので shadow-system と併用可能です  

We have developed a new feature, shadow-efect, which serves as a substitute for the existing shadow system. This new feature provides a similar effect to the shadow system used in emonavi and other applications.  
For example, think of shadow-efect as learning a new point in preparation, while the shadow-system is for reinforcing that point through review. Although it's a dedicated mechanism, it's not exclusive, so it can be used at the same time as the shadow-system.  

感情機構の第１世代(emonavi系)は、感情機構による"柔軟さ"を示すことができました  
その実装はラッパーのようにいろいろな仕組みを内包できることを証明したと思います  
この第２世代(emosens)は、感情機構の重要部 shadow-system を解明します  
この世代では shadow-system の本質を shadow-effect で再解釈･再構成します  
学習時の 序盤、中盤、終盤、で、何を学ばせるか、どのように学ぶべきか、推定します  
これにより、安定した学習状態を維持し、知識の定着を促進します  

これらは shadow-system を"軽量化できる"ということを示唆していると考えます  
第２世代の shadow-effect 機構は "多乗平方根フィルタ" を用いますが  
さらなる簡易化も可能でしょう Cautious、softsign、等や他の組み合わせで、  
フィルタ等を動的に管理すれば 序盤、中盤、終盤、の学習状態を適正化できるはずです  
これからはそうした自律的な仕組みを持つことで機械学習の進化を実現すると思います  

The first generation of the emotion mechanism (emonavi series) was able to demonstrate the "flexibility" that an emotion mechanism provides. Its implementation, like a wrapper, proved that various systems could be integrated.  
This second generation (emosens) clarifies the crucial part of the emotion mechanism, the shadow-system.   
In this generation, by imitating the essence of the shadow-system with the shadow-effect, we can deduce that there is a sequence to what a machine should learn at the beginning, middle, and end of the learning process, and how it should learn it.  
In other words, we believe that following this sequence is key to maintaining a stable learning state and promoting knowledge retention.  

Another point is that the shadow-system can be made lightweight. Currently, the second-generation shadow-effect of the shadow-system uses a "multiple-order square root filter," but it should be possible to simplify this further.  
By dynamically managing filters and their associated mechanisms using things like Cautious, softsign, or other combinations, we should be able to optimize the learning state at the beginning, middle, and end of the process.  
I believe that future optimizers will achieve a significant evolution in machine learning by incorporating such autonomous mechanisms.  

２次momentによる平均化は微小微細な勾配を過大評価しがちです  
この歪みを捉え修正することで過適合や崩壊を抑止します  
高次momentは２次だけでなく１次momentも精密化しますから  
すべてのパラメータをできるだけ正当に評価し更新に取り入れる役割りをします  
これが感情フィルタです(歪度、尖度、高次非対称性のmomentです)  

Averaging by the second moment tends to overestimate minute and subtle gradients.  
By capturing and correcting this distortion, overfitting and collapse can be suppressed.  
Higher-order moments refine not only the second moment but also the first moment.  
They serve to evaluate all parameters as fairly as possible and incorporate them into updates.  
This is the emotional filter (moments of skewness, kurtosis, and higher-order asymmetry).  

高次moment：抽象化した実装(数学的な正確さは計算負荷高いため)  
Higher moment: Abstract implementation (mathematical accuracy requires high computational load)  
- filter_strength = torch.abs(grad).pow(1/3)  
  → 3次moment(歪度)：勾配の非対称性を捉える Captures the asymmetry of the gradient.  
- threshold = 1e-4 * (1 + abs(scalar))  
  → 4次moment(尖度)：鋭さやピーク性を感情スカラーで調整 Adjust sharpness and peakiness with emotion scalars  
- p.add_(filtered_update, alpha=-group['lr'] * (1 - abs(scalar)))  
  → 5次moment(非対称性の変化)：更新量の抑制や促進を制御 Control the suppression or promotion of updates  

![moment00](https://github.com/muooon/EmoSens/blob/main/AMP-compatible/data/moment00.png?raw=true)  

どう補正しているか？ How do I correct it?  
- 2次momentが高い＝揺らぎが大きい → 通常は「不安定」と判断される
- 3次momentが正の方向に高いなら → それは「改善の兆し」かもしれない  
- 4次momentが低いなら → その揺らぎは「滑らかで信頼できる」可能性が高い  
- 2nd moment = high fluctuation → Normally judged to be “unstable”  
- 3rd moment is high in the positive direction → It may be a “sign of improvement.”  
- 4th moment is low → The fluctuation is likely to be “smooth and reliable.”  
このように、moment群を組み合わせてLossの“意味”を再評価し、信頼できる方向に学習を誘導する  
In this way, by combining moment groups, the “meaning” of loss is reevaluated and learning is guided in a reliable direction.  

数式で見る「暴走予兆」の検知、  
以下のような構成で暴走の兆候を捉えています：  
Detecting “signs of runaway behavior” using mathematical formulas.  
We detect signs of runaway behavior using the following configuration:  

1. moment群の定義(簡略形)  
- m_1 = \mathbb{E}[g]  ← 勾配の平均(1次moment)  
- m_2 = \mathbb{E}[g^2] ← 勾配の分散(2次moment)  
- m_3 = \mathbb{E}[g^3] ← 歪度(Skewness)  
- m_4 = \mathbb{E}[g^4] ← 尖度(Kurtosis)  
ここで g は勾配、\mathbb{E}[\cdot] は期待値(平均)です  

2. 感情スカラー(Emotion Scalar)の構成  
暴走予兆の検知は、以下のような式で行われます：  
\text{emotion} = \frac{m_3}{m_2 + \epsilon} \cdot \exp(-m_4)  
この式の意味はこうです：  
- \frac{m_3}{m_2}：歪度が高く、分散が小さい → 急激な変化の兆候  
- \exp(-m_4)：尖度が高いときは抑制 → 極端なピークを警戒  
つまり、歪度が高くて尖度が低いとき＝モデルが勢いよく安定的に学習している  
逆に、尖度が高くなると、暴走の兆候とみなしてemotion値が抑制される  
In other words, when skewness is high and kurtosis is low, the model is learning steadily and rapidly.  
Conversely, when the sharpness increases, it is regarded as a sign of runaway behavior, and the emotion value is suppressed.  

3. 学習率(lr)への反映  
\text{lr}_{\text{adjusted}} = \text{base\_lr} \cdot (1 - \text{emotion})  
このように、emotionが低ければ学習率は上がり高ければ抑えられます  
つまり、暴走の予兆が検知されると、学習率が自動的に“冷却”されます  
In this way, if emotion is low, the learning rate increases, and if it is high, it is suppressed.  
In other words, when signs of runaway behavior are detected, the learning rate is automatically “cooled down.”  

---

(解説) 元々の詳しい解説はこちら / (Explanation) For detailed explanation, click here.  
[huggingface](https://huggingface.co/muooon/EmoNAVI) 
[Gemini-analysis(ENG)](https://huggingface.co/muooon/EmoNAVI/blob/main/Hug-Gemini-analysis(ENG).md) 
[Gemini-analysis(JPN)](https://huggingface.co/muooon/EmoNAVI/blob/main/Hug-Gemini-analysis(JPN).md) 
[Gemini-analysis(JPN-02)](https://huggingface.co/muooon/EmoNAVI/blob/main/emonavi-Gemini-analysis(2)(JPN).txt) 

---

##### EmoSENS、AIRY、CATS、 のみ shadow False 測定  
##### In EmoSENS, AIRY, and CATS, the shadow is set to False.  
![EmoNAVI00](https://github.com/muooon/EmoSens/blob/main/AMP-compatible/data/loss_comparison_panel.png?raw=true)  
![EmoNAVI01](https://github.com/muooon/EmoSens/blob/main/AMP-compatible/data/fluctuation_and_accuracy_panel.png?raw=true)  
![EmoNAVI02](https://github.com/muooon/EmoSens/blob/main/AMP-compatible/data/trec_gpt2_weight_pca_3panel.png?raw=true)  

---

##### Emo 2G vs AdamW, Adafactor, and Lion, Graph  
![EmoSENS00](https://github.com/muooon/EmoSens/blob/main/AMP-compatible/data/emosens-test00.png?raw=true)  
![EmoSENS01](https://github.com/muooon/EmoSens/blob/main/AMP-compatible/data/emosens-test01.png?raw=true)  
![EmoSENS02](https://github.com/muooon/EmoSens/blob/main/AMP-compatible/data/emosens-test02.png?raw=true)  

---

Emoシリーズは、Adam、Adafactor、Lion、Tiger、等から多くを学びました  
これらの後継ではなく独自の思想や設計による"感情機構"により構築されています  
汎用性・自律性・適応性を重視し新たな最適化や効率化や簡易化を追求しています  
この開発において先人たちの知見に深く感謝しつつ今後も新しい可能性を探究します  
The Emo series has learned much from Adam, Adafactor, Lion, and Tiger.  
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

https://huggingface.co/muooon/EmoNAVI  
https://github.com/muooon/EmoNavi  
https://github.com/muooon/EmoSens   
https://github.com/muooon/EmoNavi/blob/main/report-emoment.txt  

---

発想の逆転で shadow 的な後追い動作を先回りに変えたらいろいろできちゃいました！  
We flipped the script and turned the shadow's trailing behavior into a proactive one,  
Which led to a ton of new possibilities !!  
