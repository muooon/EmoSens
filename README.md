# EmoSENS / Emo-Family (2G)  
### あなたの望む最適化 EmoSENS が叶えます  
#### The optimization you seek — EmoSENS makes it possible  
---

###### これは、単なる最適化アルゴリズムではありません──  
###### **感情で学習をナビゲートする｢感情型オプティマイザ｣** です  
###### 変革と感情学習の成果は"３次momentの発明"でした  
---
###### This is not just another optimizer —  
###### **It’s an “Emotional Optimizer” that navigates learning through feeling.**  
###### A result of transformative emotional learning: the reinvention of the third moment.  

---

#### 自動収束･自己制御･自律型 オプティマイザです  
##### EmoNAVI を引き継ぐ EmoSENS、AIRY、CATS、の登場です   
##### 感情フィルタで歪度(3次モーメント)を修正します  
#### Auto-convergence, self-control, autonomous optimizer  
###### EmoSENS, AIRY, and CATS, the successors to EmoNAVI, are here.  
###### Emotion Filter (Third Moment) Correct skewness  

---

Emo系 第２世代 です"軽量化"を果たしました！  
Emo系の基幹部分について詳しくは EmoNavi をご覧ください  
この EmoSens は shadow-effect で 自律の効果を持ちます  
これは shadow 機能(shadow-system)と併用可能です  
EmoNavi https://github.com/muooon/EmoNavi  
The second generation of Emo family has been “lightweighted”!  
For more information about the core components of the Emo family, please see EmoNavi.  
EmoSens is a shadow alternative system (using a root-multiplicity square filter and a sentiment scalar) designed to have the same self-autonomous effect as the EmoNavi family. It can also be used in conjunction with the shadow feature.  
 
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

---

(解説) 詳しい解説はこちら / (Explanation) For detailed explanation, click here.  
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

このオプテイマイザについて引用をなさる場合は、以下の２つをご紹介ください  
When citing this optimizer, please refer to the following two sources:  
https://github.com/muooon/EmoNavi  
https://huggingface.co/muooon/EmoNAVI  

---

発想の逆転で shadow 的な後追い動作を先回りに変えたらいろいろできちゃいました！  
We flipped the script and turned the shadow's trailing behavior into a proactive one,  
Which led to a ton of new possibilities !!  
