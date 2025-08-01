# EmoSENS / Emo-Family  
### あなたの望む最適化 EmoSENS が叶えます  
#### The optimization you seek — EmoSENS makes it possible  
---

#### 自動収束･自己制御･自律型 オプティマイザです  
##### EmoNAVI を引き継ぐ EmoSENS、AIRY、CATS、の登場です   
#### Auto-convergence, self-control, autonomous optimizer  
###### EmoSENS, AIRY, and CATS, the successors to EmoNAVI, are here.  

---

詳しくは EmoNavi をご覧ください  
こちらの EmoSens は shadow 代替システム(多乗平方根フィルターと感情スカラー)で  
EmoNavi ファミリーと同じような自己自律の効果を持つようにしています  
これは shadow 機能と併用も可能です  
EmoNavi https://github.com/muooon/EmoNavi  
For more details, please see EmoNavi.  
EmoSens is a shadow alternative system (using a root-multiplicity square filter and a sentiment scalar) designed to have the same self-autonomous effect as the EmoNavi family.  
It can also be used in conjunction with the shadow feature.  

---

### EmoSENS の主な特徴 / Main Features of EmoSENS  

---

EmoNAVI 等にある shadow 機能 ON/OFF 切替可能、多くの特徴を引き継ぎました  
過学習や発散を抑制、学習率、スケジューラ、学習の 再開、追加、積層、等 "自動調整･同期不要" 誰でも簡単です  
普段は shadow をつかわず、VRAM を軽量に保持可能となりました  
極端に難しい学習時には shadow を ON にして学習開始も可能です  
VRAM に余裕のある時は shadow を使うとより良い学習を行えます  

Engaging with EmoNAVI's core functionalities, we've introduced a new optimizer with a switchable ON/OFF shadow function. This new version retains many of our key features: it curbs overfitting and divergence, autonomously adjusts the learning rate and scheduler, and eliminates the need for synchronization when resuming, adding to, or stacking models. This makes it incredibly easy for anyone to use.  
Typically, the shadow function is kept off, which allows for minimal VRAM usage. However, for extremely difficult training tasks, you have the option to enable the shadow function. For those with ample VRAM, using the shadow function can lead to even better training performance.  

EmoNAVI/SENS 系統は既存のオプティマイザにはない｢感情駆動型｣です、  
調整の複雑なマルチモーダル学習などの新しい分野の課題への対応も期待します  
EmoNAVI/SENS system is “emotion-driven,” which is not the case with existing optimizers,  
We expect it to overcome the challenges we currently face,  
while also addressing challenges in new areas such as multimodal learning with complex coordination  

---
#### 更新履歴 / History
---

|★| EmoSENS、AIRY、CATS、は shadow機能の on/off 切替えをできるようにしました  
|★| EmoSENS、AIRY、and CATS are now able to switch the shadow function on and off.  

---

新たな機能、shadow-effect (shadow 代替システム) をつくりました  
これにより emonavi 等の shadow に近い効果を維持します  
例えますと shadow-effect：予習で重点を学ぶ、shadow-system：復習で重点を学ぶ、こういう感じです  
動的フィルター、動的学習率、をつかうことでこの機能を実現しています  
そしてこの機能は排他的ではないので shadow-system と同時に利用可能です  
We have developed a new feature, shadow-efect, which serves as a substitute for the existing shadow system.  
This new feature provides a similar effect to the shadow system used in emonavi and other applications.  
For example, think of shadow-efect as learning a new point in preparation, while the shadow-system is for reinforcing that point through review.  
This functionality is Although it's a dedicated mechanism, it's not exclusive, so it can be used at the same time as the shadow-system.  

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

Emoシリーズは、Adam、Adafactor、Lion、Tiger、等から多くを学びました  
これらの後継ではなく独自の思想や設計による"感情機構"というアプローチにより構築されています  
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
