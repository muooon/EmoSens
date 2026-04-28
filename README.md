## emo series Optimizers  

⭐ If you like this project, please give it a star ⭐  

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


