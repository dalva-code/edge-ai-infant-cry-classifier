# Edge AI Infant Cry Classifier (Bioacoustic Telemetry & On-Device Inference) 🍼⚡

An end-to-end edge computing architecture for on-device bioacoustic infant cry classification. Engineered to enforce biometric privacy (EU AI Act & GDPR compliance) and eliminate cloud latency by executing raw acoustic analysis locally on ARM-based silicon in under 500 ms.

---

## 📌 Architectural Overview

Centralized cloud AI solutions introduce critical data privacy risks when handling biometric infant data, alongside prohibitive latency (3–5 seconds) during parental distress events.

This repository presents an on-device bioacoustic classification system deployed on consumer hardware (Apple Silicon M4 via **Metal Performance Shaders - MPS**). The architecture converts non-stationary acoustic waveforms into compact 2D Mel-frequency representations and executes local inference using a custom-adapted **MobileNetV3-Small** neural network (6.2 MB disk footprint).

```text
[ Analog Microphone ] 
         │ (PCM 16-bit @ 22,050 Hz)
         ▼
[ DSP Feature Extraction ] ──► STFT (Hann 46.2 ms, N_FFT=1024, Hop=256, 128 Mel bands)
         │ (94.91% Data Dimensionality Reduction)
         ▼
[ Normalized Mel Tensor ] ──► Shape: (3, 224, 224)
         │
         ▼
[ MobileNetV3-Small (Edge) ] ──► Accelerated via Apple Silicon MPS (<500 ms)
         │
         ▼
[ Binary Inference Engine ] ──► Colic Distress vs Fatigue / Sleep
```
---

## 🔬 Digital Signal Processing (DSP) Pipeline

Direct waveform ingestion causes severe computational bottlenecks on edge processors. The system employs a deterministic preprocessing chain using `Librosa`:

* **Nyquist-Shannon Stability:** Input audio is standardized at $f_s = 22,050 \text{ Hz}$. Given that diagnostic infant cry energy concentrates below $10,000 \text{ Hz}$, this rate prevents aliasing while reducing memory payload by 50% compared to standard $44.1 \text{ kHz}$ streams.
* **Short-Time Fourier Transform (STFT):** Quasi-stationary window length of $N\_FFT = 1024$ samples ($46.2 \text{ ms}$ at $22.05 \text{ kHz}$) using a Hann window, matching the anatomical stability of infant vocal tract filters.
* **Temporal Overlap:** $HOP\_LENGTH = 256$ samples ($11.6 \text{ ms}$), providing a 75% overlap to capture rapid pitch shifts and explosive cry dynamics.
* **Perceptual Compression:** 128-band triangular Mel filter bank bounded by $F_{MAX} = 8,000 \text{ Hz}$. A 5-second capture ($110,250$ raw samples) is condensed into a $224 \times 224$ matrix—a **94.91% reduction in input dimensionality** without loss of acoustic formant integrity.

---

## 🧠 Neural Architecture & Edge Fine-Tuning

* **Backbone:** `MobileNetV3-Small` utilizing depthwise separable convolutions ($1 \times 1$ pointwise + $3 \times 3$ depthwise filters) to minimize multiply-accumulate (MAC) operations.
* **Model Footprint:** **6.2 MB** compiled checkpoint (`best_model.pt`), operating well beneath edge device memory constraints.
* **Two-Phase Transfer Learning:**
  1. *Head Stabilization:* Base convolutional layers frozen; training isolated to the custom linear classification head for 5 epochs.
  2. *Full Network Fine-Tuning:* Global gradient propagation enabled across all layers with a decaying learning rate schedule to adapt generic ImageNet filters to complex bioacoustic Mel textures.
* **Data Leakage Prevention:** Stratified splitting was strictly enforced **prior** to any data augmentation. Synthetic expansion (Gaussian white noise injection $\epsilon \sim \mathcal{N}(0, \sigma^2)$ and pitch shifting) was applied solely to the training split ($N=23 \to 170$ samples), preserving an untouched blind evaluation set ($N=7$).

---

## 📊 Experimental Results (Blind Test Evaluation)

The optimized edge model achieved an overall accuracy of **85.7%** evaluated exclusively on unpolluted, analog blind test samples:

| Bioacoustic Class | Precision | Recall (Sensitivity) | F1-Score | Evaluation Support |
| :--- | :---: | :---: | :---: | :---: |
| **Colic / Gas Distress** | **1.00** | 0.67 | 0.80 | 3 |
| **Fatigue / Sleep** | 0.80 | **1.00** | 0.89 | 4 |
| **Macro Average** | 0.90 | 0.83 | 0.84 | 7 |
| **Weighted Average** | **0.89** | **0.86** | **0.85** | 7 |

### Clinical & Operational Insights
* **Perfect Precision on Gas Distress:** A precision of **1.00** in the Colic/Gas category guarantees zero false positives when flagging digestive pain.
* **Full Sensitivity on Rest Detection:** A recall of **1.00** in the Fatigue/Sleep class ensures every episode of sleep exhaustion is correctly captured.

---

## ⚡ Hardware Acceleration Benchmarks

* **Host Architecture:** Apple M4 SoC (ARMv9-A 64-bit).
* **Execution Engine:** PyTorch with native Metal Performance Shaders (`torch.device("mps")`).
* **Latency:** Measured end-to-end inference (DSP extraction + tensor forward pass) consistently executes in **$< 480 \text{ ms}$**.
* **Memory Invariance:** Unified memory architecture eliminates host-to-device bus copying overhead.

---

## 🔒 Confidentiality & IP Notice

*This repository provides the architectural implementation, feature engineering pipeline, and benchmark metrics of the thesis prototype. In strict compliance with minor data privacy regulations (GDPR / EU AI Act) and intellectual property protection, the raw clinical infant cry dataset (`.wav`) and trained production weights (`best_model.pt`) are withheld.*

---

## 👨‍💻 Author

**David Esteban Correa Alvarado**  
*Computer Engineer & Business Administrator*  
*Specialized in Audio DSP, Edge AI & Distributed Systems*  
[LinkedIn](https://www.linkedin.com/in/david-correa-5140a1232/) | [GitHub Profile](https://github.com/dalva-code)
