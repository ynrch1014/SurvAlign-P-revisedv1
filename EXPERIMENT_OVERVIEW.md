# SurvAlign-P Experiment Overview

## Research Question

Feature-Aligned (AlignMark) achieves strong average bit accuracy, but specific reconstruction
conditions cause non-trivial **Exact-message failures and Attribution FAR**. This project tests:

> At a **fixed L2 residual-energy budget**, can limited T-F redistribution toward a
> physically-surviving subspace recover those failures without degrading perceptual quality?

---

## Scope and Honest Limitations

1. **"Decoder-free" applies only to the Survival Map**, not to the Gate training.
   The Gate is trained with the frozen AlignMark decoder's CE loss.
   Only `analytic_survival` mode is truly decoder-free.

2. **Identity STE**: Codec backward uses `x + (T(x) − x).detach()`, so the Gate
   cannot learn the codec's actual frequency selectivity through backpropagation.

3. **Same L2 ≠ same imperceptibility**. PESQ/STOI/SI-SDR must be reported alongside
   accuracy metrics; equal energy does not guarantee equal perceptual quality.

---

## Phase 1 — Controlled Diagnostic (No Training)

Phase 1 does **not** train anything. It answers: "Which T-F prior best preserves
decoding information under equal-energy masking?"

### Compared Conditions
| Condition | Description |
|---|---|
| High-Survival Top-k | Physical residual-retention × dominance |
| Low-Survival Top-k | Inverted survival (negative control) |
| Clean Gradient Saliency | `∂L_dec/∂x_wm` magnitude on clean input |
| Codec-aware Signed Utility | `−∂L_dec(attacked)/∂α(f,t)` |
| Residual-Energy Top-k | Loudest residual bins |
| Speech-Energy Top-k | Loudest clean-speech bins |
| VAD Top-k | Voice-activity energy bins |
| Random Top-k (×20) | Repeated random baseline |

### Evaluation Protocol
- Masking → **actual attack** → decoding (not clean-only decoding)
- `natural` and `equal` energy conditions separated
- Wilcoxon signed-rank and sign-flip permutation tests for paired comparisons
- Metrics: Bit Accuracy, Exact-message Accuracy, Attribution FAR, decoder CE, logit margins

### Key Caution
- Exploratory and confirmatory datasets **must be separate**
- If `survival_attacks` and `eval_attacks` share a codec family, label results as "seen-attack"

---

## Phase 2 — Gate Training and Paired Evaluation

### Attack Protocol (Must Be Disjoint)

| Stage | Role | Example |
|---|---|---|
| **Survival Map** | Physical prior generation | noise, lowpass, resample, reconstruct_nq6, spectral_proxy |
| **Train** | Decoder CE optimization | noise, lowpass, resample, reconstruct_nq6 |
| **Validation** | Checkpoint selection | bandpass, reconstruct_nq8 |
| **Test (held-out)** | Final evaluation | ffmpeg_mp3, facodec, clearervoice, encodec |

> **Critical**: Default batch scripts use `strong_speechtokenizer` as test, which shares
> the `speechtokenizer` family with Map attacks. For generalization claims, use
> `--strict_heldout` with genuinely external codecs.

### Experimental Modes

| Mode | Guide Map | Trainable? | Decoder-free? | Purpose |
|---|---|---|---|---|
| `baseline` | — | No | — | AlignMark original (reference) |
| `uniform_upper` | — | No | — | 1.1× residual amplification upper bound |
| `analytic_survival` | Survival | No | ✅ **Yes** | Pure physical prior, no optimization |
| `proposed_gate` + `survival` | Survival | Yes | Map only | **Main proposed method** |
| `proposed_gate` + `gradient_saliency` | Gradient | Yes | No | Decoder-derived alternative |
| `proposed_gate` + `codec_utility` | Codec utility | Yes | No | Attack-aware decoder alternative |
| `constant_gate` | Ones | Yes | — | "Does any gate help?" |
| `random_gate` | Random | Yes | — | "Is the map information needed?" |
| `shuffled_survival` | Shuffled Surv. | Yes | — | "Does spatial structure matter?" |
| `energy_gate` | Speech energy | Yes | — | "Is speech loudness enough?" |

### Critical Comparisons

1. **`analytic_survival` vs `proposed_gate`**: If performance gap is small,
   the Survival prior alone is sufficient (stronger claim). If large,
   decoder-supervised training adds value beyond the prior.

2. **`cap` vs `equal` projection**: `equal` mode is the strictly controlled condition.
   If improvement holds under `equal`, it proves redistribution alone is effective.

3. **Recovery Rate vs Regression Rate**: Recovery = baseline failures rescued.
   Regression = baseline successes broken. Both must be reported honestly.

---

## Metrics

### Watermark Robustness Metrics
| Metric | Description |
|---|---|
| Bit Accuracy / BER | Per-bit correct rate |
| **Exact-message Accuracy** | All 16 bits correct (strict) |
| **Attribution FAR (strict)** | Ties count as failure |
| Attribution FAR (lenient) | Only strictly closer wrong message |
| Tie Rate | Correct and nearest-wrong at same Hamming distance |
| Hamming Attribution Margin | Gap between correct and nearest-wrong distance |
| Decoder CE | Cross-entropy loss on chunk logits |
| Min/Mean Logit Margin | Correct-class logit minus best-wrong logit |
| Logit Entropy | Prediction uncertainty per chunk |

### Paired Comparison Metrics
| Metric | Description |
|---|---|
| **Failure Recovery Rate** | % of baseline failures rescued by Gate |
| **Regression Rate** | % of baseline successes broken by Gate |

### Perceptual Quality Metrics
| Metric | Description |
|---|---|
| PESQ (wideband) | Perceptual speech quality |
| STOI | Short-time objective intelligibility |
| SI-SDR | Scale-invariant signal-to-distortion ratio |
| L2 Ratio | Energy ratio vs original residual |
| Peak Amplitude | Maximum absolute sample value |
| Clipping Ratio | Fraction of samples exceeding ±1.0 |

---

## Distortion / Attack Types

### Internal Differentiable Attacks (Training & Map)
| Attack | Description | Parameters |
|---|---|---|
| `noise` | Additive white Gaussian noise | SNR=20dB |
| `noise10db` | Stronger AWGN | SNR=10dB |
| `lowpass` | FIR low-pass filter | cutoff=4kHz |
| `bandpass` | FIR band-pass filter | 300–3400Hz |
| `resample` | Down/up-sample | rate=2× |
| `reconstruct_nq6` | SpeechTokenizer n_q=6 (identity STE) | — |
| `reconstruct_nq8` | SpeechTokenizer n_q=8 (identity STE) | — |
| `strong_speechtokenizer` | SpeechTokenizer n_q=2 (aggressive) | — |
| `spectral_proxy` | Spectral compression proxy (NOT real MP3) | cutoff_ratio=0.7 |

### External Held-out Attacks (Test Only)
| Attack | Description | Requirement |
|---|---|---|
| `ffmpeg_mp3` | Real MP3 encode/decode via ffmpeg | ffmpeg on PATH |
| `clearervoice` | 10dB noise + neural denoising | External wrapper |
| `clearervoice_only` | Denoising only (no added noise) | External wrapper |
| `facodec` | FACodec neural codec | External wrapper |
| `encodec` | Meta EnCodec | External wrapper |
| `dac` | Descript Audio Codec | External wrapper |
| `vocos` | Vocos vocoder | External wrapper |
| `hifigan` | HiFi-GAN vocoder | External wrapper |

> **Proxy ≠ Real**: `spectral_proxy` is NOT MP3. `strong_speechtokenizer` is NOT FACodec.
> Always label proxy results accordingly and test with real held-out codecs for generalization claims.

---

## Primary Endpoints (Paper Table)

1. **16-bit Exact-message Accuracy** under held-out attacks (ffmpeg MP3, FACodec, ClearerVoice)
2. **Attribution FAR (strict)** under the same held-out attacks
3. **Failure Recovery Rate** and **Regression Rate** (paired with baseline)
4. **PESQ / STOI / SI-SDR** non-inferiority vs baseline
5. Decoder CE and minimum logit margin distribution shift
