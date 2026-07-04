# SurvAlign-P Architecture Flow

## Canonical Pipeline (Phase 2)

```text
                    ┌──────────────────────────────────┐
                    │  Frozen AlignMark Embedder        │
                    │  (Encoder → WM Model → Decoder    │
                    │   → AudioFusion)                  │
                    └──────────┬───────────────────────-┘
                               │
               ┌───────────────┼───────────────┐
               │               │               │
           x (clean)      x_wm (watermarked)   r₀ = x_wm − x
               │               │               │
               ▼               ▼               ▼
           STFT(x)         STFT(x_wm)      STFT(r₀)
               │                               │
       ┌───────┴──────────────────────────┐    │
       │          4-Channel Input          │    │
       │  ┌──────────────────────────────┐ │    │
       │  │ Ch.1: |STFT(x)|   (clean)   │ │    │
       │  │ Ch.2: |STFT(r₀)|  (residual)│ │    │
       │  │ Ch.3: Guide Map *            │ │    │
       │  │ Ch.4: Masking Proxy          │ │    │
       │  └──────────────────────────────┘ │    │
       └───────────┬──────────────────────-┘    │
                   │                            │
                   ▼                            │
        ┌─────────────────────┐                 │
        │  CNN Gate (3 Conv2d) │                 │
        │  hidden_dim=16       │                 │
        │  receptive field=5×5 │                 │
        └─────────┬───────────┘                 │
                  │                             │
                  ▼                             │
        scale = 1 + ε·tanh(logits)              │
              ε = 0.2 (default)                 │
              range: [0.8, 1.2]                 │
                  │                             │
                  ▼                             ▼
              Rg = scale ⊙ STFT(r₀)   ←────── STFT(r₀)
                  │
                  ▼
              ISTFT(Rg)
                  │
                  ▼
        ┌─────────────────────────┐
        │  Hard L2 Projection      │
        │  • cap:   ‖rg‖ ≤ ‖r₀‖   │
        │  • equal: ‖rg‖ = ‖r₀‖   │
        └─────────┬───────────────┘
                  │
                  ▼
          x_gated = x + rg_projected
                  │
                  ▼
        ┌─────────────────────────────────┐
        │  Attack (train / validation /    │
        │         test — must be disjoint) │
        │  † Identity STE for codec       │
        │    backward (see Limitations)    │
        └─────────┬───────────────────────┘
                  │
                  ▼
        ┌─────────────────────────┐
        │  Frozen AlignMark       │
        │  Decoder                │
        │  → CE Loss (training)   │
        │  → Metrics (evaluation) │
        └─────────────────────────┘
```

\* **Guide Map options:**

| Map Type | Source | Decoder-free? |
|---|---|---|
| `survival` | Physical residual retention × dominance across attacks | ✅ Yes |
| `gradient_saliency` | Clean-input decoder gradient magnitude | ❌ No |
| `codec_utility` | Attacked decoder CE signed gradient w.r.t. residual scale | ❌ No |
| `random` / `constant` / `shuffled` / `energy` | Ablation controls | N/A |

† **Identity STE limitation:** SpeechTokenizer reconstruction uses `x + (T(x) − x).detach()`.
Backward treats the codec as an identity function. The Gate therefore **cannot learn
the codec's actual frequency selectivity** through backpropagation. This is an inherent
limitation of the current differentiable proxy and must be acknowledged.

---

## Decoder-free Analytic Ablation

```text
Survival Map S ∈ [0, 1]
        │
        ▼
scale = (1 − ε) + 2ε · S       ← No decoder, no training
        │
        ▼
    L2 Projection
        │
        ▼
  x_gated = x + rg_projected
```

This condition uses **no decoder optimization** and isolates the pure information
content of the Survival prior. Comparing `analytic_survival` with `proposed_gate`
reveals how much value decoder-supervised training adds beyond the physical prior.

---

## Training Loss Composition

```text
L_total = L_robust + λ_dev · L_deviation + λ_mask · L_masking + λ_tv · L_tv

where:
  L_robust   = mean CE over train attacks (decoder-supervised)
  L_deviation = mean (scale − 1)²  (regularization toward identity)
  L_masking  = mean [ReLU(scale − 1) · (1 − masking_proxy)]²
               (penalizes amplification in perceptually exposed bins)
  L_tv       = total variation of scale map (spatial smoothness)
```
