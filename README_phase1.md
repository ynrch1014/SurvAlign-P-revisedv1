# Phase 1: Controlled Attribution Diagnostics

## Purpose

Phase 1은 **학습을 하지 않습니다**. 다음 질문을 공정하게 검증합니다:

> 동일한 잔차 에너지와 실제 공격 조건에서, Survival 기반 T-F 선택이 residual-energy,
> random, decoder saliency 및 codec-aware utility보다 복호 정보를 더 잘 보존하는가?

## Compared Conditions

| Condition | Type | Description |
|---|---|---|
| High-Survival Top-k | Physical prior | Residual retention × dominance |
| Low-Survival Top-k | Negative control | Worst-surviving bins |
| Clean Gradient Saliency | Decoder-derived | `∂L_dec/∂x_wm` magnitude |
| Codec-aware Signed Utility | Decoder-derived | `−∂L_dec(attacked)/∂α(f,t)` |
| Residual-Energy Top-k | Signal-level | Loudest residual bins |
| Speech-Energy Top-k | Signal-level | Loudest clean-speech bins |
| VAD Top-k | Signal-level | Voice-activity energy bins |
| Random Top-k (×20) | Statistical baseline | Repeated random selection |

## Evaluation Protocol

- 마스킹 후 **실제 공격을 통과시켜** 평가 (clean-only decoding이 아님)
- `natural` (자연 보존 에너지) / `equal` (동일 에너지 통제) 실험 분리
- Wilcoxon signed-rank 및 sign-flip permutation tests
- Bit Accuracy, Exact-message Accuracy, Attribution FAR, decoder CE, logit margins

## Important Cautions

1. `gradient_saliency`는 clean 입력에서의 gradient magnitude이고,
   `codec_utility`는 공격 후 decoder CE의 residual scale에 대한 signed gradient.
   **두 개념을 혼용하지 마세요.**

2. `survival_attacks`와 `eval_attacks`가 같은 codec family를 공유하면 결과는
   "seen-attack" 성능입니다. 일반화 주장에는 `--strict_heldout`를 사용하세요.

3. Exploratory dataset에서 발견한 결과를 confirmatory dataset에서 별도 검증해야 합니다.

## Usage

```bash
python phase1_attribution.py \
  --dataset_type librispeech --dataset_name train-clean-100 \
  --split test --random_repeats 20 \
  --survival_attacks noise,lowpass,resample,reconstruct_nq6,spectral_proxy \
  --eval_attacks clean,bandpass,ffmpeg_mp3 \
  --strict_heldout \
  --energy_modes natural,equal
```

실제 held-out 모델 평가:
```bash
  --eval_attacks clearervoice,facodec \
  --clearervoice_command "python tools/run_clearervoice.py --input {input} --output {output}" \
  --facodec_command "python tools/run_facodec.py --input {input} --output {output}"
```

## Output

- `results/phase1_confirmatory/phase1_summary.json`
- `phase1_sample_results.csv`
- `phase1_correlations.csv`
- `map_comparison.png`
