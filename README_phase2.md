# Phase 2: Gate Training and Paired Evaluation

## Purpose

Phase 2의 궁극적 목표는 대규모 포렌식 추적(Planetary-scale forensic)에서 필수적인 **Exact-message (16-bit 전원 일치) 복원율**을 극대화하는 것입니다.

> **Why SurvAlign-P?**  
> 페이로드 용량을 절반으로 깎아먹는 오류 정정 코드(ECC)에 의존하거나, 기존 베이스라인 모델을 재학습하는 비용(Retraining)을 지불할 필요가 없습니다. Phase 2는 코덱의 Burst Error 특성을 역이용하는 물리적 `Survival Map`을 바탕으로, **비가청성(에너지 총량)을 그대로 유지(Equal Energy)하면서 워터마크를 스스로 안전지대로 대피시키는 지능형 Post-hoc Gate 네트워크**를 학습합니다.

## Attack Protocol (Must Be Disjoint)

| Stage | Role | Example Attacks |
|---|---|---|
| **Survival Map** | Physical prior 생성 | noise, lowpass, resample, reconstruct_nq6, spectral_proxy |
| **Train** | Decoder CE 학습 | noise, lowpass, resample, reconstruct_nq6 |
| **Validation** | Checkpoint 선택 | bandpass, reconstruct_nq8 |
| **Test** | **Held-out 최종 평가** | ffmpeg_mp3, facodec, clearervoice, encodec |

공격 이름 또는 codec family가 겹치면 스크립트가 경고합니다.
확증 실험에서는 `--strict_heldout`으로 누출을 오류 처리하세요.

> **주의**: `strong_speechtokenizer`를 test에 사용하면 Map attacks의
> `reconstruct_nq6`과 같은 SpeechTokenizer family이므로 일반화 주장 불가.

## Experimental Modes

| Mode | Guide Map | Trainable? | Decoder-free? |
|---|---|---|---|
| `baseline` | — | No | — |
| `uniform_upper` | — | No | — |
| `analytic_survival` | Survival | No | ✅ Yes |
| `proposed_gate` + `survival` | Survival | Yes | Map only |
| `proposed_gate` + `gradient_saliency` | Gradient | Yes | No |
| `proposed_gate` + `codec_utility` | Utility | Yes | No |
| `constant_gate` | Ones | Yes | — |
| `random_gate` | Random | Yes | — |
| `shuffled_survival` | Shuffled | Yes | — |
| `energy_gate` | Energy | Yes | — |

## Critical Comparisons

1. **`analytic_survival` vs `proposed_gate`**:
   Gap이 작으면 → prior만으로 충분 (stronger claim).
   Gap이 크면 → decoder-supervised 학습이 essential.

2. **`cap` vs `equal` projection**:
   `equal`에서도 개선되면 → 에너지가 아닌 분배의 효과 입증.

3. **Recovery vs Regression**:
   Recovery = baseline 실패 구제 비율. Regression = baseline 성공 파괴 비율.
   **둘 다 정직하게 보고해야 합니다.**

## Validation & Checkpointing

최고 모델은 calibration split의 **Exact-message Accuracy** 우선,
동률일 때 CE가 낮은 checkpoint를 선택합니다.

- `--min_validation_si_sdr_delta`: SI-SDR 하락 제한
- `--max_validation_clipping_ratio`: Clipping 비율 제한

## Energy Control

| Mode | Description |
|---|---|
| `cap` | 원래 residual L2를 초과하지 않음 (실사용) |
| `equal` | 정확히 동일 L2 (통제 실험용, **논문 main table 권장**) |
| `uniform_upper` | 1.1배 residual 증폭 참고 상한선 |

L2는 비가청성의 충분조건이 아닙니다. PESQ/STOI/SI-SDR을 반드시 함께 보고하세요.

## Attribution FAR

Test 메시지는 고정된 고유 16-bit codebook입니다. 모든 후보와의 Hamming distance를
비교하여 strict/lenient Attribution FAR, tie rate, attribution margin을 계산합니다.
후보 수 민감도는 `--far_candidate_sizes`로 평가합니다.

## Usage

### Training (with proper held-out setup)
```bash
python phase2_training.py \
  --mode proposed_gate --map_type survival \
  --epochs 5 --gate_range 0.2 --projection_mode equal \
  --train_attacks noise,lowpass,resample,reconstruct_nq6 \
  --validation_attacks bandpass,reconstruct_nq8 \
  --test_attacks ffmpeg_mp3 \
  --strict_heldout
```

### Test-only with external codecs
```bash
python phase2_training.py \
  --mode proposed_gate --map_type survival --test_only \
  --load_weight checkpoints/your_gate.pth \
  --test_attacks clearervoice,facodec \
  --strict_heldout \
  --clearervoice_command "..." \
  --facodec_command "..."
```

## Output

- `results/phase2/{run_id}_{dataset}_{mode}_{map}_summary.json`
- `results/phase2/{run_id}_{dataset}_{mode}_{map}_samples.csv`
- `results/phase2/phase2_results_long.csv` (cross-run appendable)
