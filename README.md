# SurvAlign-P: Controlled Post-hoc Residual Redistribution

SurvAlign-P는 Feature-Aligned(AlignMark)가 생성한 워터마크 잔차를 **동일한 에너지 예산 안에서 시간–주파수별로 제한적으로 재가중**하고, 뉴럴 코덱·음성 복원 이후의 메시지 복호 실패를 회복할 수 있는지 검증하는 연구 코드입니다.

## Scope

> **Survival Map**은 decoder 없이 생성되는 물리적 prior입니다.
> **Gate 자체**는 고정된 AlignMark decoder의 CE loss로 학습됩니다 — decoder-free가 아닙니다.
> 최종 성능은 Map/Train 공격과 **완전히 분리된 held-out 공격**(ffmpeg MP3, FACodec, ClearerVoice 등)에서 평가해야 합니다.

## 1. Core Contributions & Defense Strategy

SurvAlign-P addresses the critical vulnerabilities of existing audio watermarking models in practical, large-scale forensic applications. Our methodology is built upon three core pillars:

1. **Planetary-Scale Robustness (The Necessity of Exact-Match):** 
   While existing studies report high bit-wise accuracy (e.g., 95-99%), this translates to a low probability of exact 16-bit message recovery ($0.95^{16} \approx 44\%$). Relying on Nearest-Neighbor (FAR) metrics is computationally infeasible and prone to catastrophic collision rates in planetary-scale databases. SurvAlign-P fundamentally restores the exact-match recovery rate without retraining the base model.
   
2. **Physical-Layer Adaptation over Capacity-Reducing ECC:**
   Error Correction Codes (ECC) can mitigate bit errors but strictly reduce the effective payload capacity (e.g., halving the payload from 16 bits to 8 bits). Instead of sacrificing capacity, SurvAlign-P operates at the physical layer, dynamically redistributing energy to inherently robust Time-Frequency (T-F) bins while strictly maintaining the perceptual energy budget. *(Note: Our evaluation includes an idealized, optimistic ECC baseline that assumes zero physical-layer codeword constraints. SurvAlign-P outperforms ECC even when the ECC is given this unfair theoretical advantage.)*

3. **Empirical Validation of Correlated Burst Errors:**
   Neural audio codecs (e.g., EnCodec) do not cause independent and identically distributed (i.i.d.) bit errors; they induce clustered (burst) errors in specific T-F regions. Our empirical analysis (`burst_error_analysis.py`) demonstrates that our physical `Survival Map` accurately predicts these localized destruction patterns, far outperforming conventional decoder-derived gradient saliency.

## 2. Pipeline 구성

| File | Role |
|---|---|
| `phase1_attribution.py` | 에너지·공격 통제 Phase 1 진단 (학습 없음) |
| `phase2_training.py` | Validation 기반 Gate 학습 및 paired 평가 |
| `burst_error_analysis.py` | 코덱 Burst Error 파괴 대역 및 Survival Map 실측 분석 |
| `survalign_p.py` | 공통 데이터셋·왜곡·AlignMark 래퍼 + legacy 실험 |
| `experiment_utils.py` | 재현성, L2 투영, Exact-message/FAR 계산 |
| `external_attacks.py` | 실제 ffmpeg MP3 및 외부 코덱 실행 어댑터 |

`survalign_p.py`의 독립 실행 파이프라인은 legacy/experimental 용도입니다.
논문용 실험은 `phase1_attribution.py`와 `phase2_training.py`를 사용하세요.

## 3. Important Evaluation Principles

1. **Map/Train/Validation/Test 공격 분리**
   같은 공격 또는 같은 codec family를 여러 단계에서 반복 사용하면 일반화 주장이 성립하지 않습니다.
   확증 실험에서는 반드시 `--strict_heldout`를 사용하세요.

2. **에너지 통제**
   - `--projection_mode cap`: 실사용 (에너지 상한)
   - `--projection_mode equal`: 엄격한 통제 실험 (**같은 에너지에서 분배만 변경**)
   - 동일 L2가 동일 비가청성을 보장하지 않으므로 PESQ/STOI/SI-SDR를 반드시 함께 보고

3. **Paired·deterministic 평가**
   Calibration/Test의 crop, 16-bit 메시지, stochastic attack seed는 샘플별로 고정.
   Test 메시지는 고유 codebook이므로 Attribution FAR 계산 가능.

4. **Proxy 명칭 구분**
   `spectral_proxy` ≠ 실제 MP3. `strong_speechtokenizer` ≠ 실제 FACodec.
   실제 코덱은 `external_attacks.py`의 held-out 어댑터를 사용.

5. **Identity STE 한계 인지**
   SpeechTokenizer reconstruction의 backward는 identity STE.
   Gate는 코덱의 실제 주파수 선택성을 backpropagation으로 학습할 수 없음.
   이 한계는 논문 Limitations에 명시 필요.

## 4. Installation

```bash
pip install -r AlignMark/requirements.txt
pip install -r requirements-survalign.txt
```

필수 가중치:
- `AlignMark/weight.pth`
- `AlignMark/speechtokenizer/pretrained_model/SpeechTokenizer.pt`

파일이 없으면 실험은 즉시 중단됩니다.

## 5. Phase 1: Controlled Diagnostic

Phase 1은 학습을 하지 않습니다. 동일 에너지 마스킹에서 어떤 T-F prior가
복호 정보를 가장 잘 보존하는지 진단합니다.

```bash
python phase1_attribution.py \
  --dataset_type librispeech \
  --dataset_name train-clean-100 \
  --split test \
  --survival_attacks noise,lowpass,resample,reconstruct_nq6,spectral_proxy \
  --eval_attacks clean,bandpass,ffmpeg_mp3 \
  --strict_heldout \
  --energy_modes natural,equal \
  --random_repeats 20
```

비교 대상: High/Low Survival, Gradient Saliency, Codec Utility,
Residual/Speech-Energy/VAD, Random×20. 자세한 내용은 `README_phase1.md` 참조.

## 6. Phase 2: Gate Training

```bash
python phase2_training.py \
  --mode proposed_gate \
  --map_type survival \
  --dataset_type librispeech \
  --dataset_name train-clean-100 \
  --epochs 5 \
  --projection_mode equal \
  --train_attacks noise,lowpass,resample,reconstruct_nq6 \
  --validation_attacks bandpass,reconstruct_nq8 \
  --test_attacks ffmpeg_mp3 \
  --strict_heldout
```

핵심 비교:
- `baseline` vs `proposed_gate`: Gate가 실제로 효과가 있는가?
- `analytic_survival` vs `proposed_gate`: 학습 없는 순수 prior만으로 충분한가?
- `cap` vs `equal` projection: 에너지 통제 하에서도 효과가 유지되는가?

자세한 내용은 `README_phase2.md` 참조.

## 7. External Held-out Attacks

외부 모델은 저장소에 포함하지 않습니다. `{input}`과 `{output}` WAV 경로를 받는
래퍼 명령을 준비한 뒤 실행합니다.

```bash
python phase2_training.py \
  --mode proposed_gate --map_type survival --test_only \
  --load_weight checkpoints/your_gate.pth \
  --test_attacks clearervoice,facodec \
  --strict_heldout \
  --clearervoice_command "python tools/run_clearervoice.py --input {input} --output {output}" \
  --facodec_command "python tools/run_facodec.py --input {input} --output {output}"
```

실제 MP3 (ffmpeg 필요):
```bash
python phase2_training.py --mode baseline --test_attacks ffmpeg_mp3 --mp3_bitrate 64k
```

외부 wrapper 규격은 `EXTERNAL_ATTACKS.md`를 참고하세요.

## 8. Output Metrics

공격별로 다음을 저장합니다:
- Bit Accuracy / BER
- 16-bit Exact-message Accuracy
- strict/lenient Attribution FAR 및 tie rate
- Hamming attribution margin
- Decoder CE, minimum/mean logit margin, entropy
- 후보 수별 FAR 진단 (`--far_candidate_sizes`, 기본 100/300/600)
- **Failure Recovery Rate / Regression Rate** (paired with baseline)
- PESQ, STOI, SI-SDR, L2 ratio, peak/clipping ratio

전체 메트릭, 공격 유형, 실험 모드에 대한 종합 정리는 `EXPERIMENT_OVERVIEW.md`를 참조하세요.
아키텍처 흐름도는 `ARCHITECTURE_FLOWCHART.md`를 참조하세요.

## 9. Data Splits

- LibriSpeech/VCTK: speaker-disjoint
- LJSpeech: 단일 화자이므로 file-disjoint만 가능
- `combined_protocol=speaker_disjoint`: LibriSpeech/VCTK 화자 격리 + LJSpeech 파일 격리
- `combined_protocol=paper`: 데이터셋별 200개 test 파일 (화자 누출 가능성 명시 필요)

**※ 주의: 본 논문의 모든 메인 리포트 결과(표 및 그래프)는 엄격한 `speaker_disjoint` 프로토콜을 기준으로 작성되었습니다.**

세부 수정 내역은 `MODIFICATION_REPORT.md`를 참고하세요.
