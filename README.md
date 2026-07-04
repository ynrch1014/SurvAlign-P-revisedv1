# SurvAlign-P: Controlled Post-hoc Residual Redistribution

SurvAlign-P는 Feature-Aligned(AlignMark)가 생성한 워터마크 잔차를 **동일한 에너지 예산 안에서 시간-주파수별로 제한적으로 재가중**하고, 뉴럴 코덱·음성 복원 이후의 메시지 복호 실패를 회복할 수 있는지 검증하는 연구 코드입니다.

## 📚 Research Documentation

All research questions, mathematical background, hypotheses, defense logic, architecture details, and experimental results have been consolidated into our master research document:
- 📖 [SURVALIGN_P_RESEARCH_MASTER.md](./SURVALIGN_P_RESEARCH_MASTER.md)

Please refer to this document for a complete understanding of the theoretical contributions, ECC evaluation strategy, TOST equivalence testing, analytic survival metrics, and differences from the original AlignMark.

## 1. Quick Start

### Installation
\\ash
pip install -r AlignMark/requirements.txt
pip install -r requirements-survalign.txt
\
필수 가중치:
- \AlignMark/weight.pth- \AlignMark/speechtokenizer/pretrained_model/SpeechTokenizer.pt
### 2. 채널 독립성 통계 검증 (TOST)
\\ash
python verify_ecc_value_independence.py
\*Note: N=100의 표본 크기를 가지며 ±15%p 마진 내에서의 동등성을 검증합니다.*

### 3. Phase 1 - Attribution 시각화 (No Training)
\\ash
python phase1_attribution.py   --dataset_type librispeech   --dataset_name train-clean-100   --split test   --survival_attacks noise,lowpass,resample,reconstruct_nq6,spectral_proxy   --eval_attacks clean,bandpass,ffmpeg_mp3   --strict_heldout   --energy_modes natural,equal   --random_repeats 20
\
### 4. Phase 2 - Gate Training & Evaluation
\\ash
python phase2_training.py   --mode proposed_gate   --map_type survival   --dataset_type librispeech   --dataset_name train-clean-100   --epochs 5   --projection_mode equal   --train_attacks noise,lowpass,resample,reconstruct_nq6   --validation_attacks bandpass,reconstruct_nq8   --test_attacks ffmpeg_mp3   --strict_heldout
\