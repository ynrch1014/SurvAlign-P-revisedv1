# SurvAlign-P 기술 분석 및 구조 보고서

> [!NOTE]
> 본 문서는 URP 연구 과제로 수행된 **SurvAlign-P 모델의 구조적 특징, 단계별 코드 분석, 그리고 종합적인 사용법**을 담은 технический 레포트입니다. 논문 작성 시 방법론(Methodology) 및 실험(Experiments) 섹션의 기초 자료로 바로 활용하실 수 있습니다.

---

## 1. SurvAlign-P 전체 구조 소개 (Architecture Overview)

SurvAlign-P는 기존 블랙박스 오디오 워터마크(AlignMark)가 극한의 음향 왜곡(MP3 압축, 잡음, 필터링 등)에서 생존율이 급감하는 문제를 해결하기 위해 고안된 **"사후 가이드형 잔차 최적화(Post-hoc Guided Residual Optimization) 네트워크"**입니다. 

원본 AlignMark의 학습 구조나 가중치를 전혀 수정하지 않은 채, 생성된 워터마크 잔차(Residual) 중 **복호(Decoding)에 가장 치명적으로 기여하는 핵심 T-F(Time-Frequency) 픽셀의 에너지만 선택적으로 증폭**하는 `Survival Gate`를 핵심 컴포넌트로 삼습니다.

### 핵심 해결 과제 (Research Questions)
1. 워터마크 잔차 중 어떤 영역을 살려야 하는가? (Phase 1)
   - **가설 A**: 물리적인 왜곡 속에서도 신호가 훼손되지 않고 끝까지 살아남는 영역 (**Survival Map**)
   - **가설 B**: 디코더가 워터마크 존재 여부를 판독할 때 가장 민감하게 반응하는 영역 (**Decoder Gradient Map**)
2. 이 가이드맵을 바탕으로 에너지를 어떻게 제약하면서 성능을 올릴 것인가? (Phase 2)

---

## 2. Phase 1: Attribution 상관 분석 엔진 (`phase1_attribution.py`)

Phase 1은 위 Research Question 1번에 대한 **통계적 해답과 인과적 증거**를 제공하는 핵심 분석 파트입니다.

### 2.1. 코드 구조 및 작동 원리
* **`get_survival_map()`**: 원본 오디오와 워터마크 오디오에 5가지 왜곡(AWGN, Lowpass, Bandpass, Resample, MP3 Proxy)을 가한 뒤, 각 픽셀별 신호 대 간섭비(SIR)를 도출하여 하위 25% 분위수로 병합한 물리적 생존 지도를 그립니다.
* **`compute_decoder_gradient_map()`**: AlignMark의 Decoder(`decode_logits_with_grad`)를 통해 CrossEntropy Loss의 역전파 Gradient 크기를 계산합니다. 즉, 디코더가 판정에 있어 가장 민감하게 반응하는 "의사결정 취약점" 지도를 그립니다.
* **상관 분석 (Correlation)**: 두 Map 간의 Pearson, Spearman 상관계수 및 상위 20% 픽셀 IoU를 계산하여 서로 얼마나 일치하는지 분석합니다.
* **인과 마스킹 실험 (Causal Ablation)**: 오디오 파형의 특정 T-F 영역의 잔차만 남겨본 뒤 디코더의 Bit Accuracy를 측정하여, 어느 Map의 정보가 **실제 복호 유틸리티**를 가지고 있는지 철저히 검증합니다.

### 2.2. 사용법 (Usage)
테스트 세트 전체에 대한 대용량 분석을 수행하려면 다음 스크립트를 실행합니다.
```bash
# 전체 테스트 데이터셋을 이용한 Phase 1 상관 분석 및 마스킹 검증
run_phase1.bat
```
실행이 완료되면 `results/phase1_summary.txt`에 통계량과 BER 수치가 자동 저장되며, `results/phase1_map_comparison.png`를 통해 두 맵의 형태를 눈으로 직접 비교해볼 수 있습니다.

---

## 3. Phase 2: Survival Gate 본학습 및 평가 엔진 (`phase2_training.py`)

Phase 1의 분석을 통해 검증된 가이드맵을 기반으로, 실제 잔차 에너지를 최적화하는 **Survival Gate** 모듈을 훈련시키고 성능을 대용량으로 평가하는 단계입니다.

### 3.1. 모델 구조 (Simplified Survival Gate)
* **입력**: 3-Channel STFT 피처 (1. 원본 파형 Mag, 2. 워터마크 잔차 Mag, 3. 가이드 맵)
* **네트워크**: 3계층 CNN과 GroupNorm을 적용해 가벼우면서도 안정적인 학습을 보장합니다. 마지막 층은 가중치가 0으로 초기화되어, 초기에는 아무런 증폭/감쇠를 하지 않는 상태(`Scale=1.0`)에서 출발합니다.
* **출력**: 주파수-시간별 증폭 비율 (Gate Scale, $1.0 \pm 0.2$ 범위 내 제어)

### 3.2. 학습 제약 및 손실 함수 (Constraints & Loss Design)
연구의 무결성과 공정한 비교를 위해 다음의 제약 조건과 손실 함수가 결합되어 있습니다.
1. **L2 Waveform Projection (Hard Constraint)**: Gate를 통과한 잔차 에너지가 Baseline의 원본 잔차 에너지를 절대 초과하지 않도록 수학적 투영(Projection)을 적용합니다. 이를 통해 단순히 전체 신호 에너지를 증폭시켜 강건성을 높이는 오류를 구조적으로 차단합니다.
2. **Robustness Loss**: 6가지 다양한 왜곡 환경을 거친 뒤 출력되는 Decoder의 CE Loss를 최소화하여 워터마크의 강건성을 극대화합니다.
3. **Deviation Loss**: Gate의 스케일 증폭률이 기준점(1.0)에서 과도하게 벗어나지 않도록 제어하는 정규화(Regularization) 항목입니다.

### 3.3. 지원 시나리오 (Ablations) 및 사용법
연구 논문 테이블을 작성하기 위한 5가지 필수 시나리오가 내장되어 있으며, 각각을 독립적으로 훈련/평가하거나 한 번에 돌릴 수 있습니다.

> [!TIP]
> **원클릭 전체 실행**
> 아래 스크립트를 더블클릭하면 1번부터 5번까지 차례대로 학습과 평가를 끝내고, 결과를 `results/phase2_results.csv`에 표 형태로 자동 누적합니다.
> `run_all_experiments.bat`

각 개별 시나리오는 다음과 같습니다.
1. `run_baseline.bat`: 아무 조작을 가하지 않은 오리지널 AlignMark (대조군 1)
2. `run_uniform.bat`: Gate 스케일 증가량만큼 수동으로 전체 에너지를 무식하게 올린 상태 (에너지 증가 효과 대조군 2)
3. `run_random_gate.bat`: 가이드맵 대신 노이즈(랜덤 픽셀)를 주입해 학습시킨 Gate (위치 정보의 가치 증명 대조군 3)
4. **`run_proposed_gate.bat` (Phase 2A)**: 물리적 생존율을 나타내는 **Survival Map** 기반 제안 기법.
5. **`run_proposed_gradient.bat` (Phase 2B)**: 취약점을 나타내는 **Gradient Map** 기반 제안 기법.

---

## 4. 데이터셋 파이프라인 (RealLibriSpeechDataset)

본 코드는 실제 사람의 발화 음성인 **LibriSpeech (`train-clean-100`)** 데이터셋을 사용합니다. 코드를 최초 실행하면 백그라운드에서 자동으로 약 6GB에 달하는 tar.gz 압축파일을 다운로드하고 풀어냅니다.

> [!IMPORTANT]
> **화자 격리 분할 (Speaker Disjoint Split)**
> 딥러닝 음성 연구의 가장 핵심적인 엄밀성 조건입니다. 학습 시에 사용된 화자(목소리)가 테스트 세트에 단 한 명도 겹치지 않도록, `RealLibriSpeechDataset` 내부에서 화자 ID를 기준으로 Train(80%), Calib(10%), Test(10%) 분할을 동적으로 수행합니다. 이는 모델이 특정 사람의 목소리에 과적합(Overfitting)되지 않았음을 증명합니다.

---

## 결론
이 패키지는 1) Phase 1을 통한 가설 검증 2) Phase 2를 통한 제안 기법 및 Ablation 학습 3) 자동 평가 및 CSV 로깅까지 이어지는 완벽한 "End-to-end 연구 팩"입니다. 분석 보고서의 각 항목을 그대로 번역하거나 요약하면 매우 완성도 높은 학술 논문을 작성하실 수 있습니다.
