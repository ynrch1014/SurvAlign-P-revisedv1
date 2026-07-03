# SurvAlign-P 실험 및 평가 구조 총정리 (Experiment Overview)

본 문서는 논문 작성 및 방어를 위해 설계된 SurvAlign-P의 전체 실험 구조, 대조군(Ablation), 그리고 왜곡(Distortion) 세트를 한눈에 파악할 수 있도록 정리한 가이드입니다.

---

## 1. 전체 실험 파이프라인 (18개 시나리오)

실험은 3개의 데이터셋과 6개의 학습/평가 모드의 조합으로 이루어지며, `run_all_experiments.bat`을 통해 **총 18개의 독립적인 훈련 및 평가 사이클**이 일괄 수행됩니다.

### 🎧 평가 대상 데이터셋 (3종)
다양한 음성 도메인에서의 범용성을 입증하기 위해 아래 3가지를 사용합니다.
1. **LibriSpeech (`train-clean-100`)**: 다화자 오디오북 낭독 (표준 벤치마크)
2. **VCTK**: 고품질, 다양한 억양(사투리 등) 및 짧은 발화 환경 (강건성 검증)
3. **LJSpeech**: 단일 화자(여성) 내레이션 환경 (단일 도메인 검증)

### 🤖 학습 및 평가 모드 (6종)
각 데이터셋별로 아래 6가지 조건(Ablation)을 통과시키며 성능을 철저히 비교합니다.
1. **`baseline`**: 아무 조작도 가하지 않은 순정 AlignMark 워터마크
2. **`uniform`**: 워터마크의 모든 스펙트로그램 픽셀 에너지를 일괄 증폭
3. **`random_gate`**: 무작위 위치에 증폭 가중치를 준 상태 (가이드 맵의 '공간적 위치 정보' 자체의 가치 입증)
4. **`energy_gate` [신규 방어 논리]**: 디코더의 지식 없이 오직 소리의 크기(음압, Magnitude)에만 비례해 증폭시키는 상태. 제안 모델이 무지성으로 소리를 키운 것이 아님을 입증하는 가장 강력한 대조군입니다.
5. **`proposed_gate (survival)`**: 물리적 생존율(Survival Map) 기반 가이드 모델 (핵심 제안 기법)
6. **`proposed_gate (gradient)`**: 디코더 역전파(Gradient Map) 기반 가이드 모델

---

## 2. 평가 단계(Test) 측정 지표

학습이 끝나면(혹은 `test_all_experiments.bat` 실행 시), Test 분할 데이터셋(학습 때 본 적 없는 격리된 화자)에 대해 아래 지표들을 평가합니다. 모든 결과는 `results/phase2_results.csv`에 자동으로 누적됩니다.

### 🎙️ 오디오 지각 품질 (Fidelity Metrics)
워터마크 주입 및 게이트 조율로 인한 음질 훼손도를 다각도로 채점합니다.
*   **PESQ**: 사람의 주관적 음질 평가 점수를 모사 (높을수록 우수)
*   **STOI**: 음성 명료도 (높을수록 우수)
*   **SI-SDR**: 원음 대비 신호 왜곡/노이즈 비율 (높을수록 우수)
*   **L2-Ratio**: 원본 잔차 에너지 대비 게이팅 후의 잔차 에너지 비율 (Energy Cheating 차단 증명)

### 🛡️ 왜곡 강건성 (Robustness / Bit Accuracy)
게이트를 통과한 워터마크 오디오를 **8가지 시나리오(Clean 포함 7개 왜곡)**에 노출시킨 뒤, 디코더의 워터마크 추출 정확도를 평가합니다.
1. **Clean**: 왜곡이 없는 깨끗한 상태
2. **Noise (AWGN)**: 20dB 수준의 백색 소음 주입
3. **Lowpass**: 4kHz 이상 고주파 대역 차단
4. **Bandpass**: 전화기 대역폭(300Hz~3.4kHz) 필터링
5. **Resample**: 다운샘플링 후 복원 (고주파 훼손)
6. **Reconstruct**: SpeechTokenizer 대용량 딥러닝 코덱 통과
7. **MP3**: 스펙트럼 마스킹 및 양자화 손실을 수반하는 전통적 MP3 압축
8. **FACodec Proxy [신규 추가]**: 최신 딥러닝 코덱 환경의 극한 병목(Bottleneck, n_q=2) 압축 및 복원 모사

---

## 3. Phase 1 (Attribution Analysis) 의의 및 통계적 검증

Phase 1(`phase1_attribution.py`)은 본격적인 Phase 2 학습 전, Survival Map이 프록시로서 타당한지 검증하는 단계입니다.

1. **상관관계 도출**: 샘플 단위로 Survival Map과 Gradient Map 간의 $r$ 값(Pearson/Spearman) 평균 및 신뢰구간 도출.
2. **마스킹 실험 (OOD 해결)**: 특정 영역만 남겼을 때 에러율을 얼마나 방어하는지 인과성을 봅니다. 이때 급격한 신호 절단으로 인한 아티팩트(Out-of-Distribution)를 막기 위해 **2D Gaussian Soft Masking**이 적용됩니다.
3. **통계적 분기 (Branching)**: `High-Survival`과 `Low-Survival` 간의 에러율 차이에 대해 **Paired t-test**를 수행하며, **`p-value < 0.05`**를 만족할 때만 Phase 2 학습으로 넘어가도록 설계하여 학술적 엄밀성을 확보했습니다.
