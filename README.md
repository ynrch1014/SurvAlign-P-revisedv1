# SurvAlign-P: Feature-Aligned Speech Watermarking with Survival Gate

성균관대학교(SKKU) AAI URP 연구 프로젝트인 **SurvAlign-P**의 공식 코드 저장소입니다.  
본 프로젝트는 기존 딥러닝 기반 블랙박스 오디오 워터마킹(e.g., `AlignMark`, ICME 2026)의 근본적인 한계를 극복하기 위해 고안된 **"사후 가이드형 잔차 최적화(Post-hoc Guided Residual Optimization)"** 네트워크입니다.

---

## 🔬 1. 연구 배경 및 필요성 (Background & Motivation)

### 블랙박스 워터마킹의 한계
기존의 오디오 워터마킹 모델들은 원본 오디오에 비밀 메시지를 숨기기 위해 인코더-디코더(Encoder-Decoder) 구조를 사용합니다. 하지만 이러한 모델들은 학습 시 경험하지 못한 **극한의 실제 환경 왜곡(MP3 압축, 밴드패스 필터링, 잔향 등)**이 발생하면 워터마크 생존율이 급감하는 치명적인 취약점을 가집니다.

### 무분별한 에너지 증폭(Energy Cheating)의 위험성
강건성(Robustness)을 높이기 위한 가장 쉬운 꼼수는 워터마크 신호의 소리(에너지) 자체를 키우는 것입니다. 하지만 이는 원음의 지각적 품질(Fidelity)을 심각하게 훼손합니다. 따라서, **"지각 품질을 전혀 훼손하지 않으면서도 왜곡에 강건하게 살아남도록 정교한 스펙트럼 에너지를 분배하는 기술"**이 본 연구의 가장 핵심적인 필요성입니다.

---

## 💡 2. 핵심 인사이트: Survival Map vs Decoder Gradient Map

SurvAlign-P의 가장 핵심적인 질문은 다음과 같습니다:
> **"워터마크가 물리적으로 잘 살아남는 시간-주파수 영역이, 디코더 인공지능이 실제로 잘 읽어내는 영역과 일치하는가?"**

이를 규명하기 위해 두 가지 맵(Map)을 정의하고 비교합니다:

| 맵(Map) | 의미 | 계산 방식 | 측정 시점 |
|:---|:---|:---|:---|
| **Survival Map** | **물리적 생존율** | 6가지 왜곡 시뮬레이터 통과 후 잔차 보존율(SIR) 산출 | 왜곡 **후** (미래 예측) |
| **Gradient Map** | **수학적 민감도** | 디코더 CE Loss의 입력 파형에 대한 오차 역전파 크기 | 왜곡 **전** (현재 상태) |

* **상관관계의 의의**: 만약 두 맵의 상관성이 높다면, 복잡하고 값비싼 역전파를 거치지 않고도 오직 직관적인 "물리적 생존율(Survival Map)"만으로 디코더의 취약점을 보완할 수 있는 강력한 이론적/수학적 근거가 완성됩니다.

---

## 🧪 3. 2-Phase 실험 설계 및 연구 흐름

본 연구는 통계적 원인 분석(Phase 1)과 실제 인공지능 최적화(Phase 2)를 엄격하게 분리하여 진행합니다.

### Phase 1 (Attribution Analysis)
- **목표**: Survival Map과 Gradient Map 간의 상관관계(Pearson, Spearman) 및 영역 교집합(Top-20% IoU)을 정량적으로 측정합니다.
- **인과 검증**: 맵의 상위 20% 영역만을 남기는 **이진 마스킹(Binary T/F Masking)** 절제 연구(Ablation)를 수행하여, 해당 영역이 실제 에러율(BER) 방어에 결정적인 인과적 기여를 하는지 직관적으로 증명합니다.

### Phase 2 (Survival Gate Training)
- **목표**: Phase 1의 통계적 발견을 바탕으로, 실제 잔차 에너지를 최적화하는 가벼운 AI 모듈(`Survival Gate`)을 학습합니다.
- **연속 점수 조율 (Soft Weighting)**: Phase 1의 거친 이진 마스크 검증과 달리, 본학습에서는 맵의 **연속적인 점수(Continuous Score)**를 가이드(Prior)로 입력받습니다. 3계층 CNN 구조를 통해 `[0.8, 1.2]` 범위의 정교한 연속 가중치를 출력하여, 주파수-시간 픽셀 단위로 워터마크의 강약을 섬세하게 조율합니다.

---

## 📐 4. 수학적·구조적 원리 (Mathematical & Structural Principles)

연구의 학술적 무결성(Integrity)과 공정한 대조군 비교를 위해 완벽한 수학적 제약을 가합니다.

1. **엄격한 L2 Waveform Projection (Hard Constraint)**
   - Gate를 통해 수정된 잔차($r_{gated}$)가 원본 잔차($r_0$)의 전체 에너지를 절대 초과하지 못하도록 **수학적 투영(Projection)** 제약을 강제로 부여합니다. 
   - $ \tilde{r} = r_{gated} \times \min\left(1, \frac{||r_0||_2}{||r_{gated}||_2}\right) $
   - 이를 통해 단순히 신호 에너지를 증폭시켜 강건성을 달성했다는 비판(Energy Cheating)을 구조적/수학적으로 원천 차단합니다.

2. **미분 가능한 디코더 역전파 (Differentiable Decoding)**
   - 원본 AlignMark 모델이 지닌 Vector Quantization(VQ) 병목을 우회하여, 입력 파형부터 최종 CE Loss까지 Autograd Gradient가 단절 없이 완벽하게 흐르도록 아키텍처를 재설계했습니다.

3. **철저한 다중 데이터셋 및 화자 격리 (Speaker Disjoint)**
   - `LibriSpeech` (다화자), `VCTK` (다화자), `LJSpeech` (단일화자) 등 음향 특성이 확연히 다른 3가지 대규모 데이터셋을 동시 지원합니다.
   - 훈련(Train)과 평가(Test) 시 **동일한 화자의 목소리가 겹치지 않도록 철저히 격리(Disjoint)** 분할하여 모델의 과적합(Overfitting)을 방지합니다.

---

## 🚀 5. 시작하기 (Setup Guide)

### 1. 가상환경 구축 및 패키지 설치
Python 3.10+ 환경에서 실행을 권장합니다.

```bash
# 가상환경 생성 및 활성화 (선택)
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # macOS/Linux

# 필수 의존성 패키지 설치
pip install -r AlignMark/requirements.txt
pip install pesq pystoi scikit-learn soundfile torchaudio
```

### 2. 사전 학습된 가중치(Pretrained Weights) 다운로드
대용량 파일 제한 정책으로 인해 아래 가중치 파일들은 GitHub에 포함되어 있지 않습니다. 아래 파일을 직접 다운로드하여 명시된 경로에 배치해 주세요:

1. **AlignMark 모델 가중치** (`weight.pth`)
   - **배치 경로**: `AlignMark/weight.pth`
2. **SpeechTokenizer 모델 가중치** (`SpeechTokenizer.pt`)
   - **배치 경로**: `AlignMark/speechtokenizer/pretrained_model/SpeechTokenizer.pt`

### 3. 데이터셋 일괄 자동 지원
- 코드 실행 시 자동으로 지정된 데이터셋을 감지하여 다운로드 및 로딩을 수행합니다.
- LibriSpeech(16kHz), VCTK(48kHz→16kHz), LJSpeech(22050Hz→16kHz) 모두 일괄된 샘플레이트와 텐서 차원 `(1, 32000)`으로 변환되어 모델에 제공됩니다.

---

## 🏃 6. 실행 및 평가 (Running & Evaluation)

사용자의 편의를 위해 마우스 더블클릭이나 단순 명령어 한 줄로 모든 실험을 진행할 수 있는 `.bat` 스크립트를 제공합니다.

### Phase 1: 상관관계 및 인과성 분석
```bash
# LibriSpeech에서 Phase 1 분석
python phase1_attribution.py --dataset_type librispeech --batch_size 4

# VCTK에서 Phase 1 분석
python phase1_attribution.py --dataset_type vctk --batch_size 4
```

### Phase 2: Survival Gate 본학습 및 평가
5가지 논문용 실험 모드를 지원합니다: `baseline`, `uniform`, `random_gate`, `proposed_gate (survival)`, `proposed_gate (gradient)`

**✅ 데이터셋 분할 및 활용 규정 (Train / Test Split)**
본 프로젝트는 리뷰어의 공격을 완벽히 방어하기 위해 **세 가지 평가 트랙**을 동시 지원합니다.

| 트랙 명칭 | `--dataset_type` | 훈련 데이터 (Train) | 평가 데이터 (Test) | 연구/방어 목적 |
|:---|:---|:---|:---|:---|
| **1. 개별 증명 트랙** | `librispeech` (등) | 1개 도메인 80% | 동일 도메인 10% (화자격리) | 도메인 독립적인 본질적 성능 입증 (Ablation) |
| **2. Cross-Dataset 트랙** | `vctk` + `--load_weight` | A 도메인 (예: LibriSpeech) | B 도메인 (예: VCTK) | 미학습 OOD(Out-of-Distribution)에 대한 극강의 일반화 성능 입증 |
| **3. 원 논문 모방 SOTA 트랙**| `combined` | 3개 데이터셋 전체 (600개 제외) | 3개 데이터셋 랜덤 600개 | 원 논문 AlignMark와 완벽히 동일한 조건에서의 1:1 최고 성능(SOTA) 비교 |

```bash
# [트랙 1] 개별 실험 실행 예시
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type vctk --epochs 5 --batch_size 8

# [트랙 2] Cross-Dataset 평가 (LibriSpeech로 훈련한 모델을 VCTK에서 테스트)
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type vctk --load_weight ./checkpoints/best_gate_librispeech_proposed_gate_survival.pth

# [트랙 3] 원 논문 방식(Test 600개)으로 SOTA 단일 모델 거대 학습 및 평가
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type combined --epochs 5 --batch_size 8

# 3 데이터셋 × 5 모드 = 총 15개 논문 실험 일괄 훈련 및 평가 (트랙 1)
run_all_experiments.bat

# 저장된 체크포인트로 전체 실험 평가만 일괄 수행
test_all_experiments.bat
```

### 주요 출력 리포트
- **Phase 1**: `results/phase1_summary_{dataset}.txt` — 상관계수 및 이진 마스킹 ablation 결과 (인과성 증명).
- **Phase 2**: `results/phase2_results.csv` — 데이터셋별 Fidelity/Robustness 지표가 정리된 최종 엑셀 테이블 (논문 첨부용).
- **시각화**: `results/phase1_map_comparison.png` — Survival Map vs Gradient Map 비교 플롯.
