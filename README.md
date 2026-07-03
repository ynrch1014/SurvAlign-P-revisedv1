# SurvAlign-P: Feature-Aligned Speech Watermarking with Survival Gate

성균관대학교(SKKU) AAI URP 연구 프로젝트인 **SurvAlign-P**의 공식 코드 저장소입니다.  
본 프로젝트는 기존 `AlignMark` (ICME 2026) 오디오 워터마킹 백본의 한계를 극복하고, 지각 품질(Fidelity) 저하 없이 다양한 실제 환경 채널 왜곡 하에서의 강건성(Robustness)을 개선하기 위해 설계되었습니다.

---

## 🔬 연구 동기 및 핵심 인사이트

### Survival Map vs Decoder Gradient Map

SurvAlign-P의 핵심 가설은 **"워터마크가 물리적으로 잘 살아남는 시간-주파수 영역이, 디코더가 실제로 잘 읽는 영역과 일치하는가?"**입니다. 이를 검증하기 위해 두 개의 독립적으로 계산되는 맵을 비교합니다:

| 맵 | 계산 방법 | 측정 대상 | 시점 |
|:---|:---|:---|:---|
| **Survival Map** | 6가지 왜곡 시뮬레이션 후 잔차 보존율 측정 | 채널의 물리적 특성 | 왜곡 **후** (미래 예측) |
| **Gradient Map** | 디코더 CE loss의 입력 파형 역전파 | 디코더의 민감도 분포 | 왜곡 **전** (현재 상태) |

**상관관계가 높다면**: AlignMark의 디코더가 이미 물리적으로 강건한 영역을 잘 읽도록 학습되어 있음을 의미합니다. 이는 비자명한(non-trivial) 발견이며, **역전파 없이 계산 가능한 Survival Map으로 비싼 Gradient Map을 대체(proxy)할 수 있는 이론적 근거**를 제공합니다.

**상관관계가 낮다면**: Survival Map은 Gradient Map이 놓치는 **"디코더가 읽지만 왜곡에 취약한 영역"** 정보를 보완적으로 제공합니다. 디코더가 민감하게 반응하는 영역이라도 왜곡 후 물리적으로 파괴되면 의미가 없기 때문입니다.

### 2-Phase 실험 설계

본 연구는 분석(Phase 1)과 적용(Phase 2)을 명확히 분리합니다:

- **Phase 1 (Attribution Analysis)**: Survival Map과 Gradient Map 간의 상관관계(Pearson, Spearman)와 영역 일치도(IoU)를 정량 분석하고, **이진 마스크 ablation**을 통해 인과관계를 검증합니다. 이진 마스킹은 "이 영역을 제거하면 BER이 얼마나 변하는가?"라는 깔끔한 인과적 증거를 위한 의도적 설계입니다.
- **Phase 2 (Gate Training)**: Survival Map을 Gate 네트워크의 **연속 score 기반 가이드(prior)**로 제공하고, Gate는 `[0.8, 1.2]` 범위의 연속 가중치를 출력합니다. 실제 파라미터 최적화는 왜곡 후 디코더 CE loss로 이루어지므로, Survival Map은 학습의 초기 방향을 잡아주는 사전 지식 역할을 합니다.

---

## ⚙️ 주요 아키텍처 특징

1. **6채널 Survival Gate**:
   - 원본 음성 스펙트로그램, AlignMark 초기 잔차, 다중 왜곡 생존율 맵, 지각 마스킹 프록시, STFT 기반 Soft VAD 및 오리지널 잔차 프라이어의 총 6가지 시간-주파수 특징을 활용해 최적의 스펙트럼 에너지 게이팅 가중치를 예측합니다.
2. **미분 가능한 디코더 직접 역전파 (Bypassing VQ Bottleneck)**:
   - 디코딩 경로 상에 Vector Quantization(VQ)이 배제되어, 실제 디코더(frozen)를 통과하여 입력 파형까지 완벽하게 autograd gradient가 전달됩니다.
3. **L2 Waveform Projection 제약**:
   - 수정 잔차의 L2 에너지가 원래 AlignMark 잔차 에너지 한도를 넘지 않도록 제한하여 단순히 에너지를 올려 성능을 향상시키는 꼼수를 차단합니다.
4. **Presence Head & Calibration**:
   - 디코딩 에비던스 특징 벡터(Entropy, Margin, 확률)에 기반한 Presence Head를 제공하며, Calibration 화자 데이터로 오탐률 1% 미만의 판단 임계값을 보정합니다.

---

## 🚀 시작하기 (Setup Guide)

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

*(연구실 공유용 구글 드라이브 혹은 드롭박스 링크를 여기에 추가하세요)*

### 3. 디렉토리 구조 검증
정상 실행을 위해 클론한 프로젝트의 경로 구성이 아래와 같은지 확인해 주세요:
```text
├── .gitignore
├── README.md
├── survalign_p.py                    # SurvAlign-P 엔진 코드
├── phase1_attribution.py             # Phase 1: 상관관계 분석
├── phase2_training.py                # Phase 2: Gate 학습 및 평가
├── run_all_experiments.bat           # 3 데이터셋 × 5 모드 일괄 실행
├── test_all_experiments.bat          # 평가 전용 일괄 실행
└── AlignMark/
    ├── weight.pth                    # [다운로드 파일]
    ├── speechtokenizer/
    │   └── pretrained_model/
    │       ├── SpeechTokenizer.pt    # [다운로드 파일]
    │       └── speechtokenizer_hubert_avg_config.json
    ...
```

---

## 📊 데이터셋

원논문(AlignMark, ICME 2026)의 실험 세팅과 일치시키기 위해 **3개 데이터셋**을 지원합니다. 코드 실행 시 자동으로 다운로드 및 전처리됩니다.

| 데이터셋 | `--dataset_type` | 용량 | 화자 수 | 분할 방식 | 샘플레이트 |
|:---|:---|:---|:---|:---|:---|
| **LibriSpeech** | `librispeech` | ~338MB (dev-clean) / ~6.3GB (train-clean-100) | 다화자 | 화자 ID 기반 80/10/10 | 16kHz |
| **VCTK** | `vctk` | ~10.9GB | 110명 (다화자) | 화자 ID 기반 80/10/10 | 48kHz → 16kHz |
| **LJSpeech** | `ljspeech` | ~2.6GB | 1명 (단일 화자) | 파일 인덱스 기반 80/10/10 (seed=42) | 22050Hz → 16kHz |

모든 데이터셋에서 동일한 `(wav, msg)` 텐서 인터페이스(`wav: (1, 32000)`, `msg: (16,)`)를 제공합니다.

---

## 🏃 실행 및 평가 (Running & Evaluation)

### Phase 1: Attribution Correlation Analysis

Survival Map과 Gradient Map 간의 상관관계 및 인과적 마스킹 검증을 수행합니다.

```bash
# LibriSpeech에서 Phase 1 분석
python phase1_attribution.py --dataset_type librispeech --batch_size 4

# VCTK에서 Phase 1 분석
python phase1_attribution.py --dataset_type vctk --batch_size 4

# LJSpeech에서 Phase 1 분석
python phase1_attribution.py --dataset_type ljspeech --batch_size 4
```

### Phase 2: Survival Gate Training & Evaluation

5가지 실험 모드를 지원합니다:

| 모드 (`--mode`) | 설명 |
|:---|:---|
| `baseline` | AlignMark 원본 (Gate 없음) |
| `uniform` | 에너지 제약 맞춤 균일 스케일링 |
| `random_gate` | 무작위 가이드 맵으로 Gate 학습 (ablation) |
| `proposed_gate` (survival) | Survival Map 가이드 Gate 학습 |
| `proposed_gate` (gradient) | Gradient Map 가이드 Gate 학습 |

```bash
# 개별 실험 실행 예시
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type vctk --epochs 5 --batch_size 8

# 저장된 체크포인트로 평가만 수행
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type ljspeech --test_only

# 3 데이터셋 × 5 모드 = 15개 실험 일괄 실행
run_all_experiments.bat

# 15개 실험 평가만 일괄 수행
test_all_experiments.bat
```

### survalign_p.py 직접 실행 (통합 파이프라인)

```bash
# 기본 설정으로 학습 진행 (dev-clean)
python survalign_p.py

# VCTK 데이터셋으로 학습
python survalign_p.py --dataset_type vctk

# 초고속 Sanity Check
python survalign_p.py --sanity_check

# 대용량 LibriSpeech로 학습
python survalign_p.py --dataset_type librispeech --dataset_name train-clean-100 --n_gate_steps 2000
```

### 주요 실행 인자
- `--dataset_type`: 사용할 데이터셋 (`librispeech`, `vctk`, `ljspeech`).
- `--dataset_name`: LibriSpeech 서브셋 이름 (`dev-clean` 또는 `train-clean-100`).
- `--mode`: Phase 2 실험 모드 (`baseline`, `uniform`, `random_gate`, `proposed_gate`).
- `--map_type`: proposed_gate 모드의 가이드 맵 유형 (`survival` 또는 `gradient`).
- `--epochs`: 학습 에폭 수 (기본값: 3).
- `--batch_size`: 배치 크기 (기본값: 8, GPU 메모리에 맞춰 조절).
- `--test_only`: 학습을 생략하고 저장된 체크포인트로 평가만 수행.
- `--sanity_check`: 초소형 스텝으로 빠른 동작 검증.

### 모델 저장 및 체크포인트
- `gate_checkpoint.pth`: Survival Gate 가중치.
- `presence_checkpoint.pth`: Presence Head 가중치 및 탐지 임계값(`tau_p`).
- `checkpoints/best_gate_{dataset}_{mode}.pth`: Phase 2 실험별 최적 Gate 가중치.

### 출력 리포트
- **Phase 1**: `results/phase1_summary_{dataset_type}.txt` — 상관계수 및 마스킹 ablation 결과.
- **Phase 2**: `results/phase2_results.csv` — 데이터셋별 Fidelity/Robustness 지표 (논문 테이블용).
- **시각화**: `results/phase1_map_comparison.png` — Survival Map vs Gradient Map 비교 플롯.

