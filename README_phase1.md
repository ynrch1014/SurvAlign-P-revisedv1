# SurvAlign-P Phase 1: Attribution Correlation Analysis

## 개요 (Overview)
본 코드는 SurvAlign-P의 가장 핵심적인 가정인 **"물리적으로 워터마크 잔차가 잘 살아남는 위치(Survival Map)가, 실제로 Decoder가 워터마크를 복호하는 데 유용하게 사용하는 위치(Decoder Utility Map)와 일치하는가?"**를 실험적으로 검증하기 위한 분석용 프레임워크입니다.

이 분석 결과는 향후 워터마크 개선 전략을 결정짓는 중요한 나침반이 됩니다. 특히, 본 Phase 1에서는 직관적인 인과관계 검증을 위해 맵을 이진화(True/False, 예: Top 20% 마스킹)하여 성과를 입증하지만, 실제 Phase 2 본학습에서는 이 이진 마스크가 아닌 연속적인 점수(Continuous Score)를 바탕으로 정교하게 가중치를 조율(Soft Weighting)하게 됩니다.

## 실행 방법 (How to Run)

본 환경은 CUDA GPU를 권장하며, 필수 패키지(`pesq`, `pystoi`, `scipy`, `matplotlib`)가 설치되어 있어야 합니다. 이제 단일 데이터셋이 아닌 다중 데이터셋(LibriSpeech, VCTK, LJSpeech)을 완벽히 지원합니다.

```bash
# 기본 실행 (데이터셋 선택 가능: librispeech, vctk, ljspeech)
python phase1_attribution.py --dataset_type librispeech --split test
```

## 주요 출력물 및 해석 가이드

실행 후 다음과 같은 지표와 시각화 결과가 산출됩니다.

### 1. 상관계수 지표 (Correlation Metrics)
* **Pearson r**: Survival Map과 Decoder Gradient Map 간의 선형 상관관계.
* **Spearman rho**: 두 맵 간의 순위(rank) 기반 비선형 단조 상관관계.
* **Top-20% IoU**: 두 맵에서 상위 20%에 해당하는 핵심 T-F(Time-Frequency) 픽셀이 얼마나 일치하는지 비율(Intersection over Union).

**[해석 기준]**
본 연구에서는 임의의 $r$ 임계값(Threshold) 대신, 각 샘플 단위로 도출된 상관계수의 **평균과 신뢰구간(Confidence Interval)**을 통해 전체적인 경향성을 파악합니다. 수치가 높을수록 Survival Map의 프록시 타당성이 높음을 의미합니다.

### 2. 마스킹 실험 결과 (Causal Verification)
상관 분석만으로는 인과관계를 단정할 수 없으므로, 실제 오디오에서 특정 영역의 잔차(residual)만 남기고 나머지는 제거했을 때의 BER(Bit Error Rate) 성능을 측정합니다.

* **Full (Baseline)**: 모든 잔차가 존재하는 일반 워터마크 상태.
* **High-Survival (Top 20%)**: Survival Map 기준 상위 20% 잔차만 남긴 상태.
* **High-Gradient (Top 20%)**: Decoder Gradient Map 기준 상위 20% 잔차만 남긴 상태.
* **Random 20%**: 무작위 20% 잔차만 남긴 상태.

**[해석 기준 및 분기 결정]**
* 마스킹 시 OOD(Out-of-Distribution) 문제를 피하기 위해 **Soft Masking과 Local Energy Noise Filling** 기법이 적용됩니다.
* **Paired t-test**: `High-Survival`과 `Low-Survival`의 평균 BER 차이에 대해 통계적 유의성(p-value < 0.05)을 검증합니다.
* 유의미한 차이가 확인되면, Survival Map이 실제 에러율 방어에 인과적으로 기여함이 증명된 것이므로 **Phase 2 (Survival Gate 학습)** 로 진입할 타당성을 확보하게 됩니다.

### 3. 시각화 결과 (Visualization)
* `results/phase1_map_comparison.png` 경로에 스펙트로그램 오버레이 이미지가 저장됩니다.
* **Survival Map**과 **Decoder Gradient Map**을 육안으로 비교할 수 있으며, 오버레이 이미지에서 Green(Survival), Red(Gradient), Yellow(Overlap) 영역을 확인할 수 있습니다.
