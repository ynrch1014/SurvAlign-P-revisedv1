# SurvAlign-P Phase 1: Attribution Correlation Analysis

## 개요 (Overview)
본 코드는 SurvAlign-P의 가장 핵심적인 가정인 **"물리적으로 워터마크 잔차가 잘 살아남는 위치(Survival Map)가, 실제로 Decoder가 워터마크를 복호하는 데 유용하게 사용하는 위치(Decoder Utility Map)와 일치하는가?"**를 실험적으로 검증하기 위한 분석용 프레임워크입니다.

이 분석 결과는 향후 워터마크 개선 전략을 결정짓는 중요한 나침반이 됩니다.

## 실행 방법 (How to Run)

본 환경은 CUDA GPU를 권장하며, 필수 패키지(`pesq`, `pystoi`, `scipy`, `matplotlib`)가 설치되어 있어야 합니다.

```bash
# 기본 실행 (Calibration 세트 중 20샘플 한정으로 빠른 평가)
python phase1_attribution.py
```

## 주요 출력물 및 해석 가이드

실행 후 다음과 같은 지표와 시각화 결과가 산출됩니다.

### 1. 상관계수 지표 (Correlation Metrics)
* **Pearson r**: Survival Map과 Decoder Gradient Map 간의 선형 상관관계.
* **Spearman rho**: 두 맵 간의 순위(rank) 기반 비선형 단조 상관관계.
* **Top-20% IoU**: 두 맵에서 상위 20%에 해당하는 핵심 T-F(Time-Frequency) 픽셀이 얼마나 일치하는지 비율(Intersection over Union).

**[해석 기준]**
* `r > 0.5`: Survival Map이 Decoder Utility를 상당히 잘 대변합니다. 기존 SurvAlign-P 설계를 신뢰할 수 있습니다.
* `r < 0.3`: 물리적으로 신호가 많이 남는 것과 Decoder가 정보를 잘 읽는 것은 별개의 문제입니다. Survival Map 기반의 보완 전략은 한계가 명확하며, Decoder-Guided 방식(Phase 2B)으로의 전환을 고려해야 합니다.

### 2. 마스킹 실험 결과 (Causal Verification)
상관 분석만으로는 인과관계를 단정할 수 없으므로, 실제 오디오에서 특정 영역의 잔차(residual)만 남기고 나머지는 제거했을 때의 BER(Bit Error Rate) 성능을 측정합니다.

* **Full (Baseline)**: 모든 잔차가 존재하는 일반 워터마크 상태.
* **High-Survival (Top 20%)**: Survival Map 기준 상위 20% 잔차만 남긴 상태.
* **High-Gradient (Top 20%)**: Decoder Gradient Map 기준 상위 20% 잔차만 남긴 상태.
* **Random 20%**: 무작위 20% 잔차만 남긴 상태.

**[해석 기준]**
* 만약 `High-Survival`의 ACC가 `Random`보다 유의미하게 높다면, Survival Map은 가치가 있습니다.
* 만약 `High-Gradient`의 ACC가 `High-Survival`보다 월등히 높다면, 향후 Gate 설계 시 Gradient Map을 사용하는 것이 훨씬 강력한 효과를 발휘할 것임을 증명합니다.

### 3. 시각화 결과 (Visualization)
* `results/phase1_map_comparison.png` 경로에 스펙트로그램 오버레이 이미지가 저장됩니다.
* **Survival Map**과 **Decoder Gradient Map**을 육안으로 비교할 수 있으며, 오버레이 이미지에서 Green(Survival), Red(Gradient), Yellow(Overlap) 영역을 확인할 수 있습니다.
