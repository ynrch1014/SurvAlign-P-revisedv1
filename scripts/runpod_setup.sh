#!/bin/bash
# SurvAlign-P RunPod 환경 복구 스크립트
# 새 Pod(마이그레이션 등으로 컨테이너가 초기화된 경우)에서 실행:
#   cd /workspace && git clone https://github.com/min-627/SurvAlign-P-revisedv1.git
#   cd SurvAlign-P-revisedv1 && bash scripts/runpod_setup.sh

set -e

echo "=== [1/5] Python 패키지 설치 ==="
pip install scipy speechtokenizer encodec vocos pyworld librosa==0.10.1

echo "=== [2/5] AlignMark 체크포인트 다운로드 ==="
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='haiyunli/AlignMark', filename='weight.pth', local_dir='AlignMark')
hf_hub_download(repo_id='haiyunli/AlignMark', filename='speechtokenizer/pretrained_model/SpeechTokenizer.pt', local_dir='AlignMark')
print('Checkpoints downloaded.')
"

echo "=== [3/5] LibriSpeech dev-clean 다운로드 ==="
mkdir -p /workspace/data
python -c "
import torchaudio
torchaudio.datasets.LIBRISPEECH('/workspace/data', url='dev-clean', download=True)
print('LibriSpeech downloaded.')
"

echo "=== [4/5] 데이터 심볼릭 링크 연결 ==="
mkdir -p data
ln -sf /workspace/data/LibriSpeech data/LibriSpeech

echo "=== [5/5] 회귀 테스트 확인 ==="
python smoke_test.py
python test_revisions.py

echo ""
echo "=== 셋업 완료 ==="
echo "AlignMark 동작 확인: cd AlignMark && python -m main embed --input ./example.wav --output ./outputs/watermarked.wav --message 1111001110101001 --device cpu"
