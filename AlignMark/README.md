# Feature-Aligned Speech Watermarking for Robustness to Reconstruction Distortions

Official implementation of the ICME 2026 paper.

**Authors**  
Haiyun Li, Shuhai Peng, Zhisheng Zhang, Jingran Xie, Xiaofeng Xie, Hanyang Peng, and Zhiyong Wu

**Contact**  
- lihaiyun24@mails.tsinghua.edu.cn

**Demo**  
- https://hyli-research.github.io/AlignMark/

## Abstract

Audio watermarking aims to embed identifiable information into audio while remaining imperceptible. Existing methods adopt high-fidelity, low-energy designs to preserve perceptual quality, but the resulting watermarks lack robustness under suppression by speech reconstruction models. Improving robustness is challenging due to the inherent robustness-fidelity trade-off in existing designs, where increasing watermark energy improves robustness but reduces fidelity.

To address this problem, we propose a feature-aligned watermarking method that aligns the watermark with the original speech feature distribution, allowing higher watermark energy to improve robustness while preserving imperceptibility. We use a pretrained speech codec to generate a pseudo-speech watermark and fuse it into the spectrogram of the input audio, with VAD loss and perceptual losses guiding embedding within voiced regions. Experiments show that our method maintains imperceptibility comparable to existing approaches while substantially improving robustness under both seen and unseen speech reconstruction models.

## Highlights

- Feature-aligned watermark design for improved robustness under reconstruction-based suppression
- Pseudo-speech watermark generation with a pretrained speech codec
- Voiced-region embedding guided by VAD and perceptual objectives

## Requirements

- Python 3.11 or later
- Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

### Embed a watermark

```bash
python -m main embed \
  --input ./example.wav \
  --output ./outputs/watermarked.wav \
  --message 1111001110101001 \
  --device cpu
```

### Decode a watermark

```bash
python -m main decode \
  --input ./outputs/watermarked.wav \
  --label 1111001110101001 \
  --device cpu
```

## Repository Structure

- `main.py` — command-line entry point
- `inferencer.py` — end-to-end embedding and detection pipeline
- `base.py` — model loading, checkpoint handling, and device setup
- `models.py` — watermarking model components
- `util.py` — helper utilities for message generation and serialization
- `speechtokenizer/` — external dependency with pretrained codec components
- `weight.pth` — released checkpoint
- `example.wav` — sample audio for quick testing
- `outputs/` — generated output files

## License

This project is released under the Apache 2.0 License.
