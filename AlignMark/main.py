import warnings

warnings.filterwarnings("ignore")

import argparse
from types import SimpleNamespace

from inferencer import WatermarkInferencer


NCHUNK_SIZE = 4
NBITS = 16
SAMPLE_RATE = 16000
CHECKPOINT_PATH = "weight.pth"


def build_cfg(device: str) -> SimpleNamespace:
    cfg = SimpleNamespace(
        device=device,
        local_rank=None,
        sample_rate=SAMPLE_RATE,
        nbits=NBITS,
        wm_mb=SimpleNamespace(nfft=256, sr=SAMPLE_RATE, nchunk_size=NCHUNK_SIZE),
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(description="AlignMark")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_embed = subparsers.add_parser("embed", help="Embed watermark into audio")
    p_embed.add_argument("--input", required=True, help="Input audio path")
    p_embed.add_argument("--output", required=True, help="Output watermarked audio path")
    p_embed.add_argument("--message", default=None, help="Binary string message, e.g., 1111000011110000")
    p_embed.add_argument("--device", default="cpu")

    p_decode = subparsers.add_parser("decode", help="Decode watermark from audio")
    p_decode.add_argument("--input", required=True, help="Input audio path")
    p_decode.add_argument("--label", default=None, help="Optional label bits (string) to compute Hamming distance")
    p_decode.add_argument("--device", default="cpu")

    args = parser.parse_args()

    cfg = build_cfg(args.device)
    inferencer = WatermarkInferencer(cfg)
    inferencer.load_model(checkpoint_path=CHECKPOINT_PATH)

    if args.cmd == "embed":
        result = inferencer.embed(args.input, args.output, message=args.message)
    else:
        result = inferencer.decode(args.input, label=args.label)
    print(result)


if __name__ == "__main__":
    main()


