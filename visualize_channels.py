# -*- coding: utf-8 -*-
"""워터마크 잔차(채널 2) vs Survival Map(채널 3) 비교 시각화.

목적: "guide 채널(Survival Map)의 내용이 성능에 영향을 주지 않는다"는 반복된 결과가
"Survival Map이 이미 워터마크 잔차와 비슷한 곳을 가리켜서 새 정보가 없기 때문"인지 눈으로
확인한다.

채널 정의는 phase2_training.py의 build_guide_map()/get_survival_map()을 그대로 재사용한다
(중복 구현 금지 -- 과거 dispatch 불일치 버그가 바로 이 중복에서 나왔다). 이 스크립트는 학습된
Gate 체크포인트 없이도 동작한다: AlignMark로 원본 오디오를 워터마킹한 결과(원본+잔차)만 있으면
2번(잔차)/3번(Survival Map) 채널을 그대로 만들 수 있다.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from types import SimpleNamespace

from experiment_utils import align_audio_tensors, stable_int_hash
from phase1_attribution import valid_correlations
from phase2_training import build_guide_map, parse_csv_list
from survalign_p import AlignMarkManager, DifferentiableDistortion, UnifiedSpeechDataset

# Phase1의 map_comparison.png(1x3, Survival/Gradient/Utility)과 달리, 여기서는 Phase2 파이프
# 라인이 실제로 쓰는 채널만(원본/잔차/Survival Map) 비교한다.
DEFAULT_SURVIVAL_ATTACKS = (
    "replacement,masking,frame_shuffle,lowpass,bandpass,highpass,"
    "ffmpeg_mp3,ffmpeg_aac,encodec,vocos"
)


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def to_db(magnitude: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return 20.0 * np.log10(magnitude + eps)


def top_quantile_mask(values: np.ndarray, quantile: float) -> np.ndarray:
    threshold = np.quantile(values, quantile)
    return values >= threshold


def make_overlap_panel(ax, residual_arr: np.ndarray, survival_arr: np.ndarray, top_fraction: float):
    residual_top = top_quantile_mask(residual_arr, 1.0 - top_fraction)
    survival_top = top_quantile_mask(survival_arr, 1.0 - top_fraction)
    code = np.zeros(residual_arr.shape, dtype=int)
    code[residual_top & ~survival_top] = 1
    code[~residual_top & survival_top] = 2
    code[residual_top & survival_top] = 3
    cmap = ListedColormap(["#eeeeee", "crimson", "royalblue", "gold"])
    ax.imshow(code, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=3)
    union = int((residual_top | survival_top).sum())
    overlap_ratio = float((code == 3).sum()) / max(1, union)
    ax.set_title(f"Top {int(top_fraction * 100)}% overlap ({overlap_ratio:.1%})")
    return overlap_ratio


def visualize_one(args, alignmark, distorter, wav, msg, metadata, device, out_dir):
    wav = wav.unsqueeze(0).to(device)
    msg = msg.unsqueeze(0).to(device)

    wav_wm, residual = alignmark.embed(wav, msg)
    wav, wav_wm, residual = align_audio_tensors(wav, wav_wm, residual)

    sample_id = metadata["sample_id"]
    context_seed = stable_int_hash(args.seed, "viz", sample_id)

    guide_args = SimpleNamespace(
        mode="proposed_gate",
        map_type="survival",
        current_msg=msg,
        survival_attack_names=args.survival_attack_names,
        survival_quantile=args.survival_quantile,
        channel_ablation="full",
        mp3_bitrate=args.mp3_bitrate,
        encodec_command=args.encodec_command,
        vocos_command=args.vocos_command,
        clearervoice_command="",
        facodec_command="",
        dac_command="",
        hifigan_command="",
        clearervoice_snr=10.0,
    )
    feature_pack, residual_spec, guide, masking_map = build_guide_map(
        guide_args, alignmark, distorter, wav, wav_wm, residual, context_seed
    )

    clean_channel = feature_pack[0, 0].detach().cpu().numpy()
    residual_channel = feature_pack[0, 1].detach().cpu().numpy()
    survival_channel = guide[0].detach().cpu().numpy()

    residual_db = to_db(np.abs(residual_spec[0].detach().cpu().numpy()))

    correlation = valid_correlations(residual_channel.reshape(-1), survival_channel.reshape(-1))
    pearson, spearman = correlation if correlation is not None else (float("nan"), float("nan"))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    im0 = axes[0, 0].imshow(clean_channel, aspect="auto", origin="lower", cmap="magma")
    axes[0, 0].set_title("Original Spectrogram (Channel 1)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(residual_channel, aspect="auto", origin="lower", cmap="magma")
    axes[0, 1].set_title("Watermark Residual (Channel 2)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(survival_channel, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Survival Map (Channel 3)")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(residual_db, aspect="auto", origin="lower", cmap="gray")
    axes[1, 0].imshow(survival_channel, aspect="auto", origin="lower", cmap="magma", alpha=0.5, vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Survival Map (magma) overlaid on Residual (gray)")

    overlap_ratio = make_overlap_panel(axes[1, 1], residual_channel, survival_channel, args.top_fraction)

    sample_n = min(args.scatter_points, residual_channel.size)
    rng = np.random.RandomState(context_seed % (2 ** 31 - 1))
    idx = rng.choice(residual_channel.size, size=sample_n, replace=False)
    axes[1, 2].scatter(
        residual_channel.reshape(-1)[idx], survival_channel.reshape(-1)[idx],
        s=4, alpha=0.3, color="tab:purple",
    )
    axes[1, 2].set_xlabel("Residual channel value (Channel 2)")
    axes[1, 2].set_ylabel("Survival Map value (Channel 3)")
    axes[1, 2].set_title(f"Pixel-wise correlation\nPearson={pearson:.3f}, Spearman={spearman:.3f}")

    fig.suptitle(f"sample={sample_id}  speaker={metadata['speaker_id']}")
    fig.tight_layout()

    out_name = f"sample_{args.sample_display_index}_{sanitize(metadata['speaker_id'])}.png"
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return {
        "sample_id": sample_id,
        "speaker_id": metadata["speaker_id"],
        "output": out_path,
        "pearson": pearson,
        "spearman": spearman,
        "top_overlap_ratio": overlap_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Residual(채널2) vs Survival Map(채널3) 비교 시각화")
    parser.add_argument("--dataset_type", default="librispeech", choices=["librispeech", "vctk", "ljspeech", "combined"])
    parser.add_argument("--dataset_name", default="dev-clean")
    parser.add_argument("--combined_protocol", default="speaker_disjoint", choices=["speaker_disjoint", "paper"])
    parser.add_argument("--latent_mode", default="public_code", choices=["public_code", "unquantized"])
    parser.add_argument("--split", default="test", choices=["train", "calib", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--sample_indices", default="",
                        help="쉼표로 구분된 명시적 데이터셋 인덱스. 지정하면 --n_samples는 무시됨.")
    parser.add_argument("--survival_attacks", default=DEFAULT_SURVIVAL_ATTACKS)
    parser.add_argument("--survival_quantile", type=float, default=0.5)
    parser.add_argument("--top_fraction", type=float, default=0.2,
                        help="겹침 비교 패널에서 '상위 몇 %%'를 비교할지 (기본 상위 20%%).")
    parser.add_argument("--scatter_points", type=int, default=3000)
    parser.add_argument("--mp3_bitrate", default="64k")
    parser.add_argument("--encodec_command", default="")
    parser.add_argument("--vocos_command", default="")
    parser.add_argument("--output_dir", default="outputs/channel_viz")
    args = parser.parse_args()

    args.survival_attack_names = parse_csv_list(args.survival_attacks)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_attack_names = set(args.survival_attack_names)
    if ("encodec" in all_attack_names and not args.encodec_command) or (
        "vocos" in all_attack_names and not args.vocos_command
    ):
        from inprocess_attacks import prewarm

        prewarm(device)

    alignmark = AlignMarkManager(device, latent_mode=args.latent_mode)
    distorter = DifferentiableDistortion(sr=16000, vae=alignmark.vae).to(device)
    dataset = UnifiedSpeechDataset(
        dataset_type=args.dataset_type,
        dataset_name=args.dataset_name,
        split=args.split,
        seed=args.seed,
        return_metadata=True,
        combined_protocol=args.combined_protocol,
    )

    if args.sample_indices:
        indices = [int(v) for v in parse_csv_list(args.sample_indices)]
    else:
        rng = np.random.RandomState(args.seed)
        indices = sorted(rng.choice(len(dataset), size=min(args.n_samples, len(dataset)), replace=False).tolist())

    results = []
    for display_index, dataset_index in enumerate(indices):
        wav, msg, metadata = dataset[dataset_index]
        args.sample_display_index = display_index
        result = visualize_one(args, alignmark, distorter, wav, msg, metadata, device, args.output_dir)
        print(
            f"[{display_index}] idx={dataset_index} sample_id={result['sample_id']} "
            f"speaker={result['speaker_id']} pearson={result['pearson']:.3f} "
            f"spearman={result['spearman']:.3f} top{int(args.top_fraction*100)}%_overlap="
            f"{result['top_overlap_ratio']:.1%} -> {result['output']}"
        )
        results.append(result)

    valid_pearson = [r["pearson"] for r in results if np.isfinite(r["pearson"])]
    valid_spearman = [r["spearman"] for r in results if np.isfinite(r["spearman"])]
    if valid_pearson:
        print(
            f"\n[SUMMARY] n={len(results)} "
            f"pearson_mean={np.mean(valid_pearson):.3f} spearman_mean={np.mean(valid_spearman):.3f}"
        )


if __name__ == "__main__":
    main()
