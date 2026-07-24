#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate E1~E6 Architecture Diagrams for SurvAlign-P Survival Gate.

Uses actual sample audio channels extracted by visualize_channels.py to build
publication-quality architecture diagrams for the 6 core experiment/ablation scenarios:

  E1: Proposed Survival Gate (Full 4-Channel Input)
  E2: No-Guide Map Ablation (Channel 3 Zeroed Out)
  E3: No-Residual Input Ablation (Channel 2 Zeroed Out)
  E4: Energy Gate Ablation (Auditory Energy Proxy Only)
  E5: Shuffled Survival Gate Ablation (Spatially Permuted Prior)
  E6: Constant Gate Ablation (Uniform 1.0 Prior Input)
"""
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from types import SimpleNamespace
from survalign_p import AlignMarkManager, DifferentiableDistortion, UnifiedSpeechDataset, align_audio_tensors
from phase2_training import build_guide_map
from experiment_utils import stable_int_hash


def create_diagram(
    scenario_id: str,
    scenario_title: str,
    ch1_img: np.ndarray,
    ch2_img: np.ndarray,
    ch3_img: np.ndarray,
    ch4_img: np.ndarray,
    output_path: str,
    ch3_label: str = "Guide Map (Survival)",
    ch2_label: str = "Residual Spectrogram",
    is_ch3_masked: bool = False,
    is_ch2_masked: bool = False,
    is_ch3_shuffled: bool = False,
    is_ch3_constant: bool = False,
    annotation_text: str = "",
):
    """Draw a publication-quality architecture diagram matching the user's reference."""
    fig = plt.figure(figsize=(16, 8.5), dpi=200, facecolor="#f8fafc")
    gs = GridSpec(1, 1, figure=fig, left=0.01, right=0.99, bottom=0.01, top=0.99)
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 850)

    # Palette
    c_navy = "#1e3a8a"
    c_teal = "#0f766e"
    c_blue = "#2563eb"
    c_dark = "#0f172a"
    c_light_bg = "#eff6ff"
    c_box_bg = "#ffffff"
    c_border = "#94a3b8"

    # --- Title Banner ---
    title_rect = patches.FancyBboxPatch(
        (30, 790), 1540, 48, boxstyle="round,pad=6",
        facecolor=c_navy, edgecolor="none"
    )
    ax.add_patch(title_rect)
    ax.text(
        800, 814, f"SurvAlign-P Architecture: {scenario_id} - {scenario_title}",
        color="white", fontsize=15, fontweight="bold", ha="center", va="center"
    )

    # --- Left: 4-Channel Input Box ---
    input_box = patches.FancyBboxPatch(
        (30, 200), 320, 560, boxstyle="round,pad=10",
        facecolor="#f1f5f9", edgecolor=c_navy, linewidth=2
    )
    ax.add_patch(input_box)

    input_header = patches.FancyBboxPatch(
        (40, 720), 300, 32, boxstyle="round,pad=4",
        facecolor=c_navy, edgecolor="none"
    )
    ax.add_patch(input_header)
    ax.text(190, 736, "4-Channel Input", color="white", fontsize=12, fontweight="bold", ha="center", va="center")

    # Inner channel specs (Ch1..Ch4)
    y_starts = [600, 480, 360, 240]
    ch_titles = [
        "1  Original Spectrogram",
        f"2  {ch2_label}",
        f"3  {ch3_label}",
        "4  Auditory Masking Proxy"
    ]
    imgs = [ch1_img, ch2_img, ch3_img, ch4_img]
    is_masked_list = [False, is_ch2_masked, is_ch3_masked, False]

    for idx in range(4):
        y = y_starts[idx]
        title = ch_titles[idx]
        img = imgs[idx]
        masked = is_masked_list[idx]

        # Channel label
        ax.text(45, y + 82, title, color=c_dark, fontsize=9.5, fontweight="bold")

        # Inset image plot
        inset_ax = fig.add_axes([50 / 1600, y / 850, 280 / 1600, 72 / 850])
        if masked:
            inset_ax.imshow(np.zeros_like(img), aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1)
            inset_ax.text(0.5, 0.5, "ABLATED (ZEROED)", color="red", fontsize=9, fontweight="bold", ha="center", va="center", transform=inset_ax.transAxes)
        elif idx == 2 and is_ch3_shuffled:
            inset_ax.imshow(img, aspect="auto", origin="lower", cmap="magma")
            inset_ax.text(0.5, 0.5, "SPATIALLY SHUFFLED", color="orange", fontsize=8, fontweight="bold", ha="center", va="center", transform=inset_ax.transAxes)
        elif idx == 2 and is_ch3_constant:
            inset_ax.imshow(np.ones_like(img), aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
            inset_ax.text(0.5, 0.5, "CONSTANT 1.0", color="blue", fontsize=9, fontweight="bold", ha="center", va="center", transform=inset_ax.transAxes)
        else:
            inset_ax.imshow(img, aspect="auto", origin="lower", cmap="magma")

        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

    # --- Middle: Lightweight 2D CNN Gate Box ---
    cnn_box = patches.FancyBboxPatch(
        (420, 440), 400, 320, boxstyle="round,pad=10",
        facecolor="#ecfdf5", edgecolor=c_teal, linewidth=2
    )
    ax.add_patch(cnn_box)

    cnn_header = patches.FancyBboxPatch(
        (430, 720), 380, 32, boxstyle="round,pad=4",
        facecolor=c_teal, edgecolor="none"
    )
    ax.add_patch(cnn_header)
    ax.text(620, 736, "Lightweight 2D CNN Gate", color="white", fontsize=12, fontweight="bold", ha="center", va="center")

    # 3 Conv blocks matching PyTorch SimplifiedSurvivalGate
    conv_xs = [445, 560, 675]
    block_labels = [
        ("Conv 3x3", "GN(4) + GELU", "16 ch"),
        ("Conv 3x3", "GN(4) + GELU", "16 ch"),
        ("Conv 3x3", "Zero-Init", "1 ch")
    ]
    for c_idx, cx in enumerate(conv_xs):
        c_rect = patches.FancyBboxPatch(
            (cx, 510), 100, 150, boxstyle="round,pad=4",
            facecolor="#ccfbf1", edgecolor=c_teal, linewidth=1.5
        )
        ax.add_patch(c_rect)
        l_name, l_sub, l_ch = block_labels[c_idx]
        ax.text(cx + 50, 620, l_name, color=c_dark, fontsize=9.5, fontweight="bold", ha="center", va="center")
        ax.text(cx + 50, 580, l_sub, color=c_teal, fontsize=8.5, fontweight="bold", ha="center", va="center")
        ax.text(cx + 50, 540, l_ch, color=c_dark, fontsize=8.5, ha="center", va="center")
        if c_idx < 2:
            ax.annotate("", xy=(cx + 115, 585), xytext=(cx + 100, 585),
                        arrowprops=dict(arrowstyle="->", color=c_teal, lw=2))

    ax.text(620, 465, "3 layers, 16 hidden channels (O(1) Params)", color=c_teal, fontsize=9.5, fontweight="bold", ha="center", va="center")

    # Connect Input Box -> CNN Gate
    ax.annotate("", xy=(420, 600), xytext=(350, 600),
                arrowprops=dict(arrowstyle="->", color=c_navy, lw=2.5))

    # --- Top Right: Scaling Map alpha(t,f) Box ---
    scale_box = patches.FancyBboxPatch(
        (890, 440), 340, 320, boxstyle="round,pad=10",
        facecolor=c_box_bg, edgecolor=c_teal, linewidth=2
    )
    ax.add_patch(scale_box)
    ax.text(1060, 736, "Scaling Map alpha(t, f)", color=c_teal, fontsize=12, fontweight="bold", ha="center", va="center")

    # Plot Scaling Map image
    scale_map_img = (ch3_img * 0.4) + 0.8  # simulate scaling map in [0.8, 1.2]
    inset_scale = fig.add_axes([910 / 1600, 470 / 850, 300 / 1600, 230 / 850])
    inset_scale.imshow(scale_map_img, aspect="auto", origin="lower", cmap="viridis", vmin=0.8, vmax=1.2)
    inset_scale.set_xticks([])
    inset_scale.set_yticks([])

    # Range badge
    range_badge = patches.FancyBboxPatch(
        (1250, 560), 140, 70, boxstyle="round,pad=6",
        facecolor="#f0fdf4", edgecolor=c_teal, linewidth=1.5, linestyle="--"
    )
    ax.add_patch(range_badge)
    ax.text(1320, 605, "Gate Range:", color=c_teal, fontsize=9, ha="center", va="center")
    ax.text(1320, 580, "alpha in [0.8, 1.2]", color=c_teal, fontsize=10, fontweight="bold", ha="center", va="center")

    # Connect CNN -> Scaling Map
    ax.annotate("", xy=(890, 600), xytext=(820, 600),
                arrowprops=dict(arrowstyle="->", color=c_teal, lw=2.5))

    # --- Bottom Flow: Element-wise Scaling ---

    # 1. Original Residual Box (Bottom Left)
    res_orig_box = patches.FancyBboxPatch(
        (30, 60), 280, 110, boxstyle="round,pad=6",
        facecolor=c_box_bg, edgecolor=c_border, linewidth=1.5
    )
    ax.add_patch(res_orig_box)
    ax.text(170, 150, "Original Residual r_0", color=c_dark, fontsize=10, fontweight="bold", ha="center", va="center")
    inset_r0 = fig.add_axes([40 / 1600, 70 / 850, 260 / 1600, 65 / 850])
    inset_r0.imshow(ch2_img, aspect="auto", origin="lower", cmap="magma")
    inset_r0.set_xticks([])
    inset_r0.set_yticks([])

    # Multiply Operator Circle
    mult_circle = patches.Circle((420, 115), 25, facecolor="#fef3c7", edgecolor="#d97706", linewidth=2)
    ax.add_patch(mult_circle)
    ax.text(420, 115, "X", color="#b45309", fontsize=18, fontweight="bold", ha="center", va="center")

    # Arrow r_0 -> (X)
    ax.annotate("", xy=(395, 115), xytext=(310, 115),
                arrowprops=dict(arrowstyle="->", color=c_dark, lw=2))

    # Arrow Scaling Map -> (X)
    ax.annotate("", xy=(420, 140), xytext=(420, 440),
                arrowprops=dict(arrowstyle="->", color=c_teal, lw=2))
    ax.text(435, 290, "Element-wise scaling", color=c_teal, fontsize=9.5, fontweight="bold", rotation=-90, va="center")

    # 2. Gated Residual Box
    gated_box = patches.FancyBboxPatch(
        (510, 60), 280, 110, boxstyle="round,pad=6",
        facecolor=c_box_bg, edgecolor=c_border, linewidth=1.5
    )
    ax.add_patch(gated_box)
    ax.text(650, 150, "Gated Residual r_g = alpha * r_0", color=c_dark, fontsize=10, fontweight="bold", ha="center", va="center")
    inset_rg = fig.add_axes([520 / 1600, 70 / 850, 260 / 1600, 65 / 850])
    inset_rg.imshow(ch2_img * scale_map_img, aspect="auto", origin="lower", cmap="magma")
    inset_rg.set_xticks([])
    inset_rg.set_yticks([])

    # Arrow (X) -> Gated Residual
    ax.annotate("", xy=(510, 115), xytext=(445, 115),
                arrowprops=dict(arrowstyle="->", color=c_dark, lw=2))

    # 3. L2 Energy Cap Box
    l2_box = patches.FancyBboxPatch(
        (840, 60), 160, 110, boxstyle="round,pad=6",
        facecolor="#fef2f2", edgecolor="#ef4444", linewidth=2
    )
    ax.add_patch(l2_box)
    ax.text(920, 140, "L2 Energy Cap", color="#b91c1c", fontsize=10, fontweight="bold", ha="center", va="center")
    ax.text(920, 100, "||r_g||2 <= ||r_0||2", color="#991b1b", fontsize=11, fontweight="bold", ha="center", va="center")

    # Arrow Gated Residual -> L2 Cap
    ax.annotate("", xy=(840, 115), xytext=(790, 115),
                arrowprops=dict(arrowstyle="->", color=c_dark, lw=2))

    # 4. Redistributed Residual Box
    redist_box = patches.FancyBboxPatch(
        (1050, 60), 280, 110, boxstyle="round,pad=6",
        facecolor=c_box_bg, edgecolor=c_blue, linewidth=2
    )
    ax.add_patch(redist_box)
    ax.text(1190, 150, "Redistributed Residual r_redist", color=c_blue, fontsize=10, fontweight="bold", ha="center", va="center")
    inset_redist = fig.add_axes([1060 / 1600, 70 / 850, 260 / 1600, 65 / 850])
    inset_redist.imshow(ch2_img * scale_map_img, aspect="auto", origin="lower", cmap="magma")
    inset_redist.set_xticks([])
    inset_redist.set_yticks([])

    # Arrow L2 Cap -> Redistributed Residual
    ax.annotate("", xy=(1050, 115), xytext=(1000, 115),
                arrowprops=dict(arrowstyle="->", color=c_blue, lw=2))

    # Right annotation note box
    target_box = patches.FancyBboxPatch(
        (1360, 60), 210, 110, boxstyle="round,pad=6",
        facecolor="#eff6ff", edgecolor=c_blue, linewidth=1.5, linestyle="--"
    )
    ax.add_patch(target_box)
    ax.text(1465, 125, "Fine-grained", color=c_blue, fontsize=9.5, fontweight="bold", ha="center", va="center")
    ax.text(1465, 105, "residual redistribution", color=c_blue, fontsize=9.5, fontweight="bold", ha="center", va="center")
    ax.text(1465, 85, "without increasing", color=c_blue, fontsize=8.5, ha="center", va="center")
    ax.text(1465, 70, "total energy budget.", color=c_blue, fontsize=8.5, ha="center", va="center")

    # Bottom Banner
    banner = patches.FancyBboxPatch(
        (160, 5), 1280, 36, boxstyle="round,pad=4",
        facecolor="#fef3c7", edgecolor="#f59e0b", linewidth=1.5
    )
    ax.add_patch(banner)
    banner_msg = annotation_text if annotation_text else "The Gate does not create a new watermark; it redistributes the existing residual within the exact same energy budget."
    ax.text(800, 23, banner_msg, color="#92400e", fontsize=10.5, fontweight="bold", ha="center", va="center")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")


def main():
    device = torch.device("cpu")
    out_dir = "assets"
    os.makedirs(out_dir, exist_ok=True)

    # Extract sample channels from librispeech:test:30
    from survalign_p import stft_audio, get_local_energy_masking_proxy, normalize_per_sample
    from inprocess_attacks import prewarm

    prewarm(device)
    alignmark = AlignMarkManager(device, latent_mode="public_code")
    distorter = DifferentiableDistortion(sr=16000, vae=alignmark.vae).to(device)
    dataset = UnifiedSpeechDataset(
        dataset_type="librispeech", dataset_name="dev-clean", split="test",
        seed=42, return_metadata=True, combined_protocol="speaker_disjoint"
    )

    wav, msg, metadata = dataset[30]
    wav = wav.unsqueeze(0).to(device)
    msg = msg.unsqueeze(0).to(device)
    wav_wm, residual = alignmark.embed(wav, msg)
    wav, wav_wm, residual = align_audio_tensors(wav, wav_wm, residual)

    context_seed = stable_int_hash(42, "gate_arch", metadata["sample_id"])

    guide_args = SimpleNamespace(
        mode="proposed_gate", map_type="survival", current_msg=msg,
        survival_attack_names=[
            "replacement","masking","frame_shuffle","lowpass","bandpass",
            "highpass","ffmpeg_mp3","ffmpeg_aac","encodec","vocos"
        ],
        survival_quantile=0.25, channel_ablation="full", mp3_bitrate="64k",
        encodec_command="", vocos_command="", clearervoice_command="",
        facodec_command="", dac_command="", hifigan_command="", clearervoice_snr=10.0,
    )
    feature_pack, residual_spec, guide, masking_map = build_guide_map(
        guide_args, alignmark, distorter, wav, wav_wm, residual, context_seed
    )

    ch1 = feature_pack[0, 0].detach().cpu().numpy()
    ch2 = feature_pack[0, 1].detach().cpu().numpy()
    ch3 = guide[0].detach().cpu().numpy()
    ch4 = masking_map[0].detach().cpu().numpy()

    # Generate E1 ~ E6 Architecture Diagrams

    # E1: Full 4-Channel Proposed Survival Gate
    create_diagram(
        "E1", "Proposed Survival Gate (Full 4-Channel Input)",
        ch1, ch2, ch3, ch4,
        os.path.join(out_dir, "gate_arch_e1_full.png"),
        annotation_text="E1 (Main Method): Uses all 4 input channels (Clean, Residual, Survival Map Prior, Auditory Masking) for optimal redistribution.",
    )

    # E2: No Guide Map Ablation (Ch 3 Zeroed Out)
    create_diagram(
        "E2", "No-Guide Map Ablation (Channel 3 Zeroed)",
        ch1, ch2, ch3, ch4,
        os.path.join(out_dir, "gate_arch_e2_no_guide.png"),
        ch3_label="Guide Map (ZEROED / ABLATED)",
        is_ch3_masked=True,
        annotation_text="E2 (No-Guide Ablation): Channel 3 (Survival Map) is zeroed out to prove the Physical Prior is essential for codec robustness.",
    )

    # E3: No Residual Input Ablation (Ch 2 Zeroed Out)
    create_diagram(
        "E3", "No-Residual Input Ablation (Channel 2 Zeroed)",
        ch1, ch2, ch3, ch4,
        os.path.join(out_dir, "gate_arch_e3_no_residual.png"),
        ch2_label="Residual Spectrogram (ZEROED / ABLATED)",
        is_ch2_masked=True,
        annotation_text="E3 (No-Residual Ablation): Channel 2 (Raw Residual) is zeroed out to test if the Gate requires raw residual spectrum features.",
    )

    # E4: Energy Gate Ablation (Simple Masking Proxy Only)
    create_diagram(
        "E4", "Energy Gate Ablation (Local Energy Proxy Only)",
        ch1, ch2, ch4, ch4,
        os.path.join(out_dir, "gate_arch_e4_energy_gate.png"),
        ch3_label="Energy Proxy (Replaces Survival Map)",
        annotation_text="E4 (Energy Gate): Replaces the attack-simulated Survival Map with a simple local audio energy proxy map.",
    )

    # E5: Shuffled Survival Gate Ablation
    rng = np.random.RandomState(42)
    ch3_shuffled = rng.permutation(ch3.ravel()).reshape(ch3.shape)
    create_diagram(
        "E5", "Shuffled Survival Gate Ablation (Permuted Prior)",
        ch1, ch2, ch3_shuffled, ch4,
        os.path.join(out_dir, "gate_arch_e5_shuffled.png"),
        ch3_label="Guide Map (Spatially Shuffled)",
        is_ch3_shuffled=True,
        annotation_text="E5 (Shuffled Survival): Spatially permutes the Survival Map to prove exact T-F bin alignment is crucial.",
    )

    # E6: Constant Gate Ablation (Uniform 1.0 Prior)
    ch3_constant = np.ones_like(ch3)
    create_diagram(
        "E6", "Constant Gate Ablation (Uniform 1.0 Prior)",
        ch1, ch2, ch3_constant, ch4,
        os.path.join(out_dir, "gate_arch_e6_constant.png"),
        ch3_label="Guide Map (Constant 1.0)",
        is_ch3_constant=True,
        annotation_text="E6 (Constant Gate): Supplies a uniform 1.0 prior (no spatial guidance) to test if unguided scaling provides any benefit.",
    )


if __name__ == "__main__":
    main()
