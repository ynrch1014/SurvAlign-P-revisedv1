# SurvAlign-P Technical Note — Revised Protocol

## Hypothesis

At a fixed Feature-Aligned residual-energy budget, limited redistribution toward a reconstruction-surviving T-F subspace may reduce Exact-message and attribution failures under strong or unseen reconstruction models.

## Important scope

- Feature-Aligned already optimizes reconstruction robustness and perceptual losses.
- This project does not claim that its authors relied only on clean decoder gradients.
- The proposed contribution is an explicit post-hoc physical-survival prior and a controlled energy-redistribution experiment.
- **"Decoder-free" applies only to the Survival Map**, not to the Gate training. The Gate is supervised by the frozen AlignMark decoder's CE loss.

## Survival score

For attack `T`, the code computes:

- residual retention: `|T(x_wm)-T(x)| / |x_wm-x|`
- residual dominance: `|T(x_wm)-T(x)| / (|T(x_wm)-T(x)| + |T(x)-x|)`

Their product is aggregated by a configurable lower quantile. This second term is not conventional SIR and is therefore named residual dominance in the revised documentation.

## Gradient comparisons

Two different decoder maps are reported:

1. Clean gradient-magnitude saliency: sensitivity only; no sign/direction.
2. Codec-aware signed utility: negative derivative of attacked decoder CE with respect to the residual scale map.

For discrete SpeechTokenizer reconstruction, backward uses an identity STE and therefore does not contain the true codec Jacobian. **This means the Gate cannot learn the codec's actual frequency selectivity through backpropagation**, and relies mostly on the clean decoder's response.

## Evaluation

All calibration/test crops, messages and stochastic attack seeds are deterministic. Test messages form a deterministic random unique codebook. Attribution FAR compares each prediction with every candidate ground-truth message by Hamming distance; strict FAR counts ties as failures.

**Generalization Claims:** If map generation attacks and held-out test attacks share a codec family (e.g. `speechtokenizer`), the results must be labeled "seen-attack". Use strict held-out codecs (e.g., `ffmpeg_mp3`, `facodec`) for true generalization claims.

## Limitations

- Same L2 does not prove equal imperceptibility. PESQ, STOI, and SI-SDR are reported to verify non-inferiority.
- External ClearerVoice/FACodec wrappers must use official models and be documented separately.
- LJSpeech cannot be speaker-disjoint.
- Public AlignMark inference code uses a quantized `SpeechTokenizer.encode` path, while the paper describes a no-quantization codec path. Both are exposed through `--latent_mode` and must not be conflated.
