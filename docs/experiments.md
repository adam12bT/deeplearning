# Experiment history

The project originally used a two-control sky/terrain model and procedural
scene crops. Those images and models were superseded by the five-control v2
pipeline. Legacy scripts, checkpoints, token sets, generated outputs and
one-off experiment folders were removed during cleanup. Active artifacts
were preserved and verified against their original hashes and inference
outputs; see `runs/cleanup_validation/`.

Current baseline: 13,824 v2 images, adapted VQ-VAE, and a fresh 4.48-million-
parameter five-control Transformer trained for 30 epochs. Best seen-validation
loss was 0.7032. The first unseen-feature test generated 32 unique sequences
with no exact training-token matches. Some reserved combinations worked,
but volcanoes, rivers and other requested features were not always present.
These finite tests do not rule out all memorization.

Quality experiments found persistent loss of fine detail in VQ-VAE
reconstructions. Signed-edge regularization modestly improved metrics over an
equally trained MSE control, but did not resolve blur. A zero-initialized
subpixel decoder detail head improved error by only about 0.5% over its
matched decoder-only control, with no convincing visible improvement. Neither
experimental model replaced the active tokenizer or existing tokens.

The tested optional detail-head and decoder-compatibility implementations
remain as reusable, regression-tested functionality. Their experiment
checkpoints and per-run tooling were removed. The default architecture and
active model remain unchanged. A decoder-only replacement must have identical
encoder/codebook weights and normalization state; the generation CLI checks
this when `--decoder-checkpoint` is explicitly supplied.

The unresolved issues are reconstruction softness and imperfect feature
adherence. More epochs or lower numerical loss alone do not establish better
visual quality. Inspect source/reconstruction comparisons before choosing
another tokenizer or running a full retraining cycle.

A subsequent direct 2x2 pixel-patch codec passed a clearer reconstruction
gate: about 62% lower pixel error and 61% lower source-edge error than the
active VQ-VAE on the same 1,632 validation images. It keeps 1,024 tokens and
2,048 code categories, but changes their meanings. A separate matching
Transformer candidate is training under `runs/pixel_patch_training/`.
The active baseline is preserved. Generation quality for this candidate
must still be evaluated; reconstruction results alone do not establish it.
