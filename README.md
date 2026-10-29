# Pixel-art landscape Transformer

## Text interface and versioned deployment

The deployed model is now the **12.62M-parameter pixel-patch Transformer**,
approved release `landscape-large-v1` (best epoch 20). It uses direct 2x2 RGB
patch decoding, rather than the older VQ-VAE pipeline described below.

`frontend/` contains the React studio. `backend/` translates text through Groq
into five validated controls and calls the Hugging Face generator. Model
releases are registered in `models/registry.json`, datasets are tracked with DVC,
and `.github/workflows/` contains CI and manual release deployment.

See **[platform setup and deployment](docs/platform.md)** for local commands,
Groq configuration, Vercel, Google Drive storage and rollback instructions.

**[MLflow dashboard](docs/mlflow.md)** tracks real training curves, model
provenance and comparison images. Start it with `python -m mlops.mlflow_server`
and open http://localhost:5000. Future training can enable tracking with `--mlflow`.

The sections below describe the retained older VQ-VAE baseline.

A local autoregressive model for 64x64 pixel-art landscapes. A VQ-VAE encodes
each image as 1,024 visual tokens; the Transformer generates those tokens
using five independent controls: **sky, landform, ground, water, objects**.

## Current model

- Transformer: `checkpoints/model_compositional_v2/best.pt` (30 epochs).
- Matching tokenizer: `checkpoints/vqvae_v2/vqvae_best.pt`.
- Validated tokens: `checkpoints/tokens_v2/`.
- Dataset: `dataset/` (13,824 images, 432 combinations).
- Run status and history: `runs/compositional_training/`.

The model can combine some unseen features, but may omit requested objects.
Reconstruction and generated images still lose fine details. See
`runs/compositional_test_epoch30/CONCLUSION.md` and `docs/experiments.md`.

## Generate images

Run commands from this project directory. Dependencies are listed in
`requirements.txt`; GPU training requires an appropriate CUDA-enabled PyTorch
installation.

```powershell
python generate_compositional.py --sky day --landform mountains --ground grass --water river --objects trees --num-images 4
```

Images and reproducibility metadata are saved in `generated_v2/`. Labels are
listed in `dataset/schema.json`. For moon + desert + volcano, use:

```powershell
python generate_compositional.py --sky night --landform volcano --ground sand --water river --objects none --num-images 4
```

That core combination was deliberately excluded from training. Reliable
unseen-combination generation is not guaranteed. The generator verifies the
exact tokenizer fingerprint recorded in the Transformer checkpoint.

## Training and evaluation

The existing training run has finished. To continue it to 40 total epochs:

```powershell
python -u training/run_compositional_training.py --resume --epochs 40
```

Track progress or request a stop after an epoch is saved:

```powershell
Get-Content runs/compositional_training/train.log -Tail 12 -Wait
New-Item -ItemType File runs/compositional_training/STOP_AFTER_EPOCH -Force
```

Do not start multiple trainers at once. `best.pt` is selected using seen
validation loss; `last.pt` includes optimizer, scheduler, scaler and RNG state.
Resume repeats any unsaved partial epoch. Closing the log viewer does not stop
training.

Inspect source images against tokenizer reconstructions:

```powershell
python evaluation/check_reconstructions.py
```

Test reserved combinations against familiar controls:

```powershell
python evaluation/test_compositional.py
```

Fine-tune the tokenizer into a separate directory:

```powershell
python training/finetune_vqvae_v2.py --out-dir checkpoints/vqvae_finetune --run-dir runs/vqvae_finetune
```

Changing the encoder/codebook requires fresh tokens and a matching Transformer.
Extraction refuses to overwrite existing output:

```powershell
python training/extract_tokens.py --checkpoint checkpoints/vqvae_finetune/vqvae_best.pt --output-dir checkpoints/tokens_finetune
```

To create a new Transformer run, use `training/train_compositional.py` with
explicit `--tokens-dir`, `--out-dir`, and `--run-dir` arguments.

## Dataset

`dataset/manifest.json` is the single active image manifest. Splits are 9,792
train, 1,632 validation, and 2,400 test images. Night + volcano + sand and
sunset + mountains + grass are reserved entirely for testing across every
water/object choice. All individual feature values still occur in training.

Build a replacement in an empty staging directory and validate before using it:

```powershell
python datasets/generate_compositional.py --output-dir dataset_build
python datasets/validate_compositional.py dataset_build
```

The generator refuses to overwrite a nonempty directory. New images do not
update existing model weights automatically.

## Layout

```text
configs/       Active dataset and model paths
dataset/       Images, schema, manifest, validation and preview
datasets/      Procedural generator, loader and validator
tokenizer/     VQ-VAE, checkpoint loader and compatibility checks
transformer/   Shared causal blocks and five-control model
training/      Token extraction, fine-tuning and Transformer training
evaluation/    Reconstruction review and held-out generation test
tests/         Model, loss and compatibility regression tests
checkpoints/   Active pipeline and current candidate checkpoints
runs/          Current training/evaluation reports and cleanup audit
docs/          Compact experiment history
```

Run regression tests:

```powershell
python -m unittest discover -s tests -v
```

## Sharp-pixel candidate

A direct 2x2 patch tokenizer is being tested to avoid reconstruction blur.
Its source/reconstruction review is in `runs/pixel_patch_review/comparison.png`.
A separate matching Transformer run lives in `runs/pixel_patch_training/`;
see that folder's README for live progress and candidate-generation commands.
The original active model above is preserved. Reconstruction improvement
does not guarantee improved free generation until the new model is evaluated.
