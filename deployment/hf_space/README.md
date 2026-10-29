---
title: Pixel Landscape Generator
emoji: 🌙
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
python_version: '3.10'
---

Autoregressive 64×64 pixel landscapes with five feature controls.
The preview uses nearest-neighbor enlargement; the original PNG is 64×64.
Requested features are not guaranteed to appear in every sample.

Upload the **contents** of this directory to a Gradio Space, including
`model.pt`, `tokenizer.pt`, `pixel_patches.py`, and `transformer/`.
Select ZeroGPU hardware in Space settings if available on your account.
Only inference artifacts are included; no training dataset or optimizer state.

Local test: install PyTorch for your machine, then
`python -m pip install gradio==5.49.1 spaces` and `python app.py`.

The queued API endpoint is `/generate`. Inputs, in order:
`sky, landform, ground, water, objects, seed, temperature, keep_seed`.
By default `keep_seed` is false: each request chooses a fresh random seed.
Enable Keep this seed to reuse the displayed seed with the same settings.
Outputs: native PNG, enlarged PNG, generation metadata, actual seed used.
React can call it using `@gradio/client` with your Space ID.
Keep private Space credentials on a server, never in browser code.
