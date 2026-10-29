# Gradio deployment

Space: https://huggingface.co/spaces/adamnouassida5/deeplearning

`python deployment/prepare_space.py` exports the current best pixel-patch model
and exact tokenizer into `deployment/hf_space/`. Upload that folder's contents
as the Space root. The export strips optimizer state and training RNG state.
It does not change training checkpoints or the existing generation defaults.

The current deployment uses the larger 12,621,696-parameter model's best
epoch-20 checkpoint, selected by the user after visual comparison. The previous
smaller epoch-38 model remains available locally and in Space commit history.
Export the smaller model for rollback using
`python deployment/prepare_space.py --checkpoint checkpoints/model_pixel_patches/best.pt`.
Initial smaller-model local inference smoke test: valid 64×64 PNG, 512×512
nearest-neighbor preview, invalid scene rejected; GPU generation took 7.73 s
on the development laptop. Hosted latency may differ, and ZeroGPU queues apply.

## React client later

Install `@gradio/client`, then call the queued endpoint:

```javascript
import { Client } from '@gradio/client';

const client = await Client.connect('adamnouassida5/deeplearning');
const result = await client.predict('/generate', {
  sky: 'night', landform: 'mountains', ground: 'rock',
  water: 'lake', objects: 'none', seed: 281030, temperature: 1.0,
  keep_seed: false,
});
// result.data[0]: native PNG file data; result.data[1]: enlarged preview;
// result.data[2]: generation metadata; result.data[3]: actual seed used.
// File data includes a URL. Set keep_seed: true to reproduce a seed.
```

For a private Space, authenticate from a backend; never embed HF tokens in React.
The Space queue permits one generation at a time, with at most ten queued requests.
