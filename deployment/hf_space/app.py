"""One-image Gradio API, compatible with CPU, CUDA and Spaces ZeroGPU."""
import hashlib
import json
import math
import secrets
import os
from pathlib import Path
import threading
import time

import gradio as gr
import spaces
import torch
from PIL import Image

from transformer.compositional import CompositionalConfig, CompositionalTransformer
from pixel_patches import PixelPatchTokenizer

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
bundle = torch.load(ROOT / 'model.pt', map_location='cpu', weights_only=True)
raw = (ROOT / 'tokenizer.pt').read_bytes()
if hashlib.sha256(raw).hexdigest() != bundle['tokenizer']['checkpoint_sha256']:
    raise RuntimeError('Tokenizer does not match the model')
codec = torch.load(ROOT / 'tokenizer.pt', map_location='cpu', weights_only=True)
tokenizer = PixelPatchTokenizer(codec['codebook']).eval()
model = CompositionalTransformer(CompositionalConfig(**bundle['config'])).eval()
model.load_state_dict(bundle['model_state'])
fields = bundle['conditions_metadata']['condition_fields']
names = bundle['conditions_metadata']['condition_names']
lock = threading.Lock()


@spaces.GPU(duration=60)
def generate(sky, landform, ground, water, objects, seed, temperature, keep_seed=False):
    values = dict(zip(('sky', 'landform', 'ground', 'water', 'objects'),
                      (sky, landform, ground, water, objects)))
    if any(values[f] not in names[f] for f in fields):
        raise gr.Error('Choose one of the supported scene features.')
    if not keep_seed:
        seed = secrets.randbelow(2147483648)
    if not math.isfinite(float(seed)) or int(seed) != seed or not 0 <= seed <= 2147483647:
        raise gr.Error('Seed must be an integer between 0 and 2147483647.')
    if not math.isfinite(float(temperature)) or not .5 <= temperature <= 1.2:
        raise gr.Error('Temperature must be between 0.5 and 1.2.')
    with lock, torch.inference_mode():
        started = time.perf_counter()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device); tokenizer.to(device)
        torch.manual_seed(int(seed))
        conditions = torch.tensor([[names[f].index(values[f]) for f in fields]], device=device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            tokens = model.generate(conditions, temperature=float(temperature), top_k=50)
        h, w = bundle['tokenizer']['grid']
        pixels = ((tokenizer.decode_from_indices(tokens.reshape(1, h, w))[0] + 1) / 2).clamp(0, 1)
        array = (pixels.permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')
        image = Image.fromarray(array)
        info = dict(features=values, seed=int(seed), temperature=float(temperature),
                    epoch=bundle['epoch'], parameters=sum(p.numel() for p in model.parameters()),
                    checkpoint_sha256=bundle.get('source_checkpoint_sha256'),
                    seconds=round(time.perf_counter() - started, 2))
        return image, image.resize((512, 512), Image.Resampling.NEAREST), info, int(seed)


with gr.Blocks(title='Pixel Landscape Generator') as demo:
    gr.Markdown('# Pixel Landscape Generator\nChoose a scene and generate native 64×64 pixel art.')
    with gr.Row():
        controls = [gr.Dropdown(names[f], value={'sky':'night','landform':'mountains',
                    'ground':'rock','water':'lake','objects':'none'}[f], label=f.title()) for f in fields]
    with gr.Row():
        seed = gr.Number(value=281030, precision=0, label='Seed used (updated after generation)')
        keep_seed = gr.Checkbox(value=False, label='Keep this seed')
        temperature = gr.Slider(.5, 1.2, value=1., step=.05, label='Variation')
    button = gr.Button('Generate', variant='primary')
    preview = gr.Image(label='Enlarged preview', format='png', type='pil')
    native = gr.Image(label='Original 64×64 PNG', format='png', type='pil')
    details = gr.JSON(label='Generation details')
    button.click(generate, inputs=controls + [seed, temperature, keep_seed],
                 outputs=[native, preview, details, seed], api_name='generate', concurrency_limit=1)

demo.queue(max_size=10, default_concurrency_limit=1, api_open=False)
if __name__ == '__main__':
    if (ROOT/'backend').exists():
        # launch() performs ZeroGPU registration; mounting alone does not.
        from backend.app import create_app
        demo.launch(server_name='0.0.0.0',ssr_mode=False,prevent_thread_lock=True)
        demo.app.mount('/api',create_app())
        demo.app.router.routes.insert(0,demo.app.router.routes.pop())
        demo.block_thread()
    else:
        demo.launch(server_name='0.0.0.0', ssr_mode=False)
