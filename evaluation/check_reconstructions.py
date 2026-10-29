"""Review source images against the active tokenizer on seen validation data."""
import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.compositional_dataset import CompositionalDataset
from tokenizer.checkpoints import load_vqvae
from training.reconstruction_losses import signed_gradient_loss


def diagnose_quantization(model, dataset, indices, device, output):
    """Bypass the codebook only for diagnosis, never for token generation."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    metrics = {name: {'pixel_mse_sum': 0., 'gradient_mae_sum': 0.} for name in ('quantized', 'continuous')}
    previews = {name: [] for name in ('source', 'quantized', 'continuous')}
    selected = set(indices)
    offset = 0
    with torch.inference_mode():
        for number, (images, _) in enumerate(loader, 1):
            images = images.to(device)
            latent = model.encoder(images)
            codes = model.vq(latent)['token_ids']
            decoded = {
                'quantized': model.decode_from_indices(codes),
                'continuous': model.decoder(latent),
            }
            for name, reconstruction in decoded.items():
                metrics[name]['pixel_mse_sum'] += ((reconstruction - images) / 2).square().mean((1, 2, 3)).sum().item()
                metrics[name]['gradient_mae_sum'] += signed_gradient_loss((reconstruction + 1) / 2, (images + 1) / 2).item() * len(images)
            for i in range(len(images)):
                if offset + i in selected:
                    previews['source'].append((offset + i, images[i].cpu()))
                    for name in decoded:
                        previews[name].append((offset + i, decoded[name][i].cpu()))
            offset += len(images)
            if number % 100 == 0:
                print(f'Quantization diagnostic: {offset}/{len(dataset)}', flush=True)
    report = {
        'validation_images': offset,
        'heldout_used': False,
        'continuous_is_diagnostic_only': True,
        'caveat': 'Decoder was trained on quantized inputs. Continuous inputs are a distribution change, not an unbiased upper bound or a usable generation mode.',
        'metrics': {name: {'pixel_mse_0_1': values['pixel_mse_sum'] / offset,
                           'gradient_mae': values['gradient_mae_sum'] / offset} for name, values in metrics.items()},
    }
    positions = {index: i for i, index in enumerate(indices)}
    sheet = Image.new('RGB', (6 * 132, 6 * 154), (25, 25, 30))
    draw = ImageDraw.Draw(sheet)
    for name in previews:
        previews[name] = dict(previews[name])
    for index in indices:
        i = positions[index]
        row, x = i // 2, (i % 2) * 396
        entry = dataset.entries[index]
        draw.text((x + 3, row * 154 + 3), f'{entry["sky"]}/{entry["landform"]}')
        draw.text((x + 3, row * 154 + 14), 'SOURCE | QUANTIZED | CONTINUOUS (diagnostic)')
        for column, name in enumerate(('source', 'quantized', 'continuous')):
            tensor = previews[name][index]
            pixels = ((tensor.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
            image = Image.fromarray(pixels).resize((128, 128), Image.Resampling.NEAREST)
            sheet.paste(image, (x + column * 132, row * 154 + 26))
    sheet.save(output / 'quantization_comparison.png')
    (output / 'quantization_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/vqvae_v2/vqvae_best.pt')
    parser.add_argument('--out-dir', default='runs/reconstruction_review')
    parser.add_argument('--diagnose-quantization', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    dataset = CompositionalDataset(split='val')
    generator = torch.Generator().manual_seed(51904)
    indices = []
    for sky in dataset.condition_names['sky']:
        for landform in dataset.condition_names['landform']:
            matches = [i for i, e in enumerate(dataset.entries) if e['sky'] == sky and e['landform'] == landform]
            indices.append(matches[torch.randint(len(matches), (1,), generator=generator).item()])
    originals = torch.stack([dataset[i][0] for i in indices])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_vqvae(Path(args.checkpoint), device)
    if args.diagnose_quantization:
        output = Path(args.out_dir)
        output.mkdir(parents=True, exist_ok=True)
        diagnose_quantization(model, dataset, indices, device, output)
        return
    reconstructed = []
    with torch.inference_mode():
        for batch in originals.split(4):
            codes = model.encode_to_indices(batch.to(device))
            reconstructed.append(model.decode_from_indices(codes).cpu())
    reconstructed = torch.cat(reconstructed)
    mse = ((reconstructed - originals) / 2).square().mean((1, 2, 3))
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new('RGB', (4 * 132, 6 * 154), (25, 25, 30))
    draw = ImageDraw.Draw(sheet)
    for i, index in enumerate(indices):
        row, column = i // 2, (i % 2) * 264
        entry = dataset.entries[index]
        draw.text((column + 3, row * 154 + 3), f'{entry["sky"]}/{entry["landform"]}: source | recon')
        for side, tensor in enumerate((originals[i], reconstructed[i])):
            pixels = ((tensor.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
            image = Image.fromarray(pixels).resize((128, 128), Image.Resampling.NEAREST)
            sheet.paste(image, (column + side * 132, row * 154 + 24))
    sheet.save(output / 'comparison.png')
    report = {
        'checkpoint': args.checkpoint,
        'validation_samples': len(indices),
        'heldout_used': False,
        'pixel_mse_0_1': mse.mean().item(),
        'mean_psnr_db': (-10 * mse.clamp_min(1e-12).log10()).mean().item(),
        'gradient_mae': signed_gradient_loss((reconstructed + 1) / 2, (originals + 1) / 2).item(),
        'filenames': [dataset.entries[i]['filename'] for i in indices],
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(f'Reconstruction review saved in {output}')


if __name__ == '__main__':
    main()
