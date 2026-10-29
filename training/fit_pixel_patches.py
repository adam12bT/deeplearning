"""Fit real RGB patch prototypes on training images, then check validation fidelity."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.compositional_dataset import CompositionalDataset
from tokenizer.checkpoints import load_vqvae
from tokenizer.pixel_patches import PixelPatchTokenizer
from training.reconstruction_losses import signed_gradient_loss


@torch.inference_mode()
def assign(points, centers, chunk_size=2048):
    labels, errors = [], []
    norms = centers.square().sum(1)
    for chunk in points.split(chunk_size):
        distance = chunk.square().sum(1, keepdim=True) - 2 * chunk @ centers.T + norms
        error, label = distance.min(1)
        labels.append(label)
        errors.append(error.clamp_min(0))
    return torch.cat(labels), torch.cat(errors)


def metrics(prediction, target):
    p, t = (prediction + 1) / 2, (target + 1) / 2
    error = (p - t).square().mean((1, 2, 3))
    exact = ((p * 255).round().byte() == (t * 255).round().byte()).all(1).float().mean((1, 2))
    edge_sum = torch.zeros((), device=p.device)
    edge_count = torch.zeros_like(edge_sum)
    for dim in (-1, -2):
        dp, dt = torch.diff(p, dim=dim), torch.diff(t, dim=dim)
        mask = dt.abs() > .05
        edge_sum += (dp - dt).abs()[mask].sum()
        edge_count += mask.sum()
    return {
        'pixel_mse_sum': error.sum().item(),
        'psnr_sum': (-10 * error.clamp_min(1e-12).log10()).sum().item(),
        'gradient_sum': signed_gradient_loss(p, t).item() * len(p),
        'exact_pixels_sum': exact.sum().item(),
        'edge_error_sum': edge_sum.item(),
        'edge_count': edge_count.item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vocab-size', type=int, default=2048)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--output', default='checkpoints/pixel_patches_candidate/tokenizer.pt')
    args = parser.parse_args()
    checkpoint_path = Path(args.output)
    if checkpoint_path.exists():
        parser.error('Candidate already exists; choose a new output path')
    torch.set_num_threads(4)
    torch.manual_seed(41873)
    generator = torch.Generator().manual_seed(41873)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, val = CompositionalDataset(split='train'), CompositionalDataset(split='val')
    groups = defaultdict(list)
    for i, entry in enumerate(train.entries):
        groups[tuple(entry[f] for f in train.condition_fields)].append(i)
    selected = []
    for ids in groups.values():
        selected.extend(ids[i] for i in torch.randperm(len(ids), generator=generator)[:6].tolist())
    schema = json.loads(Path('dataset/schema.json').read_text())
    assert not any(all(train.entries[i][k] == v for k, v in rule.items())
                   for i in selected for rule in schema['heldout_rules'])
    samples = []
    for number, index in enumerate(selected, 1):
        image, _ = train[index]
        patches = PixelPatchTokenizer.patches(image.unsqueeze(0)).reshape(-1, 12)
        patches = (patches * 255).round() / 255
        uniform = patches[torch.randperm(len(patches), generator=generator)[:32]]
        contrast = patches.reshape(-1, 3, 4)
        edge_ids = ((contrast.amax(2) - contrast.amin(2)).sum(1) > .035).nonzero().flatten()
        detail = patches[edge_ids[torch.randint(len(edge_ids), (32,), generator=generator)]] if len(edge_ids) else uniform
        samples.append(torch.cat([uniform, detail]))
        if number % 400 == 0:
            print(f'Sampled patches from {number}/{len(selected)} training images', flush=True)
    raw = torch.unique(torch.cat(samples), dim=0)
    points = raw.to(device)
    flat = (raw.reshape(-1, 3, 4).amax(2) == raw.reshape(-1, 3, 4).amin(2)).all(1)
    solids = raw[flat]
    size = min(args.vocab_size, len(raw))
    if len(solids) >= size:
        raise RuntimeError('Too many distinct solid colors for this vocabulary; increase --vocab-size')
    complex_patches = raw[~flat]
    count = size - len(solids)
    centers = torch.cat([solids, complex_patches[torch.randperm(len(complex_patches), generator=generator)[:count]]]).to(device)
    print(f'Fitting {size} prototypes from {len(raw)} unique patches; locking {len(solids)} exact solid colors', flush=True)
    with torch.inference_mode():
        for iteration in range(args.iterations):
            labels, errors = assign(points, centers)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, labels, points)
            counts = torch.bincount(labels, minlength=size)
            populated = counts > 0
            updated = sums / counts.clamp_min(1).unsqueeze(1)
            updated[~populated] = points[errors.topk(int((~populated).sum())).indices]
            updated[:len(solids)] = solids.to(device)
            centers = updated
            if (iteration + 1) % 5 == 0:
                print(f'Prototype fit {iteration + 1}/{args.iterations}; squared patch error {errors.mean().item():.6f}', flush=True)
        # Snap means back to actual training patches: reconstruction never
        # introduces colors/pixel patterns created by averaging centroids.
        nearest, _ = assign(centers, points, chunk_size=64)
        prototypes = torch.unique(raw[nearest.cpu()], dim=0)
        _, errors = assign(points, prototypes.to(device))
        keys = {bytes(row.tolist()) for row in (prototypes * 255).round().byte()}
        rows = list(prototypes)
        raw_bytes = (raw * 255).round().byte().numpy()
        for index in errors.argsort(descending=True).cpu().tolist():
            if len(rows) == size:
                break
            key = raw_bytes[index].tobytes()
            if key not in keys:
                keys.add(key)
                rows.append(raw[index])
    codebook = torch.stack(rows)
    assert len(codebook) == size
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_type': 'pixel_patch_v1', 'codebook': codebook,
        'config': {'num_embeddings': size, 'patch_size': 2, 'in_channels': 3},
        'fit': {'seed': 41873, 'training_images': len(selected), 'training_patches_unique': len(raw),
                'fixed_solid_colors': len(solids), 'iterations': args.iterations, 'heldout_used': False,
                'manifest_sha256': hashlib.sha256(Path('dataset/manifest.json').read_bytes()).hexdigest()},
    }, checkpoint_path)
    model = PixelPatchTokenizer(codebook).to(device).eval()
    baseline = load_vqvae(Path('checkpoints/vqvae_v2/vqvae_best.pt'), device)
    totals = {name: defaultdict(float) for name in ('vqvae', 'pixel_patches')}
    preview_names = json.loads(Path('runs/vqvae_v2/comparison.json').read_text())['visual_subset_filenames']
    previews = {}
    count_images = 0
    with torch.inference_mode():
        for number, (images, _) in enumerate(DataLoader(val, batch_size=8), 1):
            images = images.to(device)
            recon = {name: codec.decode_from_indices(codec.encode_to_indices(images))
                     for name, codec in [('vqvae', baseline), ('pixel_patches', model)]}
            for name, output in recon.items():
                for key, value in metrics(output, images).items():
                    totals[name][key] += value
            for i in range(len(images)):
                filename = val.entries[count_images + i]['filename']
                if filename in preview_names:
                    previews[filename] = [images[i].cpu(), recon['vqvae'][i].cpu(), recon['pixel_patches'][i].cpu()]
            count_images += len(images)
            if number % 100 == 0:
                print(f'Validation reconstruction {count_images}/{len(val)}', flush=True)
    report = {'validation_images': count_images, 'training_images_used_for_fit': len(selected),
              'vocab_size': size, 'token_grid': [32, 32], 'heldout_used': False,
              'transformer_retraining_required': True, 'generation_quality_evaluated': False,
              'active_pipeline_replaced': False, 'metrics': {}}
    for name, total in totals.items():
        report['metrics'][name] = {
            'pixel_mse_0_1': total['pixel_mse_sum'] / count_images,
            'mean_psnr_db': total['psnr_sum'] / count_images,
            'gradient_mae': total['gradient_sum'] / count_images,
            'source_edge_gradient_mae': total['edge_error_sum'] / max(1, total['edge_count']),
            'exact_rgb_pixels_fraction': total['exact_pixels_sum'] / count_images,
        }
    root = Path('runs/pixel_patch_review')
    root.mkdir(parents=True, exist_ok=True)
    sheet = Image.new('RGB', (6 * 132, 6 * 154), (25, 25, 30))
    draw = ImageDraw.Draw(sheet)
    for i, filename in enumerate(preview_names):
        row, x = i // 2, (i % 2) * 396
        draw.text((x + 3, row * 154 + 4), 'SOURCE | CURRENT VQ-VAE | PIXEL PATCHES')
        for col, tensor in enumerate(previews[filename]):
            pixels = ((tensor.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
            image = Image.fromarray(pixels).resize((128, 128), Image.Resampling.NEAREST)
            sheet.paste(image, (x + col * 132, row * 154 + 24))
    sheet.save(root / 'comparison.png')
    (root / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
