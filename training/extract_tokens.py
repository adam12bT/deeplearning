"""Extract schema-v2 visual tokens while preserving all five feature labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.compositional_dataset import CompositionalDataset
from tokenizer.checkpoints import load_tokenizer


@torch.inference_mode()
def extract_split(model, dataset, device, split_name, batch_size=8, num_workers=0):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    tokens, conditions = [], []
    model.eval()
    seen = 0
    for images, labels in loader:
        grid = model.encode_to_indices(images.to(device))
        tokens.append(grid.flatten(1).cpu())
        conditions.append(labels)
        seen += len(images)
        if len(tokens) % 100 == 0:
            print(f'{split_name}: {seen}/{len(dataset)}', flush=True)
    return {
        'tokens': torch.cat(tokens),
        'conditions': torch.cat(conditions),
        'condition_fields': dataset.condition_fields,
        'filenames': [entry['filename'] for entry in dataset.entries],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='dataset/manifest.json')
    parser.add_argument('--search-root', default='dataset')
    parser.add_argument('--checkpoint', default='checkpoints/vqvae_v2/vqvae_best.pt')
    parser.add_argument('--output-dir', default='checkpoints/tokens_v2')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('Invalid batch size or worker count')
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose a new --output-dir')
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_tokenizer(Path(args.checkpoint), device)
    with torch.inference_mode():
        probe = model.encode_to_indices(torch.zeros(1, model.cfg.in_channels, 64, 64, device=device))
    grid = list(probe.shape[1:])
    identity = {
        'checkpoint': args.checkpoint,
        'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        'dataset_root': args.search_root,
        'grid': grid,
        'tokenizer_type': getattr(model, 'tokenizer_type', 'vqvae'),
    }
    output.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        dataset = CompositionalDataset(args.manifest, args.search_root, split)
        result = extract_split(model, dataset, device, split, args.batch_size, args.num_workers)
        assert result['tokens'].shape[1] == grid[0] * grid[1]
        assert result['tokens'].min() >= 0
        assert result['tokens'].max() < model.cfg.num_embeddings
        torch.save(result, output / f'{split}_tokens.pt')
        print(f'{split}: {len(dataset)} images saved', flush=True)
    if hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() != identity['checkpoint_sha256']:
        raise RuntimeError('Tokenizer changed during extraction; discard this output')
    (output / 'condition_names.json').write_text(json.dumps({
        'condition_fields': dataset.condition_fields,
        'condition_names': dataset.condition_names,
    }, indent=2))
    (output / 'vocab_size.json').write_text(json.dumps({'vocab_size': model.cfg.num_embeddings}, indent=2))
    (output / 'tokenizer.json').write_text(json.dumps(identity, indent=2))
    print(f'Tokens ready in {output}', flush=True)


if __name__ == '__main__':
    main()
