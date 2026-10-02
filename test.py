import argparse

import torch
from torch.utils.data import DataLoader

from cocl import COCLModel, compute_metrics, format_metrics, image_folder, predict


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a trained COCL model on raw test images.')
    p.add_argument('--test_dir', required=True, help='Test images in <class>/<image> layout.')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device)
    test_set = image_folder(args.test_dir)
    if test_set.classes != ckpt['classes']:
        raise ValueError(f'Class mismatch: {test_set.classes} vs {ckpt["classes"]}')

    model = COCLModel(num_classes=len(ckpt['classes']), pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])

    loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
    labels, probs = predict(model, loader, device)
    print(f'Test | {format_metrics(compute_metrics(labels, probs))}')


if __name__ == '__main__':
    main()
