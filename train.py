import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cocl import (COCLLoss, COCLModel, PairedImageFolder, compute_metrics,
                  format_metrics, image_folder, predict, set_seed)


def parse_args():
    p = argparse.ArgumentParser(description='Train COCL (Algorithm 1).')
    p.add_argument('--data_root', required=True,
                   help='Directory containing train/, train_aug/, val/, and test/.')
    p.add_argument('--output_dir', default='checkpoints')
    p.add_argument('--lambda_c', type=float, default=0.01, help='Contrastive loss weight λ.')
    p.add_argument('--temperature', type=float, default=0.1, help='Temperature τ.')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for images, labels in loader:
        _, logits = model(images.to(device))
        total += F.cross_entropy(logits, labels.to(device), reduction='sum').item()
        n += labels.size(0)
    return total / n


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'best_model.pth')

    root = args.data_root
    train_set = PairedImageFolder(os.path.join(root, 'train'), os.path.join(root, 'train_aug'))
    val_set = image_folder(os.path.join(root, 'val'))
    test_set = image_folder(os.path.join(root, 'test'))

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

    model = COCLModel(num_classes=len(train_set.classes)).to(device)
    contrastive = COCLLoss(args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, wait = float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {'loss': 0.0, 'ce': 0.0, 'cont': 0.0}
        for raw, aug, labels in train_loader:
            images = torch.cat([raw, aug]).to(device)
            labels = torch.cat([labels, labels]).to(device)

            z, logits = model(images)
            ce_loss = F.cross_entropy(logits, labels)
            cont_loss = contrastive(z, labels)
            loss = ce_loss + args.lambda_c * cont_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sums['loss'] += loss.item()
            sums['ce'] += ce_loss.item()
            sums['cont'] += cont_loss.item()

        steps = len(train_loader)
        val_loss = validation_loss(model, val_loader, device)
        print(f'Epoch {epoch:3d} | loss {sums["loss"] / steps:.4f} '
              f'(CE {sums["ce"] / steps:.4f}, cont {sums["cont"] / steps:.4f}) '
              f'| val CE {val_loss:.4f}')

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            torch.save({'model': model.state_dict(), 'classes': train_set.classes}, ckpt_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f'Early stopping at epoch {epoch}.')
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    labels, probs = predict(model, test_loader, device)
    print(f'Test | {format_metrics(compute_metrics(labels, probs))}')


if __name__ == '__main__':
    main()
