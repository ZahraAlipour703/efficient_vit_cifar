"""
Simple interactive CLI demo for the trained Efficient ViT.
Shows top-3 predictions for random or selected test images.
"""

import os
import argparse
import torch
from torchvision import datasets, transforms
import numpy as np

from model import EfficientViT

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_model(checkpoint_path, device="cpu"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = EfficientViT(
        embed_dim=args.get("embed_dim", 192),
        depth=args.get("depth", 6),
        num_heads=args.get("num_heads", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model | Epoch: {ckpt.get('epoch')} | Best Acc: {ckpt.get('acc', 0):.2f}%")
    return model


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


@torch.no_grad()
def predict_top3(model, image_tensor, device="cpu"):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top3_prob, top3_idx = torch.topk(probs, 3)
    return [(CIFAR10_CLASSES[i], p.item()) for i, p in zip(top3_idx, top3_prob)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--data-dir", default="/tmp/cifar")
    parser.add_argument("--num", type=int, default=10, help="How many random samples to show")
    args = parser.parse_args()

    device = "cpu"
    try:
        model = load_model(args.checkpoint, device)
    except FileNotFoundError:
        print("No trained model found. Using the small demo_model.pth instead...")
        model = load_model("outputs/demo_model.pth", device)

    transform = get_transform()
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=None)

    indices = np.random.choice(len(test_set), args.num, replace=False)

    print("\n" + "="*60)
    print("Efficient ViT – Interactive Demo (Top-3 predictions)")
    print("="*60)

    correct = 0
    for i, idx in enumerate(indices, 1):
        img, true_label = test_set[idx]
        preds = predict_top3(model, transform(img), device)
        true_name = CIFAR10_CLASSES[true_label]
        top1_name, top1_conf = preds[0]

        is_correct = top1_name == true_name
        if is_correct:
            correct += 1
        mark = "✓" if is_correct else "✗"

        print(f"\n[{i}/{args.num}]  True: {true_name:12s}  {mark}")
        for rank, (name, conf) in enumerate(preds, 1):
            bar = "█" * int(conf * 20)
            print(f"   {rank}. {name:12s} {conf*100:5.1f}%  {bar}")

    print("\n" + "="*60)
    print(f"Top-1 Accuracy on these {args.num} samples: {100*correct/args.num:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()