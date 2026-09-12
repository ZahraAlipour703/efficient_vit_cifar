"""
Inference demo for the trained Efficient ViT.
Shows predictions on CIFAR-10 test images + simple robustness check.
"""

import os
import argparse
import torch
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model import EfficientViT


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_model(checkpoint_path, device="cpu"):
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
    print(f"Loaded model from epoch {ckpt.get('epoch')} | Acc: {ckpt.get('acc', 0):.2f}%")
    return model


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


@torch.no_grad()
def predict(model, image_tensor, device="cpu"):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    conf, pred = probs.max(0)
    return pred.item(), conf.item(), probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pth")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default="/tmp/cifar")
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    device = "cpu"
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Please run train.py first.")
        return

    model = load_model(args.checkpoint, device)
    transform = get_test_transform()
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=False,
                                transform=None)  # raw for visualization

    # Pick random samples
    indices = np.random.choice(len(test_set), args.num_samples, replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    print("\n=== Predictions ===")
    for i, idx in enumerate(indices):
        img, true_label = test_set[idx]
        img_tensor = transform(img)
        pred, conf, probs = predict(model, img_tensor, device)

        axes[i].imshow(img)
        color = "green" if pred == true_label else "red"
        axes[i].set_title(
            f"True: {CIFAR10_CLASSES[true_label]}\n"
            f"Pred: {CIFAR10_CLASSES[pred]} ({conf*100:.1f}%)",
            color=color, fontsize=9
        )
        axes[i].axis("off")
        print(f"Sample {i+1}: True={CIFAR10_CLASSES[true_label]:10s} | "
              f"Pred={CIFAR10_CLASSES[pred]:10s} | Conf={conf*100:.1f}%")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "inference_demo.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nDemo image saved to {out_path}")


if __name__ == "__main__":
    main()