import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import SRDDataset
from models import Model
from transforms.normalize import QU_BOUND


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on a cached patch with a trained checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the Lightning checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing val_subset.pt and test_subset.pt.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Cached split to sample from.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index within the split.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "logs" / "inference",
        help="Directory for the output .npz and .png files.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Prefix for output filenames (defaults to the checkpoint name).",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(checkpoint_path: Path, device: torch.device) -> Model:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint["hyper_parameters"]
    model = Model(
        config=hyper_parameters["config"],
        loss=None,
        metrics=None,
        extra_metrics=None,
        metrics_on_inputs=None,
        extra_metrics_on_inputs=None,
        total_steps=hyper_parameters.get("total_steps", 0),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_sample(
    data_dir: Path, split: str, index: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    split_path = data_dir / f"{split}_subset.pt"
    if not split_path.exists():
        raise FileNotFoundError(f"Cached split not found: {split_path}")
    dataset = torch.load(split_path, map_location="cpu", weights_only=False)
    if not isinstance(dataset, SRDDataset):
        raise TypeError(f"Expected SRDDataset, got {type(dataset).__name__}")
    if not 0 <= index < len(dataset):
        raise IndexError(f"Index {index} out of range for {split} split of size {len(dataset)}")
    data, resol = dataset[index]
    return data.unsqueeze(0).to(device), resol.reshape(1).to(device=device, dtype=torch.float32)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    return torch.clamp(image * QU_BOUND, -QU_BOUND, QU_BOUND)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    data, resol = load_sample(args.data_dir, args.split, args.index, device)

    with torch.no_grad():
        prediction = model(data, resol)[0].cpu()

    sample = data[0].cpu()
    q_lr = sample[3]
    u_lr = sample[4]
    q_hr = sample[5]
    u_hr = sample[6]
    q_pred = prediction[0]
    u_pred = prediction[1]

    q_pred_phys = denormalize(q_pred)
    u_pred_phys = denormalize(u_pred)
    q_hr_phys = denormalize(q_hr)
    u_hr_phys = denormalize(u_hr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.checkpoint.stem.replace(":", "-")
    stem = f"{tag}_{args.split}_{args.index:04d}"
    npz_path = args.out_dir / f"{stem}_prediction.npz"
    png_path = args.out_dir / f"{stem}_prediction.png"

    resolution = int(resol.item())
    np.savez_compressed(
        npz_path,
        resol=np.array(resolution),
        q_lr=q_lr.squeeze().numpy(),
        u_lr=u_lr.squeeze().numpy(),
        q_hr=q_hr.squeeze().numpy(),
        u_hr=u_hr.squeeze().numpy(),
        q_pred=q_pred.squeeze().numpy(),
        u_pred=u_pred.squeeze().numpy(),
        q_pred_phys=q_pred_phys.squeeze().numpy(),
        u_pred_phys=u_pred_phys.squeeze().numpy(),
        q_hr_phys=q_hr_phys.squeeze().numpy(),
        u_hr_phys=u_hr_phys.squeeze().numpy(),
    )

    panels = [
        (r"$Q_{LR}$", q_lr),
        (r"$Q_{HR}$", q_hr),
        (r"$Q_{pred}$", q_pred),
        (r"$U_{LR}$", u_lr),
        (r"$U_{HR}$", u_hr),
        (r"$U_{pred}$", u_pred),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, (title, image) in zip(axes.flat, panels):
        axis.imshow(image.squeeze().numpy(), cmap="twilight")
        axis.set_title(title, fontsize=16)
        axis.axis("off")
    figure.suptitle(
        f"{args.split}[{args.index}] target resolution {resolution} arcmin",
        fontsize=16,
    )
    figure.savefig(png_path, dpi=200)
    plt.close(figure)

    q_mae = (q_pred - q_hr).abs().mean().item()
    u_mae = (u_pred - u_hr).abs().mean().item()
    q_mae_phys = (q_pred_phys - q_hr_phys).abs().mean().item()
    u_mae_phys = (u_pred_phys - u_hr_phys).abs().mean().item()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sample: {args.split}[{args.index}], resolution {resolution} arcmin")
    print(f"Prediction shape: {tuple(prediction.shape)}")
    print(f"Normalized MAE: Q={q_mae:.6f}, U={u_mae:.6f}")
    print(f"Denormalized MAE: Q={q_mae_phys:.8f}, U={u_mae_phys:.8f}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
