"""Evaluate two already-rendered image folders.

The first folder contains watermarked renders and the second folder contains
renders from the original pretrained TensoRF model. Images are paired by file
name; no NeRF model or dataset is loaded.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_wavelets import DWTForward
from tqdm.auto import tqdm

from models.attack import Attacker
from utils import rgb_lpips, rgb_ssim


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ATTACK_NAMES = ["Blur", "Rotate", "Crop", "Resize", "noise", "JPEG_Compression"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare watermarked render images against pretrained-model render images."
    )
    parser.add_argument("watermarked_dir", type=Path, help="Folder of watermarked renders")
    parser.add_argument("gt_dir", type=Path, help="Folder of pretrained-model GT renders")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <watermarked_dir>/metrics_pretrained_gt.json)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--decoder",
        type=Path,
        default=Path(__file__).resolve().parent / "data/pretrained_decoder/16_256_decoder_whit.pth",
        help="TorchScript watermark decoder",
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        default=None,
        help="Watermark key.txt; if omitted, search watermarked_dir and its parents",
    )
    parser.add_argument("--dwt-wave", default="bior4.4")
    parser.add_argument("--dwt-level", type=int, default=2)
    parser.add_argument("--dwt-mode", default="periodization")
    parser.add_argument("--skip-robustness", action="store_true")
    return parser.parse_args()


def image_files(folder):
    return {
        path.name: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def load_rgb(path):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def find_key_file(image_dir, explicit_key_file):
    if explicit_key_file is not None:
        return explicit_key_file
    for folder in [image_dir, *image_dir.parents]:
        candidate = folder / "key.txt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find key.txt in watermarked_dir or its parents. Pass --key-file explicitly."
    )


def bit_accuracy(decoded, key):
    matches = ~torch.logical_xor(decoded > 0, key > 0)
    return matches.sum(dim=-1).float().div(matches.shape[-1]).mean().item()


@torch.no_grad()
def extract_bit_accuracy(image, decoder, key, dwt, device):
    tensor = image.to(device).permute(2, 0, 1).unsqueeze(0)
    ll, _ = dwt(tensor)
    return bit_accuracy(decoder(ll), key)


@torch.no_grad()
def evaluate(watermarked_dir, gt_dir, device, decoder, key, dwt, run_robustness):
    watermarked = image_files(watermarked_dir)
    ground_truth = image_files(gt_dir)
    paired_names = sorted(set(watermarked) & set(ground_truth))

    if not paired_names:
        raise RuntimeError(
            f"No images with matching file names in {watermarked_dir} and {gt_dir}."
        )

    missing_gt = sorted(set(watermarked) - set(ground_truth))
    missing_watermarked = sorted(set(ground_truth) - set(watermarked))
    psnrs, ssims, lpips_alex, lpips_vgg = [], [], [], []
    bit_accuracies = []
    watermarked_images = []

    for name in tqdm(paired_names, desc="Evaluating image pairs"):
        wm = load_rgb(watermarked[name])
        gt = load_rgb(ground_truth[name])
        if wm.shape != gt.shape:
            raise ValueError(
                f"Image size mismatch for {name}: watermarked={tuple(wm.shape)}, "
                f"GT={tuple(gt.shape)}"
            )

        mse = torch.mean((wm - gt) ** 2).item()
        psnrs.append(float("inf") if mse == 0 else -10.0 * np.log10(mse))
        ssims.append(float(rgb_ssim(wm, gt, 1)))
        lpips_alex.append(float(rgb_lpips(gt.numpy(), wm.numpy(), "alex", device)))
        lpips_vgg.append(float(rgb_lpips(gt.numpy(), wm.numpy(), "vgg", device)))
        bit_accuracies.append(extract_bit_accuracy(wm, decoder, key, dwt, device))
        watermarked_images.append(wm)

    metrics = {
        "watermarked_dir": str(watermarked_dir.resolve()),
        "pretrained_gt_dir": str(gt_dir.resolve()),
        "num_image_pairs": len(paired_names),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips_alex": float(np.mean(lpips_alex)),
        "lpips_vgg": float(np.mean(lpips_vgg)),
        "bit_accuracy": float(np.mean(bit_accuracies)),
        "missing_in_gt": missing_gt,
        "missing_in_watermarked": missing_watermarked,
    }

    if run_robustness:
        attacker = Attacker()
        attack_results = {}
        for attack_index, attack_name in enumerate(ATTACK_NAMES):
            accuracies = []
            for image in tqdm(watermarked_images, desc=f"Attack: {attack_name}", leave=False):
                pil_image = Image.fromarray((image.numpy() * 255).round().astype(np.uint8))
                attacked_image = attacker(pil_image, attack_index)
                if isinstance(attacked_image, np.ndarray):
                    attacked = attacked_image.copy()
                else:
                    attacked = np.asarray(attacked_image.convert("RGB")).copy()
                if attacked.ndim == 2:
                    attacked = np.repeat(attacked[..., None], 3, axis=-1)
                if attacked.shape[-1] == 4:
                    attacked = attacked[..., :3]
                attacked_tensor = torch.from_numpy(attacked).float() / 255.0
                accuracies.append(
                    extract_bit_accuracy(attacked_tensor, decoder, key, dwt, device)
                )
            attack_results[attack_name] = float(np.mean(accuracies))
        metrics["attack_bit_accuracy"] = attack_results
        metrics["attack_bit_accuracy_mean"] = float(np.mean(list(attack_results.values())))

    return metrics


if __name__ == "__main__":
    args = parse_args()
    if not args.watermarked_dir.is_dir():
        raise NotADirectoryError(args.watermarked_dir)
    if not args.gt_dir.is_dir():
        raise NotADirectoryError(args.gt_dir)

    device = torch.device(args.device)
    key_file = find_key_file(args.watermarked_dir.resolve(), args.key_file)
    key_string = key_file.read_text().strip()
    key = torch.tensor([[int(bit) for bit in key_string]], dtype=torch.float32, device=device)
    decoder = torch.jit.load(str(args.decoder), map_location=device).to(device).eval()
    dwt = DWTForward(
        wave=args.dwt_wave, J=args.dwt_level, mode=args.dwt_mode
    ).to(device)

    metrics = evaluate(
        args.watermarked_dir,
        args.gt_dir,
        device,
        decoder,
        key,
        dwt,
        not args.skip_robustness,
    )
    metrics["decoder"] = str(args.decoder.resolve())
    metrics["key_file"] = str(key_file.resolve())
    output_path = args.output or args.watermarked_dir / "metrics_pretrained_gt.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(metrics, file, indent=2)

    mean_path = args.watermarked_dir / "mean.txt"
    np.savetxt(
        mean_path,
        np.asarray([
            metrics["psnr"],
            metrics["ssim"],
            metrics["lpips_alex"],
            metrics["lpips_vgg"],
            metrics["bit_accuracy"],
        ]),
        header="PSNR SSIM LPIPS_ALEX LPIPS_VGG BIT_ACCURACY",
    )

    if "attack_bit_accuracy" in metrics:
        attack_path = args.watermarked_dir / "attack_bit_acc_mean.txt"
        with attack_path.open("w") as file:
            for name, accuracy in metrics["attack_bit_accuracy"].items():
                file.write(str({"Attack_Type": name, "bit_acc": accuracy}) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"Saved: {output_path}")
    print(f"Saved: {mean_path}")
