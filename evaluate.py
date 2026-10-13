"""Evaluation of the trained generator (MAE, PSNR, SSIM) on the validation set.

The repository training loop (src/train.py) logs generator/discriminator losses
only. This script reproduces the image-quality metrics reported in the
manuscript (MAE, PSNR and SSIM) so that the quantitative results can be
recomputed from the saved model.
"""

import argparse

import numpy as np
import tensorflow as tf

from src.config import (
    BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, MODEL_SAVE_PATH, PAIRS, SEED, VAL_RATIO,
)
from src.dataset import build_datasets


def to_unit(x):
    """Maps generator output from [-1, 1] back to [0, 1] for metric computation."""
    return (x + 1.0) / 2.0


def compute_metrics(gen, target):
    gen_u = to_unit(gen)
    tgt_u = to_unit(target)

    mae = tf.reduce_mean(tf.abs(gen_u - tgt_u))
    psnr = tf.reduce_mean(tf.image.psnr(gen_u, tgt_u, max_val=1.0))
    ssim = tf.reduce_mean(tf.image.ssim(gen_u, tgt_u, max_val=1.0))

    return mae, psnr, ssim


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained generator (MAE/PSNR/SSIM) on the validation set.")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit the number of validation batches to process.")
    args = parser.parse_args()

    generator = tf.keras.models.load_model(MODEL_SAVE_PATH)

    _, val_ds = build_datasets(PAIRS, VAL_RATIO, SEED, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)

    mae_list, psnr_list, ssim_list = [], [], []

    for batch, (hm, tx) in enumerate(val_ds):
        pred = generator(hm, training=False)
        mae, psnr, ssim = compute_metrics(pred, tx)
        mae_list.append(float(mae))
        psnr_list.append(float(psnr))
        ssim_list.append(float(ssim))

        if args.max_batches is not None and (batch + 1) >= args.max_batches:
            break

    if not mae_list:
        print("No validation samples found.")
        return

    print(f"Samples evaluated: {len(mae_list)} batches")
    print(f"MAE  : {np.mean(mae_list):.4f} +/- {np.std(mae_list):.4f}")
    print(f"PSNR : {np.mean(psnr_list):.2f} +/- {np.std(psnr_list):.2f} dB")
    print(f"SSIM : {np.mean(ssim_list):.4f} +/- {np.std(ssim_list):.4f}")


if __name__ == "__main__":
    main()
