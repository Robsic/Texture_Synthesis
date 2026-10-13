import pathlib

import numpy as np
import tensorflow as tf


def generate_files_path_multi(pairs, pattern='*.png', strict=True):
    hm_all, tx_all = [], []

    for hm_dir, tx_dir in pairs:
        hm_dir = pathlib.Path(hm_dir)
        tx_dir = pathlib.Path(tx_dir)

        if strict:
            if not hm_dir.exists():
                raise FileNotFoundError(f"Directory does not exist: {hm_dir}")
            if not tx_dir.exists():
                raise FileNotFoundError(f"Directory does not exist: {tx_dir}")
        else:
            if not hm_dir.exists() or not tx_dir.exists():
                print(f"[Warning] Skipping missing pair: {hm_dir} | {tx_dir}")
                continue

        hm_files = {p.name: p for p in hm_dir.glob(pattern)}
        tx_files = {p.name: p for p in tx_dir.glob(pattern)}

        common = sorted(set(hm_files) & set(tx_files))
        if not common:
            print(f"[Warning] No matching file names between {hm_dir} and {tx_dir}")

        hm_all.extend(str(hm_files[n]) for n in common)
        tx_all.extend(str(tx_files[n]) for n in common)

    if len(hm_all) != len(tx_all):
        raise ValueError(f"Misaligned lists: {len(hm_all)} HMs vs {len(tx_all)} TXs")

    return hm_all, tx_all


def train_val_split(hm_files, tx_files, val_ratio=0.10, seed=42):
    """Randomly partitions the file lists into train/validation sets.

    Note: the split is performed at the patch (file) level with no spatial
    grouping. Patches cut from the same source orthomosaic can therefore fall
    into both subsets, which may introduce spatial leakage; group by source
    image if a fully independent validation set is required.
    """
    assert len(hm_files) == len(tx_files), "Misaligned lists!"

    n = len(hm_files)
    idx = np.arange(n)

    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    n_val = int(np.round(n * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    hm_train = [hm_files[i] for i in train_idx]
    tx_train = [tx_files[i] for i in train_idx]

    hm_val = [hm_files[i] for i in val_idx]
    tx_val = [tx_files[i] for i in val_idx]

    return (hm_train, tx_train), (hm_val, tx_val)


def load_pair(hm_path, tx_path, img_height, img_width):
    hm = tf.io.read_file(hm_path)
    hm = tf.image.decode_png(hm, channels=1)
    hm = tf.image.convert_image_dtype(hm, tf.float32) * 2.0 - 1.0   # [-1, 1]
    hm = tf.image.resize(hm, [img_height, img_width])

    tx = tf.io.read_file(tx_path)
    tx = tf.image.decode_png(tx, channels=3)
    tx = tf.image.convert_image_dtype(tx, tf.float32) * 2.0 - 1.0   # [-1, 1]
    tx = tf.image.resize(tx, [img_height, img_width])

    return hm, tx


def augment_pair(hm, tx):
    if tf.random.uniform(()) < 0.5:
        hm = tf.image.flip_left_right(hm)
        tx = tf.image.flip_left_right(tx)

    if tf.random.uniform(()) < 0.5:
        hm = tf.image.flip_up_down(hm)
        tx = tf.image.flip_up_down(tx)

    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    hm = tf.image.rot90(hm, k)
    tx = tf.image.rot90(tx, k)

    return hm, tx


def make_dataset(hm_files, tx_files, training=True,
                  batch_size=32, img_height=256, img_width=256):
    ds = tf.data.Dataset.from_tensor_slices((hm_files, tx_files))
    ds = ds.map(lambda hm, tx: load_pair(hm, tx, img_height, img_width),
                num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(pairs, val_ratio, seed, batch_size, img_height, img_width):
    """Chains the steps above: list files -> split train/val -> build the tf.data.Dataset objects."""
    hm_files, tx_files = generate_files_path_multi(pairs)

    (train_hm, train_tx), (val_hm, val_tx) = train_val_split(
        hm_files, tx_files, val_ratio=val_ratio, seed=seed)

    train_ds = make_dataset(train_hm, train_tx, training=True,
                             batch_size=batch_size, img_height=img_height, img_width=img_width)
    val_ds = make_dataset(val_hm, val_tx, training=False,
                           batch_size=batch_size, img_height=img_height, img_width=img_width)

    return train_ds, val_ds
