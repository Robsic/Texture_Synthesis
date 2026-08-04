import pathlib

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

from src.config import MODEL_SAVE_PATH

HEIGHTMAP_PATH = "./testing_files/full_heightmap/hm_2021.png"
OUTPUT_PATH = "./results/prediction.png"

def hann2d(tile):
    """2D Hann window (0..1..0) used to smooth tile borders."""
    w = np.hanning(tile)
    W = np.outer(w, w)
    W /= W.max()
    return W.astype(np.float32)

def pad_to_cover(H, W, tile, stride):
    """Computes the padding needed to cover the whole image with the given stride."""
    def last_start(L):
        if L <= tile:
            return 0
        n_steps = int(np.ceil((L - tile) / stride))
        return n_steps * stride
    y0 = last_start(H)
    x0 = last_start(W)
    pad_h = max(0, y0 + tile - H)
    pad_w = max(0, x0 + tile - W)
    return pad_h, pad_w

def sliding_window_predict(generator, hm, tile=256, stride=128, batch_size=8):
    """
    Applies the generator to a large image using sliding windows.

    generator: trained model (input [tile,tile,1] -> output [tile,tile,3])
    hm       : np.ndarray (H,W,1) in [-1,1] (already normalized height map)
    return   : reconstructed texture (H,W,3) in [-1,1]
    """
    H, W = hm.shape[:2]

    pad_h, pad_w = pad_to_cover(H, W, tile, stride)
    if pad_h > 0 or pad_w > 0:
        hm = np.pad(hm, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")

    Hp, Wp = hm.shape[:2]

    Wblend = hann2d(tile)[..., None]

    out_sum = np.zeros((Hp, Wp, 3), np.float32)
    w_sum   = np.zeros((Hp, Wp, 3), np.float32)

    ys = list(range(0, max(1, Hp - tile + 1), stride))
    xs = list(range(0, max(1, Wp - tile + 1), stride))
    if ys[-1] != Hp - tile:
        ys.append(Hp - tile)
    if xs[-1] != Wp - tile:
        xs.append(Wp - tile)

    batch_inp, batch_pos = [], []

    def flush_batch():
        nonlocal batch_inp, batch_pos, out_sum, w_sum
        if not batch_inp:
            return
        inp_np = np.stack(batch_inp, axis=0)  # (B,tile,tile,1)
        # training=False (BatchNorm running averages), consistent with val_step.
        preds = generator.predict(inp_np, verbose=0)  # (B,tile,tile,3)
        for (y, x), pred in zip(batch_pos, preds):
            out_sum[y:y+tile, x:x+tile, :] += pred * Wblend
            w_sum[y:y+tile, x:x+tile, :]   += Wblend
        batch_inp.clear(); batch_pos.clear()

    for y in ys:
        for x in xs:
            patch = hm[y:y+tile, x:x+tile, :]  # (tile,tile,1)
            batch_inp.append(patch)
            batch_pos.append((y, x))
            if len(batch_inp) == batch_size:
                flush_batch()
    flush_batch()

    eps = 1e-6
    out = out_sum / np.maximum(w_sum, eps)
    out = out[:H, :W, :]
    return out  # in [-1,1]

def load_image(path):
    hm = tf.io.read_file(path)
    
    hm = tf.image.decode_png(hm, channels=1)
    
    hm = tf.image.convert_image_dtype(hm,tf.float32)
    
    hm = hm * 2.0 - 1.0
    
    return hm.numpy()

def save_image(pred, path):
    img = (pred + 1.0) / 2.0
    # Hann-window blending should keep values within [-1, 1], but without
    # clipping any float overflow becomes silent garbage once cast to uint8.
    img = np.clip(img, 0.0, 1.0)

    img_uint8 = (img * 255).astype(np.uint8)

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_uint8).save(path)


if __name__ == "__main__":
    generator = load_model(MODEL_SAVE_PATH)

    height_map = load_image(HEIGHTMAP_PATH)

    pred_texture = sliding_window_predict(generator, height_map, tile=256, stride=128)

    save_image(pred_texture, path=OUTPUT_PATH)
