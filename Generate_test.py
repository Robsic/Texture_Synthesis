import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def hann2d(tile):
    """Janela 2D de Hann (0..1..0) para suavizar as bordas."""
    w = np.hanning(tile)        # 1D
    W = np.outer(w, w)          # 2D
    W /= W.max()                # normaliza para pico=1
    return W.astype(np.float32) # (tile, tile)

def pad_to_cover(H, W, tile, stride):
    """Calcula o padding necessário para cobrir toda a imagem com stride."""
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
    Aplica o gerador em uma imagem grande usando janelas deslizantes.
    
    generator: modelo treinado (entrada [tile,tile,1] -> saída [tile,tile,3])
    hm       : np.ndarray (H,W,1) em [-1,1] (height map já normalizado)
    return   : textura reconstruída (H,W,3) em [-1,1]
    """
    H, W = hm.shape[:2]

    # padding reflexivo para cobrir toda a imagem
    pad_h, pad_w = pad_to_cover(H, W, tile, stride)
    if pad_h > 0 or pad_w > 0:
        hm = np.pad(hm, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")

    Hp, Wp = hm.shape[:2]

    # janela de blend
    Wblend = hann2d(tile)[..., None]  # (tile,tile,1) -> broadcast nos canais

    # acumuladores
    out_sum = np.zeros((Hp, Wp, 3), np.float32)
    w_sum   = np.zeros((Hp, Wp, 3), np.float32)

    # índices de janelas
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
        preds = generator.predict(inp_np, verbose=0)  # (B,tile,tile,3)
        for (y, x), pred in zip(batch_pos, preds):
            out_sum[y:y+tile, x:x+tile, :] += pred * Wblend
            w_sum[y:y+tile, x:x+tile, :]   += Wblend
        batch_inp.clear(); batch_pos.clear()

    # percorre todos os patches
    for y in ys:
        for x in xs:
            patch = hm[y:y+tile, x:x+tile, :]  # (tile,tile,1)
            batch_inp.append(patch)
            batch_pos.append((y, x))
            if len(batch_inp) == batch_size:
                flush_batch()
    flush_batch()

    # normaliza pelo peso acumulado e remove padding
    eps = 1e-6
    out = out_sum / np.maximum(w_sum, eps)
    out = out[:H, :W, :]
    return out  # em [-1,1]

def load_image(path):
    hm = tf.io.read_file(path)
    
    hm = tf.image.decode_png(hm, channels=1)
    
    hm = tf.image.convert_image_dtype(hm,tf.float32)
    
    hm = hm * 2.0 - 1.0
    
    return hm.numpy()

def save_image(pred,path):
    img =(pred + 1.0) / 2.0
    
    img_uint8 = (img*255).astype(np.uint8)
    
    Image.fromarray(img_uint8).save(path)
    
generator = load_model("Modelo_Treinados/Big_dataset_100k")

height_map_2021 = load_image("images/heightmap/hm_2021.png")

pred_texture_2021 = sliding_window_predict(generator,height_map_2021,tile=256,stride=128)

save_image(pred_texture_2021,path="images/Resultados/Texture_dez_2025_151.png")




