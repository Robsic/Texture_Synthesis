import pathlib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

batch_size = 32
img_height = 256
img_width = 256

def generate_files_path():
    
    hm_path = pathlib.Path('data-set/Hm_to_texture/height_map')
    tx_path = pathlib.Path('data-set/Hm_to_texture/texture')

    hm_files = {p.name: p for p in hm_path.glob('*.png')}
    tx_files = {p.name: p for p in tx_path.glob('*.png')}

    common_names = sorted(set(hm_files) & set(tx_files))

    hm_list = [str(hm_files[name]) for name in common_names]
    tx_list = [str(tx_files[name]) for name in common_names]

    return hm_list, tx_list

def manage_data_set(hm_files,tx_files, with_holes = True, training = True):
    
    dataset_path = tf.data.Dataset.from_tensor_slices((hm_files,tx_files))
    
    dataset = dataset_path.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        dataset = dataset.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    
    if with_holes:
        dataset = dataset.map(lambda hm, tx: (hm, generate_holes(tx)), num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return dataset
    

def load_pair(hm_path, tx_path):
    hm = tf.io.read_file(hm_path)
    hm = tf.image.decode_png(hm, channels=1)
    hm = tf.cast(hm, tf.float32) / 127.5 - 1.0   # [-1, 1]
    hm = tf.image.resize(hm, [img_height, img_width])
    
    tx = tf.io.read_file(tx_path)
    tx = tf.image.decode_png(tx, channels=3)
    tx = tf.cast(tx, tf.float32) / 127.5 - 1.0   # [-1, 1]
    tx = tf.image.resize(tx, [img_height,img_width])
    
    return hm,tx

def augment_pair(hm,tx):
    #flip horiontal
    if tf.random.uniform(()) < 0.5:
        hm = tf.image.flip_left_right(hm)
        tx = tf.image.flip_left_right(tx)
        
    if tf.random.uniform(()) < 0.5:
        hm = tf.image.flip_up_down(hm)
        tx = tf.image.flip_up_down(tx)
        
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    hm = tf.image.rot90(hm,k)
    tx = tf.image.rot90(tx, k)
    
    return hm,tx  

def generate_holes(tx, min_size = 32, max_size = 96, fill_value = 1.0):
      # tx em [-1,1], shape (H, W, 3)
    h = tf.shape(tx)[0]
    w = tf.shape(tx)[1]

    hole_h = tf.random.uniform([], min_size, max_size, dtype=tf.int32)
    hole_w = tf.random.uniform([], min_size, max_size, dtype=tf.int32)
    top    = tf.random.uniform([], 0, h - hole_h, dtype=tf.int32)
    left   = tf.random.uniform([], 0, w - hole_w, dtype=tf.int32)

    yy = tf.range(h)[:, None]
    xx = tf.range(w)[None, :]
    in_y = (yy >= top) & (yy < top + hole_h)
    in_x = (xx >= left) & (xx < left + hole_w)

    # mask2d: 1 fora do buraco, 0 dentro
    mask2d = tf.cast(~(in_y & in_x), tf.float32)   # (H, W)
    mask   = mask2d[..., None]                     # (H, W, 1) → broadcast para 3 canais

    # aplica buraco somente na textura
    tx_hole = tx * mask + fill_value * (1.0 - mask)  # fill=1.0 → branco em [-1,1]
    
    return tx_hole

def denorm(x):  # [-1,1] -> [0,1] para plot
    return (x + 1.0) * 0.5   
            
    

hm_files,tx_files = generate_files_path()

dataset_holes = manage_data_set(hm_files,tx_files)

dataset = manage_data_set(hm_files, tx_files, with_holes=False)

(hms_hole, txs_hole) = next(iter(dataset_holes.take(1)))
(hms_orig, txs_orig) = next(iter(dataset.take(1)))

plt.figure(figsize=(12, 8))
for i in range(3):
    # HM (mesmo nos dois)
    plt.subplot(3, 3, 3*i + 1)
    plt.imshow(tf.squeeze(denorm(hms_hole[i]), -1), cmap='gray')
    plt.title("Height Map")
    plt.axis('off')

    # Textura com buraco (pipeline com holes)
    plt.subplot(3, 3, 3*i + 2)
    plt.imshow(denorm(txs_hole[i]))
    plt.title("Textura COM buraco")
    plt.axis('off')

    # Textura original (pipeline sem holes)
    plt.subplot(3, 3, 3*i + 3)
    plt.imshow(denorm(txs_orig[i]))
    plt.title("Textura original")
    plt.axis('off')

plt.tight_layout()
plt.show()

