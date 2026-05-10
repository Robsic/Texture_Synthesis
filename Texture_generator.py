import datetime
import pathlib
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np


batch_size = 64
img_height = 256
img_width = 256

OUTPUT_CHANNELS = 3
LAMBDA = 100

def generate_files_path_multi(pairs, pattern='*.png', strict=True):
    hm_all, tx_all = [], []

    for hm_dir, tx_dir in pairs:
        hm_dir = pathlib.Path(hm_dir)
        tx_dir = pathlib.Path(tx_dir)

        if strict:
            if not hm_dir.exists():
                raise FileNotFoundError(f"Diretório não existe: {hm_dir}")
            if not tx_dir.exists():
                raise FileNotFoundError(f"Diretório não existe: {tx_dir}")
        else:
            if not hm_dir.exists() or not tx_dir.exists():
                print(f"[Aviso] Ignorando par inexistente: {hm_dir} | {tx_dir}")
                continue

        # indexa por nome de arquivo
        hm_files = {p.name: p for p in hm_dir.glob(pattern)}
        tx_files = {p.name: p for p in tx_dir.glob(pattern)}

        common = sorted(set(hm_files) & set(tx_files))
        if not common:
            print(f"[Aviso] Sem nomes em comum entre {hm_dir} e {tx_dir}")

        # estende as listas globais preservando o pareamento
        hm_all.extend(str(hm_files[n]) for n in common)
        tx_all.extend(str(tx_files[n]) for n in common)

    # checagem final: tamanhos devem casar
    if len(hm_all) != len(tx_all):
        raise ValueError(f"Listas desalinhadas: {len(hm_all)} HMs vs {len(tx_all)} TXs")

    return hm_all, tx_all

def train_val_split(hm_files, tx_files, val_ratio=0.10, seed=42):
    assert len(hm_files) == len(tx_files), "Listas desalinhadas!"

    n = len(hm_files)
    idx = np.arange(n)

    rng = np.random.RandomState(seed)   # split determinístico
    rng.shuffle(idx)

    n_val = int(np.round(n * val_ratio))
    val_idx  = idx[:n_val]
    train_idx = idx[n_val:]

    hm_train = [hm_files[i] for i in train_idx]
    tx_train = [tx_files[i] for i in train_idx]

    hm_val = [hm_files[i] for i in val_idx]
    tx_val = [tx_files[i] for i in val_idx]

    return (hm_train, tx_train), (hm_val, tx_val)
  
def make_dataset(hm_files, tx_files, training=True,
                 batch_size=32, img_height=256, img_width=256):
    ds = tf.data.Dataset.from_tensor_slices((hm_files, tx_files))
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
          
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

pairs= [
    ("Fill_Texture/data-set/2025/heightmap","Fill_Texture/data-set/2025/texture"),
    ("Fill_Texture/data-set/2021/heightmap","Fill_Texture/data-set/2021/texture"),
    ("Fill_Texture/data-set/2021/Cut_TOPO_ITA/1/heightmap","Fill_Texture/data-set/2021/Cut_TOPO_ITA/1/texture"),
    ("Fill_Texture/data-set/2021/Cut_TOPO_ITA/2/heightmap","Fill_Texture/data-set/2021/Cut_TOPO_ITA/2/texture"),
    ("Fill_Texture/data-set/2021/Cut_TOPO_ITA/3/heightmap","Fill_Texture/data-set/2021/Cut_TOPO_ITA/3/texture"),
]

hm_files,tx_files = generate_files_path_multi(pairs)

(train_hm, train_tx), (val_hm, val_tx) = train_val_split(hm_files, tx_files, val_ratio=0.10, seed=42)

train_ds = make_dataset(train_hm, train_tx, training=True,
                        batch_size=batch_size, img_height=img_height, img_width=img_width)

val_ds   = make_dataset(val_hm, val_tx, training=False,
                        batch_size=batch_size, img_height=img_height, img_width=img_width)

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

checkpoint_dir = 'Pix2Pix_Classic/Big_dataset/checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

latest = tf.train.latest_checkpoint(checkpoint_dir)

if latest:
  checkpoint.restore(latest)
  print("Checkpoint carregado")
else:
  print("Comeando do zero")

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  
log_dir="Pix2Pix_Classic/Big_dataset/logs"

summary_writer = tf.summary.create_file_writer(
  log_dir + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

@tf.function
def val_step(input_image, target):
    gen_out = generator(input_image, training=False)
    # se seu generator_loss atual usa máscara, passe uma máscara de 1s (sem peso especial)
    # ou crie uma variante de loss para o caso HM->TX (sem máscara).
    disc_fake = discriminator([input_image, gen_out], training=False)
    disc_real = discriminator([input_image, target],   training=False)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_fake, gen_out, target)  # adapte se precisar de mask
    disc_loss_ = discriminator_loss(disc_real, disc_fake)
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss_

def evaluate_val(val_ds, global_step, max_batches=None):
    # agrega métricas
    g_tot = g_gan = g_l1 = d_loss = 0.0
    n = 0
    for b, (inp, tgt) in enumerate(val_ds):
        gt, gg, gl1, dl = val_step(inp, tgt)
        g_tot += float(gt); g_gan += float(gg); g_l1 += float(gl1); d_loss += float(dl)
        n += 1
        if max_batches is not None and (b+1) >= max_batches:
            break
    if n == 0: return
    g_tot /= n; g_gan /= n; g_l1 /= n; d_loss /= n

    with summary_writer.as_default():
        tf.summary.scalar('val/gen_total_loss', g_tot, step=global_step)
        tf.summary.scalar('val/gen_gan_loss',   g_gan, step=global_step)
        tf.summary.scalar('val/gen_l1_loss',    g_l1,  step=global_step)
        tf.summary.scalar('val/disc_loss',      d_loss,step=global_step)

    print(f"[VAL] step {global_step}: gen_total={g_tot:.4f} | gen_gan={g_gan:.4f} | gen_l1={g_l1:.4f} | disc={d_loss:.4f}")

def fit(train_ds,steps):
  start = time.time()
  
  example_inp,example_tgt = next(iter(train_ds.take(1)))

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    
    if (step) % 1000 == 0:

      if step != 0:
        print(f' Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)
    
    if (step) % 2000 == 0:
      pred = generator(example_inp, training=False)
      
      with summary_writer.as_default():
        # Loga Height Map
          tf.summary.image("input_Hm", (example_inp + 1) / 2, step=step)

        # Loga ground truth
          tf.summary.image("target_texture", (example_tgt + 1) / 2, step=step)

        # Loga predição
          tf.summary.image("pred_texture", (pred + 1) / 2, step=step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)

    # Save (checkpoint) the model every 5k steps and validate the model
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
      
      evaluate_val(val_ds,global_step=int(step)+1,max_batches=None)

generator.save("Modelo_Treinados/Big_dataset_100k")







