import datetime
import pathlib
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import os

batch_size = 4
img_height = 256
img_width = 256

OUTPUT_CHANNELS = 3

def generate_files_path_multi(pairs, pattern='*.png', strict=True):
    """
    pairs: lista de tuplas (hm_dir, tx_dir), cada uma referente a uma data.
           ex.: [
               ('data-set/hm_to_texture/heightmap2021', 'data-set/hm_to_texture/texture2021'),
               ('data-set/hm_to_texture/heightmap2025', 'data-set/hm_to_texture/texture2025'),
           ]
    pattern: glob de arquivos a considerar (padrão: '*.png')
    strict:  se True, lança erro se algum diretório não existir.
             se False, ignora par inexistente e segue.
    """
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

def manage_data_set(hm_files,tx_files, with_holes = True, training = True):
    
    dataset_path = tf.data.Dataset.from_tensor_slices((hm_files,tx_files))
    
    dataset = dataset_path.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        dataset = dataset.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    
    if with_holes:
        dataset = dataset.map(make_inpanting_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return dataset
    
def make_inpanting_example(hm, tx):
    
    tx_hole, mask = generate_holes(tx)
    inp = tf.concat([hm,tx_hole,mask], axis=-1)
    tgt = tx
    
    return inp, tgt, mask        
    
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
    
    return tx_hole, mask

def denorm(x):  # [-1,1] -> [0,1] para plot
    return (x + 1.0) * 0.5   

pairs= [
    ("data-set/Hm_to_texture/2021/height_map","data-set/Hm_to_texture/2021/texture"),
    ("data-set/Hm_to_texture/2025/height_map","data-set/Hm_to_texture/2025/texture")
]

hm_files,tx_files = generate_files_path_multi(pairs)

dataset_holes = manage_data_set(hm_files,tx_files)

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
  inputs = tf.keras.layers.Input(shape=[256, 256, 5])

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

def weighted_l1(y_true, y_pred, mask, alpha=10.0, beta=1.0):
    w = alpha*mask + beta*(1.0 - mask)   # mask=1 no buraco
    return tf.reduce_mean(tf.abs(y_true - y_pred) * w)

def generator_loss(disc_generated_output, gen_output, target, mask, lambda_l1=100.0):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss  = weighted_l1(target, gen_output, mask, alpha=10.0, beta=1.0)
    total    = gan_loss + lambda_l1 * l1_loss
    return total, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 5], name='input_image')
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
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
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
  
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, mask, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target, mask)
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

def fit(train_ds,steps):
  start = time.time()
  
  example_inp,example_tgt,example_mask = next(iter(train_ds.take(1)))

  for step, (input_image, target, mask) in train_ds.repeat().take(steps).enumerate():
    
    if (step) % 1000 == 0:
      
      pred = generator(example_inp, training=False)
      
      with summary_writer.as_default():
        # Loga input (aqui só a textura furada do input, canais 1:4)
          tf.summary.image("input_tx_hole", (example_inp[..., 1:4] + 1) / 2, step=step)

        # Loga ground truth
          tf.summary.image("target_texture", (example_tgt + 1) / 2, step=step)

        # Loga predição
          tf.summary.image("pred_texture", (pred + 1) / 2, step=step)

        # Loga máscara (já está em [0,1])
          tf.summary.image("mask", example_mask, step=step)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      print(f"Step: {step//1000}k")

    train_step(input_image, target, mask, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)


    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
      
fit(dataset_holes,steps=2000)
      








