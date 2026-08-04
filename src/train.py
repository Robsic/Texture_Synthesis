import datetime
import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    BATCH_SIZE, CHECKPOINT_DIR, IMG_HEIGHT, IMG_WIDTH, LOG_DIR,
    MODEL_SAVE_PATH, PAIRS, SEED, STEPS, VAL_RATIO,
)
from src.dataset import build_datasets
from src.model import Discriminator, Generator, discriminator_loss, generator_loss


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)


@tf.function
def val_step(input_image, target):
    gen_out = generator(input_image, training=False)
    disc_fake = discriminator([input_image, gen_out], training=False)
    disc_real = discriminator([input_image, target], training=False)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_fake, gen_out, target)
    disc_loss_ = discriminator_loss(disc_real, disc_fake)
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss_


def evaluate_val(val_ds, global_step, max_batches=None):
    g_tot = g_gan = g_l1 = d_loss = 0.0
    n = 0
    for b, (inp, tgt) in enumerate(val_ds):
        gt, gg, gl1, dl = val_step(inp, tgt)
        g_tot += float(gt); g_gan += float(gg); g_l1 += float(gl1); d_loss += float(dl)
        n += 1
        if max_batches is not None and (b + 1) >= max_batches:
            break
    if n == 0:
        return
    g_tot /= n; g_gan /= n; g_l1 /= n; d_loss /= n

    with summary_writer.as_default():
        tf.summary.scalar('val/gen_total_loss', g_tot, step=global_step)
        tf.summary.scalar('val/gen_gan_loss', g_gan, step=global_step)
        tf.summary.scalar('val/gen_l1_loss', g_l1, step=global_step)
        tf.summary.scalar('val/disc_loss', d_loss, step=global_step)

    print(f"[VAL] step {global_step}: gen_total={g_tot:.4f} | gen_gan={g_gan:.4f} | gen_l1={g_l1:.4f} | disc={d_loss:.4f}")


def fit(train_ds, val_ds, steps):
    start = time.time()

    example_inp, example_tgt = next(iter(train_ds.take(1)))

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():

        if step % 1000 == 0:
            if step != 0:
                print(f' Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
            start = time.time()
            print(f"Step: {step//1000}k")

        train_step(input_image, target, step)

        if step % 2000 == 0:
            pred = generator(example_inp, training=False)

            with summary_writer.as_default():
                tf.summary.image("input_Hm", (example_inp + 1) / 2, step=step)
                tf.summary.image("target_texture", (example_tgt + 1) / 2, step=step)
                tf.summary.image("pred_texture", (pred + 1) / 2, step=step)

        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps and validate the model
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            evaluate_val(val_ds, global_step=int(step) + 1, max_batches=None)


if __name__ == "__main__":
    train_ds, val_ds = build_datasets(PAIRS, VAL_RATIO, SEED, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)

    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                      discriminator_optimizer=discriminator_optimizer,
                                      generator=generator,
                                      discriminator=discriminator)

    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest:
        checkpoint.restore(latest)
        print("Checkpoint carregado")
    else:
        print("Comecando do zero")

    summary_writer = tf.summary.create_file_writer(
        LOG_DIR + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # NOTA(davi): no Texture_generator.py original, fit() era definido mas nunca
    # chamado - o generator.save() rodava sobre o modelo recém-criado/restaurado,
    # sem treinar. Adicionei a chamada abaixo para o treino de fato acontecer;
    # STEPS em config.py foi inferido do nome da pasta de save ("...100k") e
    # precisa da sua confirmação.
    fit(train_ds, val_ds, STEPS)

    generator.save(MODEL_SAVE_PATH)
