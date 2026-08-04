# Pix2Pix — Texture Generation from Height Maps

![Example output: input height map, ground truth texture, generated texture](docs/example_output.png)

### What is this repository for? ###

This repository contains an image-to-image translation model based on the **Pix2Pix (conditional GAN)** architecture, trained to synthesise RGB terrain textures from grayscale height maps. It includes the training script, the loss/metric logging setup, checkpoint management, and a small example dataset so the code can be run end-to-end without downloading the full dataset.

*** Please cite the software as: ***

<AUTHORS>, <TITLE OF THE PAPER>, Computers & Geosciences, <YEAR>, <DOI>

---

## Table of Contents

1. [Requirements and Installation](#requirements-and-installation)
2. [Repository Contents](#repository-contents)
3. [Quick Test](#quick-test)
4. [Directory Structure](#directory-structure)
5. [Dataset Generation](#dataset-generation)
6. [Configuring the Dataset](#configuring-the-dataset)
7. [Tuning Hyperparameters](#tuning-hyperparameters)
8. [Starting the Training](#starting-the-training)
9. [Monitoring the Training](#monitoring-the-training)
10. [Checkpoints and Resuming Training](#checkpoints-and-resuming-training)
11. [Saving the Final Model](#saving-the-final-model)
12. [Inference with a Trained Model](#inference-with-a-trained-model)
13. [Who do I talk to?](#who-do-i-talk-to)
14. [License](#license)

---

## Requirements and Installation

The code was developed and tested with:

```
python >= 3.9
tensorflow >= 2.10
matplotlib
numpy
pillow
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Training on CPU is possible but slow. A CUDA-capable GPU is recommended; the results reported in the paper were obtained on `<GPU MODEL>` with `<X>` GB of memory.

---

## Repository Contents

| File / directory | Description |
|---|---|
| `train_pix2pix.py` | Main training script (data loading, model definition, training loop, checkpointing) |
| `inference.py` | Loads a saved generator and produces a texture from a height map |
| `requirements.txt` | Python dependencies |
| `example/` | Small example dataset used by the quick test (see below) |
| `Fill_Texture/data-set/` | Location where the full training dataset is expected |
| `Pix2Pix_Classic/Big_dataset/` | Checkpoints and TensorBoard logs written during training |
| `Modelo_Treinados/` | Final saved models (SavedModel format) |

> Adjust the file names above to match the actual scripts in the repository.

---

## Quick Test

A small example dataset is provided in `example/` so that the installation can be verified in a few minutes, without the full dataset:

```
example/
├── heightmap/     ← <N> grayscale .png images (1 channel)
└── texture/       ← <N> RGB .png images (3 channels), same file names
```

### 1. Test the training loop

Run a short training run on the example data:

```bash
python train_pix2pix.py --data example --steps 200 --batch_size 4
```

Expected behaviour: the console prints a progress dot every 10 steps and the elapsed time, a `checkpoints/` directory is created, and sample images appear under `logs/`. The run should finish in a few minutes on a laptop GPU. The output at 200 steps is not meant to look good — the test only verifies that the pipeline runs.

### 2. Test inference with the provided pre-trained model

A pre-trained generator is included in `Modelo_Treinados/<MODEL NAME>/`:

```bash
python inference.py --model Modelo_Treinados/<MODEL NAME> \
                    --input example/heightmap/<FILE>.png \
                    --output prediction.png
```

This writes `prediction.png`, the RGB texture synthesised from the input height map.

---

## Directory Structure

Each pair of directories must contain images with **identical file names** — pairing is done by file name.

```
project/
│
├── Fill_Texture/
│   └── data-set/
│       ├── 2025/
│       │   ├── heightmap/        ← grayscale .png images (1 channel)
│       │   └── texture/          ← colour .png images (3 channels)
│       │
│       └── 2021/
│           ├── heightmap/
│           ├── texture/
│           └── Cut_TOPO_ITA/
│               ├── 1/
│               │   ├── heightmap/
│               │   └── texture/
│               ├── 2/ ...
│               └── 3/ ...
│
├── Pix2Pix_Classic/
│   └── Big_dataset/
│       ├── checkpoints/          ← checkpoints saved automatically
│       └── logs/                 ← TensorBoard logs
│
└── Modelo_Treinados/
    └── Big_dataset_100k/         ← final saved model
```

> **Note:** height maps must be **1-channel** PNG images and textures **3-channel** PNG images. Any other format requires changing the `load_pair` functions.

---

## Dataset Generation

The height map / texture pairs used to train the model are produced with a separate tool:

https://github.com/DaviBossi/generate_data_set

---

## Configuring the Dataset

### Adding new data

To add new image pairs to the training set, append an entry to the `pairs` list with the path to the height map directory and to the corresponding texture directory:

```python
pairs = [
    # existing pairs
    ("Fill_Texture/data-set/2025/heightmap", "Fill_Texture/data-set/2025/texture"),
    ("Fill_Texture/data-set/2021/heightmap", "Fill_Texture/data-set/2021/texture"),

    # new pair — just add the line below
    ("Fill_Texture/data-set/new_batch/heightmap", "Fill_Texture/data-set/new_batch/texture"),
]
```

> The script matches file names across the two directories automatically. Files without a counterpart are skipped and reported in the console.

### Adjusting the validation split

The `val_ratio` parameter in `train_val_split` controls how much of the dataset is held out for validation (default: 10%):

```python
# Example: hold out 15% for validation
(train_hm, train_tx), (val_hm, val_tx) = train_val_split(
    hm_files, tx_files,
    val_ratio=0.15,   # ← change here
    seed=42
)
```

The `seed` keeps the split reproducible. Change it for a different shuffle.

---

## Tuning Hyperparameters

### Learning rate

The generator and the discriminator have separate learning rates:

```python
# Generator — default: 2e-4
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Discriminator — default: 1e-4
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
```

**Practical guidance:**

| Observed behaviour | Suggested adjustment |
|---|---|
| Discriminator too strong (generator stalled) | Lower the discriminator LR or raise the generator LR |
| Generator collapsing to identical images | Raise the generator LR slightly |
| Unstable training, strongly oscillating losses | Lower both LRs proportionally |

### L1 loss weight (LAMBDA)

`LAMBDA` controls how much the generator is penalised for pixel-wise differences from the ground truth. Default: `100`.

```python
LAMBDA = 100  # ← at the top of the script
```

- **Higher** values make the output closer to the ground truth, but the texture may become blurrier.
- **Lower** values give the generator more freedom, at the risk of artefacts.

### Batch size

```python
batch_size = 64  # ← at the top of the script
```

Reduce this if the GPU runs out of memory (OOM). Values such as `16` or `32` work well, with some impact on training stability.

---

## Starting the Training

Training is launched through the `fit` function, which takes the total number of **steps** (iterations over individual batches) and the training dataset:

```python
fit(train_ds, steps=100_000)
```

### Behaviour of `fit`

| Frequency | Action |
|---|---|
| Every **1,000 steps** | Prints the elapsed time and the current step to the console |
| Every **10 steps** | Prints a `.` as a progress indicator |
| Every **2,000 steps** | Writes sample images (input, ground truth, prediction) to TensorBoard |
| Every **5,000 steps** | Saves a checkpoint and evaluates the metrics on the validation set |

### Example

```python
# Train for 50,000 steps from the current state (or from scratch)
fit(train_ds, steps=50_000)
```

> **Automatic resume:** if checkpoints already exist in `Pix2Pix_Classic/Big_dataset/checkpoints/`, the script loads them before training starts. No manual change is needed to continue from where it stopped.

---

## Monitoring the Training

TensorBoard is fed automatically during training. To open it:

```bash
tensorboard --logdir Pix2Pix_Classic/Big_dataset/logs
```

Available metrics:

| Metric | Meaning |
|---|---|
| `gen_total_loss` | Total generator loss (GAN + L1) |
| `gen_gan_loss` | Adversarial component of the generator loss |
| `gen_l1_loss` | Pixel-wise component (fidelity to the ground truth) |
| `disc_loss` | Discriminator loss |
| `val/*` | Validation counterparts of the metrics above |

Besides the numerical metrics, the **Images** tab shows how the predictions evolve during training.

---

## Checkpoints and Resuming Training

Checkpoints are saved automatically every 5,000 steps in:

```
Pix2Pix_Classic/Big_dataset/checkpoints/
```

TensorFlow keeps the **5 most recent checkpoints** by default. To keep more, pass `max_to_keep` when creating the checkpoint manager:

```python
checkpoint = tf.train.Checkpoint(...)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)
```

To restore a specific checkpoint instead of the most recent one, replace the automatic restore:

```python
latest = tf.train.latest_checkpoint(checkpoint_dir)
```

with:

```python
checkpoint.restore("Pix2Pix_Classic/Big_dataset/checkpoints/ckpt-3")
```

---

## Saving the Final Model

At the end of the script, the generator can be saved in the TensorFlow **SavedModel** format:

```python
generator.save("Modelo_Treinados/Big_dataset_100k")
```

Change the path above to save it elsewhere or under a different name.

---

## Inference with a Trained Model

A saved model can be loaded for inference without recreating the architecture:

```python
import tensorflow as tf

generator = tf.keras.models.load_model("Modelo_Treinados/Big_dataset_100k")

# Inference
prediction = generator(heightmap_input, training=False)
```

> `training=False` disables Dropout during inference, which is required for deterministic results.

---

## Who do I talk to? ###

<AUTHOR NAME>, <INSTITUTION>
*** <name> <dot> <surname> <at> <domain> ***

---

## License

Copyright (C) <YEAR> <AUTHORS>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.