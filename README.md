# Pix2Pix — Texture Generation from Height Maps

This repository contains an image-to-image translation model based on the **Pix2Pix (conditional GAN)** architecture, trained to synthesise RGB terrain textures from grayscale height maps. It includes the training script, the loss/metric logging setup, checkpoint management, and a small example dataset so the code can be run end-to-end without downloading the full dataset.

*** Please cite the software as: ***

<AUTHORS>, <TITLE OF THE PAPER>, Computers & Geosciences, <YEAR>, <DOI>

---

## Table of Contents

1. [Requirements and Installation](#requirements-and-installation)
2. [Repository Contents](#repository-contents)
3. [Directory Structure](#directory-structure)
4. [Quick Test](#quick-test)
5. [Dataset Generation](#dataset-generation)
6. [Configuring the Dataset](#configuring-the-dataset)
7. [Tuning Hyperparameters](#tuning-hyperparameters)
8. [Starting the Training](#starting-the-training)
9. [Monitoring the Training](#monitoring-the-training)
10. [Checkpoints and Resuming Training](#checkpoints-and-resuming-training)
11. [Saving the Final Model](#saving-the-final-model)
12. [Inference with a Trained Model](#inference-with-a-trained-model)
13. [Pre-trained Model](#pre-trained-model)
14. [Who do I talk to?](#who-do-i-talk-to)
15. [License](#license)

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
| `src/config.py` | All hyperparameters and paths (dataset pairs, batch size, learning rate weight, checkpoint/log/model directories, training steps) |
| `src/dataset.py` | Pairs height-map/texture files by name, splits train/validation, builds the `tf.data` pipeline and augmentation |
| `src/model.py` | Generator (U-Net) and Discriminator (PatchGAN) architectures, and the GAN + L1 losses |
| `src/train.py` | Training entry point (data loading, training loop, checkpointing, TensorBoard logging) |
| `predict.py` | Loads a saved generator and runs sliding-window inference on a full-size height map |
| `requirements.txt` | Python dependencies |
| `testing_files/sample_images/` | 35 paired example tiles (`Hm/` height maps, `Tx/` textures, 256×256) used as the default training set |
| `testing_files/full_heightmap/` | Full-size height map used by `predict.py` |
| `testing_files/pre_trained_model/` | Pre-trained generator (SavedModel format) — not tracked in git, see [Pre-trained Model](#pre-trained-model) |
| `tensorboard/checkpoints/` | Training checkpoints (created automatically, gitignored) |
| `tensorboard/logs/` | TensorBoard logs (created automatically, gitignored) |
| `results/` | Output images written by `predict.py` (gitignored) |

---

## Directory Structure

Each pair of directories referenced in `PAIRS` (`src/config.py`) must contain images with **identical file names** — pairing is done by file name.

```
Fill_Texture/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
│
├── predict.py
│
├── testing_files/
│   ├── sample_images/
│   │   ├── Hm/                 ← grayscale .png height maps (1 channel)
│   │   └── Tx/                 ← colour .png textures (3 channels)
│   ├── full_heightmap/
│   │   └── hm_2021.png         ← used by predict.py
│   └── pre_trained_model/      ← downloaded from Zenodo, see below
│
├── tensorboard/
│   ├── checkpoints/            ← written automatically during training
│   └── logs/                   ← TensorBoard logs
│
└── results/                    ← predictions written by predict.py
```

> **Note:** height maps must be **1-channel** PNG images and textures **3-channel** PNG images. Any other format requires changing `load_pair` in `src/dataset.py`.

---

## Quick Test

`src/config.py` ships pointed at the 35 example pairs in `testing_files/sample_images/`, with `STEPS = 1000`, so the pipeline can be verified end-to-end in a few minutes without the full dataset:

```bash
python -m src.train
```

Expected behaviour: the console prints a progress dot every 10 steps and the elapsed time every 1,000 steps, checkpoints are written to `tensorboard/checkpoints/`, and sample images appear under `tensorboard/logs/`. The output at 1,000 steps is not meant to look good — the test only verifies that the pipeline runs.

To test inference with the pre-trained model (see [Pre-trained Model](#pre-trained-model) to obtain it):

```bash
python predict.py
```

This reads `testing_files/full_heightmap/hm_2021.png` and writes `results/prediction.png`.

---

## Dataset Generation

The height map / texture pairs used to train the model are produced with a separate tool:

https://github.com/DaviBossi/generate_data_set

---

## Configuring the Dataset

### Adding new data

To add new image pairs to the training set, append an entry to the `PAIRS` list in `src/config.py` with the path to the height map directory and to the corresponding texture directory:

```python
PAIRS = [
    ("./testing_files/sample_images/Hm", "./testing_files/sample_images/Tx"),

    # new pair — just add the line below
    ("path/to/new_batch/heightmap", "path/to/new_batch/texture"),
]
```

> `generate_files_path_multi` (`src/dataset.py`) matches file names across the two directories automatically. Files without a counterpart are skipped and reported in the console.

### Adjusting the validation split

`VAL_RATIO` in `src/config.py` controls how much of the dataset is held out for validation (default: 10%):

```python
VAL_RATIO = 0.10   # ← change here, e.g. 0.15 for 15%
SEED = 42          # keeps the split reproducible; change it for a different shuffle
```

---

## Tuning Hyperparameters

All hyperparameters live in `src/config.py`.

### Learning rate

The generator and the discriminator have separate learning rates, set in `src/train.py`:

```python
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
```

**Practical guidance:**

| Observed behaviour | Suggested adjustment |
|---|---|
| Discriminator too strong (generator stalled) | Lower the discriminator LR or raise the generator LR |
| Generator collapsing to identical images | Raise the generator LR slightly |
| Unstable training, strongly oscillating losses | Lower both LRs proportionally |

### L1 loss weight (LAMBDA)

`LAMBDA` (`src/config.py`) controls how much the generator is penalised for pixel-wise differences from the ground truth. Default: `100`.

- **Higher** values make the output closer to the ground truth, but the texture may become blurrier.
- **Lower** values give the generator more freedom, at the risk of artefacts.

### Batch size

`BATCH_SIZE` in `src/config.py` (default `4`). Reduce this if the GPU runs out of memory (OOM); increase it for faster convergence on larger datasets, with some impact on training stability.

---

## Starting the Training

```bash
python -m src.train
```

Training runs for `STEPS` (`src/config.py`) iterations over individual batches.

### Behaviour of `fit`

| Frequency | Action |
|---|---|
| Every **1,000 steps** | Prints the elapsed time and the current step to the console |
| Every **10 steps** | Prints a `.` as a progress indicator |
| Every **2,000 steps** | Writes sample images (input, ground truth, prediction) to TensorBoard |
| Every **5,000 steps** | Saves a checkpoint and evaluates the metrics on the validation set |

> **Automatic resume:** if checkpoints already exist in `tensorboard/checkpoints/`, the script loads the most recent one before training starts. No manual change is needed to continue from where it stopped.

---

## Monitoring the Training

TensorBoard is fed automatically during training. To open it:

```bash
tensorboard --logdir tensorboard/logs
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
tensorboard/checkpoints/
```

via a `tf.train.CheckpointManager`, which keeps the **5 most recent checkpoints** by default (`src/train.py`). To keep more, change `max_to_keep`:

```python
checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=10)
```

To restore a specific checkpoint instead of the most recent one, replace the automatic restore with:

```python
checkpoint.restore("tensorboard/checkpoints/ckpt-3")
```

---

## Saving the Final Model

At the end of training, the generator is saved in the TensorFlow **SavedModel** format to `MODEL_SAVE_PATH` (`src/config.py`, default `testing_files/pre_trained_model`):

```python
generator.save(MODEL_SAVE_PATH)
```

---

## Inference with a Trained Model

`predict.py` loads a saved generator and runs sliding-window inference on a full-size height map (larger than the 256×256 tiles used in training), blending overlapping tiles with a 2D Hann window:

```bash
python predict.py
```

This reads `HEIGHTMAP_PATH` and writes the reconstructed texture to `OUTPUT_PATH` (both defined at the top of `predict.py`).

For inference on a single 256×256 tile without the sliding-window logic:

```python
import tensorflow as tf

generator = tf.keras.models.load_model("testing_files/pre_trained_model")
prediction = generator(heightmap_input, training=False)
```

---

## Pre-trained Model

The pre-trained generator (`testing_files/pre_trained_model/`) is **not** stored in this git repository — the model weights exceed GitHub's file size limit. It is hosted on Zenodo instead:

<ZENODO 10.5281/zenodo.21792022 / [DOWNLOAD LINK](https://zenodo.org/records/21792022)>

Download it and extract it into `testing_files/pre_trained_model/` before running `predict.py` or the inference snippet above.

---

## Who do I talk to? ###

<AUTHOR NAME>, <INSTITUTION>
*** <name> <dot> <surname> <at> <domain> ***

---

## License

Copyright (C) <YEAR> <AUTHORS>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
