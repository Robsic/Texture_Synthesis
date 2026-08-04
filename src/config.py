# --- Dataset ---
BATCH_SIZE = 64
IMG_HEIGHT = 256
IMG_WIDTH = 256
VAL_RATIO = 0.10
SEED = 42

# TODO(davi): confira estes caminhos - foram copiados do Texture_generator.py
# original e podem não bater mais com a estrutura de pastas atual do projeto.
PAIRS = [
    ("Fill_Texture/data-set/2025/heightmap", "Fill_Texture/data-set/2025/texture"),
    ("Fill_Texture/data-set/2021/heightmap", "Fill_Texture/data-set/2021/texture"),
    ("Fill_Texture/data-set/2021/Cut_TOPO_ITA/1/heightmap", "Fill_Texture/data-set/2021/Cut_TOPO_ITA/1/texture"),
    ("Fill_Texture/data-set/2021/Cut_TOPO_ITA/2/heightmap", "Fill_Texture/data-set/2021/Cut_TOPO_ITA/2/texture"),
    ("Fill_Texture/data-set/2021/Cut_TOPO_ITA/3/heightmap", "Fill_Texture/data-set/2021/Cut_TOPO_ITA/3/texture"),
]

# --- Modelo ---
OUTPUT_CHANNELS = 3
LAMBDA = 100

# TODO(davi): "Pix2Pix_Classic/Big_dataset" é nome de outro projeto/pasta -
# troque para o caminho correto dentro de Fill_Texture antes de treinar.
CHECKPOINT_DIR = 'Pix2Pix_Classic/Big_dataset/checkpoints'
LOG_DIR = 'Pix2Pix_Classic/Big_dataset/logs'

# TODO(davi): mesma observação de caminho acima.
MODEL_SAVE_PATH = 'Modelo_Treinados/Big_dataset_100k'

# --- Treino ---
# Não estava definido no arquivo original (fit() nunca era chamado - ver train.py).
# 100_000 é um chute a partir do nome da pasta de save ("...100k"); confirme antes de rodar.
STEPS = 100_000
