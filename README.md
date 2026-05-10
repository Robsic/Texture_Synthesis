# Pix2Pix — Geração de Textura a partir de Mapa de Altura

Modelo de geração de imagens baseado na arquitetura **Pix2Pix (cGAN)** que aprende a sintetizar texturas RGB a partir de mapas de altura em escala de cinza.

---

## Índice

1. [Requisitos](#requisitos)
2. [Estrutura de Diretórios](#estrutura-de-diretórios)
3. [Configurando o Dataset](#configurando-o-dataset)
4. [Ajustando Hiperparâmetros](#ajustando-hiperparâmetros)
5. [Iniciando o Treinamento](#iniciando-o-treinamento)
6. [Monitorando o Treinamento](#monitorando-o-treinamento)
7. [Checkpoints e Retomada de Treino](#checkpoints-e-retomada-de-treino)
8. [Salvando o Modelo Final](#salvando-o-modelo-final)

---

## Requisitos

```
tensorflow >= 2.x
matplotlib
numpy
```

---

## Geração de data - set

Para gerar os pares de heightmap - textura utilizar o repositório https://github.com/DaviBossi/generate_data_set

---

## Estrutura de Diretórios

Cada par de diretórios deve conter imagens com **nomes idênticos** — o pareamento é feito pelo nome do arquivo.

```
projeto/
│
├── Fill_Texture/
│   └── data-set/
│       ├── 2025/
│       │   ├── heightmap/        ← imagens .png em escala de cinza (1 canal)
│       │   └── texture/          ← imagens .png coloridas (3 canais)
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
│       ├── checkpoints/          ← checkpoints salvos automaticamente
│       └── logs/                 ← logs do TensorBoard
│
└── Modelo_Treinados/
    └── Big_dataset_100k/         ← modelo final salvo
```

> **Atenção:** Heightmaps devem ser imagens PNG de **1 canal** e texturas PNG de **3 canais**. Qualquer outro formato exige alteração nas funções `load_pair`.

---

## Configurando o Dataset

### Adicionando novos dados

Para incluir novos pares de imagens ao treinamento, adicione uma entrada na lista `pairs`, informando o caminho para o diretório de heightmaps e o diretório de texturas correspondente:

```python
pairs = [
    # pares existentes
    ("Fill_Texture/data-set/2025/heightmap", "Fill_Texture/data-set/2025/texture"),
    ("Fill_Texture/data-set/2021/heightmap", "Fill_Texture/data-set/2021/texture"),

    # novo par — basta adicionar a linha abaixo
    ("Fill_Texture/data-set/novo_lote/heightmap", "Fill_Texture/data-set/novo_lote/texture"),
]
```

> O script cruza automaticamente os nomes de arquivo entre os dois diretórios. Arquivos sem par correspondente são ignorados com um aviso no console.

### Ajustando a proporção de validação

O parâmetro `val_ratio` em `train_val_split` controla quanto do dataset é reservado para validação (padrão: 10%):

```python
# Exemplo: reservar 15% para validação
(train_hm, train_tx), (val_hm, val_tx) = train_val_split(
    hm_files, tx_files,
    val_ratio=0.15,   # ← altere aqui
    seed=42
)
```

O `seed` garante que o split seja sempre o mesmo. Mude o valor se quiser um embaralhamento diferente.

---

## Ajustando Hiperparâmetros

### Learning Rate

As taxas de aprendizado do gerador e do discriminador são definidas individualmente:

```python
# Gerador — padrão: 2e-4
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Discriminador — padrão: 1e-4
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
```

**Orientações práticas:**

| Situação observada | Ajuste sugerido |
|---|---|
| Discriminador muito forte (gerador estagnado) | Reduza o LR do discriminador ou aumente o do gerador |
| Gerador colapsando para imagens iguais | Aumente levemente o LR do gerador |
| Treinamento instável, losses oscilando muito | Reduza ambos os LRs proporcionalmente |

### Peso da Loss L1 (LAMBDA)

O parâmetro `LAMBDA` controla quanto o gerador é penalizado por diferenças pixel a pixel em relação ao ground truth. Padrão: `100`.

```python
LAMBDA = 100  # ← linha no topo do script
```

- Valores **maiores** tornam as saídas mais fiéis ao ground truth, mas podem deixar a textura mais "borrada".
- Valores **menores** dão mais liberdade criativa ao gerador, mas podem gerar artefatos.

### Tamanho do Batch

```python
batch_size = 64  # ← linha no topo do script
```

Reduza se houver erros de memória (OOM na GPU). Valores menores como `16` ou `32` funcionam bem, com possível impacto na estabilidade do treino.

---

## Iniciando o Treinamento

O treinamento é iniciado pela função `fit`. Os parâmetros usados são o número total de **steps** (iterações sobre batches individuais) e o data-set destinado ao treinamento:

```python
fit(train_ds, steps=100_000)
```

### Comportamento da função `fit`

| Frequência | Ação |
|---|---|
| A cada **1.000 steps** | Exibe o tempo decorrido e o step atual no console |
| A cada **10 steps** | Imprime um `.` para indicar progresso |
| A cada **2.000 steps** | Salva amostras visuais (input, ground truth, predição) no TensorBoard |
| A cada **5.000 steps** | Salva um checkpoint e avalia as métricas no conjunto de validação |

### Exemplo de uso

```python
# Treinar por 50.000 steps a partir do estado atual (ou do zero)
fit(train_ds, steps=50_000)
```

> **Retomada automática:** se já existirem checkpoints em `Pix2Pix_Classic/Big_dataset/checkpoints/`, o script os carrega automaticamente antes do treino. Não é necessário nenhum ajuste manual para continuar de onde parou.

---

## Monitorando o Treinamento

O TensorBoard é alimentado automaticamente durante o treino. Para abrir:

```bash
tensorboard --logdir Pix2Pix_Classic/Big_dataset/logs
```

As métricas disponíveis são:

| Métrica | O que indica |
|---|---|
| `gen_total_loss` | Loss total do gerador (GAN + L1) |
| `gen_gan_loss` | Componente adversarial da loss do gerador |
| `gen_l1_loss` | Componente pixel-a-pixel (fidelidade ao ground truth) |
| `disc_loss` | Loss do discriminador |
| `val/*` | Versões de validação das métricas acima |

Além das métricas numéricas, a aba **Images** exibe a evolução visual das predições ao longo do treino.

---

## Checkpoints e Retomada de Treino

Checkpoints são salvos automaticamente a cada 5.000 steps em:

```
Pix2Pix_Classic/Big_dataset/checkpoints/
```

O TensorFlow mantém os **5 checkpoints mais recentes** por padrão. Para aumentar esse número, adicione o parâmetro `max_to_keep` na criação do checkpoint:

```python
checkpoint = tf.train.Checkpoint(...)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)
```

Para restaurar um checkpoint específico (em vez do mais recente), substitua a restauração automática: 

```python
latest = tf.train.latest_checkpoint(checkpoint_dir)
```

por :

```python
checkpoint.restore("Pix2Pix_Classic/Big_dataset/checkpoints/ckpt-3")
```

---

## Salvando o Modelo Final

Ao final do script, o modelo do gerador pode ser salvo no formato **SavedModel** do TensorFlow:

```python
generator.save("Modelo_Treinados/Big_dataset_100k")
```

Para alterar o destino ou o nome, modifique o caminho acima. O modelo salvo pode ser carregado posteriormente para inferência sem precisar recriar a arquitetura:

```python
import tensorflow as tf

generator = tf.keras.models.load_model("Modelo_Treinados/Big_dataset_100k")

# Inferência
predicao = generator(heightmap_input, training=False)
```

> `training=False` desativa o Dropout durante a inferência, o que é necessário para resultados determinísticos.
