# LLMVIT

Language Models as Vision Transformer Backbones.
A Python package for leveraging pretrained Language Models (LLMs) as vision transformer backbones.

## Features
- Use any Hugging Face transformer LLM as a vision backbone.
- Multiple image embedding strategies, adapted to the word embeddings of the LLM.
- Configurable and customizable vision encoders.
- Support for QLoRA fine-tuning and mixed precision training.

## Installation

```bash
pip install llmvit
```

Or install from source with Poetry:

```bash
git clone https://github.com/BjBodner/llmvit
cd llmvit
poetry install
```

## Quick Start
Create an LLMVIT model with your LLM of choice and, using a lightweight ViT encoder and a maximum similarity embedder.

```python
from llmvit import LLMVIT, EncoderConfig, EmbedderConfig
from transformers import AutoModelForSequenceClassification

# Initialize base transformer model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)

# Configure vision encoder and embedder
encoder_config = EncoderConfig(
    img_size=256,
    patch_size=7,
    hidden_dim=1024,
    embed_dim=768
)

embedder_config = EmbedderConfig(
    type="max",
)

# Create LLMVIT model
llm_vit = LLMVIT(
    model=model,
    img_processor_config=encoder_config,
    embedder_config=embedder_config
)

# Forward pass
outputs = llm_vit(images, labels)
```

## Architecture

The model consists of three main components:

1. **Vision Encoder**: Processes images into patch embeddings using a Vision Transformer architecture.
2. **Embedder**: Maps vision features to the LLM's word embedding space.
3. **LLM Backbone**: Pre-trained transformer that processes the embedded image features.

## Available Embedders

1. **KNN Embedder**: Maps image features to k-nearest word embeddings with temperature-based weighting.

2. **Max Embedder**: Selects word embeddings based on maximum similarity.

3. **Direct Embedder**: Simple pass-through embedder.


## Available Encoders

LLMVIT supports two types of image encoders:

1. **Patch Encoder**: A convolutional encoder that processes image patches through a single convolutional layer.

2. **Vision Transformer (ViT)**: A full transformer-based encoder that processes image patches through multiple self-attention layers.




## Configuration

### Encoder Configuration
```python
EncoderConfig(
    type="vit",              # Encoder type: "vit" or "patch_embed"
    img_size=28,            # Input image size
    patch_size=7,           # Size of image patches
    in_channels=3,          # Number of input channels
    hidden_dim=256,         # Hidden dimension size
    embed_dim=768,          # Output embedding dimension
    num_encoder_layers=2,   # Number of transformer layers
    dropout=0.0            # Dropout rate
)
```

### Embedder Configuration
```python
EmbedderConfig(
    type="knn",            # Embedder type: "knn", "max", or "direct"
    k=4,                   # Number of nearest neighbors (KNN only)
    temperature=0.1,       # Temperature for softmax (KNN only)
    similarity_metric="cosine_similarity"  # Similarity metric (Max only)
)
```


## Running the Demo

LLMVIT includes a demo script that trains a model on the MNIST dataset. To run it:

1. First, set up your environment variables:
```bash
# Create a .env file with your HuggingFace token
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

2. Run the demo script:
```bash
python -m llmvit.demo
```

The demo uses the following configuration:
```python
model = LLMVIT(
    model=get_classification_model(
        "distilbert-base-uncased",  # Base LLM
        depth=2,                    # Number of transformer layers
        num_classes=10,            # MNIST has 10 classes
        lora_config=DEFAULT_LORA_CFG,
        bnb_config=DEFAULT_BNB_CFG,
    ),
    img_processor_config=EncoderConfig(
        type="patch_embed",        # Use simple patch embedding
        in_channels=1,            # MNIST images are grayscale
    ),
    embedder_config=EmbedderConfig(
        type="direct",            # Direct mapping to embedding space
        k=2,                     # Only used for KNN embedder
        temperature=1.0,         # Only used for KNN embedder
    ),
)
```

The demo uses:
- QLoRA for efficient fine-tuning.
- 4-bit quantization with bitsandbytes.
- Weights & Biases for experiment tracking (disabled by default).
- HuggingFace's Trainer for training loop management.

You can modify the training arguments in the demo script to adjust:
- Batch size
- Learning rate
- Number of epochs
- Optimization settings
- Logging frequency

The demo will save the model checkpoints to `./outputs/mnist_vit/



## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{bodner2024llmvit,
  author = {Benjamin Jacob Bodner},
  title = {LLMVIT: Language Models as Vision Transformer Backbones},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/BjBodner/llmvit}
}
```
