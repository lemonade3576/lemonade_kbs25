# Entity Matching with LLM Embeddings and SLM Compression

This repository contains code for entity matching using Large Language Model (LLM) embeddings with compression to Small Language Model (SLM) dimensions. The project supports various compression methods including autoencoder, PCA, linear projection, and knowledge distillation.

## Features

- **Multiple Embedding Models**: Support for OpenAI embeddings, BGE, E5, RoBERTa, DeBERTa, GTE-Qwen, and more
- **Compression Methods**: Autoencoder, PCA, linear projection, first/last/random dimension selection, and knowledge distillation
- **Counterfactual Training**: Support for counterfactual samples to improve model robustness
- **Multiple Datasets**: Pre-configured support for various entity matching datasets (Amazon-Google, DBLP-ACM, Walmart-Amazon, etc.)

## Requirements

- Python 3.8+
- PyTorch 2.5.0
- CUDA-capable GPU (recommended)

See `requirements.txt` for the complete list of dependencies.

## Installation

1. Clone this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional, defaults to OpenAI
```

4. Configure model paths:
   - Update the `lm_map` dictionary in `util.py` with paths to your local embedding models
   - Set model paths via command-line arguments or edit `run.sh`

## Configuration

### Model Paths

Before running, you need to configure the following paths:

1. **SLM Path**: Path to your Small Language Model (e.g., DeBERTa, RoBERTa)
2. **Embedding Model Paths**: Update `lm_map` in `util.py` with paths to your local embedding models
3. **Config File**: Path to `config.json` (default: `config.json` in the project root)
4. **Result Directory**: Directory to save training results
5. **Checkpoint Directory**: Directory to save model checkpoints

### Dataset Configuration

Edit `config.json` to configure dataset paths. The configuration includes:
- Table A and Table B paths
- Training, validation, and test set paths
- Embedding file directories for different embedding models

## Usage

### Basic Training

```bash
python main.py \
    --task AG \
    --config_file config.json \
    --result_dir ./results \
    --checkpoint_dir ./checkpoints \
    --slm <path-to-slm> \
    --embedding_tool openai
```

### Using Shell Script

Edit `run.sh` to set your paths, then run:

```bash
bash run.sh
```

## Project Structure

```
.
├── main.py              # Main training script
├── util.py              # Utility functions and model definitions
├── config.json          # Dataset configuration
├── run.sh               # Example training script
├── requirements.txt     # Python dependencies
├── data/                # Dataset directory
│   ├── amazon-google/
│   ├── dblp-acm_1/
│   └── ...
└── README.md           # This file
```

## Supported Datasets

- Amazon-Google (AG)
- DBLP-ACM (DA1, DA2)
- Walmart-Amazon (WA1, WA2)
- Abt-Buy (AB)
- Beer (BR)
- WDC Products (WDC50, WDC100)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If you encounter any issues or have questions, please open an issue on the repository.
