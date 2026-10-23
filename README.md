# Fraud Detection with QLoRA

A notebook-based experiment that fine-tunes **Qwen2.5-1.5B-Instruct** to classify financial transactions by fraud risk using **4-bit quantization** and **LoRA**.

> **Status:** Experimental learning project. This model is not validated for real-world fraud prevention and should not be used to approve, reject, or investigate live transactions without substantial evaluation.

## Overview

The notebook converts transaction records into short chat-style prompts and trains the model to respond with one of two labels:

- `HIGH` — transaction is labeled as fraudulent in the source dataset.
- `LOW` — transaction is labeled as non-fraudulent in the source dataset.

It uses supervised fine-tuning (SFT) with parameter-efficient fine-tuning (PEFT), so training focuses on LoRA adapter parameters rather than updating every model parameter.

## Tech Stack

- Python
- Hugging Face Datasets
- Hugging Face Transformers
- TRL (`SFTTrainer`, `SFTConfig`)
- PEFT (LoRA)
- BitsAndBytes (4-bit NF4 quantization)
- PyTorch

## Dataset and preprocessing

The notebook loads the training split of [`CiferAI/Cifer-Fraud-Detection-Dataset-AF`](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF).

Current notebook workflow:

1. Limits processing to at most the first **1,000,000** rows.
2. Separates records using the `isFraud` label.
3. Randomly samples **500 fraudulent** and **500 non-fraudulent** transactions with seed `42`.
4. Combines and shuffles the samples, producing a balanced dataset of up to **1,000 examples**.
5. Formats each transaction as a chat conversation using transaction type, amount, sender balances, recipient balances, and a `HIGH`/`LOW` target label.

The training sample is deliberately small for experimentation. It is not representative of the full class distribution and does not establish real-world performance.

## Model and training configuration

| Setting | Value in notebook |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Quantization | 4-bit NF4 |
| Compute dtype | FP16 |
| Fine-tuning method | LoRA with TRL SFT |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| Epochs | 3 |
| Per-device batch size | 4 |
| Gradient accumulation steps | 2 |
| Learning rate | `2e-4` |
| Maximum sequence length | 512 |
| Output directory | `./fraud-detector` |

Actual runtime and memory use depend on the GPU, driver, CUDA, and package versions.

## Run the notebook

### 1. Requirements

Use a Python environment with a compatible **CUDA-enabled NVIDIA GPU**. The notebook loads the model with `device_map="auto"`, uses FP16 training, and later creates a text-generation pipeline on CUDA.

Recommended: run the notebook in a GPU-enabled environment such as Google Colab or a compatible local setup.

### 2. Install dependencies

The notebook installs its main dependencies in notebook cells. To prepare an environment manually:

```bash
pip install -U torch transformers datasets trl peft bitsandbytes accelerate scikit-learn huggingface_hub
```

For best reproducibility, use a clean environment and record the package versions that work with your GPU and CUDA setup.

### 3. Execute the notebook

Open `fraud_detection_qlora.ipynb` in Jupyter or Google Colab and run the cells in order. The notebook:

1. Loads and samples the dataset.
2. Builds chat-formatted training examples.
3. Loads the quantized base model.
4. Configures LoRA and supervised fine-tuning.
5. Trains the model.
6. Uploads the trained model to the Hugging Face Hub.
7. Loads the uploaded model for a sample inference.

**Note:** The uploaded notebook filename currently has a duplicated `.ipynb` extension. Rename it to `fraud_detection_qlora.ipynb` in your repository for a cleaner project layout.

### 4. Hugging Face Hub upload

The notebook is configured to upload to:

[`shloksutar/fraud-detector`](https://huggingface.co/shloksutar/fraud-detector)

Before running the upload cells, authenticate with Hugging Face when prompted and ensure your account has permission to create or update that repository. Do not commit access tokens or credentials to GitHub.

## Example inference

The notebook tests a transaction prompt with fields such as transaction type, amount, sender balance before/after, and recipient balance before/after. It then generates a response using the fine-tuned model.

Example input (illustrative):

```text
Analyze this transaction for fraud risk:
- Type: TRANSFER
- Amount: $85,000.00
- Sender Balance Before: $85,000.00
- Sender Balance After: $0.00
- Recipient Balance Before: $0.00
- Recipient Balance After: $85,000.00
```

The intended output label is `HIGH` or `LOW`. The example demonstrates the inference workflow only; it is **not evidence of model accuracy**.

## Evaluation and limitations

The notebook does not currently report a held-out test set, precision, recall, F1-score, confusion matrix, or calibration results. Therefore, no accuracy or reliability claims are made here.

Before considering any practical use:

- Split data into training, validation, and test sets before fine-tuning.
- Evaluate precision, recall, F1-score, confusion matrix, and false-negative rates.
- Compare against a conventional baseline such as logistic regression or gradient-boosted trees.
- Check for data leakage, distribution shift, and class imbalance.
- Test whether model outputs reliably follow the expected label format.
- Keep a human review process and avoid using generated labels as the sole basis for consequential financial decisions.

## Repository structure

```text
.
├── README.md
└── fraud_detection_qlora.ipynb
```

## Security and responsible use

- Never upload API keys, Hugging Face tokens, private transaction data, or other secrets to a public repository.
- Use only data you are authorized to process.
- Treat predictions as experimental signals, not definitive fraud findings.
- This project is for educational and research purposes and is not financial, legal, or compliance advice.

## License

No project license is specified in the notebook. Add a `LICENSE` file before inviting reuse, and verify the dataset and base model license terms separately.
