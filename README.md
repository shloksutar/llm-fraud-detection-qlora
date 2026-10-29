# Fraud Detection with QLoRA

A notebook-based experiment that fine-tunes **Qwen2.5-1.5B-Instruct** to classify financial transactions by fraud risk using **4-bit quantization** and **LoRA**, plus an evaluation notebook that compares the fine-tuned model against the untouched base model.

> **Status:** Experimental learning project. This model is not validated for real-world fraud prevention and should not be used to approve, reject, or investigate live transactions.

## Results at a glance

Evaluated on **100 held-out transactions** (50 fraudulent, 50 legitimate). `HIGH` (fraud) is the positive class.

| Metric    | Base model | Fine-tuned |
|-----------|-----------:|-----------:|
| Accuracy  | 48%        | **60%**    |
| Precision | 0%         | **58%**    |
| Recall    | 0%         | **74%**    |
| F1 score  | 0%         | **65%**    |

**How to read this:**

- Fine-tuning taught the model the task. The base model never correctly flagged a fraud case, while the fine-tuned model catches about three in four.
- Overall accuracy is only modestly above the 50% chance level for a balanced test set. With 100 samples, a difference of this size is not strong evidence.
- Precision of 58% means roughly 4 in 10 `HIGH` flags are false alarms, so the model over-flags legitimate transactions.
- Responses containing neither `HIGH` nor `LOW` are counted as `LOW`. The base model's 0% therefore partly reflects not following the required output format, not only a lack of fraud reasoning.
- These are results from a small experiment, not a claim of production-grade performance. See [Limitations](#evaluation-and-limitations).

## Overview

The training notebook converts transaction records into short chat-style prompts and trains the model to respond with one of two labels:

- `HIGH`: the transaction is labeled as fraudulent in the source dataset.
- `LOW`: the transaction is labeled as non-fraudulent in the source dataset.

It uses supervised fine-tuning (SFT) with parameter-efficient fine-tuning (PEFT), so training focuses on LoRA adapter parameters rather than updating every model parameter.

## Tech stack

- Python
- Hugging Face Datasets
- Hugging Face Transformers
- TRL (`SFTTrainer`, `SFTConfig`)
- PEFT (LoRA)
- BitsAndBytes (4-bit NF4 quantization)
- PyTorch
- scikit-learn (evaluation metrics)

## Dataset and preprocessing

The notebooks use the training split of [`CiferAI/Cifer-Fraud-Detection-Dataset-AF`](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF).

**Training data** (`fraud_detection_qlora.ipynb`):

1. Limits processing to at most the first **1,000,000** rows.
2. Separates records using the `isFraud` label.
3. Randomly samples **500 fraudulent** and **500 non-fraudulent** transactions with seed `42`.
4. Combines and shuffles the samples into a balanced dataset of up to **1,000 examples**.
5. Formats each transaction as a chat conversation using transaction type, amount, sender balances, recipient balances, and a `HIGH`/`LOW` target label.

**Evaluation data** (`evaluate_fraud_detector.ipynb`):

1. Samples **50 fraudulent** and **50 legitimate** transactions from the dataset with seed `99`.
2. Combines and shuffles them into a 100-transaction test set.
3. Builds the same prompt format used in training, without the label.

The samples are deliberately small and balanced for experimentation. They are not representative of the real-world class distribution, where fraud is rare.

## Model and training configuration

| Setting                     | Value                                  |
|-----------------------------|----------------------------------------|
| Base model                  | `Qwen/Qwen2.5-1.5B-Instruct`           |
| Quantization                | 4-bit NF4                              |
| Compute dtype               | FP16                                   |
| Fine-tuning method          | LoRA with TRL SFT                      |
| LoRA rank (`r`)             | 16                                     |
| LoRA alpha                  | 32                                     |
| LoRA dropout                | 0.05                                   |
| Target modules              | `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| Epochs                      | 3                                      |
| Per-device batch size       | 4                                      |
| Gradient accumulation steps | 2                                      |
| Learning rate               | `2e-4`                                 |
| Maximum sequence length     | 512                                    |
| Output directory            | `./fraud-detector`                     |

Actual runtime and memory use depend on the GPU, driver, CUDA, and package versions.

## Run the notebooks

### 1. Requirements

Use a Python environment with a compatible **CUDA-enabled NVIDIA GPU**. Both notebooks run on a free Google Colab T4. Training uses FP16, and the evaluation loads text-generation pipelines on CUDA.

### 2. Install dependencies

Each notebook installs its main dependencies in its first cells. To prepare an environment manually:

```bash
pip install -U torch transformers datasets trl peft bitsandbytes accelerate scikit-learn huggingface_hub
```

### 3. Train: `fraud_detection_qlora.ipynb`

Run the cells in order. The notebook:

1. Loads and samples the dataset.
2. Builds chat-formatted training examples.
3. Loads the quantized base model.
4. Configures LoRA and supervised fine-tuning.
5. Trains the model.
6. Uploads the trained model to the Hugging Face Hub.
7. Loads the uploaded model for a sample inference.

The notebook uploads to [`shloksutar/fraud-detector`](https://huggingface.co/shloksutar/fraud-detector). Before running the upload cells, authenticate with Hugging Face and make sure your account can create or update that repository. Do not commit access tokens or credentials to GitHub.

### 4. Evaluate: `evaluate_fraud_detector.ipynb`

Run the cells in order. The notebook:

1. Loads the fine-tuned model from the Hugging Face Hub and the original `Qwen/Qwen2.5-1.5B-Instruct` base model.
2. Builds the 100-transaction test set (50 fraud, 50 legitimate).
3. Prompts both models with each transaction and maps the generated text to `HIGH` or `LOW`.
4. Computes accuracy, precision, recall, and F1 for both models and prints a side-by-side comparison.

To evaluate your own fine-tuned model, change the `repo_name` variable to your Hugging Face repository.

## Example inference

The training notebook tests a transaction prompt with fields such as transaction type, amount, sender balance before/after, and recipient balance before/after, then generates a response using the fine-tuned model.

Example input (illustrative):

```
Analyze this transaction for fraud risk:
- Type: TRANSFER
- Amount: $85,000.00
- Sender Balance Before: $85,000.00
- Sender Balance After: $0.00
- Recipient Balance Before: $0.00
- Recipient Balance After: $85,000.00
```

The intended output label is `HIGH` or `LOW`. This example demonstrates the workflow only and is **not evidence of model accuracy**. Use the [results above](#results-at-a-glance) for that.

## Evaluation and limitations

The evaluation is a first, small-scale check. Keep these limitations in mind:

- **Small test set.** 100 transactions give wide uncertainty. A few predictions changing would move the scores noticeably.
- **Balanced classes.** The test set is 50/50, whereas real fraud is a tiny fraction of transactions. Precision would be far lower at realistic fraud rates.
- **Over-flagging.** Precision of 58% means many legitimate transactions are labeled `HIGH`.
- **Lenient parsing.** The evaluation looks for `HIGH` or `LOW` in the first 10 generated tokens and defaults to `LOW` if neither appears. This affects the base model's score most.
- **No explicit overlap check.** Test transactions were sampled independently of the training sample, but the notebooks do not verify that no test row also appears in the training set.
- **No baseline.** Results are not compared to a conventional model such as logistic regression or gradient-boosted trees.
- **Single run.** Scores come from one run with fixed seeds. Different samples can produce different numbers.

### Planned improvements

- [ ] Evaluate on a larger test set with a realistic class ratio
- [ ] Add a confusion matrix and false-negative analysis
- [ ] Tune the decision threshold to reduce false alarms
- [ ] Benchmark against a classical baseline (logistic regression / XGBoost)
- [ ] Verify there is no overlap between training and test samples
- [ ] Train on more data with a proper train/validation/test split

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── fraud_detection_qlora.ipynb      # QLoRA fine-tuning
└── evaluate_fraud_detector.ipynb    # base vs fine-tuned evaluation
```

## Security and responsible use

- Never upload API keys, Hugging Face tokens, private transaction data, or other secrets to a public repository.
- Use only data you are authorized to process.
- Treat predictions as experimental signals, not definitive fraud findings.
- Keep a human review process and do not use generated labels as the sole basis for consequential financial decisions.
- This project is for educational and research purposes and is not financial, legal, or compliance advice.

## License

No project license is specified yet. Add a `LICENSE` file before inviting reuse, and verify the dataset and base model license terms separately.
