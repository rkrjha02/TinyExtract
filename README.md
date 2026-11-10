# TinyExtract

Fine-tuning a small language model (1.5B parameters) with QLoRA to extract structured JSON from messy product descriptions. Demonstrates that a tiny, task-specific model can achieve strong performance at a fraction of the cost of large general-purpose models like GPT-4 or Claude.

## The Problem

Product descriptions on e-commerce sites, marketplaces, and listings come in wildly different formats — formal spec sheets, casual used listings, marketing copy, terse pipe-delimited text. Extracting structured data from these requires understanding context, handling missing information, and producing consistent output.

Large language models (GPT-4, Claude) handle this well but are expensive to call at scale. This project shows that a 1.5B parameter model, fine-tuned on just 640 examples, can match or approach their quality on this specific task.

## Results

| Metric | Base Model | Fine-Tuned | Improvement |
|---|---|---|---|
| Valid JSON | 97.5% | 100.0% | +2.5% |
| Product Name | 46.2% | 63.7% | +17.5% |
| Brand | 88.5% | 91.2% | +2.7% |
| Category | 39.7% | 76.2% | +36.5% |
| Price | 96.2% | 98.8% | +2.6% |
| Condition | 87.2% | 91.2% | +4.0% |

Category extraction improved from **39.7% to 76.2%** — the model learned the exact category naming conventions from the training data. Product name accuracy jumped from **46.2% to 63.7%**. Valid JSON output reached **100%**, meaning every response is properly structured.

## How It Works

### Input (Messy Product Description)
```
Nothing Ear (2) transparent design earbuds. 11.6mm custom driver, LHDC 5.0,
personalized ANC, IP54 rated. 4 hours + 22.5 with case. $149. New.
```

### Output (Structured JSON)
```json
{
  "product_name": "Nothing Ear (2)",
  "brand": "Nothing",
  "category": "Earbuds",
  "price": {"amount": 149, "currency": "USD"},
  "key_specs": {
    "driver_size": "11.6mm",
    "codec": "LHDC 5.0",
    "anc": "Personalized ANC",
    "ip_rating": "IP54",
    "battery_life": "4 hours + 22.5 with case"
  },
  "condition": "new"
}
```

## Technical Approach

### Model
- **Base model:** Qwen2.5-1.5B-Instruct
- **Quantization:** 4-bit NF4 (QLoRA) — compresses the model from ~3GB to ~0.75GB
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) with rank 8
- **Trainable parameters:** 2.18M out of 1.55B total (0.14%)
- **Training hardware:** Google Colab free tier (T4 GPU, 15GB VRAM)
- **Training time:** ~12 minutes for 4 epochs

### Dataset
- **800 synthetic examples** across 10 product categories
- **Training:** 640 examples (80%)
- **Validation:** 80 examples (10%)
- **Test:** 80 examples (10%)

### Product Categories
Smartphones, Laptops, Headphones/Earbuds, Televisions, Cameras, Smartwatches, Tablets, Bluetooth Speakers, Computer Monitors, Wireless Routers

### Dataset Characteristics
- **Difficulty levels:** Easy (clean specs), medium (marketing copy), hard (messy informal text), edge cases (missing fields)
- **Writing styles:** Formal spec sheets, casual used listings, marketing copy, terse pipe-delimited, structured key-value
- **Currencies:** USD, INR, GBP, EUR, CAD, JPY, CNY
- **Conditions:** New, used, refurbished
- **Edge cases:** Unknown brand (null), missing price (null), ambiguous descriptions

## Project Structure

```
TinyExtract/
├── README.md
├── notebooks/
│   ├── BaseModelBenchmark.ipynb   # Baseline evaluation
│   ├── Fine_Tuning.ipynb       # QLoRA fine-tuning
│   └── FineTunedModelBenchmark.ipynb             # Fine-tuned model evaluation & comparison
├── data/
│   ├── dataset_combined.json               # All 800 examples
│   ├── dataset_train.json                  # 640 training examples
│   ├── dataset_val.json                    # 80 validation examples
│   └── dataset_test.json                   # 80 test examples
├── results/
│   ├── benchmark_report.json               # Base model scores
│   ├── benchmark_predictions.json          # Base model predictions on test set
│   ├── comparison_report.json              # Side-by-side comparison
│   └── finetuned_predictions.json          # Fine-tuned model predictions on test set
└── adapter/
    └── (LoRA adapter weights)              # ~16MB trained adapter
```

## How to Reproduce

### Requirements
- Google account (for Colab)
- Hugging Face account

### Steps

1. **Open any notebook in Google Colab**
   - Upload the notebook to Colab or open directly from GitHub

2. **Set GPU runtime**
   - Runtime → Change runtime type → T4 GPU

3. **Run Phase 2** (`Phase2_Base_Model_Benchmark.ipynb`)
   - Upload `dataset_test.json` when prompted
   - Runs the base Qwen2.5-1.5B-Instruct model on 80 test examples
   - Saves baseline scores

4. **Run Phase 3** (`Phase3_LoRA_Fine_Tuning.ipynb`)
   - Upload `dataset_train.json` and `dataset_val.json` when prompted
   - Fine-tunes the model using QLoRA
   - Saves the LoRA adapter (~16MB)
   - Training takes ~12-16 minutes

5. **Run Phase 4** (`Phase4_Evaluation.ipynb`)
   - Upload the adapter zip, `dataset_test.json`, and `benchmark_report.json`
   - Runs the fine-tuned model on the same 80 test examples
   - Produces the comparison report

## Training Details

### LoRA Configuration
| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Task type | CAUSAL_LM |

### Training Configuration
| Parameter | Value |
|---|---|
| Epochs | 4 |
| Batch size | 4 |
| Gradient accumulation steps | 4 |
| Effective batch size | 16 |
| Learning rate | 2e-4 |
| Scheduler | Cosine with 10% warmup |
| Max sequence length | 512 |

### Training Loss Curve
| Epoch | Training Loss | Validation Loss | Token Accuracy |
|---|---|---|---|
| 1 | 0.586 | 0.557 | 88.6% |
| 2 | 0.464 | 0.452 | 90.3% |
| 3 | 0.430 | 0.435 | 90.7% |
| 4 | 0.428 | 0.434 | 90.7% |

## Key Takeaways

1. **Small models can be highly effective** — A 1.5B parameter model achieved 76-99% accuracy on structured extraction after task-specific fine-tuning.

2. **QLoRA makes fine-tuning accessible** — Training only 0.14% of parameters (2.18M out of 1.55B) on a free GPU in 12 minutes produces meaningful improvements.

3. **Category and product name benefit most from fine-tuning** — These fields improved by 36.5% and 17.5% respectively because the model learned specific naming conventions from the training data.

4. **The base model already handles structured output well** — 97.5% valid JSON rate before any training shows that instruct-tuned models understand structured output formats.

## Tools & Libraries

- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — Model and tokenizer loading
- [PEFT](https://huggingface.co/docs/peft) — LoRA adapter implementation
- [TRL](https://huggingface.co/docs/trl) — Supervised fine-tuning trainer
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — 4-bit quantization
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) — Base model
