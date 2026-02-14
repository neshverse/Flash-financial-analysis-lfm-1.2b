# ⚡ Flash-Financial-Analysis-LFM-1.2B

**Lightning-fast financial intelligence for structured data analysis**

A blazing-fast, customized, lightweight language model optimized for real-time sales & stock analytics, inventory insights, and financial reporting based on the LiquidAI 1.2B base model supervised fine-tuned FP16 model.

---
license: apache-2.0
language:
  - en
base_model: LiquidAI/LFM2.5-1.2B-Base
tags:
  - finance
  - sales-analysis
  - structured-data
  - unsloth
  - lora
  - lfm
  - data-analysis
  - time-series
  - business-intelligence
  - fast-inference
  - efficient-finetuning
pipeline_tag: text-generation
library_name: transformers
---

## Model Details

| Attribute | Value |
|-----------|-------|
| **Base Architecture** | [LiquidAI/LFM2.5-1.2B-Base](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base) |
| **Fine-tuning** | LoRA (r=4, alpha=8) |
| **Context Window** | 1,024 tokens |
| **Precision** | FP16 |
| **Parameters** | 1.2B base + ~500K LoRA |

## Training Summary

- **Total Samples**: 39,435 (37,463 train / 1,972 validation)
- **Training Duration**: 2.4 hours
- **Final Loss**: 0.497 (train) / 0.508 (validation)
- **Hardware**: Consumer GPU (T4)

## Capabilities

- **Sales Analytics**: Real-time sales data querying and analysis
- **Stock Analytics**: Inventory levels, turnover rates, stock movement
- **Financial Reporting**: Automated report generation from structured data
- **Inventory Insights**: Product performance, seasonal trends, demand forecasting

## Performance

- **Inference Speed**: ~0.55 it/s (T4 GPU)
- **Memory Usage**: ~6GB (4-bit loaded)
- **Batch Size**: 4 (effective 8 with grad accum)
- **Max Sequence**: 1,024 tokens
  
## Limitations

- Optimized for structured financial/sales data queries
- Context window limited to 1,024 tokens
- Training data from 2022-2023; may not reflect current market conditions
- Best performance on English language inputs

## Model Files

| File | Format | Size | Description | Use Case |
|------|--------|------|-------------|----------|
| `model.safetensors` | FP16 | ~2.4 GB | Original full precision | Maximum quality, GPU inference |
| `flash-financial-analysis-q8_0.gguf` | Q8_0 | ~1.2 GB | 8-bit quantized (llama.cpp) | CPU inference, Ollama, LM Studio |

## Quantized Version (Q8_0)

We now provide a **Q8_0 quantized version** for easier deployment:

- **Format**: GGUF (llama.cpp compatible)
- **Size**: ~50% smaller than FP16 (1.2 GB vs 2.4 GB)
- **Quality**: ~99.9% of original performance
- **Tools**: Works with llama.cpp, Ollama, LM Studio, llama-cpp-python

### Download Q8_0

```bash
# Using huggingface-cli
huggingface-cli download NeshVerse/Flash-financial-analysis-lfm-1.2b flash-financial-analysis-q8_0.gguf

# Or direct download
wget https://huggingface.co/NeshVerse/Flash-financial-analysis-lfm-1.2b/resolve/main/flash-financial-analysis-q8_0.gguf

## Quick Start

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "NeshVerse/Flash-financial-analysis-lfm-1.2b",
    max_seq_length=1024,
    load_in_4bit=True,
    trust_remote_code=True,
)
