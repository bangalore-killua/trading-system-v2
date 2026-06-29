"""
finetune_minicpm_sentiment.py — LoRA fine-tuning for MiniCPM-1B financial sentiment.

Prerequisites (run once):
    pip install peft accelerate datasets transformers
    # Model weights via Ollama:  ollama pull minicpm
    # OR HuggingFace:            huggingface-cli login (model: openbmb/MiniCPM-1B-sft-bf16)

Usage:
    python scripts/finetune_minicpm_sentiment.py
    ollama create minicpm-sentiment -f models/minicpm_sentiment_lora/Modelfile

Then in config.json:
    "news": {"sentiment_model": "minicpm-sentiment"}
"""

import json
import os
from pathlib import Path

HF_MODEL_ID  = "openbmb/MiniCPM-1B-sft-bf16"
DATASET_PATH = Path(__file__).parent.parent / "data" / "minicpm_sentiment_train.jsonl"
OUTPUT_DIR   = Path(__file__).parent.parent / "models" / "minicpm_sentiment_lora"
EPOCHS       = 3
BATCH_SIZE   = 4
MAX_SEQ_LEN  = 256
LORA_R       = 8
LORA_ALPHA   = 16


def load_dataset():
    records = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt(text: str, sentiment: str = "") -> str:
    instruction = (
        "You are a financial news sentiment analyser. "
        'Reply with JSON only: {"sentiment": "positive|neutral|negative", "score": <float -1 to 1>}\n\n'
        f"Text: {text}"
    )
    if sentiment:
        return instruction + f'\n\n{{"sentiment": "{sentiment}", "score": '
    return instruction


def main():
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        import torch
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install peft accelerate datasets transformers")
        return

    print(f"Loading base model: {HF_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    records = load_dataset()
    print(f"Loaded {len(records)} training examples")

    def tokenize(batch):
        texts = []
        for text, sentiment, score in zip(batch["text"], batch["sentiment"], batch["score"]):
            prompt = build_prompt(text)
            completion = f'{{"sentiment": "{sentiment}", "score": {score:.2f}}}'
            texts.append(prompt + "\n\n" + completion)
        return tokenizer(texts, max_length=MAX_SEQ_LEN, truncation=True, padding="max_length")

    ds = Dataset.from_list(records)
    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Adapter saved to {OUTPUT_DIR}")

    modelfile = OUTPUT_DIR / "Modelfile"
    modelfile.write_text(
        "FROM minicpm\n"
        f"ADAPTER {OUTPUT_DIR / 'adapter_model.safetensors'}\n"
        "PARAMETER temperature 0.0\n"
        'SYSTEM "You are a financial news sentiment analyser. '
        'Reply with JSON only: {\\"sentiment\\": \\"positive|neutral|negative\\", \\"score\\": <float -1 to 1>}"\n'
    )
    print(f"Modelfile written to {modelfile}")
    print("\nNext step:")
    print(f"  ollama create minicpm-sentiment -f {modelfile}")
    print('  Then in config.json: "news": {"sentiment_model": "minicpm-sentiment"}')


if __name__ == "__main__":
    main()
