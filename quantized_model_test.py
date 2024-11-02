import torch
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, set_seed
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import numpy as np
# Custom per-channel scaling file
from src.per_channel_scaling import create_calibration_dataloader, calibrate_model 


# Hyperparameter
BATCH_SIZE = 16
MAX_SAMPLES_PER_LAYER = 10


def load_model_and_tokenizer(model_name, model_dir, bnb_config, is_peft_model=False):
    """
    Load and return the quantized model and tokenizer for the given model name and directory. Option for PEFT model

    Args:
        model_name (str): The name of the model to load.
        model_dir (str): The directory where the fine-tuned model is saved.
        bnb_config (): BitsAndBytes quantization configuration
        is_peft_model (default=False): Toggle for PEFT model

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

    if is_peft_model:
        model = get_peft_model(
            model,
            peft_config
        )
        model = model.merge_and_unload()

    return model, tokenizer

def load_and_preprocess_prompt_engineering_dataset(dataset_name, tokenizer):
    """
    Load and preprocess the dataset with prompt engineering for the base model.

    Args:
        dataset_name (str): The name of the dataset to load.
        tokenizer: The tokenizer to use for preprocessing.

    Returns:
        tokenized_dataset: The tokenized dataset ready for evaluation with the base model and prompt engineering.
    """
    dataset = load_dataset(dataset_name)

    def preprocess_function(examples):
        # Placeholder for prompt engineering
        # Customize the prompt here
        # For example:
        # prompt_template = "Summarize the following dialogue in one or two sentences:\n"
        prompt_template = "<PROMPT TEMPLATE>"  # Replace this with your actual prompt

        # Apply the prompt template to each dialogue in the dataset
        inputs = [prompt_template + dialogue for dialogue in examples["dialogue"]]
        model_inputs = tokenizer(inputs, truncation=True, max_length=512, padding="max_length")
        labels = tokenizer(examples["summary"], truncation=True, max_length=128, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["id", "dialogue", "summary", "topic"],
    )
    return tokenized_dataset


def load_and_preprocess_dataset(dataset_name, tokenizer):
    """
    Load and preprocess the dataset using the given tokenizer and accelerator.

    Args:
        dataset_name (str): The name of the dataset to load.
        tokenizer: The tokenizer to use for preprocessing.
        accelerator: The Accelerator object to ensure proper synchronization.

    Returns:
        tokenized_dataset: The tokenized dataset.
    """
    dataset = load_dataset(dataset_name)

    def preprocess_function(examples):
        inputs = ["Summarize the following dialogue: " + dialogue for dialogue in examples["dialogue"]]
        model_inputs = tokenizer(inputs, truncation=True, max_length=512, padding="max_length")
        labels = tokenizer(examples["summary"], truncation=True, max_length=128, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["id", "dialogue", "summary", "topic"],
    )

    return tokenized_dataset

def create_test_dataloader(tokenized_dataset, tokenizer, model):
    """
    Create and return a DataLoader object for the test dataset.

    Args:
        tokenized_dataset: The tokenized dataset.
        tokenizer: The tokenizer used for collating data.
        model: The model used for generating batches.

    Returns:
        test_dataloader: DataLoader for the test set.
    """
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, return_tensors="pt")
    test_dataloader = DataLoader(tokenized_dataset["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator, drop_last=True)
    return test_dataloader

def evaluate_model(model, test_dataloader, tokenizer, device):
    """
    Evaluate the model on the test dataset and calculate ROUGE, BLEU, and Perplexity scores.

    Args:
        model: The model to evaluate.
        test_dataloader: DataLoader for the test dataset.
        tokenizer: The tokenizer used for decoding.
        accelerator: The Accelerator object for distributed evaluation.
    """
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")
    total_loss = 0
    all_predictions = []
    all_labels = []

    model.eval()
    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            # Generate predictions
            generated_tokens = model.generate(batch["input_ids"].to(device), max_length=128)
            labels = batch["labels"]
            all_predictions.extend(generated_tokens.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    metrics = compute_metrics((all_predictions, all_labels), tokenizer, rouge_metric, bleu_metric)

    print(f"Test Loss: {avg_loss}")
    print(f"Perplexity: {perplexity}")
    print(f"ROUGE and BLEU Scores: {metrics}")

def compute_metrics(eval_pred, tokenizer, rouge_metric, bleu_metric):
    """
    Compute ROUGE and BLEU scores for the model's predictions.

    Args:
        eval_pred (tuple): A tuple containing predictions and labels.
        tokenizer: The tokenizer used for decoding.
        rouge_metric: The ROUGE metric to compute.
        bleu_metric: The BLEU metric to compute.

    Returns:
        dict: The computed ROUGE and BLEU scores rounded to 4 decimal places.
    """
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels_bleu = [[label] for label in decoded_labels]  # BLEU expects a list of lists

    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_bleu)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    rouge_result["gen_len"] = np.mean(prediction_lens)

    result = {k: round(v, 4) for k, v in rouge_result.items()}
    result["bleu"] = round(bleu_result["bleu"], 4)
    return result


def main():
    """
    Main function to initialize components, run evaluation, and print metrics.
    """
    set_seed(42)
    model_name = "google/flan-t5-base"
    is_base_model = input("Are you testing the base model? (yes/no): ").strip().lower() == "yes"
    if is_base_model:
        model_dir = model_name
    else:
        model_dir = "../peft_fine-tuned-flan-t5-base-v3"  # input("Input model directory: ")
    is_peft_model = input("Is this a PEFT model? (yes/no): ").strip().lower() == "yes"

    k_bit = input("Set the bit-width for quantization (4/8): ")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # If base model then use prompt engineering dataset
    if is_base_model:
        tokenized_dataset = load_and_preprocess_prompt_engineering_dataset("knkarthick/dialogsum", tokenizer)
    else:
        # Load and preprocess normal dataset
        tokenized_dataset = load_and_preprocess_dataset("knkarthick/dialogsum", tokenizer)

    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device if it's not already there
    model.to(device)

    # Create calibration dataloader
    calibration_dataset = create_calibration_dataloader(model, tokenizer, tokenized_dataset)

    # Calibrate model
    calibrate_model(model, calibration_dataset, device, MAX_SAMPLES_PER_LAYER)

    # Saving calibrated model path
    if is_base_model:
        calibrated_model_path = "./flan-t5-base-calibrated"
    else:
        calibrated_model_path = model_dir + "-calibrated"
    model.save_pretrained(calibrated_model_path)

    # Setup BitsAndBytes Configuration for either 4 or 8 bit
    if k_bit == "4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    # Load calibrated model for quantization model with the bnb_config (and tokenizer but this is redundant)
    model, tokenizer = load_model_and_tokenizer(model_name, calibrated_model_path, bnb_config, is_peft_model)

    # Create dataloader
    test_dataloader = create_test_dataloader(tokenized_dataset, tokenizer, model)

    # Evaluate the model on the test set
    evaluate_model(model, test_dataloader, tokenizer, device)

if __name__ == "__main__":
    main()
