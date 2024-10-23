import torch
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, set_seed
from accelerate import Accelerator
import numpy as np
from peft import PeftModel

# Hyperparameter
BATCH_SIZE = 16

def initialize_accelerator():
    """
    Initialize and return an Accelerator object for distributed training.
    """
    return Accelerator()

# Load model and tokenizer with option for PEFT model
def load_model_and_tokenizer(model_dir, model_name, is_peft_model=False):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if is_peft_model:
        model = PeftModel.from_pretrained(model, model_dir)
        model = model.merge_and_unload()  # Merge LoRA weights into the base model
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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

def load_and_preprocess_dataset(dataset_name, tokenizer, accelerator):
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

    with accelerator.main_process_first():
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

def evaluate_model(model, test_dataloader, tokenizer, accelerator):
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
            generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"], max_length=128)
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
        model_dir = input("Input model directory: ")

    is_peft_model = input("Is this a PEFT model? (yes/no): ").strip().lower() == "yes"

    model, tokenizer = load_model_and_tokenizer(model_dir, model_name, is_peft_model)
    
    # If base model then use prompt engineering dataset
    if is_base_model:
        tokenized_dataset = load_and_preprocess_prompt_engineering_dataset("knkarthick/dialogsum", tokenizer)
    else:
        # Load and preprocess normal dataset
        tokenized_dataset = load_and_preprocess_dataset("knkarthick/dialogsum", tokenizer)

    accelerator = initialize_accelerator()
    #tokenized_dataset = load_and_preprocess_dataset("knkarthick/dialogsum", tokenizer, accelerator)
    test_dataloader = create_test_dataloader(tokenized_dataset, tokenizer, model)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    evaluate_model(model, test_dataloader, tokenizer, accelerator)

if __name__ == "__main__":
    main()
