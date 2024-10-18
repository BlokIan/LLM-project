import torch
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup, set_seed
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import os

# Hyperparameter recommendations
NUM_EPOCHS = 5  # Increased epoch for better learning
LEARNING_RATE = 1e-5  # Reduced learning rate for better convergence in later epochs
BATCH_SIZE = 32  # Increase batch size to leverage more GPU memory, adjust according to availability

def initialize_accelerator():
    """
    Initialize and return an Accelerator object for distributed training.
    """
    return Accelerator()

def load_model_and_tokenizer(model_name):
    """
    Load and return the model and tokenizer for the given model name and directory.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

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

def create_dataloaders(tokenized_dataset, tokenizer, model):
    """
    Create and return DataLoader objects for the training and evaluation datasets.

    Args:
        tokenized_dataset: The tokenized dataset.
        tokenizer: The tokenizer used for collating data.
        model: The model used for generating batches.

    Returns:
        train_dataloader: DataLoader for the training set.
        eval_dataloader: DataLoader for the evaluation set.
    """
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, return_tensors="pt")
    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator, drop_last=True)
    eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator, drop_last=True)
    return train_dataloader, eval_dataloader

def setup_optimizer_and_scheduler(model, train_dataloader, num_epochs=NUM_EPOCHS):
    """
    Set up and return the optimizer and learning rate scheduler.

    Args:
        model: The model to optimize.
        train_dataloader: DataLoader for the training set.
        num_epochs (int): Number of epochs to train.

    Returns:
        optimizer: The optimizer for training.
        scheduler: The learning rate scheduler.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    return optimizer, scheduler

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

def training_loop(model, optimizer, scheduler, train_dataloader, eval_dataloader, tokenizer, accelerator, num_epochs=NUM_EPOCHS):
    """
    Train and evaluate the model for the given number of epochs.

    Args:
        model: The model to train.
        optimizer: The optimizer for training.
        scheduler: The learning rate scheduler.
        train_dataloader: DataLoader for training data.
        eval_dataloader: DataLoader for evaluation data.
        tokenizer: The tokenizer used for decoding.
        accelerator: The Accelerator object for distributed training.
        num_epochs (int): Number of epochs to train.
    """
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")

    for epoch in range(num_epochs):
        model.train()
        for _, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        all_predictions = []
        all_labels = []
        for _, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
                
                # Generate predictions
                generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"], max_length=128)
                labels = batch["labels"]
                all_predictions.extend(generated_tokens.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = eval_loss / len(eval_dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        metrics = compute_metrics((all_predictions, all_labels), tokenizer, rouge_metric, bleu_metric)

        print(f"Epoch {epoch + 1}, Evaluation Loss: {avg_loss}")
        print(f"Perplexity: {perplexity}")
        print(f"ROUGE and BLEU Scores: {metrics}")

    accelerator.wait_for_everyone()
    accelerator.end_training()

def save_model(model, accelerator, output_dir="./peft_fine-tuned-flan-t5-base"):
    """
    Save the fine-tuned model to the specified output directory.

    Args:
        model: The model to save.
        accelerator: The Accelerator object to ensure proper synchronization.
        output_dir (str): The directory where the model should be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )

def main():
    """
    Main function to initialize components, run evaluation, and print metrics.
    """
    set_seed(42)
    accelerator = initialize_accelerator()
    model_name = "google/flan-t5-base"
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Prepare LoRA configuration and apply it to the model
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # Specify the task type
        r=8,  # Rank of the update matrices
        lora_alpha=16,  # Scaling factor
        lora_dropout=0.1,  # Dropout probability
    )
    model = get_peft_model(model, lora_config)
    tokenized_dataset = load_and_preprocess_dataset("knkarthick/dialogsum", tokenizer, accelerator)
    train_dataloader, eval_dataloader = create_dataloaders(tokenized_dataset, tokenizer, model)
    optimizer, scheduler = setup_optimizer_and_scheduler(model, train_dataloader)
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, scheduler)
    training_loop(model, optimizer, scheduler, train_dataloader, eval_dataloader, tokenizer, accelerator)
    save_model(model, accelerator)

if __name__ == "__main__":
    main()
