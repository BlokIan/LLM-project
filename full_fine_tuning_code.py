import torch
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup, set_seed
from accelerate import Accelerator
import numpy as np
import os

def initialize_accelerator():
    """
    Initialize and return an Accelerator object for distributed training.
    """
    return Accelerator()

def load_model_and_tokenizer(model_name):
    """
    Load and return the model and tokenizer for the given model name.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token="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")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_and_preprocess_dataset(dataset, tokenizer, accelerator):
    """
    Load and preprocess the dataset using the given tokenizer and accelerator.

    Args:
        dataset (str): The name of the dataset to load.
        tokenizer: The tokenizer to use for preprocessing.
        accelerator: The Accelerator object to ensure proper synchronization.

    Returns:
        tokenized_dataset: The tokenized dataset.
    """
    dataset = load_dataset(dataset)

    def preprocess_function(examples):
        # Preprocess the input dialogues and summaries
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
    Create and return DataLoader objects for training and evaluation.

    Args:
        tokenized_dataset: The tokenized dataset.
        tokenizer: The tokenizer used for collating data.
        model: The model used for generating batches.

    Returns:
        train_dataloader: DataLoader for the training set.
        eval_dataloader: DataLoader for the evaluation set.
    """
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, return_tensors="pt")
    train_dataloader = DataLoader(tokenized_dataset["train"].select(range(100)), batch_size=16, shuffle=True, collate_fn=data_collator, drop_last=True)
    eval_dataloader = DataLoader(tokenized_dataset["validation"].select(range(20)), batch_size=16, shuffle=False, collate_fn=data_collator, drop_last=True)
    return train_dataloader, eval_dataloader

def setup_optimizer_and_scheduler(model, train_dataloader, num_epochs=3):
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    return optimizer, scheduler

def compute_metrics(eval_pred, tokenizer, metric):
    """
    Compute evaluation metrics for the model's predictions.

    Args:
        eval_pred (tuple): A tuple containing predictions and labels.
        tokenizer: The tokenizer used for decoding.
        metric: The evaluation metric to compute (e.g., ROUGE).

    Returns:
        dict: The computed metrics rounded to 4 decimal places.
    """
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

def training_loop(model, optimizer, scheduler, train_dataloader, eval_dataloader, tokenizer, accelerator, num_epochs=3):
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
    metric = evaluate.load("rouge")

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            if step % 1 == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        all_predictions = []
        all_labels = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                eval_loss += outputs.loss.item()

                # Generate predictions
                generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"], max_length=128)
                labels = batch["labels"]
                gathered_predictions = accelerator.gather(generated_tokens)
                all_predictions.extend(gathered_predictions.numpy())
                gathered_labels = accelerator.gather(labels)
                all_labels.extend(gathered_labels.numpy())

        eval_loss /= len(eval_dataloader)
        eval_metric = compute_metrics((all_predictions, all_labels), tokenizer, metric)
        print(f"Epoch {epoch + 1}, Evaluation Loss: {eval_loss}, ROUGE Scores: {eval_metric}")
    accelerator.wait_for_everyone()
    accelerator.end_training()

def save_model(model, accelerator, output_dir="./fine-tuned-flan-t5-base"):
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
    Main function to initialize components, run training, and save the model.
    """
    set_seed(42)
    accelerator = initialize_accelerator()
    model, tokenizer = load_model_and_tokenizer("google/flan-t5-base")
    tokenized_dataset = load_and_preprocess_dataset("knkarthick/dialogsum", tokenizer, accelerator)
    train_dataloader, eval_dataloader = create_dataloaders(tokenized_dataset, tokenizer, model)
    optimizer, scheduler = setup_optimizer_and_scheduler(model, train_dataloader)
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, scheduler)
    training_loop(model, optimizer, scheduler, train_dataloader, eval_dataloader, tokenizer, accelerator)
    save_model(model, accelerator)

if __name__ == "__main__":
    main()
