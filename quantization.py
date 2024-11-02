import torch
from datasets import load_dataset
import torch.nn as nn
import torch.quantization
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPTQConfig, AwqConfig
from awq.awq import AWQQuantizer
from torch.utils.data import DataLoader
from accelerate import Accelerator
from peft import PeftModel
import numpy as np

def load_model_and_tokenizer(model_dir, model_name, is_peft_model=False):
    """
    Load the model and tokenizer with optional Parameter-Efficient Fine-Tuning (PEFT) for sequence-to-sequence tasks.

    Args:
        model_dir (str): Directory containing the pretrained model.
        model_name (str): Name of the model to load.
        is_peft_model (bool): Whether to apply PEFT (e.g., LoRA) to the model. Defaults to False.

    Returns:
        Tuple[torch.nn.Module, transformers.PreTrainedTokenizer]: The loaded model and tokenizer.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if is_peft_model:
        model = PeftModel.from_pretrained(model, model_dir)
        model = model.merge_and_unload()  # Merge LoRA weights into the base model
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_and_preprocess_dataset(dataset_name, tokenizer):
    """
    Load a dataset and preprocess it for model training by tokenizing inputs and labels.

    Args:
        dataset_name (str): Name of the dataset to load.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding text data.

    Returns:
        Dataset: The tokenized dataset ready for training with tokenized inputs and labels.
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

def create_calibration_dataloader(tokenized_dataset, batch_size=8):
    """
    Create a DataLoader for the calibration dataset used during model quantization.

    Args:
        tokenized_dataset (Dataset): The tokenized dataset to use for calibration.
        batch_size (int): Batch size for the DataLoader. Defaults to 8.

    Returns:
        DataLoader: A DataLoader for the selected calibration data.
    """
    return DataLoader(tokenized_dataset["train"].select(range(100)), batch_size=batch_size, shuffle=False)

def apply_gptq_quantization(model_name, calibration_dataloader, quant_path_gptq):
    """
    Apply the GPTQ (Gradient-based Post-Training Quantization) strategy to quantize a model.

    Args:
        model_name (str): Name of the pretrained model to quantize.
        calibration_dataloader (DataLoader): DataLoader for calibration data.
        quant_path_gptq (str): Path to save the quantized model.

    Returns:
        None
    """
    # Configure GPTQ
    gptq_config = GPTQConfig(
        bits=4,
        group_size=128,
        dataset='wikitext2',
        desc_act=False,
    )

    # Assuming the model is compatible
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=gptq_config)
    model.eval()

    model.model.decoder.layers[0].self_attn.q_proj.__dict__

    # Save quantized model
    model.save_pretrained(quant_path_gptq)

def apply_awq_quantization(model, quant_path_awq, calibration_dataloader):
    """
    Apply the AWQ (Adaptive Weight Quantization) strategy to quantize a model using a calibration dataset.

    Args:
        model (torch.nn.Module): The model to quantize.
        quant_path_awq (str): Path to save the quantized model.
        calibration_dataloader (DataLoader): DataLoader for calibration data.

    Returns:
        None
    """
    # Modify the config file so that it is compatible with transformers integration
    awq_config = AwqConfig(
        bits=4,
        group_size=128,
        zero_point=True,
        version="GEMM",
    ).to_dict()

    # Quantizer instance
    quantizer = AWQQuantizer(model, awq_config)
    model.eval()

    # Calibration step
    for batch in calibration_dataloader:
        with torch.no_grad():
            quantizer.calibrate(**batch)  # Calibration pass

    # Apply quantization
    quantizer.apply_quantization()
    
    # Save the quantized model
    model.save_pretrained(quant_path_awq)

def main():
    accelerator = Accelerator()

    # Dataset and model names
    model_name = "google/flan-t5-base"
    model_dir = input("Input model directory: ")
    is_peft_model = input("Is this a PEFT model? (yes/no): ").strip().lower() == "yes"
    dataset_name = "knkarthick/dialogsum"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir, model_name, is_peft_model)
    model = accelerator.prepare(model)  # Prepare model for multi-GPU use
    tokenized_dataset = load_and_preprocess_dataset(dataset_name, tokenizer)
    calibration_dataloader = create_calibration_dataloader(tokenized_dataset)
    calibration_dataloader = accelerator.prepare(calibration_dataloader)  
    
    apply_awq_quantization(model, "./awq_model", tokenizer, calibration_dataloader)
    apply_gptq_quantization(model_name, calibration_dataloader, "./gptq_model")

if __name__ == "__main__":
    main()
