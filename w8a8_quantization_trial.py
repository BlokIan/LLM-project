import torch
from datasets import load_dataset
import torch.quantization
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from peft import PeftModel
import numpy as np

# Load and preprocess dataset
def load_and_preprocess_dataset(dataset_name, tokenizer):
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

# Create a calibration dataset loader
def create_calibration_dataloader(tokenized_dataset, batch_size=8):
    return DataLoader(tokenized_dataset["train"].select(range(100)), batch_size=batch_size, shuffle=False)

# Scale salient weights and quantize
def scale_and_quantize_model(model, quant_path):
    model.eval()
    activation_magnitudes = []

    # Step 1: Forward pass on calibration data to collect activation magnitudes
    calibration_dataloader = create_calibration_dataloader(load_and_preprocess_dataset("knkarthick/dialogsum", AutoTokenizer.from_pretrained("google/flan-t5-base")))
    for batch in calibration_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            activation_magnitudes.append(outputs.last_hidden_state.abs().mean(dim=1).cpu().numpy())
    
    # Step 2: Identify salient weight channels based on activation magnitudes
    activation_magnitudes = np.concatenate(activation_magnitudes, axis=0)
    threshold = np.percentile(activation_magnitudes, 99)  # Top 1% of activations are considered salient
    salient_mask = activation_magnitudes >= threshold

    # Step 3: Apply scaling factor of 10 to salient weights
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            if salient_mask.any():
                param.data = torch.where(salient_mask, param * 10.0, param)

    # Step 4: Save the modified model
    model.save_pretrained("scaled_model")

    # Step 5: Load the model with 8-bit weights using bitsandbytes
    model_int8 = AutoModelForSeq2SeqLM.from_pretrained("scaled_model", load_in_8bit=True, device_map='auto')

    # Step 6: Set backend for quantization
    torch.backends.quantized.engine = 'fbgemm'

    # Define a quantization configuration to quantize activations
    qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_int8.qconfig = qconfig

    # Prepare the model for activation quantization
    model_prepared = torch.quantization.prepare(model_int8, inplace=True)

    # Calibration step (running calibration dataset through the model)
    for batch in calibration_dataloader:
        with torch.no_grad():
            model_prepared(**batch)  # Calibration pass

    # Convert to quantized version (activations only)
    model_quantized = torch.quantization.convert(model_prepared, inplace=True)

    # Save the fully quantized model
    model_quantized.save_pretrained(quant_path)

# Main function
def main():
    # Dataset and model names
    model_name = "google/flan-t5-base"

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Apply scaling and quantization
    scale_and_quantize_model(model, "./quantized_model")

if __name__ == "__main__":
    main()
