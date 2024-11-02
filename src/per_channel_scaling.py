import torch
from torch import nn
from datasets import load_dataset
import torch.quantization
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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
def create_calibration_dataloader(model, tokenizer, tokenized_dataset, batch_size=8):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, return_tensors="pt")
    return DataLoader(tokenized_dataset["train"].select(range(1000)), batch_size=batch_size, shuffle=True, collate_fn=data_collator, drop_last=True)

# Quantization function for scaled weights
def quantize_weight(W_scaled):
    W_abs_max = W_scaled.abs().max()
    qmax = 128  # For int8 quantization
    scale = W_abs_max / qmax if W_abs_max != 0 else 1.0
    W_q = (W_scaled / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return W_q, scale

def calibrate_model(model, calibration_dataloader, device, max_samples_per_layer=1000):
    # Mapping from module to unique name
    module_names = {}
    module_ids = {}
    activations = {}
    activation_stats = {} # For activation quantization
    
    # Collect all Linear layers in the model
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module_names[module] = name
            module_ids[name] = module
    
    # Define the hook function to capture inputs
    def activation_hook(module, input, output):
        module_name = module_names[module]
        # Collect input activations for weight calibration
        if module_name not in activations:
            activations[module_name] = {'inputs': []}
        if len(activations[module_name]['inputs']) < max_samples_per_layer:
            input_activation = input[0].detach().cpu()
            activations[module_name]['inputs'].append(input_activation)

        # Collect activation statistics for activation quantization
        activation = output.detach()
        # Initialize activation_stats for the module if it doesn't exist
        if module_name not in activation_stats:
            activation_stats[module_name] = {'min': float('inf'), 'max': float('-inf')}
        activation_stats[module_name]['min'] = min(activation_stats[module_name]['min'], activation.min().item())
        activation_stats[module_name]['max'] = max(activation_stats[module_name]['max'], activation.max().item())

    
    # Register hooks
    hooks = []
    for module in module_names:
        hook = module.register_forward_hook(activation_hook)
        hooks.append(hook)
    
    # Perform forward pass over the calibration dataloader
    model.eval()
    with torch.no_grad():
        for batch in calibration_dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward pass
            outputs = model(**batch)
            # Break early if enough samples are collected
            if all(len(activations[name]['inputs']) >= max_samples_per_layer for name in activations):
                break
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # For each linear layer, compute scaling factors
    for module_name in activations:
        inputs_list = activations[module_name]['inputs']
        # Concatenate inputs
        X = torch.cat(inputs_list, dim=0)  # Shape: [samples, ...]
        # Flatten over all dimensions except the last
        X = X.view(-1, X.shape[-1])  # Shape: [N, in_features]
        # Limit the number of samples
        if X.shape[0] > max_samples_per_layer:
            X = X[:max_samples_per_layer]
        s_X = X.abs().mean(dim=0) + 1e-8  # Shape: [in_features], avoid division by zero
        alphas = torch.arange(0, 1.1, 0.1)  # Alphas from 0 to 1 in steps of 0.1
        min_loss = None
        best_alpha = None
        W = module_ids[module_name].weight.data.cpu().clone()  # Shape: [out_features, in_features]
        for alpha in alphas:
            s = s_X.pow(alpha)
            s_inv = 1.0 / s
            # Scale weights
            W_scaled = W * s.unsqueeze(0)  # Multiply each column by s
            # Quantize W_scaled
            W_q, scale_w = quantize_weight(W_scaled)
            # Scale inputs
            X_scaled = X * s_inv.unsqueeze(0)
            # Compute outputs
            Y_q = X_scaled @ W_q.t().float() * scale_w
            Y_fp = X @ W.t()
            # Compute loss
            loss = (Y_q - Y_fp).pow(2).mean()
            if min_loss is None or loss < min_loss:
                min_loss = loss
                best_alpha = alpha
        # Apply scaling with best_alpha
        s = s_X.pow(best_alpha)
        module_ids[module_name].weight.data *= s.unsqueeze(0).to(device)
        print(f"Applied scaling to module {module_name} with alpha {best_alpha}")

def visualize_activations(activations):

    # Visualize activations
    for layer_name in activations:
        # Concatenate collected activations
        layer_activations = torch.cat(activations[layer_name], dim=0)
        # Flatten over all dimensions except the last (feature dimension)
        activation_flat = layer_activations.view(-1, layer_activations.shape[-1])
        # Compute mean activation per channel
        mean_activations = activation_flat.mean(dim=0).numpy()
        
        # Plot histogram of activations
        plt.figure(figsize=(10, 6))
        plt.hist(mean_activations, bins=100, color='blue', alpha=0.7)
        plt.title(f'Activation Histogram for {layer_name}')
        plt.xlabel('Mean Activation Value per Channel')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

# Main function
def main():
    # Dataset and model names
    model_name = "google/flan-t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained("./fine-tuned-flan-t5-base-v2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device if it's not already there
    model.to(device)

    # Load and tokenize the dataset
    tokenized_dataset = load_and_preprocess_dataset("knkarthick/dialogsum", tokenizer)
    
    # Create calibration dataloader
    calibration_dataset = create_calibration_dataloader(model, tokenizer, tokenized_dataset)

    # Calibrate model and quantize weights
    calibrate_model(model, calibration_dataset, device)
    model.save_pretrained("./calibrated-fine-tunedflan-t5-base-v2")

if __name__ == "__main__":
    main()
