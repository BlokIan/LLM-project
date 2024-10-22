import torch
from torch import nn
from datasets import load_dataset
import torch.quantization
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class QuantizedLinear(nn.Linear):
    def forward(self, input):
        # Quantize input activation
        if hasattr(self, 'activation_scale') and hasattr(self, 'activation_zero_point'):
            scale = self.activation_scale
            zero_point = self.activation_zero_point
            qmin = 0
            qmax = 256
            input_q = ((input / scale).round() + zero_point).clamp(qmin, qmax).to(torch.uint8)
            # Dequantize back to float for computation
            input_dq = (input_q.float() - zero_point) * scale
        else:
            input_dq = input  # If no quantization parameters, use input as is

        # Use quantized weights if weights have been quantized
        if hasattr(self, 'weight_scale'):
            weight_scale = self.weight_scale
            weight_q = self.weight.data.to(torch.int8)
            weight_dq = weight_q.float() * weight_scale
        else:
            weight_dq = self.weight

        # Perform linear operation
        output = torch.nn.functional.linear(input_dq, weight_dq, self.bias)

        # Optionally, you can quantize the output here if needed
        return output

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

def calibrate_model(model, calibration_dataloader, device):
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

        return activation_stats

def compute_activation_quantization_parameters(model, activation_stats, device):
    """
    Computes activation quantization parameters and stores them in the model modules.

    Args:
        model: The neural network model.
        activation_stats: A dictionary containing min and max activation values for each module.
        device: The device on which the model is located.
    """
    # Get all Linear modules with their names
    module_ids = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    activation_quant_params = {}

    # Compute quantization parameters for each module
    for module_name in activation_stats:
        a_min = activation_stats[module_name]['min']
        a_max = activation_stats[module_name]['max']
        qmin = 0   # For uint8 quantization
        qmax = 256 # For uint8 quantization
        if a_max - a_min == 0:
            scale = 1.0
            zero_point = 0
        else:
            scale = (a_max - a_min) / (qmax - qmin)
            zero_point = int(qmin - a_min / scale)
            zero_point = max(qmin, min(qmax, zero_point))
        activation_quant_params[module_name] = {'scale': scale, 'zero_point': zero_point}

    # Store activation quantization parameters in the model modules
    for module_name, module in module_ids.items():
        if module_name in activation_quant_params:
            quant_params = activation_quant_params[module_name]
            module.register_buffer('activation_scale', torch.tensor(quant_params['scale'], device=device))
            module.register_buffer('activation_zero_point', torch.tensor(quant_params['zero_point'], device=device))

    print("Activation quantization parameters computed and stored in the model.")

def quantize_model_weights(model):
    # Quantize weights of all Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Assume scaling factors have already been applied to module.weight during calibration
            W_scaled = module.weight.data
            # Quantize the scaled weights
            W_q, scale = quantize_weight(W_scaled)
            # Replace weights with quantized weights
            module.weight.data = W_q.float().to(W_scaled.device)
            # Store the scale for use during inference
            module.register_buffer('weight_scale', torch.tensor(scale, device=W_scaled.device))

def replace_linear_with_quantized_linear(model):
    for name, module in model.named_children():
        # Check for T5-specific linear layers
        if isinstance(module, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
            quant_linear = QuantizedLinear(module.in_features, module.out_features, bias=module.bias is not None)
            # Copy existing weights and biases
            quant_linear.weight = module.weight
            quant_linear.bias = module.bias
            # Copy activation quantization parameters
            if hasattr(module, 'activation_scale'):
                quant_linear.register_buffer('activation_scale', module.activation_scale)
            if hasattr(module, 'activation_zero_point'):
                quant_linear.register_buffer('activation_zero_point', module.activation_zero_point)
            # Copy weight quantization parameters
            if hasattr(module, 'weight_scale'):
                quant_linear.register_buffer('weight_scale', module.weight_scale)
            # Replace the module
            setattr(model, name, quant_linear)
        elif isinstance(module, nn.Linear):
            # Same as before
            quant_linear = QuantizedLinear(module.in_features, module.out_features, bias=module.bias is not None)
            quant_linear.weight = module.weight
            quant_linear.bias = module.bias
            if hasattr(module, 'activation_scale'):
                quant_linear.register_buffer('activation_scale', module.activation_scale)
            if hasattr(module, 'activation_zero_point'):
                quant_linear.register_buffer('activation_zero_point', module.activation_zero_point)
            if hasattr(module, 'weight_scale'):
                quant_linear.register_buffer('weight_scale', module.weight_scale)
            setattr(model, name, quant_linear)
        else:
            # Recursively apply to child modules
            replace_linear_with_quantized_linear(module)

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

    # layers_to_visualize = [
    #     'decoder.block.4.layer.2.DenseReluDense.wi_0',
    #     'decoder.block.4.layer.2.DenseReluDense.wi_1',
    #     'decoder.block.4.layer.2.DenseReluDense.wo'
    # ]
    # # visualize_activations(model, calibration_dataset, layers_to_visualize)

    # Calibrate model and quantize weights
    activation_stats = calibrate_model(model, calibration_dataset, device)
    compute_activation_quantization_parameters(model, activation_stats, device)
    quantize_model_weights(model)

    # Replace Linear layers with QuantizedLinear layers to implement activation quantization
    replace_linear_with_quantized_linear(model)

    # The model is now quantized and ready for inference
    # Example inference
    model.eval()
    input_text = "Summarize the following dialogue: #Person1#: Excuse me, did you see a set of keys? #Person2#: What kind of keys? #Person1#: Five keys and a small foot ornament. #Person2#: What a shame! I didn't see them. #Person1#: Well, can you help me look for it? That's my first time here. #Person2#: Sure. It's my pleasure. I'd like to help you look for the missing keys. #Person1#: It's very kind of you. #Person2#: It's not a big deal.Hey, I found them. #Person1#: Oh, thank God! I don't know how to thank you, guys. #Person2#: You're welcome."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    print("Generated Summary:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
