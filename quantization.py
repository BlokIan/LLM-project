import torch, load_dataset
import torch.nn as nn
import torch.quantization
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPTQConfig, AwqConfig
from autoawq import AWQQuantizer, AWQConfig



# Load model and tokenizer
def load_model_and_tokenizer(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# Load and preprocess dataset, function from full_fine_tuning_code
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


# W8A8 Quantization Strategy
# not sure if this is the right way to apply W8A8 quantization
def apply_w8a8_quantization(model, quant_path_w8a8, tokenizer, tokenized_dataset):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('x86')  

    # prepare the model for quantization
    model_prepared = torch.quantization.prepare(model)
    inputs = tokenizer("This is a sample sentence.", return_tensors="pt")
    with torch.no_grad():
        model_prepared(**inputs)  # Calibration
    
    # convert the model to quantized version
    model_quantized = torch.quantization.convert(model_prepared)

    # Save the quantized model
    model.save_pretrained(quant_path_w8a8)
    tokenizer.save_pretrained(quant_path_w8a8)


# GPTQ Quantization Strategy
def apply_gptq_quantization(model_name, dataset_name, quant_path_gptq, tokenizer, tokenized_dataset):
    gptq_config = GPTQConfig(
        bits=4,
        group_size=128,
        dataset=dataset_name,
        desc_act=False,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=gptq_config)
    
    # Save quantized model
    model.save_quantized(quant_path_gptq)
    tokenizer.save_pretrained(quant_path_gptq)

    # Load the quantized model for inference
    loaded_quantized_model = AutoModelForSeq2SeqLM.from_pretrained(quant_path_gptq)

    # Use input from the dataset
    sample = tokenized_dataset["train"][0]  # Get the first sample from the training set
    input_text = "Summarize the following dialogue: " + sample["dialogue"]
    inputs = tokenizer(input_text, return_tensors="pt").to("cpu") # could be GPU i guess 

    # Generate output using the quantized model
    output = loaded_quantized_model.generate(**inputs, max_new_tokens=5)
    decoded_output = tokenizer.decode(output[0])

    print("Generated output:", decoded_output)

    #return model


# AWQ Quantization Strategy
def apply_awq_quantization(model, quant_path_awq, tokenizer, tokenized_dataset):
    # modify the config file so that it is compatible with transformers integration
    awq_config = AwqConfig(
        bits=4,
        group_size=128,
        zero_point=True,
        version="GEMM", # not sure why this is needed
    ).to_dict()

    # Quantize the model
    model.quantize(tokenizer, quant_config=awq_config)
    model.save_quantized(quant_path_awq)
    tokenizer.save_pretrained(quant_path_awq)

    # Load the quantized model for inference
    loaded_quantized_model = AutoModelForSeq2SeqLM.from_pretrained(quant_path_awq)
                                                                   
    # Use input from the dataset
    sample = tokenized_dataset["train"][0]  # Get the first sample from the training set
    input_text = "Summarize the following dialogue: " + sample["dialogue"]
    inputs = tokenizer(input_text, return_tensors="pt").to("cpu")  # could be GPU i guess 

    # Generate output using the quantized model
    output = loaded_quantized_model.generate(**inputs, max_new_tokens=5)
    decoded_output = tokenizer.decode(output[0])
    print("Generated output:", decoded_output)
    
    #return model


# Main function
# have not tested anything, some thing wil not run in cpu can be fixed, do not know how yet. 
def main():
    # Dataset and model names
    model_name = "google/flan-t5-base"
    data_set = "knkarthick/dialogsum"
    quant_path_gptq = "./quantized_gptq_model"  # Path to save the quantized model

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    tokenized_dataset = load_and_preprocess_dataset(data_set, tokenizer)

    # Quantization Methods
    model_awq = apply_awq_quantization(model, "./awq_model", tokenizer, tokenized_dataset)
    model_gptq = apply_gptq_quantization(model_name, data_set, "./gptq_model",tokenizer, tokenized_dataset)
    model_W8A8 = apply_w8a8_quantization(model_name, data_set, "./w8a8_model",tokenizer, tokenized_dataset)


if __name__ == "__main__":
    main()
