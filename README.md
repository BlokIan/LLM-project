# LLM-project
 
## How to run
Python 3.10 was used for this project
The following files can be run as follows after having installed all packages (requirements.txt), please note that the awq library requires a gpu:
- full_fine_tuning_code.py - Fully fine-tunes the base flan model, can be executed from terminal.
- model_test.py - A file to test base models and their fine-tuned versions, requires some terminal input when executed from terminal.
- peft_fine_tuning_code.py - Parameter-efficient fine-tuning for the base flan model, can be executed from terminal.
- quantized_model_test.py - A file which quantizes a model using per-channel scaling (per the AWQ methodology) and tests it, requires some terminal inputs.

For more information, please see our report.
