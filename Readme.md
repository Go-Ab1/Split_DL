# Deep Learning  for Embedded Device Deployment



## Overview


This repository's focus is development, training, evaluation, and deployment of deep learning models optimized for embedded devices. 

  Model Training: Implementation of lightweight architectures MobileNetV2 modified to small size of inputs for classification tasks, with support for CPU/GPU training.

  Testing & Evaluation: CPU based comprehensive evaluation of model performance using accuracy metrics, confusion matrices, classification predictions, and inference latency measurements.

  Post-Training Quantization: Techniques to reduce model size and improve inference speed on resource-constrained devices without significant loss in accuracy.

  <!-- Deployment-ready Models: Export of models in TorchScript format for easy deployment on embedded systems. -->

  Dataset: CIFAR-10 Based

This project is intended for deploying efficient models on low-resource platforms such as Raspberry Pi, NVIDIA Jetson, or other embedded hardware.


## Code Structure

```bash
Scripts/
│
├── main.py                 # Entry point (train, test, quantize)
├── train.py                # Training module
├── test.py                 # Evaluation and testing
├── quantize_model.py       # Quantization pipeline
├── data_loader.py          # Dataset loading utilities
├── model_network.py        # Model architecture (MobileNetV2)
├── helpers/                # utils
│   ├── global_import.py
│   └── quantize_helper.py

```



## Installation

Follow the steps below to set up the required Python environment and install the dependencies.

```bash
cd /path/to/this/project

# Recommended
    python3 -m venv pytorch_env
    # ubuntu
    source pytorch_env/bin/activate
    # Windows CMD
    pytorch_env\Scripts\activate.bat
    # Windows PowerShell
    pytorch_env\Scripts\Activate.ps1

pip install -r requirements.txt

```





## Usage

### Run Tasks Using main

You can run training, testing, and post-training quantization (PTQ) tasks using the `main.py` script. Tasks can be executed individually or sequentially by passing arguments to the script:
`
To run a single task, provide the task name as an argument.

To run multiple tasks, provide a space-separated list of task names.

```bash
# training
python3 main.py train

# testing
python3 main.py test

# quantization
python3 main.py quantize

# training -> testing
python3 main.py train test

# training -> quantization
python3 main.py train quantize
```



# Configuration Parameters (config.yaml)

The following parameters can be modified to adjust as required.
```yaml
data_dir: Dataset
batch_size: 64
n_calib_batch: 32
num_epochs: 25
learning_rate: 0.01
alpha: 1.0
media_log_dir: media
model_log: training_log.txt
test_log: test_log.txt
val_split: 0.1
num_workers: 4
visualize_losses: true
visualize_accuracies: true
models_dir: models
compare_models: true
comparison_log_name: comparison_log.txt
trained_model_name: final_model.pth
quantized_model_name: quantized_model.pth
quantization: false 

```


# Results


`References:`

> 



<div align="right">
<b>@Nov, 2025!!!<br>
Goitom</b>
</div>
