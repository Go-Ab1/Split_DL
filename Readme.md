## Overview
This repo focuses on the **development, training, evaluation, and deployment evaluation of lightweight deep learning models** optimized for **resource-constrained embedded devices**.  

Key features:

- **Model Training**: Modified MobileNetV2 architecture for small input sizes
- **Testing & Evaluation**: CPU-based evaluation using accuracy, confusion matrices, classification predictions, and inference latency.  
- **Post-Training Quantization**: Reduces model size and improves inference speed on embedded devices with minimal accuracy loss.  
- **Dataset**: CIFAR-10 
---

## Code Structure
```bash
Scripts/
│
├── main.py                 # Entry point for training, testing, quantization[or combined via argument at run time]
├── train.py                # Model training module
├── test.py                 # Evaluation and testing 
├── quantize_model.py       # Post-training quantization pipeline
├── data_loader.py          # Dataset loading 
├── model_network.py        # Model architecture
├── plot_log.py             # Model architecture
├── helpers/                # utils functions
│   ├── global_import.py    # packages
│   └── quantize_helper.py  
```
## Installation

Follow the steps below to set up the required Python environment and install the dependencies.
1. Navigate to the project directory:
```bash
cd /path/to/this/project
```
2. Create and activate a Python virtual environment[Recommended]
```bash
 python3 -m venv pytorch_env
 #Ubuntu
 source pytorch_env/bin/activate 
```
3. Install dependencies:
```bash
# The packages listed are used for testing
# Adjust the PyTorch installation to match your CUDA version.
pip install -r requirements.txt
```
## Usage

Once the dependencies are installed, run the scripts to reproduce the experiments.

```bash 
# ==========================================
#               PROJECT USAGE
#   (Training, Testing, Quantization )
# ==========================================

# ------------------------------------------
# 1. TRAINING
# ------------------------------------------
# Trains 
# - Downloads dataset on first run
# - Logs training/validation loss & accuracy
# - Saves training log: media/
# - Saves trained model: models/

python3 Scripts/main.py train

# Visualize logs after training:
# - Plots loss curves
# - Plots accuracy curves
python3 Scripts/plot_log.py

# ------------------------------------------
# 2. TESTING
# ------------------------------------------
# Evaluates the trained model
# Produces:
# - Overall accuracy
# - Classification
# - Confusion matrix
# - Inference latency (batch & per-image)
python3 Scripts/main.py test

# ------------------------------------------
# 3. POST-TRAINING QUANTIZATION (PTQ)
# ------------------------------------------
# Runs quantization on the trained model.
# Produces:
# - Quantized model
# - Model size comparison
# - Latency comparison (FP32 vs INT8)
# - Accuracy comparison

python3 Scripts/main.py quantize

# ------------------------------------------
# 4. CHAINED WORKFLOWS
# ------------------------------------------

# Train → Test
# (Complete pipeline for performance evaluation)
python3 Scripts/main.py train test

# OR

# Train → Quantize
# (Complete pipeline for optimizing deployed models)
python3 Scripts/main.py train quantize
```


## Configuration Parameters
All parameters can be modified in ```config.yaml``` and can be adjusted as required.
```yaml
data_dir: Dataset
batch_size: 64
num_epochs: 100
learning_rate: 0.01
alpha: 1.0
media_log_dir: media
model_log: training_log.txt
test_log: test_log.txt
val_split: 0.1
num_workers: 4
models_dir: models
compare_models: true
quant_backend: fbgemm
num_runs_latency: 5
comparison_log_name: comparison_log.txt
trained_model_name: final_model.pth
quantized_model_name: quantized_model.pth
```
# Results

## Training Curve (Accuracy & Loss)

The training and validation performance of the model is shown below.

<table>
<tr>
<td align="center">
  <b>Training/Validation Accuracy</b><br>
  <img src="eval_result_100/accuracy_vs_epoch_100.png" width="800" alt="Training Accuracy">
</td>
<td align="center">
  <b>Training/Validation Loss</b><br>
  <img src="eval_result_100/loss_vs_epoch_100.png" width="800" alt="Training Loss">
</td>
</tr>
</table>

</div>

## Sample Predictions & Confusion Matrix

Below are example predictions and the confusion matrix for CIFAR-10 test images.

<table>
<tr>
<td align="left">
  <b>Sample Predictions</b><br>
  <img src="eval_result_100/sample_prediction_100.png" width="800" alt="Sample Predictions">
</td>
<td align="right">
  <b>Confusion Matrix</b><br>
  <img src="eval_result_100/confusion_100.png" width="800" alt="Confusion Matrix">
</td>
</tr>
</table>



## Model Comparison Summary After Quantization

| Metric                | Full-Precision Model | Quantized (PTQ) Model | Change |
|----------------------|----------------------|------------------------|--------|
| **Accuracy**         | 0.9139               | 0.9125                 | ↓ 0.15% |
| **Model Size (MB)**  | 9.2820               | 2.6434                 | ↓ 71.52% |
| **Latency / Batch(32) (s)** | 0.5283            | 0.2276                 | ↓ 56.92% |
| **Latency / Image (s)** | 0.0165            | 0.0071                 | ↓ 56.92% |



<div align="right">
<b>Nov, 2025!!!<br>
@Goitom</b>
</div>
