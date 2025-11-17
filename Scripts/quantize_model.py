from helpers.global_import import *
from helpers.quantize_helper import *
from data_loader import DatasetLoader
from model_network import ModifiedMobileNetV2
from test import TestChecker

# ================================
# CompressionManager
# Manages model compression and quantization processes.
# ================================
class CompressionManager:
    """
    Manages model quantization processes.
        Args:
        model (nn.Module): The model to be quantized.
        backend (str): Quantization backend, either 'qnnpack(ARM)' or 'fbgemm(x86)'.
    """
    def __init__(self, model, backend="fbgemm"):
      
        self.model = model
        self.backend = backend  
        torch.backends.quantized.engine = backend

    @staticmethod
    def relu_changer(module):
        """
        Changes ReLU6 activations to ReLU
        quantization compatibility
        Inplace = True: Inputs are modified in place
        Inplace = False: Inputs are not modified in place
        """
        for name, mod in module.named_children():
            CompressionManager.relu_changer(mod)
            if isinstance(mod, (nn.ReLU, nn.ReLU6)):
                module._modules[name] = nn.ReLU(inplace=False)

    @staticmethod
    def calibrate(model, dataloader, n_batches=64):
        """
        Calibration process for Post-Training Quantization (PTQ).
        To determine qunantization parameters [scale, zero-point].
        From FP32 to INT8
        Args:
            model (nn.Module): The quantized model to be calibrated.
            dataloader (DataLoader): DataLoader for calibration data.
            n_batches (int): Number of batches to use for calibration.
        """
        model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                inputs = inputs.to(torch.device("cpu"))
                model(inputs)
                if i + 1 >= n_batches:
                    break

    def apply_ptq(self, state_dict, calib_loader, example_inputs, n_calib_batch=64):
        """
        Applies Post-Training Quantization (PTQ)
        Torch FX-based quantization workflow.
        Args:
            state_dict (dict): State dictionary of the pre-trained model.
            calib_loader (DataLoader): DataLoader for calibration data.
            example_inputs (tuple): Example inputs for model tracing.
            n_calib_batch (int): Number of batches to use for calibration.
        Returns:
            nn.Module: The quantized model.

        """
        self.model.load_state_dict(state_dict)
        self.relu_changer(self.model)
        qconfig = {"": get_default_qconfig(self.backend)}
        self.model = quantize_fx.prepare_fx(self.model.eval(), qconfig, example_inputs)
        self.calibrate(self.model, calib_loader, n_calib_batch)
        self.model = quantize_fx.convert_fx(self.model.eval())
        return self.model

# ================================
# Evaluates model performance, size, and latency. 
# ================================
class CIFAR10Evaluator:
    """
    Evaluator .
        Args:
        data_dir (str): Directory containing the CIFAR-10 dataset.
        model_dir (str): Directory to save/load models.
        batch_size (int): Batch size for data loading.
        n_calib_batch (int): Number of batches for calibration.
        backend (str): Quantization backend, either 'qnnpack(ARM)' or 'fbgemm(x86)'.
        compare (bool): Whether to compare results (quantized vs. original).
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, data_dir, model_dir="models", batch_size=64, n_calib_batch=64, backend="fbgemm" , compare = True, seed=1000): 
       
        self.compare = compare
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir))
        os.makedirs(self.model_dir, exist_ok=True)
        self.batch_size = batch_size
        self.n_calib_batch = n_calib_batch
        self.backend = backend
        self.seed = seed
        self.device = torch.device("cpu") 

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
        ])
     
        loader = DatasetLoader(self.data_dir, batch_size=self.batch_size)
        _, _, self.test_loader = loader.get_dataloaders()

        calib_dataset = CIFAR10(root=self.data_dir, train=True, download=True, transform=transform)
        self.calib_loader = DataLoader(calib_dataset, batch_size=self.batch_size, shuffle=False)

        # Model
        self.model = ModifiedMobileNetV2(output_size=10).to(self.device)
        self.quant_manager = CompressionManager(self.model, backend=self.backend)
        self.example_inputs = (torch.randn(1, 3, 32, 64),) 
  

    def load_state_dict(self, model_file):
        """
        Loads the model state dictionary from a file.
        Args:
            model_file (str): Path to the model file.
        Returns:
            dict: The loaded state dictionary.
        """
        state_dict = torch.load(model_file, map_location="cpu")
        self.model.load_state_dict(state_dict)
        return state_dict



    def run_ptq(self, model_file, save_name):
        """
        Runs Post-Training Quantization (PTQ) on the model.
        Args:
            model_file (str): Path to the pre-trained model file.
            save_name (str): Path to save the quantized model.
        """
        state_dict = self.load_state_dict(model_file)
        self.model = self.quant_manager.apply_ptq(state_dict, self.calib_loader, self.example_inputs, self.n_calib_batch)
        self.save_scripted_model(save_name)

    def evaluate(self):
        """
        Evaluates the model on the test dataset.
        Returns:
            tuple: Accuracy, true labels, and predicted labels.
        """

        tester = TestChecker(self.model, self.test_loader, device=self.device)
        acc, labels, preds = tester.evaluate()
        print(f"Test Accuracy: {acc:.4f}")
        return acc, labels, preds

    def measure_model_size(self):
        """
        Measures the size of the model in megabytes.
        
        Returns:
            float: Model size in MB.
        """
        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(self.model.cpu()), buffer)
        size_mb = buffer.getbuffer().nbytes / 1e6
        print(f"Model size: {size_mb:.2f} MB")
        return size_mb    

    def measure_inference_latency(self, num_runs=5, batch_size=64):
        """
        Measures the average inference latency of the model on the CPU.
        Args:
            num_runs (int): Number of runs to average latency over.
            batch_size (int): Number of images per batch.
        Returns:
            tuple: Average latency per batch and per image in seconds.
        """
        self.model.to(self.device).eval()
        images, _ = next(iter(self.test_loader))
        images = images[:batch_size]
        images = images.to(self.device)
        
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(images)

        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(images)
        end = time.time()

        avg_latency_batch = (end - start) / num_runs 
        print(f"Avg CPU latency (batch={images.size(0)}): {avg_latency_batch:.4f} s")
        latency_per_image = avg_latency_batch / images.size(0)
        print(f"Avg CPU latency per image: {latency_per_image:.4f} s")
        return avg_latency_batch, latency_per_image

    def save_scripted_model(self, path):
        """
        Saves the scripted model to the specified path.
        Args:
            path (str): Path to save the scripted model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.jit.save(torch.jit.script(self.model.cpu()), path)
        print(f"Saved scripted model: {path}")
