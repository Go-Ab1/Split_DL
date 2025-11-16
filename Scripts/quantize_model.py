from helpers.global_import import *
from helpers.quantize_helper import *
from data_loader import DatasetLoader
from model_network import ModifiedMobileNetV2
from test import TestChecker

# -------------------------------
# Compression / Quantization Manager
# -------------------------------
class CompressionManager:
    def __init__(self, model, backend="fbgemm"):
        """
        Manages model compression and quantization processes.
         Args:
            model (nn.Module): The model to be quantized.
            backend (str): Quantization backend, either 'qnnpack' or 'fbgemm'.
        """

        self.model = model
        self.backend = backend  # qnnpack or fbgemm
        torch.backends.quantized.engine = backend

    @staticmethod
    def relu_changer(module):
        """
        Changes ReLU6 activations to ReLU
        quantization compatibility
        Inplace = True: Inputs are modified 
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

# ------------------------------- 
# CIFAR-10 Evaluation / Workflow
# -------------------------------
class CIFAR10Evaluator:
    def __init__(self, data_dir, model_dir="models", batch_size=64, n_calib_batch=64, backend="fbgemm" , compare = True, seed=1000): 
        # self.data_dir = os.path.expanduser(data_dir)
        self.compare = compare

        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir))

        os.makedirs(self.model_dir, exist_ok=True)


        self.batch_size = batch_size
        self.n_calib_batch = n_calib_batch
        self.backend = backend
        self.seed = seed
        self.device = torch.device("cpu") 

        # Dataset
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
        state_dict = torch.load(model_file, map_location="cpu")
        self.model.load_state_dict(state_dict)
        return state_dict



    def run_ptq(self, model_file, save_name):
        state_dict = self.load_state_dict(model_file)
        self.model = self.quant_manager.apply_ptq(state_dict, self.calib_loader, self.example_inputs, self.n_calib_batch)
        self.save_scripted_model(save_name)

    def evaluate(self):
        tester = TestChecker(self.model, self.test_loader, device=self.device)
        acc, labels, preds = tester.evaluate()
        print(f"Test Accuracy: {acc:.4f}")
        return acc, labels, preds

    def measure_model_size(self):
        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(self.model.cpu()), buffer)
        size_mb = buffer.getbuffer().nbytes / 1e6
        print(f"Model size: {size_mb:.2f} MB")
        return size_mb    

    def measure_inference_latency(self, num_runs=5, batch_size=64):
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.jit.save(torch.jit.script(self.model.cpu()), path)
        print(f"Saved scripted model: {path}")

        

# --- - -- -----------------------
# Usage [Test...]
# ---------------------------------
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    evaluator = CIFAR10Evaluator(
        data_dir=config.get("data_dir"),
        compare=config.get("compare_models")
    )

    logger = QuantizationLogger(
        base_dir=os.path.join(os.path.dirname(__file__), "..", config.get("media_log_dir", "media")),
        log_name=config.get("comparison_log_name", "comparison_log.txt"),
        clear_log=True
    )

    logger.log_to_file(f"Backend: {evaluator.backend}")
    logger.log_to_file(f"Device: {evaluator.device}\n")

    # --- Full Precision Evaluation ---
    print("Evaluating full-precision model...")
    evaluator.load_state_dict(config.get("trained_model_name"))
    full_acc, _, _ = evaluator.evaluate()
    full_size = evaluator.measure_model_size()
    full_latency_batch, full_latency_image = evaluator.measure_inference_latency(num_runs=5, batch_size=64)

    logger.log_section("Full-Precision Model", {
        "Accuracy": full_acc,
        "Model Size (MB)": full_size,
        "Latency (s per batch)": full_latency_batch,
        "Latency (s per image)": full_latency_image
    })

    # --- Run PTQ ---
    print("\nRunning PTQ...")
    evaluator.run_ptq(
        model_file=config.get("trained_model_name"),
        save_name=config.get("quantized_model_name")
    )

    ptq_acc, _, _ = evaluator.evaluate()
    ptq_size = evaluator.measure_model_size()
    ptq_latency_batch, ptq_latency_image = evaluator.measure_inference_latency(num_runs=5, batch_size=64)

    logger.log_section("Quantized (PTQ) Model", {
        "Accuracy": ptq_acc,
        "Model Size (MB)": ptq_size,
        "Latency (s per batch)": ptq_latency_batch ,
        "Latency (s per image)": ptq_latency_image
    })

    # --- Comparison ---
    if evaluator.compare:
        print("\n--- Comparison: Full-precision vs PTQ ---")
        logger.log_to_file("========== Comparison Summary ==========")
        logger.compare_and_log("Accuracy", full_acc, ptq_acc)
        logger.compare_and_log("Model Size (MB)", full_size, ptq_size)
        logger.compare_and_log("Latency (s)", full_latency_batch, ptq_latency_batch)
        logger.compare_and_log("Latency (s per image)", full_latency_image, ptq_latency_image)

    logger.finalize()
