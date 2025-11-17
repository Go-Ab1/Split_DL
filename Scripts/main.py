#!/usr/bin/env python3
from helpers.global_import import *
from helpers.quantize_helper import *
from train import MainModelTrain
from test import TestChecker
from quantize_model import CIFAR10Evaluator
from data_loader import DatasetLoader
from model_network import ModifiedMobileNetV2


# ================================
# Task Registry
# Registers and manages tasks globally.
# ================================
TASKS = {}

def task(name):
    """
    Registers and manages tasks
    Args:
        name (str): Name of the task to register.
    Returns:
        function: Decorator function to register the task.
    """
    def task_reg(func):
        TASKS[name.lower()] = func
        return func
    return task_reg


# ================================
# ModelPipeline
# Manages the overall model pipeline: training, testing, quantization.
# ================================
class ModelPipeline:
    """
    Manages the overall model pipeline: training, testing, quantization.
    """

    def __init__(self, config_path=None):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        self.config = self._load_config()
        self.data_dir = os.path.join(self.project_root, self.config["data_dir"])    
        self.media_dir = os.path.join(self.project_root, self.config["media_log_dir"])
        self.models_dir = os.path.join(self.project_root, self.config["models_dir"])


    def _load_config(self):
        """
        Load configuration from the YAML file.
        """
        if not os.path.exists(self.config_path):
            print("config not found!")
            sys.exit(1)
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    # ================================
    # TASK DEFINITIONS
    # ================================
    @task("train")
    def train(self):
        """
        Starts the training process.
        """
        print("Starting training...")
        trainer = MainModelTrain(
            data_dir=self.data_dir,
            batch_size=self.config.get("batch_size"),
            num_epochs=self.config.get("num_epochs"),
            val_split=self.config.get("val_split"),
            num_workers=self.config.get("num_workers"),
            alpha=self.config.get("alpha"),
            log_file=self.config.get("model_log"),
            save_dir=self.media_dir,
            lr=self.config.get("learning_rate"),
        )
        trainer.train()
        print("Training completed.\n")

    @task("test")
    def test(self):
        """
        Starts the testing process.
        """
        print("Starting testing...")
        model_path = os.path.join(self.models_dir, self.config.get("trained_model_name"))


        loader = DatasetLoader(self.data_dir, batch_size=self.config.get("batch_size"))
        _, _, test_loader = loader.get_dataloaders()
        class_names = [str(i) for i in range(10)]

        # Model 
        model = ModifiedMobileNetV2(output_size=10, alpha=self.config.get("alpha"))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        tester = TestChecker(
            model,
            test_loader,
            device="cpu",
            class_names=class_names,
            save_dir=self.media_dir,
            log_file=self.config.get("test_log"),

        )
        latency_batch, latency_image = tester.measure_inference_latency(num_runs=self.config.get("num_runs_latency"), 
                                                                        batch_size=self.config.get("batch_size"))
        acc, labels, preds = tester.evaluate()
        print(f"Test Accuracy: {acc:.4f}")

        report = tester.classification_report(labels, preds)
        cm = tester.compute_confusion_matrix(labels, preds)
        tester.save_results(acc, report, cm, latency_batch=latency_batch, latency_image=latency_image)
        tester.visualize_samples(num_samples=30)
        print("Testing completed.\n")


    @task("quantize")
    @task("quant")
    def quantize(self):
        """
        Starts the quantization process and performs a full comparison.
        """
        print("Starting Quantization + Full Comparison...")

        evaluator = CIFAR10Evaluator(
            data_dir=self.data_dir,
            compare=self.config.get("compare_models"),
            backend=self.config.get("quant_backend"),
            batch_size=self.config.get("batch_size")
        )


        logger = QuantizationLogger(
                        base_dir=self.config.get("media_log_dir"),
                        log_name=self.config.get("comparison_log_name"),
                        clear_log=True,
                        root_dir=self.project_root   
        )
        

        logger.log_to_file(f"Backend: {evaluator.backend}")
        logger.log_to_file(f"Device: {evaluator.device}\n")

        # ===========================
        # FULL PRECISION MODEL EVAL
        # ===========================
        print("Evaluating full-precision model...")
        fp_model_path = os.path.join(self.models_dir, self.config.get("trained_model_name"))
        evaluator.load_state_dict(fp_model_path)

        fp_acc, _, _ = evaluator.evaluate()
        fp_size = evaluator.measure_model_size()
        fp_lat_batch, fp_lat_img = evaluator.measure_inference_latency(
            num_runs=self.config.get("num_runs_latency"), batch_size=self.config.get("batch_size")
        )

        logger.log_section("Full-Precision Model", {
            "Accuracy": fp_acc,
            "Model Size (MB)": fp_size,
            "Latency (s per batch)": fp_lat_batch,
            "Latency (s per image)": fp_lat_img
        })

        # ===========================
        # RUN PTQ
        # ===========================
        print("\nRunning PTQ...")

        evaluator.run_ptq(
                    model_file=os.path.join(self.models_dir, self.config.get("trained_model_name")),
                    save_name=os.path.join(self.models_dir, self.config.get("quantized_model_name"))
                    )
        # ===========================
        # QUANTIZED MODEL EVAL
        # ===========================
        ptq_acc, _, _ = evaluator.evaluate()
        ptq_size = evaluator.measure_model_size()
        ptq_lat_batch, ptq_lat_img = evaluator.measure_inference_latency(
            num_runs=self.config.get("num_runs_latency"), batch_size=self.config.get("batch_size")
        )

        logger.log_section("Quantized (PTQ) Model", {
            "Accuracy": ptq_acc,
            "Model Size (MB)": ptq_size,
            "Latency (s per batch)": ptq_lat_batch,
            "Latency (s per image)": ptq_lat_img
        })

        # ===========================
        # COMPARISON SUMMARY
        # ===========================
        if evaluator.compare:
            print("\n--- Comparison: Full-precision vs Quantized ---")
            logger.log_to_file("========== Comparison Summary ==========")
            logger.compare_and_log("Accuracy", fp_acc, ptq_acc)
            logger.compare_and_log("Model Size (MB)", fp_size, ptq_size)
            logger.compare_and_log("Latency (s per batch)", fp_lat_batch, ptq_lat_batch)
            logger.compare_and_log("Latency (s per image)", fp_lat_img, ptq_lat_img)

        logger.finalize()
        print("Quantization + Comparison completed.\n")

    # ================================
    # TASK DISPATCHER
    # ================================
    def run(self, args):
        if not args:
            print("\nUsage ---> Please Specify a Task:")
            print("\nUse Either args:")
            for name in TASKS:
                print(f"  python3 Scripts/main.py {name}")
            print("python3 Scripts/main.py train test")
            print("python3 Scripts/main.py train quant\n")
            sys.exit(0)

        for arg in args:
            task_func = TASKS.get(arg.lower())
            if task_func:
                task_func(self)
            else:
                print(f"Unknown arg ====> use \"\\train\" or \"test\" or \"quant\": {arg}")


# ================================
# MAIN 
# ================================
def main():
    args = sys.argv[1:]
    pipeline = ModelPipeline()
    pipeline.run(args)


if __name__ == "__main__":
    main()
