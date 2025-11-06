#!/usr/bin/env python3
from helpers.global_import import *
from train import MainModelTrain
from test import TestChecker
from quantize_model import CIFAR10Evaluator
from data_loader import DatasetLoader
from model_network import ModifiedMobileNetV2


# -------------------------
# GLOBAL TASK REGISTRY 
# -------------------------
TASKS = {}

def task(name):
    """Register tasks globally"""
    def task_reg(func):
        TASKS[name.lower()] = func
        return func
    return task_reg

class ModelPipeline:
    """Encapsulates all training, testing, and quantization tasks."""

    def __init__(self, config_path=None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        self.config = self._load_config()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _load_config(self):
        if not os.path.exists(self.config_path):
            print("config not found!")
            sys.exit(1)
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    # -------------------------
    # TASK DEFINITIONS
    # -------------------------
    @task("train")
    def train(self):
        print("Starting training...")
        trainer = MainModelTrain(
            data_dir=self.config.get("data_dir"),
            batch_size=self.config.get("batch_size"),
            num_epochs=self.config.get("num_epochs"),
            alpha=self.config.get("alpha"),
            log_file=self.config.get("model_log"),
            save_dir=self.config.get("media_log_dir"),
            lr=self.config.get("learning_rate"),
        )
        trainer.train()
        print("Training completed.\n")

    @task("test")
    def test(self):
        print("Starting testing...")
        models_dir = os.path.join(self.project_root, self.config.get("models_dir", "models"))
        model_path = os.path.join(models_dir, self.config.get("trained_model_name", "final_model.pth"))

        # Dataset
        loader = DatasetLoader(self.config.get("data_dir"), batch_size=self.config.get("batch_size"))
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
            save_dir=self.config.get("media_log_dir", "media"),
            log_file=self.config.get("test_log", "test_log.txt"),
        )

        latency_batch, latency_image = tester.measure_inference_latency(num_runs=5, batch_size=32)
        acc, labels, preds = tester.evaluate()
        print(f"Test Accuracy: {acc:.4f}")

        report = tester.classification_report(labels, preds)
        cm = tester.compute_confusion_matrix(labels, preds)
        tester.save_results(acc, report, cm, latency_batch=latency_batch, latency_image=latency_image)
        tester.visualize_samples(num_samples=30)
        print("Testing completed.\n")

    @task("quantize")
    @task("quant")  # alias
    def quantize(self):
        print("Starting Post-Training Quantization (PTQ)...")
        evaluator = CIFAR10Evaluator(
            data_dir=self.config.get("data_dir"),
            compare=self.config.get("compare_models", True),
            batch_size=self.config.get("batch_size"),
        )
        evaluator.run_ptq(
            model_file=self.config.get("trained_model_name", "final_model.pth"),
            save_name=self.config.get("quantized_model_name", "quantized_model.pth"),
        )
        print("Quantization completed.\n")

    # -------------------------
    # TASK DISPATCHER
    # -------------------------
    def run(self, args):
        if not args:
            print("\nUsage:")
            for name in TASKS:
                print(f"  python3 main.py {name}")
            print("  python3 main.py train test")
            print("  python3 main.py train quantize\n")
            sys.exit(0)

        for arg in args:
            task_func = TASKS.get(arg.lower())
            if task_func:
                task_func(self)
            else:
                print(f"Unknown command: {arg}")


def main():
    args = sys.argv[1:]
    pipeline = ModelPipeline()
    pipeline.run(args)


if __name__ == "__main__":
    main()
