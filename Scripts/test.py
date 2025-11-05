from helpers.global_import import *
from model_network import ModifiedMobileNetV2
from data_loader import DatasetLoader

class TestChecker:
    def __init__(self, model , test_loader: DataLoader, device="cpu", class_names=None, save_dir="media", log_file="test_log.txt"):
        """
        Args:
            model: Trained PyTorch model
            test_loader: DataLoader for test dataset
            device: 'cpu' 

            log_file: file to save test metrics
        """
        self.test_loader = test_loader
        self.device = device
        self.model = model.to(self.device)   
        self.class_names = class_names
        # self.criterion = nn.CrossEntropyLoss()
        self.save_dir = save_dir
        self.log_file = log_file
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad() 
    # @torch.inference_mode()
    def evaluate(self):
        self.model.eval()
        # total_loss = 0
        correct, total = 0, 0
        all_preds, all_labels = [], []

        for images, labels in self.test_loader:
            # print("Testing on device:", self.device)
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            # loss = self.criterion(outputs, labels)
            # total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            # Cpu: for NP, SKLearn, PLT[ ONLY WORK ON CPU]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # avg_loss = total_loss / total
        accuracy = correct / total
        return  accuracy, np.array(all_labels), np.array(all_preds)

    def compute_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
        return cm

    def classification_report(self, labels, preds):
        report = classification_report(labels, preds, target_names=self.class_names)
        print(report)
        return report
    
    def measure_inference_latency(self, num_runs=5, batch_size=32):
        """
        Measure average inference latency on CPU.
        """
        # print(f"Measuring inference latency on device: {self.device}")
        self.model.eval().to(self.device)
        images, _ = next(iter(self.test_loader))
        images = images[:batch_size].to(self.device)

        # Warm-up runs
        # with torch.no_grad():
        with torch.inference_mode():
            for _ in range(10):
                _ = self.model(images)
        start = time.time()
        # with torch.no_grad():
        with torch.inference_mode():
            for _ in range(num_runs):
                _ = self.model(images)
        end = time.time()
        avg_latency_batch = (end - start) / num_runs 
        avg_latency_image = avg_latency_batch / images.size(0)
        print(f"Avg CPU inference latency(batch_size:{batch_size}): {avg_latency_batch:.4f} s")
        print(f"Avg CPU inference latency per image: {avg_latency_image:.4f} s")
        return avg_latency_batch, avg_latency_image


    def save_results(self, accuracy, report, cm, latency_batch=None, latency_image=None):
        path = os.path.join(self.save_dir, self.log_file)
        with open(path, "w") as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            if latency_batch is not None:
                f.write(f"Avg CPU Inference Latency: {latency_batch:.4f} s\n")
            if latency_image is not None:
                f.write(f"Avg CPU Inference Latency per image: {latency_image:.4f} s\n")
            f.write("\nClassification Report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm))
        print(f"Saved test results: {path}")

   
    def visualize_samples(self, num_samples=30, save_fig=True):
        class_labels = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
                        5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

        images, labels = next(iter(self.test_loader))
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self.model(images)
        _, preds = torch.max(outputs, 1)

        num_rows, num_cols = 5, 6
        plt.figure(figsize=(12, 10))

        for i in range(min(num_samples, num_rows * num_cols)):
            plt.subplot(num_rows, num_cols, i + 1)
            img = images[i].cpu().permute(1, 2, 0).numpy()
            plt.imshow((img * 0.247 + 0.491).clip(0, 1)) 
            pred_idx = int(preds[i].item())
            true_idx = int(labels[i].item())
            color = 'green' if pred_idx == true_idx else 'red'

            plt.title(f"P: {pred_idx} ({class_labels[pred_idx]})\n"
                    f"T: {true_idx} ({class_labels[true_idx]})",
                    fontsize=8, color=color)
            plt.axis('off')

        plt.tight_layout()
        if save_fig:
            fig_path = os.path.join(self.save_dir, "sample_predictions.png")
            plt.savefig(fig_path)
            print(f"Saved sample images: {fig_path}")
        plt.show()


# # ======================
# # Usage
# # ======================
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    
    data_dir = config.get("data_dir")
    media_log_dir = config.get("media_log_dir") 
    test_log_name = config.get("test_log")  

    models_dir = os.path.join(project_root, config.get("models_dir", "models"))
    model_path = os.path.join(models_dir, config.get("trained_model_name", "final_model.pth"))


    loader = DatasetLoader(data_dir, batch_size=64)
    _, _, test_loader = loader.get_dataloaders()
    class_names = [str(i) for i in range(10)]  

    model = ModifiedMobileNetV2(output_size=10, alpha=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    tester = TestChecker(model, test_loader, device="cpu", class_names=class_names, save_dir=media_log_dir, log_file=test_log_name)
    latency_batch, latency_image = tester.measure_inference_latency(num_runs=5, batch_size=32)

    acc, labels, preds = tester.evaluate()
    print(f"Test Accuracy: {acc:.4f}")
    report = tester.classification_report(labels, preds)
    cm = tester.compute_confusion_matrix(labels, preds)
    tester.save_results(acc, report, cm, latency_batch=latency_batch, latency_image=latency_image)
    tester.visualize_samples(num_samples=30)
