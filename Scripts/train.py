from model_network import *
from data_loader import DatasetLoader


##======================================================
# MainModelTrain
# Handles training of ModifiedMobileNetV2 on CIFAR-10 dataset.
# Supports configurable batch size, number of epochs, learning rate,
# width multiplier alpha for model scaling, logging, and model saving.
#======================================================

class MainModelTrain:
    """
    MainModelTrain Class
    Handles training, validation, evaluation, logging, model saving, and measuring model size 
    for the ModifiedMobileNetV2 model on CIFAR-10 dataset.

    Attributes:
        device (torch.device): Computes on GPU if available, else CPU.
        inference_device (torch.device): Fixed CPU device for inference.
        num_epochs (int): Number of training epochs.
        lr (float): Initial learning rate.
        batch_size (int): Mini-batch size.
        save_dir (str): Directory path for saving models and logs.
        log_file (str): CSV file path for training statistics logging.
        train_loader, val_loader, test_loader (DataLoader): Data loaders for respective datasets.
        model (nn.Module): MobileNetV2-based classification model.
        criterion: CrossEntropyLoss for classification.
        optimizer: SGD with momentum and L2 weight decay.
        scheduler: Learning rate scheduler reducing LR on plateau of validation loss.
    """
    def __init__(self, data_dir, batch_size=64, num_epochs=25, alpha=1, lr=0.01, log_file="training_log.txt", save_dir="media"):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        self.inference_device = torch.device('cpu')  # Inference on CPU
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.save_dir = os.path.expanduser(save_dir)
        self.save_dir_checker(self.save_dir)
        self.log_file = os.path.join(self.save_dir, "training_log.txt")
        # Dataset
        data_loader = DatasetLoader(data_dir, batch_size=self.batch_size, val_split=0.2)
        self.train_loader, self.val_loader, self.test_loader = data_loader.get_dataloaders()

        # Model
        self.model = ModifiedMobileNetV2(output_size=10, alpha=alpha).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # weight_decay = L2 regularization--> 4e-5(default for cifar10)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=4e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=2)

        # Initialize log file
        with open(self.log_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"])
    
    # @staticmethod
    def save_dir_checker(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)  


    # Per Epoch Training
    def train_epoch(self):
        """
        Executes training for one epoch over the entire train dataset.

        Returns:
            avg_training_loss (float): Average loss over all samples.
            training_accuracy (float): Training accuracy over the epoch.
        """
        self.model.train()
        training_loss = 0
        correct, total = 0, 0
        for images, labels in self.train_loader: 
            # print("training on device:", self.device)
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad() 
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward() 
            self.optimizer.step() 

            training_loss += loss.item() * images.size(0)  
            _, predicted = outputs.max(1) 
            total += labels.size(0) 
            correct += predicted.eq(labels).sum().item() 

        return training_loss / total, correct / total 

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Evaluate model performance on provided DataLoader (validation or test).

        Args:
            loader (DataLoader): DataLoader for evaluation dataset.

        Returns:
            avg_loss (float): Average loss on evaluation data.
            accuracy (float): Accuracy metric over evaluation data.
        """
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        for images, labels in loader:
            # print("evaluating(val dataset) on device:", self.device)
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return total_loss / total, correct / total     

    # MODEL SIZE IN MB[] at for now removed after checked
    def measure_model_size(self):
        """
        Saves the model to disk to measure model size in megabytes.

        Returns:
            size_mb (float): Size of saved model file in megabytes.
        """
           
        torch.save(self.model.state_dict(), "temp.pth")
        size_mb = os.path.getsize("temp.pth") / 1e6
        os.remove("temp.pth")
        return size_mb
    

    def save_model(self, filename="final_model.pth"):
        """
        Saves the trained model’s state_dict to a models directory.

        Args:
            filename (str): Name of the file to save the model.

        Returns:
            str: Path to the saved model file.
        """
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        os.makedirs(models_dir, exist_ok=True)

        save_path = os.path.join(models_dir, filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")
        return save_path

    def train(self):
        """
        Main training loop over num_epochs.

        Logs training and validation losses and accuracies file,
        adjusts learning rate on validation loss plateaus,
        and saves the final trained model at the end.
        """
        training_start_time = time.time()
        iteration = 0
        for epoch in range(self.num_epochs):
            if epoch % 5 == 0:
                print(f"Starting epoch {epoch+1}/{self.num_epochs}...")
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader)
            # Maybe could be tested for test_loader as well but not ideal
            self.scheduler.step(val_loss) 
            current_lr = self.optimizer.param_groups[0]['lr']
            end_time = time.time()
            # print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

            # Save to txt
            with open(self.log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc, current_lr])
            if epoch %10 ==0:
                print(f"Epoch {epoch+1}/{self.num_epochs} | "
                                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

            iteration += 1
        
        # torch.save(self.model.state_dict(), "models/final_model.pth")
        self.save_model(filename="final_model.pth") 
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        print(f"Training Done with: {total_training_time/60:.2f} minutes")

        """Size and params for the model """
        size_mb = self.measure_model_size()
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {num_params} and Model size (MB): {size_mb:.2f}")

        # Summary to log file
        with open(self.log_file, "a") as f:
            f.write(f"\nModel size (MB): {size_mb:.2f}\n")
            f.write(f"Total training time (minutes): {total_training_time/60:.2f} for total epochs {self.num_epochs}\n")
            f.write(f"Total parameters: {num_params}\n")


# ===============================
# Run[]
# ===============================
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_dir = config.get("data_dir")
    media_log_dir = config.get("media_log_dir")
    trainer = MainModelTrain(data_dir=data_dir, batch_size=config.get("batch_size"), num_epochs=config.get("num_epochs"), alpha=config.get("alpha"), log_file=config.get("model_log"), save_dir=media_log_dir)
    trainer.train()
