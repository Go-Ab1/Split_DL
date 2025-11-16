import pandas as pd
import matplotlib.pyplot as plt

class LogPlotter:
    """
    TXT file plotter 
    """
    def __init__(self, log_path):
        self.log_path = log_path
        self.data = None

    def load_log(self):
        self.data = pd.read_csv(self.log_path)
        self.data = self.data[pd.to_numeric(self.data['epoch'], errors='coerce').notnull()]
        self.data['epoch'] = self.data['epoch'].astype(int)

        expected_cols = {
            'epoch': int,
            'train_loss': float,
            'val_loss': float,
            'train_acc': float,
            'val_acc': float,
            'lr': float
        }
        for col, dtype in expected_cols.items():
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(dtype)

    # ----------------------------------------------------------
    # NEW: Separate loss plot
    # ----------------------------------------------------------
    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['epoch'], self.data['train_loss'], label="Train Loss")
        plt.plot(self.data['epoch'], self.data['val_loss'], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------
    # NEW: Separate accuracy plot
    # ----------------------------------------------------------
    def plot_accuracy(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['epoch'], self.data['train_acc'], label="Train Accuracy")
        plt.plot(self.data['epoch'], self.data['val_acc'], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------
    # OPTIONAL: Keep combined plot if needed
    # ----------------------------------------------------------
    def plot_all(self):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(self.data['epoch'], self.data['train_loss'], 'r-', label='Train Loss')
        ax1.plot(self.data['epoch'], self.data['val_loss'], 'r--', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2 = ax1.twinx()
        ax2.plot(self.data['epoch'], self.data['train_acc'], 'g-', label='Train Acc')
        ax2.plot(self.data['epoch'], self.data['val_acc'], 'b-', label='Val Acc')
        ax2.set_ylabel('Accuracy')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='center right')

        plt.title('Training/Validation Loss and Accuracy')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plotter = LogPlotter(log_path="media/training_log.txt")    
    plotter.load_log()
    
    plotter.plot_loss()
    plotter.plot_accuracy()
    # plotter.plot_all()
