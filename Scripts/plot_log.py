import pandas as pd
import matplotlib.pyplot as plt

class LogPlotter:
    """
    TXT file plotter.
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

    def plot_all(self):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(self.data['epoch'], self.data['train_loss'], 'r-', label='Train Loss')
        ax1.plot(self.data['epoch'], self.data['val_loss'], 'r--', label='Val Loss')
        ax1.set_xlabel('Epoch', color='black')
        ax1.set_ylabel('Loss', color='black')
        ax1.tick_params(axis='y', colors='black')
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)

        ax2 = ax1.twinx()
        ax2.plot(self.data['epoch'], self.data['train_acc'], 'g-', label='Train Acc')
        ax2.plot(self.data['epoch'], self.data['val_acc'], 'b-', label='Val Acc')
        ax2.set_ylabel('Accuracy', color='black')
        ax2.tick_params(axis='y', colors='black')
        ax2.set_ylim(bottom=0, top=1)

        # Create a third y-axis for learning rate
        # ax3 = ax1.twinx()
        # ax3.spines["right"].set_position(("axes", 1.1))  # offset to avoid overlap with ax2
        # ax3.plot(self.data['epoch'], self.data['lr'], 'k--', label='Learning Rate')
        # ax3.set_ylabel('Learning Rate', color='black')
        # ax3.tick_params(axis='y', colors='black')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2,
                loc='center right', facecolor='white', edgecolor='black')

        plt.title('Training/Validation Loss and Accuracy', color='black')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plotter = LogPlotter(log_path="/home/goi-tom/ANT_DC12_UniTrento/media/training_log.txt")
    plotter.load_log()
    plotter.plot_all()
