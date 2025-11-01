import os
import datetime

class QuantizationLogger:
    """
    Handles experiment logging for quantization and model comparison.
    """
    def __init__(self, base_dir="media", log_name="comparison_log.txt", clear_log=True):
        self.log_dir = os.path.expanduser(base_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, log_name)

        # Clear old log 
        if clear_log:
            open(self.log_path, "w").close()

        # Add timestamp header [time runnned]
        self._write_header()

    def _write_header(self):
        run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_to_file("========== Model Comparison Log ==========")
        self.log_to_file(f"Run Time: {run_time}")
        self.log_to_file("==========================================\n")

    def log_to_file(self, content):
        """Appends content to the log file."""
        with open(self.log_path, "a") as f:
            f.write(content + "\n")

    def log_section(self, title, data_dict):
        """Logs a formatted section (e.g., model metrics)."""
        self.log_to_file(f"========== {title} ==========")
        for k, v in data_dict.items():
            if isinstance(v, float):
                self.log_to_file(f"{k}: {v:.4f}")
            else:
                self.log_to_file(f"{k}: {v}")
        self.log_to_file("")

    def compare_and_log(self, metric_name, before, after):
        """Logs comparison results between two metrics."""
        diff = after - before
        pct_change = (diff / before) * 100 if before != 0 else 0
        msg = (f"{metric_name}: {'Increased' if diff >= 0 else 'Decreased'} "
               f"by {abs(pct_change):.2f}% ({before:.4f} → {after:.4f})")
        print(msg)
        self.log_to_file(msg)

    def finalize(self):
        self.log_to_file("\n==========================================")
        print(f"\n Done and Log saved @: {self.log_path}")
