import os
import datetime

class QuantizationLogger:
    def __init__(self, base_dir="media", log_name="comparison_log.txt",
                 clear_log=True, root_dir=None):
        """
        Handles logging of quantization/model comparison results.

        Creates a log file, writes headers, allows structured logging of metrics,
        comparisons, and finalization. Supports optional root directory.
        """
        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))

        self.log_dir = os.path.join(root_dir, base_dir)

        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = os.path.join(self.log_dir, log_name)

        if clear_log:
            open(self.log_path, "w").close()

        self._write_header()



    def _write_header(self):
        """Writes the header section to the log file."""
        run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_to_file("========== Model Comparison Log ==========")
        self.log_to_file(f"Run Time: {run_time}")
        self.log_to_file("==========================================\n")

    def log_to_file(self, content):
        """Appends content to the log file."""
        with open(self.log_path, "a") as f:
            f.write(content + "\n")

    def log_section(self, title, data_dict):
        
        """
        Logs a formatted section with a title and key-value pairs.

        Parameters:
            title (str): The section title.
            data_dict (dict): Dictionary of metrics to log.
        """
        self.log_to_file(f"========== {title} ==========")
        for k, v in data_dict.items():
            if isinstance(v, float):
                self.log_to_file(f"{k}: {v:.4f}")
            else:
                self.log_to_file(f"{k}: {v}")
        self.log_to_file("")

    def compare_and_log(self, metric_name, before, after):
        
        """
        Compares two metric values and logs the percentage change.
        Parameters:
            metric_name (str): Name of the metric being compared.
            before (float): Metric value before quantization.
            after (float): Metric value after quantization.
        """

        diff = after - before
        pct_change = (diff / before) * 100 if before != 0 else 0
        msg = (f"{metric_name}: {'Increased' if diff >= 0 else 'Decreased'} "
               f"by {abs(pct_change):.2f}% ({before:.4f} → {after:.4f})")
        print(msg)
        self.log_to_file(msg)

    def finalize(self):

        self.log_to_file("\n==========================================")
        print(f"\n Done and Log saved @: {self.log_path}")
