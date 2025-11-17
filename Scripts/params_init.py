from helpers.global_import import *

# ================================
# WeightsInitializer
# Initializes model parameters.
# ================================

class WeightsInitializer:
    """Utility class to initialize model parameters."""
    @staticmethod
    def initialize(model):
        """
        Initialize the weights of a PyTorch model.

        Convolutional layers: He initialization (normal distribution)
        BatchNorm layers: weights=1, bias=0

        Args:
            model (nn.Module): The PyTorch model to initialize.
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
