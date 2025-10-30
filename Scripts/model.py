from import_utils import *
from params_init import WeightsInitializer 
from dataset_loader import DatasetLoader

# ===========================================================
# BaseBlock (Inverted Residual Bottleneck)
# ===========================================================

class BaseBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, t=6, downsample=False):
        """
        BaseBlock: Inverted Residual Block used in MobileNetV2.
        Args:
            input_channel: number of input channels
            output_channel: number of output channels
            t: expansion factor (how much to expand before depthwise conv)
            downsample: whether to apply stride=2 (reduce spatial dimension)
        """

        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.shortcut = (not downsample) and (input_channel == output_channel) 

        # Apply width multiplier (alpha)
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)
        c = t * input_channel  # expanded channels

        # 1x1 pointwise conv (expansion)
        self.conv1 = nn.Conv2d(input_channel, c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)

        # 3x3 depthwise conv(feature extraction)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=self.stride,
                               padding=1, groups=c, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

        # 1x1 pointwise conv (projection back to output_channel)
        self.conv3 = nn.Conv2d(c, output_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self, inputs):
        x = F.relu6(self.bn1(self.conv1(inputs)), inplace=True)
        x = F.relu6(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))
        if self.shortcut:
            x = x + inputs
        return x


# ===========================================================
# MobileNetV2 Model
# ===========================================================
class MobileNetV2(nn.Module):
    def __init__(self, output_size=10, alpha=1):
        super(MobileNetV2, self).__init__()
        self.output_size = output_size

        # First convolution layer (stride=1 for CIFAR-10)
        self.conv0 = nn.Conv2d(3, int(32 * alpha), kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(int(32 * alpha))

        # Define bottleneck sequence
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, t=1, downsample=False),
            BaseBlock(16, 24, downsample=False),
            BaseBlock(24, 24),
            BaseBlock(24, 32, downsample=False),
            BaseBlock(32, 32),
            BaseBlock(32, 32),
            BaseBlock(32, 64, downsample=True),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 96, downsample=False),
            BaseBlock(96, 96),
            BaseBlock(96, 96),
            BaseBlock(96, 160, downsample=True),
            BaseBlock(160, 160),
            BaseBlock(160, 160),
            BaseBlock(160, 320, downsample=False)
        )

        # Last conv and FC
        self.conv1 = nn.Conv2d(int(320 * alpha), 1280, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, output_size)

        # self.weights_init()
        WeightsInitializer.initialize(self) 

   

    def forward(self, inputs):
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace=True)
        x = self.bottlenecks(x)
        x = F.relu6(self.bn1(self.conv1(x)), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


