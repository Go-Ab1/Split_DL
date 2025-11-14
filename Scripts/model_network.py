from helpers.global_import import *
from params_init import WeightsInitializer 
from data_loader import DatasetLoader


# ===========================================================
# InvResidualBlock (Inverted Residual Bottleneck)
# This block implements the inverted residual bottleneck as described
# in the MobileNetV2 architecture. It uses expansion with a pointwise 
# 1x1 convolution, followed by a depthwise 3x3 convolution, and a final 
# pointwise convolution projection. A skip (residual) connection is 
# applied if input & output channels match without downsampling.
# ===========================================================

class InvResidualBlock(nn.Module):
 
    alpha = 1  # Width multiplier for channel scaling
    def __init__(self, input_channel, output_channel, t=6, downsample=False):
       

        """
        Initializes the inverted residual block.

        Args:
            input_channel (int): Number of input channels.
            output_channel (int): Number of output channels.
            t (int): Expansion factor for the bottleneck (default 6).
            downsample (bool): If True, block downsamples spatial resolution
                               by stride 2, else stride 1.
        """
       
        super(InvResidualBlock, self).__init__()  
       
        if downsample:
            self.stride = 2
        else:
            self.stride = 1

        self.skip_connection = (not downsample) and (input_channel == output_channel) 

        # Apply width multiplier alpha to channels for model scaling
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)
        c = t * input_channel # Expanded channels

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
        """
        Forward pass through block.

        Applies:
            - Expansion conv + ReLU6
            - Depthwise conv + ReLU6
            - Projection conv (no activation)
            - Residual addition if applicable

        Args:
            inputs (Tensor): Input feature map.

        Returns:
            Tensor: Output feature map after block transformations.
        """
        x = F.relu6(self.bn1(self.conv1(inputs)), inplace=True)
        x = F.relu6(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))
        if self.skip_connection:
            x = x + inputs
        return x


# ===========================================================
# ModifiedMobileNetV2 Network
# MobileNetV2 variant classifier network, 
# incorporating inverted residual bottleneck blocks.
# Supports scaling with width multiplier alpha.
# ===========================================================

class ModifiedMobileNetV2(nn.Module):
    def __init__(self, output_size=10, alpha=1):

        """
        Initializes the MobileNetV2 network.

        Args:
            output_size (int): Number of classification output classes.
            alpha (float): Width multiplier for channel scaling.
        """
         
        super(ModifiedMobileNetV2, self).__init__()
        self.output_size = output_size # Number of classes

        # First convolution (32*32*3 -> 32*32*32) 
        self.conv0 = nn.Conv2d(3, int(32 * alpha), kernel_size=3,
                               stride=1, padding=1, bias=False) 
        
        self.bn0 = nn.BatchNorm2d(int(32 * alpha))

        
        InvResidualBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            InvResidualBlock(32, 16, t=1, downsample=False), 
            InvResidualBlock(16, 24, downsample=False), 
            InvResidualBlock(24, 24),                    
            InvResidualBlock(24, 32, downsample=False), 
            InvResidualBlock(32, 32),                     
            InvResidualBlock(32, 32),                    
            InvResidualBlock(32, 64, downsample=True), 
            InvResidualBlock(64, 64),                    
            InvResidualBlock(64, 64),           
            InvResidualBlock(64, 64),                   
            InvResidualBlock(64, 96, downsample=False), 
            InvResidualBlock(96, 96), 
            InvResidualBlock(96, 96),  
            InvResidualBlock(96, 160, downsample=True), 
            InvResidualBlock(160, 160), 
            InvResidualBlock(160, 160), 
            InvResidualBlock(160, 320, downsample=False) 
        )

        self.conv1 = nn.Conv2d(int(320 * alpha), 1280, kernel_size=1, bias=False) 
        # self.avgpool = nn.AvgPool2d(7) 
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, output_size) 
        WeightsInitializer.initialize(self)  

    def forward(self, inputs):
        """
        Forward pass through the MobileNetV2 network.

        Args:
            inputs (Tensor): Input image tensor (batch_size x 3 x H x W)

        Returns:
            Tensor: Logits tensor (batch_size x output_size)
        """
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace=True) 
        x = self.bottlenecks(x)
        x = F.relu6(self.bn1(self.conv1(x)), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


