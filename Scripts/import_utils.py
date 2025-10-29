# !/usr/bin/env python3
import os
import time
import argparse
import math
import random
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10

# source ~/pytorch_env/bin/activate

