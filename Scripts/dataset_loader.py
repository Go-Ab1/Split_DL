from import_utils import *


'''
Main For Dataset Loading!
'''
class DatasetLoader:
    def __init__(self, data_dir, batch_size=32, val_split=0.2, shuffle=True, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.val_split = val_split

        # Check if Folder exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    # CIFAR-10 dataset folder
    def dataset_exists(self):
        dataset_check = os.path.exists(os.path.join(self.data_dir, 'cifar-10-batches-py'))
        return dataset_check    

    def get_dataloaders(self):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        
        # Check if dataset is already downloaded
        download_flag = not self.dataset_exists()

        full_trainset = CIFAR10(root=self.data_dir, train=True, download=download_flag, transform=transform_train)
        val_size = int(len(full_trainset) * self.val_split)
        train_size = len(full_trainset) - val_size
        train_dataset, val_dataset = random_split(full_trainset, [train_size, val_size])

        test_dataset = CIFAR10(root=self.data_dir, train=False, download=download_flag, transform=transform_test)

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader
    

# USAGE
loader = DatasetLoader(data_dir='../Dataset', batch_size=64)
train_loader, val_loader, test_loader = loader.get_dataloaders()

# Check dataset sizes and shape of one batch
def check_dataset_shapes(loader, name):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    print(f"{name} dataset:")
    print(f"  Number of batches: {len(loader)}")
    print(f"  Batch shape: {images.shape}")  # (batch_size, channels, H, W)
    print(f"  Labels shape: {labels.shape}")

check_dataset_shapes(train_loader, "Train")
check_dataset_shapes(val_loader, "Validation")
check_dataset_shapes(test_loader, "Test")
