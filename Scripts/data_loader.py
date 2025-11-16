from helpers.global_import import *


# ==================================================
# DatasetLoader
# Handles loading and preprocessing of the CIFAR-10 dataset.
# Provides train, validation, and test DataLoader objects.
# Supports data augmentation on training data and configurable
# batch size, validation split, and number of workers for loading.
# ==================================================

class DatasetLoader:
    '''
    Dataset loading and preprocessing class for CIFAR-10.

    Args:
        data_dir (str): Directory path where dataset is stored or will be downloaded.
        batch_size (int): Batch size used by DataLoader.
        val_split (float): Fraction of training data reserved for validation.
        num_workers (int): Number of subprocesses used for data loading.

    Returns:
        train_loader, val_loader, test_loader (DataLoader): PyTorch DataLoaders
        for training, validation, and test sets respectively.
    '''
    def __init__(self, data_dir, batch_size=64, val_split=0.1, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.check_make_dataset_dir(self.data_dir)

    def check_make_dataset_dir(self, folder):
        """
        Verify if dataset directory exists; create if not.

        Args:
            folder (str): Dataset directory path.
        """
        # CIFAR-10 dataset folder
        if not os.path.exists(folder):
            os.makedirs(folder)

    def dataset_exists(self)->bool:
        """
        Checks if CIFAR-10 dataset files exist locally.

        Returns:
            bool: True if dataset folder exists, False otherwise.
        """
        dataset_check = os.path.exists(os.path.join(self.data_dir, 'cifar-10-batches-py'))
        # print("Dataset exists?", dataset_check)
        return dataset_check    

    def get_dataloaders(self):
        
        """
        Create train, validation, and test DataLoader objects with data augmentations.

        Training data is augmented with random crops, flips, rotations, and color jitter.

        Returns:
            tuple: train_loader, val_loader, test_loader (PyTorch DataLoader instances)
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),           
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
                
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        
        download_flag = not self.dataset_exists()

        train_dataset = CIFAR10(root=self.data_dir, train=True, download=download_flag, transform=transform_train)
        val_size = int(len(train_dataset) * self.val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        test_dataset = CIFAR10(root=self.data_dir, train=False, download=download_flag, transform=transform_test)

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        print(f"[INFO] Loaded {self.data_dir} -> "
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    
