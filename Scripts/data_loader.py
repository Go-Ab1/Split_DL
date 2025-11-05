from helpers.global_import import *

'''
Main For Dataset Loading!
'''
class DatasetLoader:
    '''Dataset loading and preprocessing for CIFAR-10
    Args:
        data_dir: Directory where dataset is stored/ downloaded
        batch_size: Batch size for DataLoader
        val_split: Fraction of training data to use for validation
        num_workers: Number of subprocesses to use for data loading
    Outputs..[returns]:
        train_loader, val_loader, test_loader: DataLoaders for training, validation, and test sets
    '''
    def __init__(self, data_dir, batch_size=64, val_split=0.2, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.check_make_dataset_dir(self.data_dir)

    def check_make_dataset_dir(self, folder):
        # CIFAR-10 dataset folder
        if not os.path.exists(folder):
            os.makedirs(folder)

    def dataset_exists(self)->bool:
        dataset_check = os.path.exists(os.path.join(self.data_dir, 'cifar-10-batches-py'))
        # print("Dataset exists?", dataset_check)
        return dataset_check    

    def get_dataloaders(self):
        ''''
        To make sure the model generalizes well, we apply data augmentation techniques to the training data.
        '''
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
    
    
