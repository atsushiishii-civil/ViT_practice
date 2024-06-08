import torch
from torchvision import datasets, transforms


def get_dataloader():
    transform = transforms.ToTensor()
    full_test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    # Splitting the test dataset into eval and test
    eval_size = int(0.5 * len(full_test_dataset))  # 50% for eval
    test_size = len(full_test_dataset) - eval_size  # Remaining for test
    val_dataset, test_dataset = torch.utils.data.random_split(full_test_dataset, [eval_size, test_size])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

