import deeplake
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


from src.model import DeepPoseRegressor
from src.dataset import LSPDataset
from src.train_test import train, test_pcp


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DeepPose parser')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    



    transform = transforms.Compose([
        transforms.Resize((220, 220)),
        transforms.ToTensor()
    ])

    train_ds = deeplake.load("hub://activeloop/lsp-extended")
    test_ds = deeplake.load("hub://activeloop/lsp-test")
    train_dataset = LSPDataset(train_ds, transform=transform)
    train_loader = DataLoader(train_dataset,**train_kwargs)
    test_dataset = LSPDataset(test_ds, transform=transform)
    test_loader = DataLoader(test_dataset,**train_kwargs)

    model = DeepPoseRegressor().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, )

    
    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_n.pt")
            test_pcp(args, model, device, test_loader)
            print(f"Model saved at epoch {epoch}")
    

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")
    


if __name__ == '__main__':
    main()
