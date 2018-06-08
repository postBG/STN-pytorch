import argparse
import torch

from models import Net
from trainer import Trainer

parser = argparse.ArgumentParser(description='PyTorch STN example')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
parser.add_argument('--epoch', type=int, default=20, help='epoch (default: 20)')
args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()

    trainer = Trainer(model, device=device, lr=args.lr, epoch=args.epoch)
    trainer.train_and_eval()


if __name__ == '__main__':
    main()
