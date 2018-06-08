import shutil

import torch
import torch.nn.functional as F
import torch.optim as optim

from datasets import train_loader, test_loader


def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class Trainer(object):
    def __init__(self, model, device='cpu', epoch=20, lr=0.01):
        self.device = device
        self.epoch = epoch
        self.lr = lr
        self.model = model.to(self.device)

        self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr)

    def train(self):
        for epoch in range(1, self.epoch + 1):
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        self.model.train()
        for batch_idx, (xs, ys) in enumerate(train_loader):
            xs = xs.to(self.device)
            ys = ys.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(xs)
            loss = F.nll_loss(output, ys)
            loss.backward()
            self.optimizer.step()

            if batch_idx % 300 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(xs), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        if epoch % 5 == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict()
            })

    def test(self):
        with torch.no_grad():
            self.model.eval()
            test_loss = 0
            correct = 0
            for xs, ys in test_loader:
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                output = self.model(xs)

                test_loss += F.nll_loss(output, ys, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(ys.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                  .format(test_loss, correct, len(test_loader.dataset),
                          100. * correct / len(test_loader.dataset)))
