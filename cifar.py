import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_memlab import profile, MemReporter

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 12000)
        self.fc2 = nn.Linear(12000, 84)
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.CrossEntropyLoss()

    @profile
    def forward(self, x, labels=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        outputs = {'y': y}
        if labels is not None:
            outputs['loss'] = self.criterion(x, labels)

        return outputs

@profile
def backward(outputs):
    outputs['loss'].backward()

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=256,
                                              shuffle=True,
                                              num_workers=2)

    net = Net().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    reporter = MemReporter(net)
    reporter.report(verbose=True)

    print('Start Training')

    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = net(inputs, labels)
            backward(outputs)
            optimizer.step()

            running_loss += outputs['loss'].item()
            if i % 100 == 0:
                print(f'[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Training Finished')
    reporter.report(verbose=True)

if __name__ == '__main__':
    main()
