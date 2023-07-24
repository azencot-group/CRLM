'''Train CIFAR10 with PyTorch.'''
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument("--model", default="resNet18", type=str, choices=["resNet18", "resNet50", "resNet101",
                                                                      "VGG13", "VGG16", "VGG19"],
                    help="model name")
parser.add_argument("--dataset", type=str, default='cifar10',
                    choices=["cifar10", "cifar100"],
                    help='dataset to analyze (default: cifar10)')
parser.add_argument('--data_path', type=str, default='./data', help='data saving path')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/',
                    help='location of model checkpoints')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--seed', type=int, default=2023,
                    help='random seed (default: 2023)')
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
torch.autograd.set_detect_anomaly(True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

cifar_10_mean, cifar_10_std  = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
cifar_100_mean, cifar_100_std = (0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)

if args.dataset == 'cifar10':
    mean, std = cifar_10_mean, cifar_10_std
    num_classes = 10
elif args.dataset == 'cifar100':
    mean, std = cifar_100_mean, cifar_100_std
    num_classes = 100

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(
        root=args.data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root=args.data_path, train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
model_dict = {
    'VGG13': VGG('VGG13',num_classes=num_classes),
    'VGG16': VGG('VGG16',num_classes=num_classes),
    'VGG19': VGG('VGG19',num_classes=num_classes),
    'resNet18': ResNet18(num_classes=num_classes),
    'resNet50': ResNet18(num_classes=num_classes),
    'resNet101': ResNet18(num_classes=num_classes),
}
net = model_dict[args.model]

net = net.to(device)


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}, accuracy: {3:.3f}".format(batch_idx + 1, epoch + 1, (train_loss / (batch_idx + 1)), 100. * correct / total))



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}, accuracy: {3:.3f}".format(batch_idx + 1, epoch + 1, (
                            test_loss / (batch_idx + 1)), 100. * correct / total))


    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.checkpoints_dir):
            os.mkdir(args.checkpoints_dir)
        torch.save(state, os.path.join(args.checkpoints_dir, f'ckpt_{args.model}_{args.dataset}.pth'))
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
    scheduler.step()
