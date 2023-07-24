import torchvision.transforms as transforms
import torchvision
import torch


def get_cifar(args):
    cifar_10_mean, cifar_10_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    cifar_100_mean, cifar_100_std = (0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)

    if args.data_set == 'cifar10':
        mean, std = cifar_10_mean, cifar_10_std
    elif args.data_set == 'cifar100':
        mean, std = cifar_100_mean, cifar_100_std

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

    if args.data_set == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=args.data_path, train=False, download=True, transform=transform_test)
    elif args.data_set == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root=args.data_path, train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader
