from torchvision import datasets, transforms
from models.Nets import CNNCifar, MobileNetCifar, CNNMnist
from models.ResNet import ResNet18, ResNet50
from models.tinyresnet import TinyResNet18
from models.resnet_imagenet import ResNet18_ImageNet
from utils.sampling import iid, noniid, iid_unbalanced, noniid_unbalanced
from models.speechresnet import SpeechResNet18
from models.TextCNN import TextCNN


def get_auto_model(dataset):
    """Return the default model for a given dataset."""
    if dataset == 'imagenet100':
        return 'resnet18_imagenet'
    elif dataset == 'tinyimagenet':
        return 'tinyresnet18'
    elif dataset in ['cifar10', 'cifar100', 'cinic10']:
        return 'resnet18'
    elif dataset in ['mnist', 'fmnist']:
        return 'cnn'
    elif dataset == 'speechcommands':
        return 'speechresnet'
    elif dataset == 'yahooanswers':
        return 'textcnn'
    elif dataset == 'agnews':
        return 'textcnn'
    elif dataset == '20newsgroup':
        return 'textcnn'
    return 'resnet18'

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def get_data(args, env='fed'):
    if env == 'single':
        if args.dataset == 'cifar10':
            dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
            dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
            
        elif args.dataset == 'cifar100':
            dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
            dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        return dataset_train, dataset_test
    
    elif env == 'fed':
        if args.unbalanced:
            if args.dataset == 'cifar10':
                dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
                dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
                if args.iid:
                    dict_users_train = iid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size)
                    dict_users_test = iid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size)
                else:
                    dict_users_train, rand_set_all = noniid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user)
                    dict_users_test, rand_set_all = noniid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar100':
                dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
                dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
                if args.iid:
                    dict_users_train = iid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size)
                    dict_users_test = iid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size)
                else:
                    dict_users_train, rand_set_all = noniid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user)
                    dict_users_test, rand_set_all = noniid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user, rand_set_all=rand_set_all)
            else:
                exit('Error: unrecognized dataset')

        else:
            if args.dataset == 'mnist':
                dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
                dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
                # sample users
                if args.iid:
                    dict_users_train = iid(dataset_train, args.num_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, args.num_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar10':
                dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
                dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
                if args.iid:
                    dict_users_train = iid(dataset_train, args.num_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, args.num_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar100':
                dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
                dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
                if args.iid:
                    dict_users_train = iid(dataset_train, args.num_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, args.num_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            else:
                exit('Error: unrecognized dataset')

        return dataset_train, dataset_test, dict_users_train, dict_users_test

def get_model(args):
    if args.model == 'auto':
        args.model = get_auto_model(args.dataset)
    if args.model == 'cnn' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'mobile' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = MobileNetCifar(num_classes=args.num_classes).to(args.device)
    elif args.model == 'resnet18' and args.dataset in ['cifar10', 'cifar100', 'cinic10']:
        net_glob = ResNet18(num_classes=args.num_classes).to(args.device)
    elif args.model == 'tinyresnet18' and args.dataset == 'tinyimagenet':
        net_glob = TinyResNet18(num_classes=args.num_classes).to(args.device)
    elif args.model == 'resnet18_imagenet' and args.dataset == 'imagenet100':
        net_glob = ResNet18_ImageNet(num_classes=args.num_classes).to(args.device)
    elif args.model == 'resnet50' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = ResNet50(num_classes=args.num_classes).to(args.device)
    elif args.model == 'cnn' and args.dataset in ['mnist','fmnist']:
        net_glob = CNNMnist(args=args).to(args.device)
        print('cnnmnist')
    elif args.model == 'speechresnet' and args.dataset == 'speechcommands':
        net_glob = SpeechResNet18(num_classes=args.num_classes).to(args.device)
    elif args.model == 'textcnn' and args.dataset == 'yahooanswers':
        net_glob = TextCNN(hidden_dim=100, num_channels=100, kernel_size=[3,4,5], max_len=200, dropout=0.5,
                           padding_idx=0, vocab_size=10000, num_classes=10).to(args.device)
        print('textcnn')
    elif args.model == 'textcnn' and args.dataset == 'agnews':
        net_glob = TextCNN(hidden_dim=100, num_channels=100, kernel_size=[3,4,5], max_len=200, dropout=0.5,
                           padding_idx=0, vocab_size=20000, num_classes=4).to(args.device)
    # elif args.model == 'mlp' and args.dataset == 'mnist':
    #     net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    elif args.model == 'textcnn' and args.dataset == '20newsgroup':
        net_glob = TextCNN(hidden_dim=100, num_channels=100, kernel_size=[3,4,5], max_len=400, dropout=0.5,
                           padding_idx=0, vocab_size=20000, num_classes=20).to(args.device)
    else:
        exit('Error: unrecognized model')

    return net_glob


def get_fedta_model(args):
    """Wrap the base model with the unified FedTA adapter."""
    from models.FedTA import FedTAWrapper

    base = get_model(args)
    return FedTAWrapper(
        base=base,
        num_classes=args.num_classes,
        num_ie=getattr(args, 'num_ie', 10),
        num_ta=getattr(args, 'num_ta', 100),
        topn=getattr(args, 'fedta_topn', 3),
        alpha=getattr(args, 'fedta_alpha', 0.5),
        tau=getattr(args, 'fedta_tau', 0.1),
    )
