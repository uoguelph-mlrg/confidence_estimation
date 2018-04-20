import pdb
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torch.autograd import Variable

import seaborn as sns

from models.vgg import VGG
from models.densenet import DenseNet3
from models.wideresnet import WideResNet
from utils.ood_metrics import tpr95, detection
from utils.datasets import GaussianNoise, UniformNoise

ind_options = ['cifar10', 'svhn']
ood_options = ['tinyImageNet_crop',
               'tinyImageNet_resize',
               'LSUN_crop',
               'LSUN_resize',
               'iSUN',
               'Uniform',
               'Gaussian',
               'all']
model_options = ['densenet', 'wideresnet', 'vgg13']
process_options = ['baseline', 'ODIN', 'confidence', 'confidence_scaling']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--ind_dataset', default='cifar10', choices=ind_options)
parser.add_argument('--ood_dataset', default='tinyImageNet_resize', choices=ood_options)
parser.add_argument('--model', default='vgg13', choices=model_options)
parser.add_argument('--process', default='confidence', choices=process_options)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--T', type=float, default=1000., help='Scaling temperature')
parser.add_argument('--epsilon', type=float, default=0.001, help='Noise magnitude')
parser.add_argument('--checkpoint', default='cifar10_vgg13_budget_0.3_seed_0', type=str,
                    help='filepath for checkpoint to load')
parser.add_argument('--validation', action='store_true', default=False,
                    help='only use first 1000 samples from OOD dataset for validation')

args = parser.parse_args()
cudnn.benchmark = True  # Should make training should go faster for large models

filename = args.checkpoint

if args.ind_dataset == 'svhn' and args.model == 'wideresnet':
    args.model = 'wideresnet16_8'

print args

###########################
### Set up data loaders ###
###########################

if args.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
else:
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

transform = transforms.Compose([transforms.ToTensor(),
                                normalize])

# tinyImageNet_crop and LSUN_crop are 36x36, so crop to 32x32
crop_transform = transforms.Compose([transforms.CenterCrop(size=(32, 32)),
                                     transforms.ToTensor(),
                                     normalize])

if args.ind_dataset == 'cifar10':
    num_classes = 10
    ind_dataset = datasets.CIFAR10(root='data/',
                                   train=False,
                                   transform=transform,
                                   download=True)

elif args.ind_dataset == 'svhn':
    num_classes = 10
    ind_dataset = datasets.SVHN(root='data/',
                                split='test',
                                transform=transform,
                                download=True)

data_path = 'data/'

if args.ood_dataset == 'tinyImageNet_crop':
    ood_dataset = datasets.ImageFolder(root=data_path + 'TinyImageNet_crop', transform=crop_transform)
elif args.ood_dataset == 'tinyImageNet_resize':
    ood_dataset = datasets.ImageFolder(root=data_path + 'TinyImagenet_resize', transform=transform)
elif args.ood_dataset == 'LSUN_crop':
    ood_dataset = datasets.ImageFolder(root=data_path + 'LSUN_crop', transform=crop_transform)
elif args.ood_dataset == 'LSUN_resize':
    ood_dataset = datasets.ImageFolder(root=data_path + 'LSUN_resize', transform=transform)
elif args.ood_dataset == 'iSUN':
    ood_dataset = datasets.ImageFolder(root=data_path + 'iSUN', transform=transform)
elif args.ood_dataset == 'Uniform':
    ood_dataset = UniformNoise(size=(3, 32, 32), n_samples=10000, low=0., high=1.)
elif args.ood_dataset == 'Gaussian':
    ood_dataset = GaussianNoise(size=(3, 32, 32), n_samples=10000, mean=0.5, variance=1.0)
elif args.ood_dataset == 'all':
    ood_dataset = torch.utils.data.ConcatDataset([
        datasets.ImageFolder(root=data_path + 'TinyImageNet_crop', transform=crop_transform),
        datasets.ImageFolder(root=data_path + 'TinyImagenet_resize', transform=transform),
        datasets.ImageFolder(root=data_path + 'LSUN_crop', transform=crop_transform),
        datasets.ImageFolder(root=data_path + 'LSUN_resize', transform=transform),
        datasets.ImageFolder(root=data_path + 'iSUN', transform=transform)])

ind_loader = torch.utils.data.DataLoader(dataset=ind_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=2)

ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=2)

if args.validation:
    # Limit dataset to first 1000 samples for validation and fine-tuning
    # Based on validation procedure from https://arxiv.org/abs/1706.02690
    if args.ood_dataset in ['Gaussian', 'Uniform']:
        ood_loader.dataset.data = ood_loader.dataset.data[:1000]
        ood_loader.dataset.n_samples = 1000
    elif args.ood_dataset == 'all':
        for i in range(len(ood_dataset.datasets)):
            ood_loader.dataset.datasets[i].imgs = ood_loader.dataset.datasets[i].imgs[:1000]
        ood_loader.dataset.cummulative_sizes = ood_loader.dataset.cumsum(ood_loader.dataset.datasets)
    else:
        ood_loader.dataset.imgs = ood_loader.dataset.imgs[:1000]
        ood_loader.dataset.__len__ = 1000
else:
    # Use remaining samples for test evaluation
    if args.ood_dataset in ['Gaussian', 'Uniform']:
        ood_loader.dataset.data = ood_loader.dataset.data[1000:]
        ood_loader.dataset.n_samples = 9000
    elif args.ood_dataset == 'all':
        for i in range(len(ood_dataset.datasets)):
            ood_loader.dataset.datasets[i].imgs = ood_loader.dataset.datasets[i].imgs[1000:]
        ood_loader.dataset.cummulative_sizes = ood_loader.dataset.cumsum(ood_loader.dataset.datasets)
    else:
        ood_loader.dataset.imgs = ood_loader.dataset.imgs[1000:]
        ood_loader.dataset.__len__ = len(ood_loader.dataset.imgs)

##############################
### Load pre-trained model ###
##############################

if args.model == 'wideresnet':
    cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10).cuda()
elif args.model == 'wideresnet16_8':
    cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8).cuda()
elif args.model == 'densenet':
    cnn = DenseNet3(depth=100, num_classes=num_classes, growth_rate=12, reduction=0.5).cuda()
elif args.model == 'vgg13':
    cnn = VGG(vgg_name='VGG13', num_classes=num_classes).cuda()

model_dict = cnn.state_dict()

pretrained_dict = torch.load('checkpoints/' + filename + '.pt')
cnn.load_state_dict(pretrained_dict)
cnn = cnn.cuda()

cnn.eval()


##############################################
### Evaluate out-of-distribution detection ###
##############################################

def evaluate(data_loader, mode):
    out = []
    xent = nn.CrossEntropyLoss()
    for data in data_loader:
        if type(data) == list:
            images, labels = data
        else:
            images = data

        images = Variable(images, requires_grad=True).cuda()
        images.retain_grad()

        if mode == 'confidence':
            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'confidence_scaling':
            epsilon = args.epsilon

            cnn.zero_grad()
            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence).view(-1)
            loss = torch.mean(-torch.log(confidence))
            loss.backward()

            images = images - args.epsilon * torch.sign(images.grad)
            images = Variable(images.data, requires_grad=True)

            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'baseline':
            # https://arxiv.org/abs/1610.02136
            pred, _ = cnn(images)
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            out.append(pred)

        elif mode == 'ODIN':
            # https://arxiv.org/abs/1706.02690
            T = args.T
            epsilon = args.epsilon

            cnn.zero_grad()
            pred, _ = cnn(images)
            _, pred_idx = torch.max(pred.data, 1)
            labels = Variable(pred_idx)
            pred = pred / T
            loss = xent(pred, labels)
            loss.backward()

            images = images - epsilon * torch.sign(images.grad)
            images = Variable(images.data, requires_grad=True)

            pred, _ = cnn(images)

            pred = pred / T
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            out.append(pred)

    out = np.concatenate(out)
    return out


ind_scores = evaluate(ind_loader, args.process)
ind_labels = np.ones(ind_scores.shape[0])

ood_scores = evaluate(ood_loader, args.process)
ood_labels = np.zeros(ood_scores.shape[0])

labels = np.concatenate([ind_labels, ood_labels])
scores = np.concatenate([ind_scores, ood_scores])

fpr_at_95_tpr = tpr95(ind_scores, ood_scores)
detection_error, best_delta = detection(ind_scores, ood_scores)
auroc = metrics.roc_auc_score(labels, scores)
aupr_in = metrics.average_precision_score(labels, scores)
aupr_out = metrics.average_precision_score(-1 * labels + 1, 1 - scores)

print("")
print("Method: " + args.process)
print("TPR95 (lower is better): ", fpr_at_95_tpr)
print("Detection error (lower is better): ", detection_error)
print("Best threshold:", best_delta)
print("AUROC (higher is better): ", auroc)
print("AUPR_IN (higher is better): ", aupr_in)
print("AUPR_OUT (higher is better): ", aupr_out)

ranges = (np.min(scores), np.max(scores))
plt.figure()
sns.distplot(ind_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='In-distribution')
sns.distplot(ood_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Out-of-distribution')
plt.xlabel('Confidence')
plt.ylabel('Density')
plt.legend()
plt.show()
