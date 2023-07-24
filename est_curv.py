import argparse
import numpy as np

from models import *
from utils.analysis_utils import get_layered_data
from utils.curv_utils import save_curvature

parser = argparse.ArgumentParser(description='Curvature Estimation')
parser.add_argument("--model_name", default="ResNet18", type=str, choices=["ResNet18", "ResNet50", "ResNet101",
                                                                           "VGG13", "VGG16", "VGG19"],
                    help="model name")
parser.add_argument("--data_set", type=str, default='cifar10',
                    choices=["cifar10", "cifar100"],
                    help='dataset to analyze (default: cifar10)')
parser.add_argument('--root_path', type=str, default='./',
                    help='root path of the project')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                    help='location of model checkpoints')
parser.add_argument('--data_path', type=str, default='./data',
                    help='data saving path')
parser.add_argument('--num_samples', type=int, default=5000,
                    help='number of samples for curvature estimation (default: 5000)')
parser.add_argument('--n_runs', type=int, default=10,
                    help='number of runs for id evaluation (default: 10)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
parser.add_argument('--caml_batch_size', type=int, default=8, help='batch size for the curvature estimation algorithm')
parser.add_argument('--seed', type=int, default=2023,
                    help='random seed (default: 2023)')

args = parser.parse_args()

np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_gpu = torch.cuda.is_available()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 10 if args.data_set == "cifar10" else 100
model_dict = {
    'VGG13': VGG('VGG13', num_classes=num_classes),
    'VGG16': VGG('VGG16', num_classes=num_classes),
    'VGG19': VGG('VGG19', num_classes=num_classes),
    'ResNet18': ResNet18(num_classes=num_classes),
    'ResNet50': ResNet18(num_classes=num_classes),
    'ResNet101': ResNet18(num_classes=num_classes),
}
model = model_dict[args.model_name].to(args.device)

activations = {}

# select layers for which we want to compute the curvature
if 'ResNet' in args.model_name:
    get_layered_data(model.modules(), activations, 'BasicBlock', )
    get_layered_data(model.modules(), activations, 'Bottleneck', )
    get_layered_data(model.modules(), activations, 'AvgPool2d')
    get_layered_data(nn.Sequential(model.linear), activations)
else:
    get_layered_data(model.modules(), activations, 'MaxPool2d')
    get_layered_data(nn.Sequential(model.classifier), activations)

save_curvature(args, model, activations)