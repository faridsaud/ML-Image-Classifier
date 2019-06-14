import argparse

from utils import load_data, init_model, train_model, save_model, predict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Train a neural network to identify flowers.')

parser.add_argument("data_dir", help="directory of the  data", default="")

parser.add_argument("--save_dir", help="directory where the checkpoints will be saved", default="models_checkpoints")

parser.add_argument("--arch", help="architecture of the pre-trained network",
                    choices=['resnet18', 'alexnet', 'vgg16', 'squeezenet1_0', 'densenet121', 'inception_v3',
                             'googlenet', 'shufflenet_v2_x1_0', 'mobilenet_v2', 'resnext50_32x4d'],
                    default='densenet121')

parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.003)

parser.add_argument("--drop_rate", type=float, help="drop rate", default=0.2)

parser.add_argument("--hidden_units", type=int, help="hidden units for each hidden layer", nargs='+')

parser.add_argument("--epochs", type=int, help="epochs", default=5)

parser.add_argument("--gpu", type=str2bool, nargs='?',
                    const=True, default=False, help="use cuda instead of cpu")

# additional parameters
parser.add_argument("--input_size", type=int, help="number of input nodes", default=1024)
parser.add_argument("--output_size", type=int, help="number of output nodes", default=102)
parser.add_argument("--batch_size", type=int, help="batch size for the data loader", default=32)

args = parser.parse_args()

data_dir = "./" + args.data_dir
save_dir = "./" + args.save_dir
device = 'cuda' if args.gpu == True else 'cpu'

dataloaders, image_datasets = load_data(data_dir, args.batch_size)

model, optimizer, criterion = init_model(args.arch, args.input_size, args.output_size, args.hidden_units,
                                         args.drop_rate, args.learning_rate)

model, optimizer, criterion, steps = train_model(dataloaders, image_datasets, model, optimizer, criterion, device, args.epochs)

save_model(model, save_dir, optimizer, args.input_size, args.output_size,
           args.arch, args.drop_rate, args.epochs)

