import argparse

from utils import load_model, predict, is_cuda_available


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

parser.add_argument("image", help="path to image")

parser.add_argument("checkpoint", help="path to checkpoint")

parser.add_argument("--category_names", help="path to json file with the category names", default="./cat_to_name.json")

parser.add_argument("--top_k", type=int, help="number of classes to resolve with", default=5)

parser.add_argument("--gpu", type=str2bool, nargs='?',
                    const=True, default=False, help="use cuda instead of cpu")

args = parser.parse_args()

device = 'cuda' if (args.gpu == True and is_cuda_available() == True) else 'cpu'

model, optimizer = load_model(args.checkpoint)


predictions = predict(args.image, model, args.category_names, args.top_k, device)

print(predictions)
