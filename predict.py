import argparse

parser = argparse.ArgumentParser(description='Train a neural network to identify flowers.')

parser.add_argument("image", help="path to image")

parser.add_argument("checkpoint", help="path to checkpoint")

parser.add_argument("--category_names ", help="path to json file with the category names")

parser.add_argument("--gpu", help="use cuda instead of cpu")

args = parser.parse_args()


print(args)