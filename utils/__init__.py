import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
import numpy as np
from PIL import Image

from utils.network import Network


def load_data(data_dir, batch_size = 48):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'training': transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]),
        'validation': transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),
        'testing': transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
        'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing'])

    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=batch_size, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=batch_size, shuffle=True),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=batch_size),

    }

    return dataloaders, image_datasets


def get_classifier(input_size, output_size, hidden_units, drop_rate=0.2):
    return Network(input_size, output_size, hidden_units, drop_rate)


def init_pretrained_model(model_name):
    dispatch = {
        'resnet18': models.resnet18,
        'alexnet': models.alexnet,
        'vgg16': models.vgg16,
        'squeezenet1_0': models.squeezenet1_0,
        'densenet121': models.densenet121,
        'inception_v3': models.inception_v3,
        'googlenet': models.googlenet,
        'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
        'mobilenet_v2': models.mobilenet_v2,
        'resnext50_32x4d': models.resnext50_32x4d
    }
    return dispatch[model_name]


def init_model(architecture, input_size, output_size, hidden_units, drop_rate=0.2, learning_rate=0.003):
    model = init_pretrained_model(architecture)(pretrained=True)
    model.classifier = get_classifier(input_size, output_size, hidden_units, drop_rate)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    return model, optimizer, criterion


def train_model(dataloaders, image_datasets, model, optimizer, criterion, device='cpu', epochs=1):
    model.to(device)
    model.class_to_idx = image_datasets['training'].class_to_idx

    steps = 0
    running_loss = 0

    def reset_running_loss():
        global running_loss
        running_loss = 0

    for epoch in range(epochs):
        epoch_steps = 0
        for inputs, labels in dataloaders['training']:
            steps += 1
            epoch_steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print('Step: ', steps)
        print_training_progress(dataloaders, model, device, criterion, epoch, epochs, running_loss, epoch_steps,
                                reset_running_loss)
    return model, optimizer, criterion, steps


def print_training_progress(dataloaders, model, device, criterion, epoch, epochs, running_loss, epoch_steps,
                            reset_running_loss):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['validation']:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Epoch {epoch + 1}/{epochs}.. "
          f"Train loss: {running_loss / epoch_steps:.3f}.. "
          f"Validation loss: {test_loss / len(dataloaders['validation']):.3f}.. "
          f"Validation accuracy: {accuracy / len(dataloaders['validation']):.3f}")
    reset_running_loss()
    model.train()


def get_accuracy(model, test_set, device):
    model.to(device)
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_set:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return accuracy / len(test_set)


def save_model(model, path, optimizer, input_size, output_size, architecture, drop_rate, epochs):
    file_name = 'checkpoint.pth'

    torch.save({
        'model_state': model.state_dict(),
        'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
        'input_size': input_size,
        'output_size': output_size,
        'architecture': architecture,
        'classes': model.class_to_idx,
        'drop_rate': drop_rate,
        'optimizer_state': optimizer.state_dict(),
        'epochs': epochs,
    }, path + '/' + file_name)


def load_model(path):
    state = torch.load(path)
    model = init_pretrained_model(state['architecture'])(pretrained=True)
    model.classifier = get_classifier(state['input_size'], state['output_size'], state['hidden_layers'],
                                      state['drop_rate'])
    model.load_state_dict(state['model_state'])
    model.class_to_idx = state['classes']
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    optimizer.load_state_dict(state['optimizer_state'])
    return model, optimizer


def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    return transform(image)


def get_classes_labels(classes_file_path):
    with open(classes_file_path) as json_file:
        classes_labels = json.load(json_file)
    return classes_labels


def get_class_label(classes_labels, model, class_idx):
    reversed_classes = {v: k for k, v in model.class_to_idx.items()}
    return classes_labels[reversed_classes[class_idx]]


def predict(image_path, model, classes_file_path, topk=5, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = process_image(image_path)
        image = image.to(device)
        output = model.forward(image[None, :, :, :])
    values, indices = output.exp().topk(topk)
    values, indices = values.to('cpu'), indices.to('cpu')
    classes = np.vectorize(lambda t: get_class_label(get_classes_labels(classes_file_path), model, t))(
        indices.numpy()[0])
    return parse_results(values, indices, classes)


def parse_result(value, index, class_name):
    return {
        'probability': value,
        'idx': index,
        'class': class_name,
    }


def parse_results(values, indices, classes):
    values_arr = values.numpy()[0]
    indices_arr = indices.numpy()[0]
    classes_arr = classes
    return [parse_result(t, indices_arr[i], classes_arr[i]) for i, t in enumerate(values_arr)]

def is_cuda_available():
    return torch.cuda.is_available()
