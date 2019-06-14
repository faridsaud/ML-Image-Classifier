import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
import numpy as np
from PIL import Image

from utils.network import Network


def load_data(data_dir):
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
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64),

    }

    return dataloaders


def get_classifier(input_size, output_size, hidden_units, drop_rate = 0.2):
    return Network(input_size, output_size, hidden_units, drop_rate)


def init_model(architecture, input_size, output_size, hidden_units, drop_rate = 0.2, learning_rate = 0.003):
    model = models[architecture](pretrained=True)
    model.classifier = get_classifier(input_size, output_size, hidden_units, drop_rate)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    return model, optimizer, criterion

def train_model(dataloaders, model, optimizer, criterion, device = 'cpu', epochs = 1):
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in dataloaders['training']:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print('Step: ', steps)

            if steps % print_every == 0:
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
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {test_loss / len(dataloaders['validation']):.3f}.. "
                      f"Validation accuracy: {accuracy / len(dataloaders['validation']):.3f}")
                running_loss = 0
                model.train()
    return model, optimizer, criterion, steps


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
    return accuracy/len(test_set)


def save_model(model, path, train_set, optimizer, input_size, output_size, architecture, drop_rate):
    file_name = 'checkpoint.pth'

    torch.save({
        'model_state': model.state_dict(),
        'hidden_layers': [each.out_features for each in model.hidden_layers],
        'input_size': input_size,
        'output_size': output_size,
        'architecture': architecture,
        'classes': train_set.class_to_idx,
        'drop_rate': drop_rate,
        'optimizer_state': optimizer.state_dict(),
    }, path + '/' + file_name)


def load_model(path):
    file_name = 'checkpoint.pth'
    state = torch.load(path + '/' + file_name)
    model = models[state['architecture']](pretrained=True)
    model.classifier = get_classifier(state['input_size'], state['output_size'], state['hidden_layers'], state['drop_rate'])
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

def get_classes_labels(classes_file):
    with open(classes_file) as json_file:
        classes_labels = json.load(json_file)
    return classes_labels

def get_class_label(classes_labels, model, class_idx):
    reversed_classes = {v: k for k, v in model.class_to_idx.items()}
    return classes_labels[reversed_classes[class_idx]]

def predict(image_path, model, classes_file, topk=5):
    model.eval()
    with torch.no_grad():
        image = process_image(image_path)
        output = model.forward(image[None, :, :, :])
    values, indices = output.exp().topk(topk)
    classes = np.vectorize(lambda t: get_class_label(get_classes_labels(classes_file), model, t))(indices.numpy()[0])
    return parse_results(values, indices, classes)

def parse_result(value, index, class_name):
    return {
        {
            'probability': value,
            'idx': index,
            'class': class_name,
        }
    }

def parse_results(values, indices, classes):
    values_arr = values.numpy()[0]
    indices_arr = indices.numpy()[0]
    classes_arr = classes
    np.vectorize(lambda i, t: parse_result(t, indices_arr[i], classes_arr[i]))(values_arr)
