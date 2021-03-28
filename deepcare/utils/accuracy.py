import time
import os

import torch


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}') 
    
    model.train()

    return float(num_correct)/float(num_samples)*100


def check_accuracy_on_classes(loader, model, device, classes=['A', 'C', 'G', 'T']):
    class_correct = [0.0 for i in range(len(classes))]
    class_total = [0.0 for i in range(len(classes))]
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predicted = scores.max(1)
            c = (predicted == y).squeeze()
            for i in range(4):
                class_correct[i] += c[i].item()
                class_total[i] += 1


    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    model.train()