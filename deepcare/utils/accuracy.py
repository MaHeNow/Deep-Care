import time
import os
from tqdm import tqdm

import torch


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in tqdm(loader, ascii=True, desc="Testing accuracy"):

            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            i += 1
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}') 
    
    model.train()

    return float(num_correct)/float(num_samples)*100
