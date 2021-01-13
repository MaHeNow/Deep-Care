import torch

if __name__ == "__main__":

    arr = (torch.tensor([0, 1, 2, 3]),)
    y = 1 in arr

    print(y)