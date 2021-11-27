import torch
import torch.optim
import numpy as np
import matplotlib.pyplot as plt

def train_cnn(X_trn, y_trn, model, epochs = 50, lr=0.05, momentum=0.9, weight_decay=0.001):
    
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(epochs):
        y_pred = model(X_trn)
        l = loss(y_pred, y_trn)
        model.zero_grad()
        l.backward()    

        optimizer.step()
        print("Epoch %3d: training loss = %.4f" % (epoch, l.item()))

    return model