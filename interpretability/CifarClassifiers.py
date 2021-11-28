import torch
from torch.nn.modules import activation
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def train_cnn(trn_sets, model, epochs = 50, batch_size=1000, lr=0.05, momentum=0.9, weight_decay=0.001):
    #model.train()
    X_trn = trn_sets[0]
    y_trn = trn_sets[1]

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(epochs):
        for i in range(0, len(X_trn), batch_size):
            X_b = X_trn[i:i+batch_size]
            y_b = y_trn[i:i+batch_size]
            y_pred = model(X_b)
            l = loss(y_pred, y_b)
            model.zero_grad()
            l.backward()
            optimizer.step()
        
        if epoch % 5 == 0:
            print("Epoch %3d: training loss = %.4f" % (epoch, l.item()))
        

    return model

def test_cnn(test_sets, model):
    #model.eval()

    X_tst = test_sets[0]
    y_test = test_sets[1]

    N = list(y_test.size())[0]

    with torch.no_grad():

        output = model(X_tst)

        # Get higheest prob class
        _, preds = torch.max(output, 1)
        
        avg_acc = (preds == y_test).float().mean().item()


    
    return preds, avg_acc

def evaluate_decision_tree(trn_sets, tst_sets, param_grid, **kwargs):

    X_trn, y_trn = trn_sets
    X_tst, y_tst = tst_sets

    tree_clf = DecisionTreeClassifier(**kwargs)
    gs = GridSearchCV(tree_clf, param_grid=param_grid, return_train_score=True).fit(X_trn, y_trn)

    trn_acc = gs.best_estimator_.score(X_trn, y_trn)
    tst_acc = gs.best_estimator_.score(X_tst, y_tst)

    print(tst_acc)
    


    



