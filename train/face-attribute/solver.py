
import numpy as np
from tqdm import tqdm
import torch
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device is not None:
    print("\nComputation device:", torch.cuda.get_device_name(torch.cuda.current_device()))


class EarlyStopping():
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def pred_acc(pred, target):
    return torch.sum(torch.eq(torch.round(pred), target)) / len(pred)

    
def save_checkpoint(state, args):
    print("=> Saving checkpoint...")
    torch.save(state, args.filename)


def train(train_loader, model, criterion, optimizer, scheduler):
    model = model.to(device)   
    train_his = {'acc': [], 'loss': []}
    
    loop = tqdm(train_loader, total=len(train_loader))
    for (data, targets) in loop:
        data = data.to(device)
        targets = targets.to(torch.float).to(device)
            
        Y_pred = model(data)
            
        #   Calculate loss
        _loss = criterion(Y_pred, targets)
        train_his['loss'].append(_loss.item())
            
        #   Calculate accuracy
        _acc = []
        for i, target in enumerate(targets):
            _acc.append(pred_acc(Y_pred[i], target).item())
        train_his['acc'].append(np.mean(_acc))
            
        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss = _loss.item(), acc = np.mean(_acc))
        
    return train_his


def validate(test_loader, model, criterion):
    model.eval().to(device)
    val_his = {'acc': [], 'loss': []}
    
    with torch.no_grad():
        for (data, targets) in test_loader:
            data = data.to(device)
            targets = targets.to(torch.float).to(device)
            
            Y_pred = model(data)
            
            #   Calculate loss
            _loss = criterion(Y_pred, targets)
            val_his['loss'].append(_loss.item())
            
            #   Calculate accuracy
            _acc = []
            for i, target in enumerate(targets):
                _acc.append(pred_acc(Y_pred[i], target).item())
            val_his['acc'].append(np.mean(_acc))
        
        return val_his


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, no_epochs):
    #   Train first time
    since = time.time()
    model_his = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0
    for epoch in range(no_epochs):    
        train_his = train(train_loader, model, criterion, optimizer, scheduler)
        model_his['loss'] += train_his['loss']
        model_his['acc'] += train_his['acc']
        val_his = validate(val_loader, model, criterion)
        model_his['val_loss'] += val_his['loss']
        model_his['val_acc'] += val_his['acc']
        
        print(f"Epoch [{epoch+1}/{no_epochs}]: loss={round(np.mean(train_his['loss']), 3)}, acc={round(np.mean(train_his['acc']), 3)}, val_loss={round(np.mean(val_his['loss']), 3)}, val_acc={round(np.mean(val_his['acc']), 3)}")
        
        #   Save best model
        if np.mean(model_his['val_acc']) > best_acc:
            checkpoint = {'model': model.state_dict(), 
                          'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)
            best_acc = np.mean(model_his['val_acc'])
        
        #   Early-stopping
        early_stopping = EarlyStopping()
        early_stopping(np.mean(val_his['loss']))
        if early_stopping.early_stop:
            break
        
        #   Learning_rate scheduler
        scheduler.step(np.mean(val_his['acc']))
    elaps_time = time.time() - since

    print(f"\nTraining complete in: {elaps_time//3600:.0f}h{elaps_time%3600//60:.0f}m{elaps_time%60:.0f}s")


