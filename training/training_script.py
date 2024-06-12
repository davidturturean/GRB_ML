import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import os
import time
import math
import random
import pandas as pd
from pathlib import Path

# Usual configuration class
class CFG:
    load_model = False
    best_loss = 1e10
    best_score = 1e10
    debug = False
    print_freq = 50  # Print frequency for monitoring progress
    num_workers = 0  # ADJUST THIS; Currently set to 0 for quick usage
    scheduler = "ReduceLROnPlateau"
    model_name = "1dresnet-DR12Q-Redshift"
    epochs = 10  # ADJUST THIS; Reduced for quick testing, increase as needed
    T_max = 5
    lr = 0.2e-4
    min_lr = 1e-7
    batch_size = 16  # ADJUST THIS; Modest batch size for CPU efficiency
    val_batch_size = 16  # ADJUST THIS; Modest validation batch size for CPU efficiency
    weight_decay = 1e-5
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    factor = 0.2
    patience = 1
    eps = 1e-7
    seed = 1127802826
    n_fold = 1  # ADJUST THIS; Simplified for quick testing
    trn_fold = [0]
    target_col = "target"
    train = True
    SAVEDIR = Path('./')
    device = torch.device("cpu")  # ADJUST THIS 

# Set seed
def seed_torch(seed=42):
    print("Setting seed")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(CFG.seed)

# Define Residual Block
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=500, padding=250, stride=stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=200, padding=100, stride=stride)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=15, padding=6, stride=1)
        if use_1x1conv:
            self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            print("No 1x1 convolution used.")
            self.conv4 = None
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        print("Forward pass through residual block") #EXCESSIVE PRINT STATEMENT; USEFUL FOR NOW
        y1 = F.relu(self.bn1(self.conv1(x)))
        y2 = F.relu(self.bn2(self.conv2(y1)))
        y3 = self.bn3(self.conv3(y2))
        if self.conv4:
            x = self.conv4(x)
        return F.relu(y3 + x)

def resnet_block(in_channels, out_channels, num_residuals, stride=1):
    print("Creating ResNet block")
    blk = []
    for i in range(num_residuals):
        print(f"Adding residual layer {i + 1}")
        if i == 0:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=stride))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

def resnet():
    print("Defining ResNet Model")
    model = nn.Sequential()
    model.add_module('resnet_block_1', resnet_block(in_channels=1, out_channels=32, num_residuals=5, stride=1))
    model.add_module('resnet_block_2', resnet_block(in_channels=32, out_channels=64, num_residuals=1, stride=1))
    model.add_module('resnet_block_3', resnet_block(in_channels=64, out_channels=32, num_residuals=1, stride=1))
    model.add_module('resnet_block_4', resnet_block(in_channels=32, out_channels=1, num_residuals=1, stride=1))
    model.add_module('flatten', nn.Flatten())
    model.add_module('fc1', nn.Sequential(nn.Linear(in_features=4618, out_features=796, bias=True), nn.ReLU()))
    model.add_module('fc2', nn.Sequential(nn.Linear(in_features=796, out_features=199, bias=True), nn.ReLU()))
    model.add_module('fc3', nn.Linear(in_features=199, out_features=1, bias=True))
    return model

print("Instantiating ResNet Model")
model = resnet().to(CFG.device)

# Custom loss function based on Mean Squared Error
def z_loss(pred, label):
    return nn.MSELoss()(pred, label)

# Define average meter for tracking metrics
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Training function
def train_fn(files, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    print("Starting training loop for epoch", epoch + 1)
    model.train()
    start = end = time.time()
    global_step = 0
    train_loader = DataLoader(files, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)

    for step, d in enumerate(train_loader):
        print(f'Step {step + 1}/{len(train_loader)}')  # Potentially excesive print statement; useful for now
        data_time.update(time.time() - end)
        x = d[0].to(device)
        labels = d[1].to(device)
        
        batch_size = labels.size(0)
        y_preds = model(x)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
    
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
    
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Access learning rate directly from optimizer
        current_lr = optimizer.param_groups[0]['lr']

        if step % CFG.print_freq == 0:
           print(f'Epoch: [{epoch + 1}/{CFG.epochs}][{step}/{len(train_loader)}] Loss: {losses.val:.9f} ({losses.avg:.9f}) Grad: {grad_norm:.9f} LR: {current_lr:.9f} Elapsed: {(time.time() - start):.2f}s')


    print("Finished training loop for epoch", epoch + 1)
    return losses.avg

# Validation function
def valid_fn(files, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    print("Starting validation")
    model.eval()
    start = end = time.time()
    valid_loader = DataLoader(files, batch_size=CFG.val_batch_size, shuffle=False, num_workers=CFG.num_workers)

    preds = []
    with torch.no_grad():
        for step, d in enumerate(valid_loader):
            print(f'Validation Step {step + 1}/{len(valid_loader)}')  # Added print for current validation step
            data_time.update(time.time() - end)
            x = d[0].to(device)
            labels = d[1].to(device)

            y_preds = model(x)
            loss = criterion(y_preds, labels)
            losses.update(loss.item(), labels.size(0))

            preds.append(y_preds.to('cpu').numpy())
            batch_time.update(time.time() - end)
            end = time.time()
            if step % CFG.print_freq == 0:
                print(f'EVAL: [{step}/{len(valid_loader)}] Loss: {losses.val:.9f} ({losses.avg:.9f}) Elapsed: {(time.time() - start):.2f}s')

    print("Finished validation")
    predictions = np.concatenate(preds).reshape(-1)
    return losses.avg, predictions

# Training loop
def train_loop(train_files, val_files, fold=0, load_model=False):
    print(f'========== fold: {fold} training ==========')
    
    def get_scheduler(optimizer):
        if CFG.scheduler == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, eps=CFG.eps)
        elif CFG.scheduler == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    print("Moving model to device")
    model.to(CFG.device)
    if load_model:
        model.load_state_dict(torch.load(CFG.SAVEDIR / f"{CFG.model_name}_fold{fold}_best_loss.pth")["model"])
        print(f"========== Model loaded: {CFG.model_name}_fold{fold}_best_score.pth ==========")

    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = get_scheduler(optimizer)
    criterion = nn.MSELoss()
    
    if load_model:
        best_score = CFG.best_score
        best_loss = CFG.best_loss
    else:
        best_score = np.inf
        best_loss = np.inf

    for epoch in range(CFG.epochs):
        start_time = time.time()
        
        avg_loss = train_fn(train_files, model, criterion, optimizer, epoch, scheduler, CFG.device)
        avg_val_loss, preds = valid_fn(val_files, model, criterion, CFG.device)
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        score = np.mean(np.abs((preds - val_files.tensors[1].numpy()) / (1 + val_files.tensors[1].numpy())))
        elapsed = time.time() - start_time

        print(f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.9f}  avg_val_loss: {avg_val_loss:.9f}  score: {score:.9f}  time: {elapsed:.0f}s')
            
        if score < best_score:
            best_score = score
            print(f'Epoch {epoch + 1} - Save Best Score: {best_score:.9f} Model')
            torch.save({'model': model.state_dict(), 'preds': preds}, CFG.SAVEDIR / f'{CFG.model_name}_fold{fold}_best_score.pth')
            
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            print(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.9f} Model')
            torch.save({'model': model.state_dict(), 'preds': preds}, CFG.SAVEDIR / f'{CFG.model_name}_fold{fold}_best_loss.pth')

    print("Training loop completed")

# Main script
if __name__ == "__main__":
    print(f"********** Start at {time.asctime(time.localtime(time.time()))} **********")
    
    processed_dir = '/Users/research/Documents/DR12Q Training Dupe/data/processed'

    def load_with_native_byte_order(np_file_path):
        array = np.load(np_file_path)
        if array.dtype.byteorder not in ('=', '|'):
            array = array.byteswap().newbyteorder()
        return array
    
    print("Loading training and validation data")
    # Loading full dataset
    train_fluxes = torch.Tensor(load_with_native_byte_order(os.path.join(processed_dir, 'train_fluxes.npy'))).view(-1, 1, 4618)
    val_fluxes = torch.Tensor(load_with_native_byte_order(os.path.join(processed_dir, 'val_fluxes.npy'))).view(-1, 1, 4618)
    train_labels = torch.Tensor(load_with_native_byte_order(os.path.join(processed_dir, 'train_labels.npy'))).view(-1, 1)
    val_labels = torch.Tensor(load_with_native_byte_order(os.path.join(processed_dir, 'val_labels.npy'))).view(-1, 1)
    
    # Calculating subset size (1/500th of the dataset)
    subset_size_train = len(train_fluxes) // 500
    subset_size_val = len(val_fluxes) // 500
    
    print(f"Subset size for training: {subset_size_train}")
    print(f"Subset size for validation: {subset_size_val}")
    
    # Randomly sampling indices for training and validation sets
    np.random.seed(CFG.seed)
    indices_train = np.random.choice(len(train_fluxes), subset_size_train, replace=False)
    indices_val = np.random.choice(len(val_fluxes), subset_size_val, replace=False)
    
    # Selecting the subset
    train_fluxes_subset = train_fluxes[indices_train]
    train_labels_subset = train_labels[indices_train]
    val_fluxes_subset = val_fluxes[indices_val]
    val_labels_subset = val_labels[indices_val]
    
    # Starting the training process
    train_files = TensorDataset(train_fluxes_subset, train_labels_subset)
    val_files = TensorDataset(val_fluxes_subset, val_labels_subset)
    
    print("Starting training process")
    if CFG.train:
        train_loop(train_files, val_files, load_model=CFG.load_model)
    print("Training process completed")