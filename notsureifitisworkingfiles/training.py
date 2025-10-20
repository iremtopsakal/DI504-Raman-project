from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix

"""
This script defines training and evaluation utilities for a regression model using PyTorch

1. run_epoch: Performs one full pass over the dataset (train or validation).
  - Computes MSE loss, MAE, RMSE, R², weighted Cohen's kappa, and confusion matrix
  - Supports optional rounding of predictions to nearest integer log10 value

2. get_predictions: Generates model predictions from a DataLoader

Notes:
- Designed for log-scaled targets (log10 concentrations)
"""


def run_epoch(epoch, model, dataloader, cuda, training=False, optimizer=None):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_samples = 0

    all_preds = []
    all_targets = []
    batch_losses = []

    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} {'Train' if training else 'Val'}")):
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets.float())

        outputs = model(inputs).squeeze()
        loss = nn.MSELoss()(outputs, targets)
        batch_losses.append(loss.item())

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

    avg_loss = total_loss / total_samples

    # === Optional: Round predictions to nearest integer log10 value ===
    # You can comment or delete this block if it hurts performance
    all_preds = [min(-5, max(-9, int(round(pred)))) for pred in all_preds]

    mae = mean_absolute_error(all_targets, all_preds)
    rmse = root_mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    # Round targets to match the format of predictions
    rounded_targets = [min(-5, max(-9, int(round(t)))) for t in all_targets]

# Compute weighted kappa and confusion matrix
    kappa = cohen_kappa_score(rounded_targets, all_preds, weights='quadratic')
    conf_mat = confusion_matrix(rounded_targets, all_preds, labels=[-5, -6, -7, -8, -9])
    return avg_loss, mae, rmse, r2, kappa, conf_mat, batch_losses


def get_predictions(model, dataloader, cuda, get_probs=False):
    preds = []
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets.float())
        outputs = model(inputs)
        if get_probs:
            probs = torch.nn.functional.softmax(outputs, dim=1)
            if cuda: probs = probs.data.cpu().numpy()
            else: probs = probs.data.numpy()
            preds.append(probs)
        else:
            predicted = outputs.squeeze()
            preds += list(predicted.detach().cpu().numpy())
    if get_probs:
        return np.vstack(preds)
    else:
        return np.array(preds)
