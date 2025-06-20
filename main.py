import torch
from torch.utils.data import DataLoader, random_split
from resnet import ResNet
from training import run_epoch
import matplotlib.pyplot as plt
from data import RamanSpectraDataset  
# from data2 import RamanSpectraDataset  
import numpy as np
import optuna
from optuna.trial import Trial
from augment import AugmentedWrapper


# ===== Setup =====
data_path = "Data/ASL baseline corrected merged"
# Automatically determine input dimension from first sample
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Load dataset =====
dataset = RamanSpectraDataset(
    data_path,
    augment=False,        # No live augment
    offline_aug=False     # No offline aug yet
)
sample_x, _ = dataset[0]
input_dim = sample_x.shape[-1]

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_set = AugmentedWrapper(train_set, num_aug=3)  # for example

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

print(f"Raw dataset size (before split): {len(dataset)}")
print(f"Training samples: {len(train_set)}")
print(f"Validation samples: {len(val_set)}")
print(f"Test samples: {len(test_set)}")
print(f"Input dimension: {input_dim}")

# ===== Optuna for hyperparameter tuning =====
def objective(trial: Trial):
    hidden_sizes = [
        trial.suggest_categorical("hidden_1", [32, 64, 128]),
        trial.suggest_categorical("hidden_2", [64, 128, 256]),
        trial.suggest_categorical("hidden_3", [128, 256, 512])
    ]
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    model = ResNet(
        hidden_sizes=hidden_sizes,
        num_blocks=[2, 2, 2],
        input_dim=input_dim,
        n_classes=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Short training for tuning
    for epoch in range(1, 6):
        train_loss, *_ = run_epoch(epoch, model, train_loader, cuda=torch.cuda.is_available(),
                                   training=True, optimizer=optimizer)
        val_loss, *_ = run_epoch(epoch, model, val_loader, cuda=torch.cuda.is_available(),
                                 training=False)

    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)  # Increase n_trials for better tuning

print("Best params:", study.best_params)





# ===== Define model =====
best_hidden = [
    study.best_params["hidden_1"],
    study.best_params["hidden_2"],
    study.best_params["hidden_3"]
]
best_lr = study.best_params["lr"]

model = ResNet(
    hidden_sizes=best_hidden,
    num_blocks=[2, 2, 2],
    input_dim=input_dim,
    n_classes=1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
# ===== Metric storage =====
train_losses, val_losses = [], []
train_maes, val_maes = [], []
train_rmses, val_rmses = [], []
train_r2s, val_r2s = [], []
train_loss_iters, val_loss_iters = [], []

# ===== Train model =====
for epoch in range(1, 21):
    train_loss, train_mae, train_rmse, train_r2, train_kappa, train_conf, train_batch_losses = run_epoch(
        epoch, model, train_loader, cuda=torch.cuda.is_available(), training=True, optimizer=optimizer
    )
    val_loss, val_mae, val_rmse, val_r2, val_kappa, val_conf, val_batch_losses = run_epoch(
        epoch, model, val_loader, cuda=torch.cuda.is_available(), training=False
    )

    # Save metrics
    train_loss_iters.extend(train_batch_losses)
    val_loss_iters.extend(val_batch_losses)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_maes.append(train_mae)
    val_maes.append(val_mae)
    train_rmses.append(train_rmse)
    val_rmses.append(val_rmse)
    train_r2s.append(train_r2)
    val_r2s.append(val_r2)

    print(f"Epoch {epoch}")
    print(f"  Train: Loss={train_loss:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}, R2={train_r2:.4f}, Kappa={train_kappa:.4f}")
    print(f"  Val  : Loss={val_loss:.4f}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R2={val_r2:.4f}, Kappa={val_kappa:.4f}")

# ===== Save model =====
torch.save(model.state_dict(), "cv_model.ckpt")
print("Model saved as cv_model.ckpt")


# ===== Plot Metrics =====
epochs = range(1, 21)

def plot_loss_graph(loss_list):
    """Plot smoothed training and validation loss curves using moving average.
    Validation loss is interpolated to match the number of training iterations.
    """
    plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    train_loss = loss_list[0]
    val_loss = loss_list[1]

    # Match validation loss length to training loss
    if len(train_loss) > len(val_loss):
        val_loss_interp = np.interp(
            np.linspace(0, len(val_loss) - 1, len(train_loss)),
            np.arange(len(val_loss)),
            val_loss
        )
    else:
        val_loss_interp = val_loss

    filter_size = max(1, len(train_loss) // 10)
    kernel = np.ones(filter_size) / filter_size
    train_smoothed = np.convolve(train_loss, kernel, mode='valid')
    val_smoothed = np.convolve(val_loss_interp, kernel, mode='valid')

    plt.plot(train_smoothed, label='Train Loss')
    plt.plot(val_smoothed, label='Validation Loss')
    plt.legend()
    plt.title("Smoothed Loss Curves")
    plt.grid(True)
    plt.savefig("loss.png")

# Use this instead of the old individual plot calls:
plot_loss_graph([train_loss_iters, val_loss_iters])
plt.savefig("loss_plot_smoothed.png")



def plot_metric(train_vals, val_vals, ylabel, title, filename):
    plt.figure()
    plt.plot(epochs, train_vals, label='Train')
    plt.plot(epochs, val_vals, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)


plot_metric(train_losses, val_losses, "Loss", "Loss over Epochs", "loss_plot.png")
plot_metric(train_maes, val_maes, "MAE", "Mean Absolute Error over Epochs", "mae_plot.png")
plot_metric(train_rmses, val_rmses, "RMSE", "Root Mean Squared Error over Epochs", "rmse_plot.png")
plot_metric(train_r2s, val_r2s, "R²", "R² Score over Epochs", "r2_plot.png")

# ===== Final Confusion Matrix =====
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

test_loss, test_mae, test_rmse, test_r2, test_kappa, test_conf, test_batch_losses = run_epoch(
    "Test", model, test_loader, cuda=torch.cuda.is_available(), training=False
)

print("\nFinal Test Confusion Matrix:")
test_confusion_df = pd.DataFrame(test_conf, index=[-5, -6, -7, -8, -9], columns=[-5, -6, -7, -8, -9])
print(test_confusion_df)

# Optional: plot confusion matrix for test
plt.figure(figsize=(6,5))
sns.heatmap(test_confusion_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.savefig("confusion_matrix_test.png")


# ===== Classification Report and Accuracy =====
from sklearn.metrics import classification_report, accuracy_score

# Re-run test_loader to collect predictions and actuals
test_preds = []
test_actuals = []

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x).squeeze()
        preds = [min(-5, max(-9, int(round(p.item())))) for p in outputs]
        labels = [int(round(t.item())) for t in y]

        test_preds.extend(preds)
        test_actuals.extend(labels)

# Print classification report
print("\nClassification Report:")
target_names = ['1e-5', '1e-6', '1e-7', '1e-8', '1e-9']
print(classification_report(test_actuals, test_preds, labels=[-5, -6, -7, -8, -9], target_names=target_names))

# Print test accuracy
test_acc = accuracy_score(test_actuals, test_preds)
print(f"Test Accuracy: {test_acc:.3f}")