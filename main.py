import torch
from torch.utils.data import DataLoader, random_split
from resnet import ResNet
#from data import RamanSpectraDataset
from training import run_epoch
import matplotlib.pyplot as plt
# from data import RamanSpectraDataset  # for all concentrations
from data2 import RamanSpectraDataset  # for only X-1 concentrations

# ===== Setup =====
data_path = "Data/ASL baseline corrected"
# Automatically determine input dimension from first sample
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Load dataset =====
dataset = RamanSpectraDataset(
    data_path,
    augment=False,        # no random live augment
    offline_aug=False,     # YES, make in-memory virtual augmentations
    num_aug=4             # 2 per file
)
sample_x, _ = dataset[0]
input_dim = sample_x.shape[-1]

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# ===== Define model =====
model = ResNet(
    hidden_sizes=[64, 128, 256],
    num_blocks=[2, 2, 2],
    input_dim=input_dim,
    n_classes=1
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ===== Metric storage =====
train_losses, val_losses = [], []
train_maes, val_maes = [], []
train_rmses, val_rmses = [], []
train_r2s, val_r2s = [], []

# ===== Train model =====
for epoch in range(1, 21):
    train_loss, train_mae, train_rmse, train_r2 = run_epoch(
        epoch, model, train_loader, cuda=torch.cuda.is_available(), training=True, optimizer=optimizer
    )
    val_loss, val_mae, val_rmse, val_r2 = run_epoch(
        epoch, model, val_loader, cuda=torch.cuda.is_available(), training=False
    )

    # Save metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_maes.append(train_mae)
    val_maes.append(val_mae)
    train_rmses.append(train_rmse)
    val_rmses.append(val_rmse)
    train_r2s.append(train_r2)
    val_r2s.append(val_r2)

    print(f"Epoch {epoch}")
    print(f"  Train: Loss={train_loss:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}, R2={train_r2:.4f}")
    print(f"  Val  : Loss={val_loss:.4f}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R2={val_r2:.4f}")

# ===== Save model =====
torch.save(model.state_dict(), "cv_model.ckpt")
print("✅ Model saved as cv_model.ckpt")

# ===== Plot Metrics =====
epochs = range(1, 21)

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
