# train.py
import os
import csv
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model_architecture import HybridCNNConvFormer

# ============================================================
# CONFIGURATION  (aligned with §3.7)
# ============================================================
AUGMENTED_MANIFEST = "buildFiles/augmented_manifest.csv"
CHECKPOINT_DIR     = "checkpoints"
LOG_CSV            = "training_log.csv"
BEST_MODEL_PATH    = "checkpoints/best_model.pth"

NUM_CLASSES   = 4
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
DROPOUT_RATE  = 0.4
RANDOM_SEED   = 42

# Early stopping (§3.7.3)
EARLY_STOP_PATIENCE = 10

# Focal loss parameters (§3.7.1)
FOCAL_GAMMA   = 2.0

# Cosine annealing (§3.7.2)
T_0           = 10   # restart period in epochs
T_MULT        = 2    # period multiplier after each restart

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ============================================================
# DATASET
# ============================================================
class SteatosisDataset(Dataset):
    """
    Loads steatosis tiles from augmented_manifest.csv.
    Applies normalization transforms for model input.

    Normalization uses ImageNet mean/std — appropriate
    for EfficientNet-B4 pretrained on ImageNet.
    """
    def __init__(self, manifest_csv, split, transform=None):
        self.samples   = []
        self.transform = transform

        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.samples.append((
                        row["file"],
                        int(row["label"])
                    ))

        print(f"  {split:>5} set: {len(self.samples):>6} tiles loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    """
    Inference-time transforms only.
    Augmentation already applied and saved to disk in Step 4.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize
    ])
    return train_transform, eval_transform


def get_dataloaders(manifest_csv):
    train_tf, eval_tf = get_transforms()

    print("\nLoading datasets...")
    train_ds = SteatosisDataset(manifest_csv, "train", train_tf)
    val_ds   = SteatosisDataset(manifest_csv, "val",   eval_tf)
    test_ds  = SteatosisDataset(manifest_csv, "test",  eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader


# ============================================================
# CLASS WEIGHTS  (§3.7.1 — handles class imbalance)
# ============================================================
def compute_class_weights(manifest_csv):
    """
    Computes inverse-frequency class weights from
    training set to address grade imbalance.
    Passed to Focal Loss (loss function) as alpha weights.
    """
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    with open(manifest_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "train":
                counts[int(row["label"])] += 1

    total   = sum(counts.values())
    weights = torch.tensor([
        total / (NUM_CLASSES * counts[i])
        for i in range(NUM_CLASSES)
    ], dtype=torch.float32).to(DEVICE)

    print("\nClass weights:")
    for i, w in enumerate(weights):
        print(f"  Grade {i}: {w.item():.4f}  "
              f"(n={counts[i]})")
    return weights


# ============================================================
# FOCAL LOSS  (§3.7.1)
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Reduces loss contribution from easy examples,
    focusing training on hard misclassifications.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha  (Tensor): Per-class weights [num_classes].
        gamma  (float) : Focusing parameter. Default 2.0.
    """
    def __init__(self, alpha=None, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            reduction="none"
        )
        pt      = torch.exp(-ce_loss)
        fl      = (1 - pt) ** self.gamma * ce_loss
        return fl.mean()




# ============================================================
# METRICS
# ============================================================
def compute_metrics(all_labels, all_preds):
    """
    Computes per-epoch classification metrics.
    Aligned with §3.8 evaluation criteria.
    """
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(
        all_labels, all_preds,
        average="weighted", zero_division=0
    )
    prec = precision_score(
        all_labels, all_preds,
        average="weighted", zero_division=0
    )
    rec  = recall_score(
        all_labels, all_preds,
        average="weighted", zero_division=0
    )
    return {
        "accuracy":  round(acc,  4),
        "f1":        round(f1,   4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4)
    }


# ============================================================
# TRAINING + VALIDATION LOOPS
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss      = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics    = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, split="Val"):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc=f"  {split}", leave=False):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits, _ = model(imgs)
        loss      = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics    = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics, all_labels, all_preds


# ============================================================
# TRAINING LOGGER
# ============================================================
def init_log(filename=LOG_CSV):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_acc", "train_f1",
            "val_loss",   "val_acc",   "val_f1",
            "lr", "epoch_time_sec"
        ])


def log_epoch(epoch, train_loss, train_m,
              val_loss, val_m, lr, elapsed,
              filename=LOG_CSV):
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            round(train_loss, 6),
            train_m["accuracy"], train_m["f1"],
            round(val_loss, 6),
            val_m["accuracy"],   val_m["f1"],
            round(lr, 8),
            round(elapsed, 1)
        ])


# ============================================================
# EARLY STOPPING  (§3.7.3)
# ============================================================
class EarlyStopping:
    """
    Monitors validation loss. Stops training if no
    improvement is observed for `patience` consecutive
    epochs. Restores best model weights at stopping point.
    """
    def __init__(self, patience=EARLY_STOP_PATIENCE,
                 min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float("inf")
        self.best_epoch = 0

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_epoch = 0
            self.counter    = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            return False   # continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # stop
            return False


# ============================================================
# MAIN TRAINING ORCHESTRATOR
# ============================================================
def train(manifest_csv=AUGMENTED_MANIFEST):
    set_seed()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    init_log()

    print(f"\n{'='*50}")
    print(f"  HYBRID CNN-CONVFORMER — TRAINING")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}")
    print(f"{'='*50}")

    # --- Data ---
    train_loader, val_loader, test_loader = get_dataloaders(
        manifest_csv
    )

    # --- Model ---
    model = HybridCNNConvFormer(
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(DEVICE)

    # --- Loss ---
    class_weights = compute_class_weights(manifest_csv)
    criterion     = FocalLoss(
        alpha=class_weights,
        gamma=FOCAL_GAMMA
    )

    # --- Optimizer (§3.7.2) ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --- LR Scheduler (§3.7.2) ---
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_MULT
    )

    # --- Early Stopping (§3.7.3) ---
    early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # --------------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------------
    print(f"\n{'─'*50}")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_m = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_m, _, _ = evaluate(
            model, val_loader, criterion, "Val"
        )

        scheduler.step(epoch)
        elapsed = time.time() - t0
        lr      = optimizer.param_groups[0]["lr"]

        # Log to CSV
        log_epoch(
            epoch, train_loss, train_m,
            val_loss, val_m, lr, elapsed
        )

        # Console output
        print(
            f"Epoch {epoch:>3}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} "
            f"Acc: {train_m['accuracy']:.4f} "
            f"F1: {train_m['f1']:.4f} | "
            f"Val Loss: {val_loss:.4f} "
            f"Acc: {val_m['accuracy']:.4f} "
            f"F1: {val_m['f1']:.4f} | "
            f"LR: {lr:.6f} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping check
        if early_stop.step(val_loss, model):
            print(f"\n⏹ Early stopping at epoch {epoch}. "
                  f"Best val loss: {early_stop.best_loss:.4f}")
            break

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------
    print(f"\n{'─'*50}")
    print("Loading best model for final test evaluation...")
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    )

    test_loss, test_m, test_labels, test_preds = evaluate(
        model, test_loader, criterion, "Test"
    )

    cm = confusion_matrix(test_labels, test_preds)

    print(f"\n{'='*50}")
    print(f"  FINAL TEST RESULTS")
    print(f"{'='*50}")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"  Accuracy  : {test_m['accuracy']:.4f}")
    print(f"  F1-Score  : {test_m['f1']:.4f}")
    print(f"  Precision : {test_m['precision']:.4f}")
    print(f"  Recall    : {test_m['recall']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")

    # Save test results
    _save_test_results(test_loss, test_m, cm)

    return model, test_m


def _save_test_results(loss, metrics, cm,
                       filename="test_results.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["test_loss",  round(loss, 6)])
        for k, v in metrics.items():
            writer.writerow([k, v])
        writer.writerow([])
        writer.writerow(["Confusion Matrix"])
        writer.writerow(["", "Pred_0", "Pred_1",
                         "Pred_2", "Pred_3"])
        labels = ["True_0", "True_1", "True_2", "True_3"]
        for i, row in enumerate(cm):
            writer.writerow([labels[i]] + list(row))
    print(f"Test results saved: {filename}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    model, results = train()
