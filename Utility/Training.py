import seaborn as sns
import torch
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np


# ---------------------------------------------------------------------------
# Default training/eval path: model takes edge attributes (paper Section 3.1.4).
# ---------------------------------------------------------------------------

def train(train_loader, model, args, device="cuda"):
    """
    Trains the model using the provided DataLoader. The model is expected to
    accept (x_dict, edge_index_dict, edge_attr_dict, batch).

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        args (dict): Dictionary containing training arguments like learning rate and epochs.
        device (str): The device to run the training on (default is "cuda").
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, threshold=0.01, min_lr=0.00001
    )
    for epoch in range(args['epochs']):
        total_loss = 0
        model.train()
        num_graphs = 0
        for batch in tqdm(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        total_loss /= num_graphs
        train_acc = test(train_loader, model, device)
        scheduler.step(train_acc)
        current_lr = optimizer.param_groups[0]['lr']
        log = "Epoch {}: Train: {:.4f}, Loss: {:.4f}, Lr: {:.6f}"
        print(log.format(epoch + 1, train_acc, total_loss, current_lr))


def test(loader, model, device='cuda'):
    """Evaluates the model on the provided DataLoader and calculates accuracy."""
    model.eval()
    correct = 0
    num_graphs = 0
    for batch in loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch).max(dim=1)[1]
            label = batch.y
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs
    return correct / num_graphs


def test_cm(loader, model, device='cuda'):
    """Evaluates the model and returns (accuracy, predictions, labels)."""
    model.eval()
    correct = 0
    num_graphs = 0
    all_preds = []
    all_labels = []
    for batch in tqdm(loader):
        batch.to(device)
        with torch.no_grad():
            pred = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch).max(dim=1)[1]
            label = batch.y
        all_preds.append(pred.cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels)
    return correct / num_graphs, all_preds, all_labels


def calculate_metrics(y_pred, y_true):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro', zero_division=0)}")
    print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro', zero_division=0)}")
    print(f"F1 (macro): {f1_score(y_true, y_pred, average='macro', zero_division=0)}")


# ---------------------------------------------------------------------------
# Backward-compat helpers: kept under their original names so old notebooks
# continue to work. Both paths now route through the same edge-attr-aware
# functions above; the ablation path (no edge attr) is offered separately.
# ---------------------------------------------------------------------------

def train_with_edge_Att(train_loader, model, args, device="cuda"):
    """Alias kept for backward-compat. Identical to train()."""
    return train(train_loader, model, args, device=device)


def test_edge(loader, model, device='cuda'):
    """Alias kept for backward-compat. Identical to test()."""
    return test(loader, model, device=device)


def test_cm_with_edge_att(loader, model, device='cuda'):
    """Alias kept for backward-compat. Identical to test_cm()."""
    return test_cm(loader, model, device=device)


# ---------------------------------------------------------------------------
# Ablation: train/eval a model that does NOT take edge attributes
# (e.g., the legacy HeteroGNN_SAGE class).
# ---------------------------------------------------------------------------

def train_no_edge_attr(train_loader, model, args, device="cuda"):
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, threshold=0.01, min_lr=0.00001
    )
    for epoch in range(args['epochs']):
        total_loss = 0
        model.train()
        num_graphs = 0
        for batch in tqdm(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x_dict, batch.edge_index_dict, batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        total_loss /= num_graphs
        train_acc = test_no_edge_attr(train_loader, model, device)
        scheduler.step(train_acc)
        current_lr = optimizer.param_groups[0]['lr']
        log = "Epoch {}: Train: {:.4f}, Loss: {:.4f}, Lr: {:.6f}"
        print(log.format(epoch + 1, train_acc, total_loss, current_lr))


def test_no_edge_attr(loader, model, device='cuda'):
    model.eval()
    correct = 0
    num_graphs = 0
    for batch in loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch.x_dict, batch.edge_index_dict, batch).max(dim=1)[1]
            label = batch.y
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs
    return correct / num_graphs


def test_cm_no_edge_attr(loader, model, device='cuda'):
    model.eval()
    correct = 0
    num_graphs = 0
    all_preds = []
    all_labels = []
    for batch in tqdm(loader):
        batch.to(device)
        with torch.no_grad():
            pred = model(batch.x_dict, batch.edge_index_dict, batch).max(dim=1)[1]
            label = batch.y
        all_preds.append(pred.cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels)
    return correct / num_graphs, all_preds, all_labels
