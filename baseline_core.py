import copy,time,json,csv,platform
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import build_HDF5_feat_dataset,mil_collate_fn
from utils import load_config,seed_everything,infer_feature_dim,classification_metrics

class Classifier1FC(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(droprate) if droprate > 0 else nn.Identity()
        # Completed 2A: one linear layer from n_channels to n_classes.
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        # Completed 2B: apply dropout, then the classifier.
        return self.fc(self.dropout(x))


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim):
        super().__init__()
        # Completed 2C: project patch features to m_dim and use ReLU.
        self.fc = nn.Linear(n_channels, m_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Completed 2D: linear projection followed by ReLU.
        return self.relu(self.fc(x))


class AttentionGated(nn.Module):
    def __init__(self, L=256, D=128, K=1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights_w = nn.Linear(D, K)

    def forward(self, x):
        # x: [B, N, L]
        # Completed 2E: implement the two gated branches.
        a_v = self.attention_V(x)
        a_u = self.attention_U(x)

        a = self.attention_weights_w(a_v * a_u)
        return a.transpose(1, 2)


class ABMIL(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = AttentionGated(conf.D_inner, conf.D_attn, K=1)
        self.classifier = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)

    def forward(self, x, return_attention=False, mask=None):
        # x: [B, N, D_feat]
        med_feat = self.dimreduction(x)       # [B, N, D_inner]
        a_logits = self.attention(med_feat)   # [B, 1, N]


        # Completed 2G: normalize attention logits over the instance dimension.
        # Exclude padded patches when a batch contains different bag sizes.
        if mask is not None:
            if not mask.any(dim=1).all():
                raise ValueError("Every slide must have at least one valid patch.")
            a_logits = a_logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attention = F.softmax(a_logits, dim=-1)

        # Completed 2H: weighted aggregation. torch.bmm is useful here.
        bag_feat = torch.bmm(attention, med_feat).squeeze(1)

        # Completed 2I: classify the slide representation.
        logits = self.classifier(bag_feat)

        if return_attention:
            return logits, attention
        return logits

def train_one_epoch(model, criterion, data_loader, optimizer, device):
    # Completed 4A: put the model in training mode.
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for data in data_loader:
        # Completed 4B: move input and labels to the selected device.
        bag = data["input"].to(device, dtype=torch.float32)
        labels = data["label"].to(device, dtype=torch.long)

        # Completed 4C: clear gradients from the previous iteration.
        optimizer.zero_grad(set_to_none=True)

        # Completed 4D: forward pass.
        logits = model(bag, mask=data["mask"].to(device))

        # Completed 4E: compute the loss.
        loss = criterion(logits, labels)

        # Completed 4F: back-propagate gradients.
        loss.backward()

        # Completed 4G: update the model parameters.
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return {
        'loss': total_loss / total_examples,
        'acc': 100.0 * total_correct / total_examples,
    }

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf):
    # Completed 5A: switch to evaluation mode.
    model.eval()

    total_loss = 0.0
    total_examples = 0
    probabilities_all = []
    labels_all = []

    for data in data_loader:
        # Completed 5B: move data to device.
        bag = data["input"].to(device, dtype=torch.float32)
        labels = data["label"].to(device, dtype=torch.long)

        # Completed 5C: forward pass and loss.
        logits = model(bag, mask=data["mask"].to(device))
        loss = criterion(logits, labels)

        # Completed 5D: convert class logits into probabilities.
        probabilities = F.softmax(logits, dim=1)

        total_loss += loss.item() * labels.shape[0]
        total_examples += labels.shape[0]
        probabilities_all.append(probabilities)
        labels_all.append(labels)

    probabilities_all = torch.cat(probabilities_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    metrics = classification_metrics(probabilities_all, labels_all, conf.n_class)
    metrics['loss'] = total_loss / total_examples
    return metrics
