import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyAnchorLoss(nn.Module):

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = 0.1,
        alpha: float = 32,
    ):
        super(ProxyAnchorLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha

        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        proxies = F.normalize(self.proxies, p=2, dim=1)
        sim_matrix = torch.matmul(embeddings, proxies.t())

        label_mask = torch.zeros_like(sim_matrix)
        label_mask[torch.arange(embeddings.size(0)), labels] = 1

        pos_term = torch.exp(-self.alpha * (sim_matrix - self.margin)) * label_mask
        pos_term = torch.log(1 + pos_term.sum(dim=0))
        num_valid = (label_mask.sum(dim=0) > 0).sum()
        pos_loss = torch.sum(pos_term) / num_valid if num_valid > 0 else 0

        neg_term = torch.exp(self.alpha * (sim_matrix + self.margin)) * (1 - label_mask)
        neg_loss = torch.log(1 + neg_term.sum(dim=0)).sum() / self.num_classes

        return pos_loss + neg_loss


class ProxyAnchorLossWithProxies(nn.Module):

    def __init__(self, margin: float = 0.1, alpha: float = 32):
        super(ProxyAnchorLossWithProxies, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor, proxies: torch.Tensor
    ):
        num_classes = proxies.size(0)
        sim_matrix = torch.matmul(embeddings, proxies.t())

        label_mask = torch.zeros_like(sim_matrix)
        label_mask[torch.arange(embeddings.size(0)), labels] = 1

        pos_term = torch.exp(-self.alpha * (sim_matrix - self.margin)) * label_mask
        pos_term = torch.log(1 + pos_term.sum(dim=0))
        num_valid = (label_mask.sum(dim=0) > 0).sum()
        pos_loss = (
            pos_term.sum() / num_valid
            if num_valid > 0
            else torch.tensor(0.0, device=embeddings.device)
        )

        neg_term = torch.exp(self.alpha * (sim_matrix + self.margin)) * (1 - label_mask)
        neg_loss = torch.log(1 + neg_term.sum(dim=0)).sum() / num_classes

        return pos_loss + neg_loss
