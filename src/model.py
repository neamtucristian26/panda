import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionNetwork(nn.Module):

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: list = [],
        embedding_dim: int = 1024,
        dropout: float = 0.3,
        num_classes: int = 112,
    ):
        super(ProjectionNetwork, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.projection = nn.Sequential(*layers)

        # classifier head used for energy/logit-based OOD scoring
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

    def forward(self, x, return_embeddings=False):
        embeddings = self.projection(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        logits = self.classifier(embeddings)

        if return_embeddings:
            return embeddings, logits
        return logits

    def get_embeddings(self, x):
        embeddings = self.projection(x)
        return F.normalize(embeddings, p=2, dim=1)

    def get_logits(self, embeddings):
        return self.classifier(embeddings)


class ProxyAnchorModel(nn.Module):

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: list = [],
        embedding_dim: int = 1024,
        dropout: float = 0.3,
        num_classes: int = 112,
        temperature: float = 0.1,
    ):
        super(ProxyAnchorModel, self).__init__()

        self.temperature = temperature

        self.projection_net = ProjectionNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout,
            num_classes=num_classes,
        )

        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, x):
        embeddings = self.projection_net.get_embeddings(x)
        normalized_proxies = F.normalize(self.proxies, p=2, dim=1)
        return embeddings, normalized_proxies

    def get_logits(self, embeddings):
        normalized_proxies = F.normalize(self.proxies, p=2, dim=1)
        logits = torch.matmul(embeddings, normalized_proxies.t())
        return logits / self.temperature


def create_model(
    input_dim: int = 1024,
    hidden_dims: list = [],
    embedding_dim: int = 1024,
    dropout: float = 0.3,
    num_classes: int = 112,
    temperature: float = 0.1,
) -> ProxyAnchorModel:
    return ProxyAnchorModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim,
        dropout=dropout,
        num_classes=num_classes,
        temperature=temperature,
    )
