import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.data.dataloader 

import pytorch_lightning as pl


class ProdLDAEncoder(nn.Module):
    def __init__(self, vocab_size: int, num_topics: int, hidden_dim: int, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(vocab_size, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense_mu = nn.Linear(hidden_dim, num_topics)
        self.dense_logvar = nn.Linear(hidden_dim, num_topics)

        self.bn_mu = nn.BatchNorm1d(num_topics, affine=False)
        self.bn_logvar = nn.BatchNorm1d(num_topics, affine=False)

    def forward(self, inputs: torch.Tensor):
        h = F.softplus(self.dense1(inputs))
        h = F.softplus(self.dense2(h))

        h = self.dropout(h)

        logtheta_loc = self.bn_mu(self.dense_mu(h))
        logtheta_logvar = self.bn_logvar(self.dense_logvar(h))

        logtheta_scale = torch.exp(logtheta_logvar * .5)

        return logtheta_loc, logtheta_scale 

class ProdLDADecoder(nn.Module):
    def __init__(self, vocab_size: int, num_topics: int, dropout: float):
        super().__init__()

        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor):
        inputs = self.dropout(inputs)

        return F.softmax(self.bn(self.beta(inputs)), dim=1)

class ProdLDA(pl.LightningModule):
    def __init__(self, vocab_size: int, num_topics: int, hidden_dim: int, dropout: float = .25):
        super().__init__()

        self.vocab_size = vocab_size 
        self.num_topics = num_topics 

        self.encoder = ProdLDAEncoder(vocab_size, num_topics, hidden_dim, dropout)
        self.decoder = ProdLDADecoder(vocab_size, num_topics, dropout)
    
    def forward(self, docs: torch.Tensor):
        logtheta_loc, logtheta_scale = self.encoder(docs)

        dist = torch.distributions.Normal(logtheta_loc, logtheta_scale)

        if self.training:
            theta = dist.rsample()
        else:
            theta = dist.mean 
        
        x_recon = self.decoder(theta)

        return x_recon, dist 
    
    def training_step(self, batch: dict, batch_idx: int):
        x = batch['bow'].float()

        x_recon, dist = self.forward(x)
        recon, kl = self.objective(x, x_recon, dist)

        loss = recon + kl 

        self.log_dict({'train/loss': loss,
                       'train/recon': recon,
                       'train/kl': kl},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        
        return loss 
    
    def validation_step(self, batch: dict, batch_idx: int):
        x = batch['bow'].float()

        x_recon, dist = self.forward(x)
        recon, kl = self.objective(x, x_recon, dist)

        loss = recon + kl

        self.log_dict({'val/loss': loss,
                       'val/recon': recon,
                       'val/kl': kl},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=.001)
    
    def objective(self, x: torch.Tensor, x_recon: torch.Tensor, dist: torch.distributions.Distribution):
        recon = -torch.sum(x * x_recon, dim=1).mean()
        prior = torch.distributions.Normal(
            torch.zeros(self.num_topics, device=x.device), 
            torch.ones(self.num_topics, device=x.device)
        )

        kl = 2. * torch.distributions.kl_divergence(dist, prior).mean()

        return recon, kl 

    def get_topics(self, vocab: dict[str, int], path: str):
        model = ProdLDA.load_from_checkpoint(path, vocab_size=self.vocab_size, num_topics=self.num_topics, hidden_dim=128)

        model.eval()
        model.freeze()

        vocab_id2word = {v: k for k, v in vocab.items()}

        topics = model.decoder.beta.weight.detach().cpu().numpy().T
        topics = topics.argsort(axis=1)[:, ::-1]

        topics = topics[:, :10]
        topics = [[vocab_id2word[i] for i in topic] for topic in topics]

        return topics 
