import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, VGAE
from torch_geometric.utils import negative_sampling


class LPDecoder(torch.nn.Module):
    """Maps pairs of node embeddings → [0,1] link probability."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.lin = torch.nn.Linear(2 * in_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, z, edge_index):
        src, dst = edge_index
        h = torch.cat([z[src], z[dst]], dim=1)  # [E, 2·latent_dim]
        h = F.relu(self.lin(h))  # [E, hidden_dim]
        return torch.sigmoid(self.out(h)).view(-1)


class VGAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=8, concat=True, dropout=0.2
        )
        self.conv2 = GATConv(
            hidden_channels * 8, hidden_channels, heads=1, concat=False, dropout=0.2
        )
        self.conv_mu = GATConv(hidden_channels, out_channels, heads=1, concat=False)
        self.conv_logstd = GATConv(hidden_channels, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        h = F.elu(
            self.conv1(x, edge_index)[0]
            if isinstance(self.conv1(x, edge_index), tuple)
            else self.conv1(x, edge_index)
        )
        h = F.elu(
            self.conv2(h, edge_index)[0]
            if isinstance(self.conv2(h, edge_index), tuple)
            else self.conv2(h, edge_index)
        )
        mu = (
            self.conv_mu(h, edge_index)[0]
            if isinstance(self.conv_mu(h, edge_index), tuple)
            else self.conv_mu(h, edge_index)
        )
        logstd = (
            self.conv_logstd(h, edge_index)[0]
            if isinstance(self.conv_logstd(h, edge_index), tuple)
            else self.conv_logstd(h, edge_index)
        )
        return mu, logstd


class DualVGAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, lp_hidden):
        super().__init__()
        # VGAE wraps the encoder and provides sampling + decoder
        self.encoder = VGAE(VGAEEncoder(in_channels, hidden_channels, latent_dim))
        # Decoder for link probability scoring
        self.prob_decoder = LPDecoder(latent_dim, lp_hidden)

    def forward(self, x, edge_index_obs, edge_splits):
        # Encode: returns mean and logstd
        mu, logstd = self.encoder.encoder(x, edge_index_obs)  # [N, latent_dim]
        z = self.encoder.reparametrize(mu, logstd)  # Reparameterization trick

        # Reconstruct adjacency from z on observed edges (used in BCE loss)
        pos_rec = self.encoder.decoder(z, edge_index_obs)

        # Negative edges from observed graph (not ground truth masked ones)
        neg_obs = negative_sampling(
            edge_index_obs, num_nodes=x.size(0), num_neg_samples=edge_index_obs.size(1)
        )
        neg_rec = self.encoder.decoder(z, neg_obs)

        # Link probability decoder on masked edges
        pos_p = self.prob_decoder(z, edge_splits["train_pos"])
        neg_p = self.prob_decoder(z, edge_splits["train_neg"])

        return z, pos_rec, neg_rec, pos_p, neg_p, mu, logstd

    def loss(self, pos_rec, neg_rec, pos_p, neg_p, mu, logstd):
        # 1. Reconstruction loss
        pos_loss = F.binary_cross_entropy(pos_rec, torch.ones_like(pos_rec))
        neg_loss = F.binary_cross_entropy(neg_rec, torch.zeros_like(neg_rec))
        rec_loss = pos_loss + neg_loss

        # 2. KL divergence loss
        kl_div = -0.5 * torch.mean(
            torch.sum(1 + logstd - mu.pow(2) - logstd.exp(), dim=1)
        )

        # 3. Link probability loss
        p_pos_loss = F.binary_cross_entropy(pos_p, torch.ones_like(pos_p))
        p_neg_loss = F.binary_cross_entropy(neg_p, torch.zeros_like(neg_p))
        prob_loss = p_pos_loss + p_neg_loss

        # 4. Total loss (you can tune the weights 0.1 and 0.01 if needed)
        total_loss = rec_loss + 0.1 * prob_loss + 0.01 * kl_div
        return total_loss

    def predict(self, z, edge_index_test):
        # Use the learned latent z to predict link probabilities on test edges
        return self.prob_decoder(z, edge_index_test)
