import torch
import torch.nn as nn

class FeatureFusionAttentionNet(nn.Module):
    def __init__(self, embedding, args):
        super(FeatureFusionAttentionNet, self).__init__()

        # Embedding Layer
        self.num_amino = len(embedding)
        self.embedding_dim = len(embedding[0])
        self.embedding = nn.Embedding(self.num_amino, self.embedding_dim, padding_idx=self.num_amino-1)
        if not (args.blosum is None or args.blosum.lower() == 'none'):
            self.embedding = self.embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)

        # Attention layers
        self.attn = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=args.heads)

        # Dense Layer
        self.size_hidden1_dense = 2 * args.lin_size
        self.size_hidden2_dense = 1 * args.lin_size
        self.net_pep_dim = args.max_len_pep * self.embedding_dim
        self.net_tcr_dim = args.max_len_tcr * self.embedding_dim
        self.net = nn.Sequential(
            nn.Linear(self.net_pep_dim + self.net_tcr_dim, self.size_hidden1_dense),
            nn.BatchNorm1d(self.size_hidden1_dense),
            nn.Dropout(args.drop_rate * 2),
            nn.SiLU(),
            nn.Linear(self.size_hidden1_dense, self.size_hidden2_dense),
            nn.BatchNorm1d(self.size_hidden2_dense),
            nn.Dropout(args.drop_rate),
            nn.SiLU(),
            nn.Linear(self.size_hidden2_dense, 1),
            nn.Sigmoid()
        )

    def forward(self, pep, tcr):

        # Embedding
        pep = self.embedding(pep)  # batch * len * dim
        tcr = self.embedding(tcr)  # batch * len * dim

        # Combine pep and tcr features before attention
        combined = torch.cat((pep, tcr), dim=1)  # concatenate along the sequence length axis

        # Apply Attention to combined sequence
        combined = torch.transpose(combined, 0, 1)  # (seq_len, batch, feature_dim)
        combined, _ = self.attn(combined, combined, combined)

        combined = torch.transpose(combined, 0, 1)  # (batch, seq_len, feature_dim)

        # Flatten and pass through the dense network
        combined = combined.reshape(-1, 1, combined.size(-2) * combined.size(-1)).squeeze(-2)
        combined = self.net(combined)

        return combined
