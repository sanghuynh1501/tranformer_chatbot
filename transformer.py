from torch import nn

from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, device, en_weight_matrix, de_weight_matrix):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, device, en_weight_matrix).to(device)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, device, de_weight_matrix).to(device)
        self.out = nn.Linear(d_model, trg_vocab).to(device)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output