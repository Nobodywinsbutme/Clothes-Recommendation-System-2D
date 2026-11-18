"""POG-inspired CLIP encoder/decoder for experiments, not a paper reproduction.

Frozen CLIP features are supplied externally. This version uses a profile token,
teacher forcing and full-catalog cross entropy. FOM pretraining, the paper's
multimodal item encoder and original sampled objective are not implemented.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 512
    hidden_dim: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    max_history: int = 50
    max_outfit: int = 6


class ClipPOG(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        if min(config.embedding_dim, config.hidden_dim, config.heads, config.layers,
               config.max_history, config.max_outfit) < 1 or config.hidden_dim % config.heads:
            raise ValueError("Invalid dimensions; hidden_dim must be divisible by heads")
        if not 0 <= config.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.config = config
        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(config.hidden_dim, config.heads,
                config.hidden_dim * 4, config.dropout, batch_first=True),
            config.layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(config.hidden_dim, config.heads,
                config.hidden_dim * 4, config.dropout, batch_first=True), config.layers)
        self.start = nn.Parameter(torch.randn(config.hidden_dim) * 0.02)
        self.end = nn.Parameter(torch.randn(config.embedding_dim) * 0.02)
        self.positions = nn.Embedding(config.max_outfit + 1, config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.embedding_dim)
        # Transformer stacks clone layer parameters initially; initialize each
        # matrix independently so deeper layers do not start with equal weights.
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, history, profile, prefix, history_padding=None):
        """[B,H,E], [B,E], [B,T,E] -> [B,T+1,E]. Profile is mandatory.

        No target item is provided at the position that predicts it. Prefix
        padding only affects unused training positions, removed by loss masking.
        """
        if history.ndim != 3 or prefix.ndim != 3 or profile.ndim != 2:
            raise ValueError("Invalid history/profile/prefix ranks")
        b, h, e = history.shape
        if profile.shape != (b, self.config.embedding_dim) or e != self.config.embedding_dim:
            raise ValueError("Profile/history dimensions do not match model")
        if prefix.shape[0] != b or prefix.shape[2] != e:
            raise ValueError("Prefix dimensions do not match history")
        if h > self.config.max_history or prefix.shape[1] > self.config.max_outfit:
            raise ValueError("Sequence length exceeds model limits")
        if not all(torch.isfinite(x).all() for x in (history, profile, prefix)):
            raise ValueError("Features must be finite")
        if (profile.norm(dim=-1) <= 1e-8).any():
            raise ValueError("Provide a nonzero profile for cold-start users")
        if history_padding is None:
            history_padding = torch.zeros((b, h), dtype=torch.bool, device=history.device)
        if history_padding.shape != (b, h) or history_padding.dtype != torch.bool:
            raise ValueError("Invalid history padding mask")
        memory_input = self.input_projection(torch.cat((profile[:, None], history), dim=1))
        memory_padding = torch.cat((torch.zeros((b, 1), dtype=torch.bool,
                                   device=history.device), history_padding), dim=1)
        memory = self.encoder(memory_input, src_key_padding_mask=memory_padding)
        target = torch.cat((self.start.expand(b, 1, -1),
                            self.input_projection(prefix)), dim=1)
        length = target.shape[1]
        target = target + self.positions(torch.arange(length, device=target.device))[None]
        causal = torch.triu(torch.ones((length, length), dtype=torch.bool,
                                      device=target.device), diagonal=1)
        output = self.decoder(target, memory, tgt_mask=causal,
                              memory_key_padding_mask=memory_padding)
        return F.normalize(self.output_projection(output), dim=-1)

    def logits(self, history, profile, prefix, candidates, history_padding=None):
        if candidates.ndim != 2 or candidates.shape[1] != self.config.embedding_dim:
            raise ValueError("Candidate embedding dimensions do not match model")
        if not torch.isfinite(candidates).all() or (candidates.norm(dim=-1) <= 1e-8).any():
            raise ValueError("Candidates must be finite and nonzero")
        predicted = self(history, profile, prefix, history_padding)
        bank = F.normalize(torch.cat((candidates.detach(), self.end[None]), dim=0), dim=-1)
        return (predicted @ bank.T) * 10.0

    @torch.no_grad()
    def generate(self, history, profile, candidates, item_ids, slots, *,
                 layout=("top", "bottom"), excluded_ids=(), history_padding=None):
        """Constrained greedy inference. Explicit layout ends at its last slot.

        Caller must load a trained checkpoint before exposing this to users.
        END is learned by the loss; this constrained MVP fixes the outfit layout.
        """
        if self.training:
            raise RuntimeError("Call eval() before generation")
        if len(item_ids) != len(candidates) or len(slots) != len(item_ids):
            raise ValueError("IDs, slots and candidate vectors must align")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("Candidate IDs must be unique")
        if history.shape[0] != 1 or len(layout) > self.config.max_outfit:
            raise ValueError("Generation supports one user within max_outfit")
        if tuple(layout) not in (("top", "bottom"), ("dress",)):
            raise ValueError("MVP layouts are top+bottom or dress")
        prefix = candidates.new_empty((1, 0, self.config.embedding_dim))
        excluded, result = set(excluded_ids), []
        for slot in layout:
            allowed = [i for i, value in enumerate(slots)
                       if value == slot and item_ids[i] not in excluded]
            if not allowed:
                raise ValueError(f"No eligible candidates for slot {slot}")
            scores = self.logits(history, profile, prefix, candidates, history_padding)[0, -1]
            best = allowed[int(scores[allowed].argmax())]
            result.append({"item_id": item_ids[best], "slot": slot,
                           "score": float(scores[best])})
            excluded.add(item_ids[best])
            prefix = torch.cat((prefix, candidates[best][None, None]), dim=1)
        return result

    def metadata(self):
        return {"architecture": "clip-pog-experimental-v1", "config": asdict(self.config)}
