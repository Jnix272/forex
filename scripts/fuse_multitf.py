#!/usr/bin/env python3
"""
scripts/fuse_multitf.py
Fuse 1m/5m/15m HAELT checkpoints into a single MultiTimeframeAttention model.
Run AFTER all three checkpoints exist:
  checkpoints/haelt_1m/haelt_1m_best.pt
  checkpoints/haelt_5m/haelt_5m_best.pt
  checkpoints/haelt_15m/haelt_15m_best.pt
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from types import SimpleNamespace

# Ensure project root on path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ensemble import MultiTimeframeAttention
from models.architectures import HAELTHybrid


def load_haelt_encoder(ckpt_path: Path, n_features: int, seq_len: int, d_model: int, nhead: int, num_layers: int, dropout: float) -> tuple:
    """
    Load HAELT checkpoint and extract the Transformer encoder branch.
    Returns (proj, pos_emb, transformer_encoder) for injection into MTF.
    """
    # Build a HAELT model with matching architecture
    args = SimpleNamespace(
        model="haelt",
        hidden_size=128,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        pair_embed_dim=0,
        multitask=False,
        seq_len=seq_len,
    )
    model = HAELTHybrid(
        input_size=n_features,
        seq_len=seq_len,
        lstm_hidden=args.hidden_size // 2,
        d_model=args.d_model // 2,
        nhead=max(2, args.nhead // 2),
        n_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=1,
    )
    
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Handle various checkpoint formats
    if isinstance(state, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    
    model.load_state_dict(state, strict=False)
    model.eval()
    
    # Extract the transformer branch components
    # The transformer branch consists of: proj -> pos_emb -> trf (TransformerEncoder)
    proj = model.proj
    pos_emb = model.pos_emb
    trf = model.trf
    
    return proj, pos_emb, trf


def main():
    p = argparse.ArgumentParser(description="Fuse 1m/5m/15m HAELT into MultiTimeframeAttention")
    p.add_argument("--ckpt-1m", default="checkpoints/haelt_1m/haelt_1m_best.pt")
    p.add_argument("--ckpt-5m", default="checkpoints/haelt_5m/haelt_5m_best.pt")
    p.add_argument("--ckpt-15m", default="checkpoints/haelt_15m/haelt_15m_best.pt")
    p.add_argument("--n-features", type=int, default=145)
    p.add_argument("--seq-len-1m", type=int, default=60)
    p.add_argument("--seq-len-5m", type=int, default=12)
    p.add_argument("--seq-len-15m", type=int, default=4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--output", default="checkpoints/haelt_mtf_fused.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load encoder components from each HAELT checkpoint
    print(f"Loading 1m encoder from {args.ckpt_1m}...")
    proj_1m, pos_emb_1m, trf_1m = load_haelt_encoder(
        Path(args.ckpt_1m), args.n_features, args.seq_len_1m,
        args.d_model, args.nhead, args.num_layers, args.dropout
    )
    
    print(f"Loading 5m encoder from {args.ckpt_5m}...")
    proj_5m, pos_emb_5m, trf_5m = load_haelt_encoder(
        Path(args.ckpt_5m), args.n_features, args.seq_len_5m,
        args.d_model, args.nhead, args.num_layers, args.dropout
    )
    
    print(f"Loading 15m encoder from {args.ckpt_15m}...")
    proj_15m, pos_emb_15m, trf_15m = load_haelt_encoder(
        Path(args.ckpt_15m), args.n_features, args.seq_len_15m,
        args.d_model, args.nhead, args.num_layers, args.dropout
    )

    # Build MultiTimeframeAttention with per-timeframe encoders
    print("Building MultiTimeframeAttention with per-timeframe encoders...")
    
    class MTFWithPerTFEncoders(MultiTimeframeAttention):
        """Extended MTF that uses separate encoders per timeframe."""
        
        def __init__(
            self,
            input_size: int,
            d_model: int = 128,
            nhead: int = 4,
            n_tf_layers: int = 2,
            dropout: float = 0.1,
            timeframes: list[int] = [1, 5, 15],
        ):
            # Initialize base but we'll replace the encoder
            super().__init__(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                n_tf_layers=n_tf_layers,
                dropout=dropout,
                timeframes=timeframes,
            )
            # We'll replace the shared encoder with per-timeframe encoders
            self.per_tf_proj = nn.ModuleList()
            self.per_tf_pos_emb = nn.ModuleList()
            self.per_tf_encoder = nn.ModuleList()
        
        def set_per_tf_encoders(self, projs, pos_embs, encoders):
            """Set per-timeframe encoders."""
            self.per_tf_proj = nn.ModuleList(projs)
            self.per_tf_pos_emb = nn.ModuleList(pos_embs)
            self.per_tf_encoder = nn.ModuleList(encoders)
            
        def forward(self, x_list: list[torch.Tensor]) -> torch.Tensor:
            """
            x_list: list of (B, T_i, input_size) tensors, one per timeframe.
            All T_i can differ (15-min bars will have fewer rows).
            """
            encoded = []
            for i, x in enumerate(x_list):
                # Use per-timeframe encoder
                h = self.per_tf_proj[i](x)
                # Inject positional embedding
                T = h.size(1)
                if T <= self.per_tf_pos_emb[i].num_embeddings:
                    pos = self.per_tf_pos_emb[i].weight[:T]
                    h = h + pos.unsqueeze(0)
                else:
                    idx = torch.arange(T, device=h.device) % self.per_tf_pos_emb[i].num_embeddings
                    pos = self.per_tf_pos_emb[i].weight[idx]
                    h = h + pos.unsqueeze(0)
                h = self.per_tf_encoder[i](h)
                encoded.append(h[:, -1, :])  # Take last bar embedding

            # Cross-timeframe attention: fine (1m) queries, coarse (5m, 15m) as K/V
            if len(encoded) > 1:
                query   = encoded[0].unsqueeze(1)            # (B, 1, d)
                context = torch.stack(encoded[1:], dim=1)    # (B, n_tf-1, d)
                attn_out, _ = self.cross_attn(query, context, context)
                fine = self.cross_norm(encoded[0] + attn_out.squeeze(1))
                encoded[0] = fine

            fused = torch.cat(encoded, dim=-1)
            return self.fuse(fused).squeeze(-1)

    mtf = MTFWithPerTFEncoders(
        input_size=args.n_features,
        d_model=args.d_model,
        nhead=args.nhead,
        n_tf_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # Inject per-timeframe encoders
    mtf.set_per_tf_encoders(
        projs=[proj_1m, proj_5m, proj_15m],
        pos_embs=[pos_emb_1m, pos_emb_5m, pos_emb_15m],
        encoders=[trf_1m, trf_5m, trf_15m],
    )

    mtf.eval()

    # Save fused checkpoint
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": mtf.state_dict(),
        "config": {
            "n_features": args.n_features,
            "seq_len_1m": args.seq_len_1m,
            "seq_len_5m": args.seq_len_5m,
            "seq_len_15m": args.seq_len_15m,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        "source_checkpoints": {
            "1m": args.ckpt_1m,
            "5m": args.ckpt_5m,
            "15m": args.ckpt_15m,
        }
    }, output_path)
    
    print(f"\nFused model saved to: {output_path}")
    print("Ready for inference or ensemble meta-training.")


if __name__ == "__main__":
    main()