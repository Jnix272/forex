"""
monitoring/attention_logger.py
================================
Logs attention weights from transformer-based models during validation.
Writes to logs/attention/<run_name>_ep<epoch>_attn.npz every N epochs.
"""
import numpy as np
from pathlib import Path
from typing import Optional
import torch.nn as nn

class AttentionLogger:
    def __init__(self, run_name: str, log_dir: str = 'logs/attention', every_n_epochs: int = 5):
        self.run_name = run_name
        self.log_dir = Path(log_dir)
        self.every_n_epochs = every_n_epochs
        self._hooks = []
        self._attn_weights = {}
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def register_hooks(self, model: nn.Module) -> None:
        """Attach forward hooks to all MultiheadAttention layers."""
        self.remove_hooks()
        
        def hook_fn(name):
            def hook(module, input, output):
                # MultiheadAttention output is a tuple: (attn_output, attn_output_weights)
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        if name not in self._attn_weights:
                            self._attn_weights[name] = []
                        self._attn_weights[name].append(attn_weights.detach().cpu().numpy())
            return hook
            
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                h = module.register_forward_hook(hook_fn(name))
                self._hooks.append(h)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
    
    def on_epoch_end(self, epoch: int) -> Optional[Path]:
        """Save attention weights if this is a logging epoch. Returns path or None."""
        if epoch % self.every_n_epochs != 0 or not self._attn_weights:
            self.clear()
            return None
            
        save_dict = {}
        for name, weights_list in self._attn_weights.items():
            if not weights_list:
                continue
            try:
                # Average attention weights across batches
                all_weights = np.concatenate(weights_list, axis=0)
                mean_weights = np.mean(all_weights, axis=0)
                save_dict[name] = mean_weights
            except Exception:
                pass
                
        out_path = None
        if save_dict:
            out_path = self.log_dir / f"{self.run_name}_ep{epoch}_attn.npz"
            np.savez(out_path, **save_dict)
            
        self.clear()
        return out_path
    
    def clear(self) -> None:
        """Clear accumulated attention weights without saving."""
        self._attn_weights.clear()
