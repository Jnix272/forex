import os
import sys
import requests
import json
import re
import argparse
from typing import Dict, Any, Optional

try:
    from infrastructure.discord_notifier import (
        send_training_alert, send_fix_notification, send_tune_notification
    )
    _DISCORD = True
except ImportError:
    try:
        # Fallback: run directly as python infrastructure/ollama_helper.py
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from infrastructure.discord_notifier import (
            send_training_alert, send_fix_notification, send_tune_notification
        )
        _DISCORD = True
    except ImportError:
        _DISCORD = False

class OllamaHelper:
    def __init__(self, model: str = "gemma4:e2b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{self.base_url}/api/chat"
        self._last_tune_epoch: Optional[int] = None
        self._last_applied: Dict[str, Any] = {}
        self._last_pre_metrics: Dict[str, Any] = {}
        self._cooldown_epochs = 3
        self._tune_apply_count = 0

    def _generate_response(self, prompt: str, system: str = "") -> Optional[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"\n[OllamaHelper] Failed to connect to Ollama: {e}")
            return None
        except Exception as e:
            print(f"\n[OllamaHelper] Error querying Ollama: {e}")
            return None

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Attempt to extract JSON from a markdown response."""
        try:
            # Look for JSON code blocks
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            # Fallback 1: regex for braces
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            # Fallback 2: aggressive substring extraction
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except Exception as e:
            print(f"[OllamaHelper] JSON parsing error: {e}")
        return None

    def monitor_training(self, epoch: int, train_loss: float, val_loss: float, additional_metrics: Dict[str, Any] = None) -> None:
        """Monitors training per epoch. Only queries Ollama when anomalies are detected."""
        if self._last_applied and additional_metrics:
            try:
                prev_vl = float(self._last_pre_metrics.get("val_loss", 0.0))
                prev_sh = float(self._last_pre_metrics.get("val_sharpe", 0.0))
                cur_sh = float(additional_metrics.get("sharpe", additional_metrics.get("val_sharpe", 0.0)) or 0.0)
                if prev_vl > 0.0 and (val_loss > prev_vl) and (cur_sh < prev_sh):
                    print(f"[OllamaHelper] Rollback signal after tune {self._last_applied}: "
                          f"val_loss {prev_vl:.4f}->{val_loss:.4f}, sharpe {prev_sh:.4f}->{cur_sh:.4f}")
            except Exception:
                pass
        # Avoid spamming on every epoch — only alert when there is a suspected issue:
        #   overfitting: val_loss > train_loss * 1.3
        #   exploding:   val_loss > 100.0 or train_loss > 100.0
        #   stagnation:  not checked here (needs history), handled by chunk early stop
        overfitting  = val_loss > train_loss * 1.3 and epoch > 3
        exploding    = val_loss > 100.0 or train_loss > 100.0
        if not (overfitting or exploding):
            return

        prompt = (f"Training Epoch: {epoch}\n"
                  f"Train Loss: {train_loss:.4f}\n"
                  f"Validation Loss: {val_loss:.4f}\n")
        if additional_metrics:
            prompt += f"Additional Metrics: {json.dumps(additional_metrics)}\n"
        if overfitting:
            prompt += (
                "\nValidation loss is much higher than training loss (overfitting). "
                "Suggest regularization fixes only: increase dropout or weight decay, "
                "enable or tighten early stopping, add more data augmentation — "
                "do NOT suggest raising learning rate."
            )
        prompt += "\nAn anomaly has been detected. Explain in 2 sentences and suggest a fix."
        system_prompt = (
            "You are an AI monitoring a deep learning training run. Be brief and actionable. "
            "When validation loss exceeds training loss, recommend LOWER learning rate, "
            "higher dropout, higher weight decay, or earlier stopping — never suggest "
            "increasing learning rate for overfitting."
        )

        response = self._generate_response(prompt, system=system_prompt)
        if response:
            msg = response.strip()
            print(f"\n[Ollama Epoch {epoch} \u26a0\ufe0f  Alert]: {msg}")
            if _DISCORD:
                try:
                    send_training_alert(epoch, train_loss, val_loss, msg, additional_metrics)
                except Exception:
                    pass

    def auto_fix_error(self, error_traceback: str, script_context: str = "") -> None:
        """Sends an error traceback to Ollama, receives a JSON patch, applies it, and restarts."""
        print(f"\n[OllamaHelper] Analyzing error and attempting auto-fix with {self.model}...")
        
        system_prompt = (
            "You are an expert Python auto-fix agent. You must output exactly ONE JSON block with your fix.\n"
            "Format:\n"
            "```json\n"
            "{\n"
            '  "file_to_edit": "path/to/file.py",\n'
            '  "search_string": "exact original code to replace",\n'
            '  "replace_string": "the new code"\n'
            "}\n"
            "```\n"
            "Provide the exact string matches including indentation."
        )
        prompt = f"Error Traceback:\n```python\n{error_traceback}\n```\nContext: {script_context}\nProvide the JSON fix."
        
        response = self._generate_response(prompt, system=system_prompt)
        if not response:
            return
            
        fix_data = self._extract_json(response)
        if not fix_data or "search_string" not in fix_data or "replace_string" not in fix_data:
            print("[OllamaHelper] Could not parse a valid fix from Ollama's response.")
            print("Response was:", response)
            return
            
        target_file = fix_data.get("file_to_edit", "")
        # Fallback to the current script if not provided
        if not target_file or not os.path.exists(target_file):
            target_file = sys.argv[0] if sys.argv else ""
            
        if not target_file or not os.path.exists(target_file):
            print("[OllamaHelper] Cannot determine which file to fix.")
            return
            
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()
                
            if fix_data["search_string"] in content:
                new_content = content.replace(fix_data["search_string"], fix_data["replace_string"])
                with open(target_file, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"[OllamaHelper] Successfully patched {target_file}. Restarting script...")
                if _DISCORD:
                    try:
                        send_fix_notification(target_file, response[:400] if response else "Patch applied.")
                    except Exception:
                        pass
                # Auto-restart the script safely
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                print(f"[OllamaHelper] Failed to apply patch: search_string not found in {target_file}.")
        except Exception as e:
            print(f"[OllamaHelper] Auto-fix application failed: {e}")

    def auto_tune_model(self, model_name: str, metrics: Dict[str, Any], config_path: str = "run.yaml") -> None:
        """Gets hyperparameter suggestions, updates run.yaml, and restarts."""
        print(f"\n[OllamaHelper] Generating auto-tuning suggestions for {model_name}...")
        
        system_prompt = (
            "You are an ML auto-tuner. Output a JSON block with recommended parameters to update in the config.\n"
            "Format:\n"
            "```json\n"
            "{\n"
            '  "lr": 0.0001,\n'
            '  "batch_size": 512,\n'
            '  "dropout": 0.2\n'
            "}\n"
            "```"
        )
        prompt = f"Model: {model_name}\nFinal Metrics: {json.dumps(metrics, indent=2)}\nBased on these metrics, what hyperparameters should we use for the next run?"
        
        response = self._generate_response(prompt, system=system_prompt)
        if not response:
            return
            
        new_params = self._extract_json(response)
        if not new_params:
            print("[OllamaHelper] Could not parse tuning JSON.")
            print("Response:", response)
            return

        # Guardrails: advisor + safe autopilot only.
        epoch = int(metrics.get("epoch", 0) or 0)
        if epoch > 0 and self._last_tune_epoch is not None:
            if (epoch - self._last_tune_epoch) < self._cooldown_epochs:
                print(f"[OllamaHelper] Cooldown active ({epoch - self._last_tune_epoch}/{self._cooldown_epochs} epochs). Skipping auto-apply.")
                return
        elif epoch <= 0 and self._tune_apply_count > 0:
            print("[OllamaHelper] Cooldown active (epoch not provided; only one auto-apply allowed per process run). Skipping auto-apply.")
            return

        protected = {"loss", "label_method", "seq_len", "folds", "data_source", "amp"}
        filtered: Dict[str, Any] = {}
        for k, v in new_params.items():
            if k in protected:
                continue
            filtered[k] = v

        # Bounded updates
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_text = f.read()
        else:
            cfg_text = ""

        def _read_current(key: str) -> Optional[float]:
            m = re.search(rf"^\s*{re.escape(key)}\s*:\s*([0-9eE\.\-]+)\s*$", cfg_text, flags=re.MULTILINE)
            if not m:
                return None
            return _to_float(m.group(1))

        safe_params: Dict[str, Any] = {}
        cur_lr = _read_current("lr")
        val_loss = _to_float(metrics.get("val_loss"))
        train_loss = _to_float(metrics.get("train_loss"))
        overfit = (
            val_loss is not None and train_loss is not None
            and val_loss > train_loss * 1.2
        )
        if "lr" in filtered and cur_lr is not None:
            new_lr = _to_float(filtered["lr"])
            if new_lr is not None:
                if overfit:
                    lo, hi = cur_lr * 0.5, cur_lr * 0.95
                else:
                    lo, hi = cur_lr * 0.85, cur_lr * 1.15
                safe_params["lr"] = min(max(new_lr, lo), hi)

        cur_do = _read_current("dropout")
        if "dropout" in filtered and cur_do is not None:
            new_do = _to_float(filtered["dropout"])
            if new_do is not None:
                lo, hi = cur_do - 0.03, cur_do + 0.03
                safe_params["dropout"] = max(0.0, min(0.9, min(max(new_do, lo), hi)))

        if "batch_size" in filtered:
            try:
                bs = int(filtered["batch_size"])
                if bs > 0:
                    safe_params["batch_size"] = bs
            except Exception:
                pass

        if not safe_params:
            print("[OllamaHelper] No safe tuning keys to apply after guardrails.")
            return
            
        print("\n" + "="*50)
        print(f"🤖 OLLAMA APPLYING TUNING FOR {model_name.upper()}:")
        print(json.dumps(safe_params, indent=2))
        print("="*50 + "\n")
        if _DISCORD:
            try:
                send_tune_notification(model_name, safe_params)
            except Exception:
                pass
        
        # Update config/run.yaml using regex to preserve comments
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_text = f.read()
                    
                for k, v in safe_params.items():
                    # Look for lines like "lr: 0.00005" or "  batch_size:  256"
                    # We match numbers including decimals, negatives, and scientific notation (e.g., 1e-4)
                    pattern = rf"^(\s*{k}\s*:\s*)[0-9\.eE\-]+(.*)$"
                    config_text = re.sub(pattern, rf"\g<1>{v}\g<2>", config_text, flags=re.MULTILINE)
                    
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(config_text)
                if epoch > 0:
                    self._last_tune_epoch = epoch
                self._tune_apply_count += 1
                self._last_applied = dict(safe_params)
                self._last_pre_metrics = {
                    "val_loss": float(metrics.get("val_loss", 0.0) or 0.0),
                    "val_sharpe": float(metrics.get("val_sharpe", 0.0) or 0.0),
                }
                print(f"[OllamaHelper] Updated {config_path}. Restarting training...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as e:
                print(f"[OllamaHelper] Failed to update config: {e}")

# Singleton-like instance
ollama = OllamaHelper()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama Helper CLI")
    parser.add_argument("action", choices=["analyze", "tune", "fix"])
    parser.add_argument("--model", type=str, default="haelt")
    parser.add_argument("--traceback", type=str, default="Unknown error")
    parser.add_argument("--metrics", type=str, default='{"best_val_loss": 1.0}')
    args = parser.parse_args()
    
    helper = OllamaHelper()
    
    if args.action == "analyze":
        helper.auto_fix_error(args.traceback, "CLI manual analysis")
    elif args.action == "fix":
        helper.auto_fix_error(args.traceback, "CLI auto-fix request")
    elif args.action == "tune":
        try:
            metrics = json.loads(args.metrics)
            helper.auto_tune_model(args.model, metrics)
        except json.JSONDecodeError:
            print("Invalid JSON for metrics")
