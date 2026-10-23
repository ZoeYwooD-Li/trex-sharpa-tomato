"""Small behavioral check for LoRA freezing, update, save, and merge."""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import torch
from torch import nn

from sharpa_lora import inject_lora, lora_state_dict, merge_lora_weights, save_lora


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj_action = nn.Linear(4, 4, bias=False)
        layer.self_attn.q_proj_tactile = nn.Linear(4, 4, bias=False)
        layer.mlp_action = nn.Module()
        layer.mlp_action.gate_proj = nn.Linear(4, 4, bias=False)
        layer.mlp_tactile = nn.Module()
        layer.mlp_tactile.gate_proj = nn.Linear(4, 4, bias=False)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])
        self.model.gradient_checkpointing = False
        self.visual = nn.Linear(4, 4, bias=False)
        self.final_layer = nn.Module()
        self.final_layer.mlp = nn.Sequential(nn.Linear(4, 4, bias=False))
        self.final_layer_tactile = nn.Module()
        self.final_layer_tactile.mlp = nn.Sequential(nn.Linear(4, 4, bias=False))

    def forward(self, x):
        layer = self.model.layers[0]
        a = layer.self_attn.q_proj_action(x) + layer.mlp_action.gate_proj(x)
        t = layer.self_attn.q_proj_tactile(x) + layer.mlp_tactile.gate_proj(x)
        return self.final_layer.mlp(a) + self.final_layer_tactile.mlp(t)


def test():
    torch.manual_seed(42)
    model = Tiny()
    original = copy.deepcopy(model)
    x = torch.randn(3, 4)
    config = inject_lora(model, rank=2, alpha=4, dropout=0.0)
    assert model.model.gradient_checkpointing
    assert len(config["target_modules"]) == 6
    assert torch.allclose(model(x), original(x), atol=1e-6)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(name.endswith((".lora_a", ".lora_b")) for name in trainable)
    assert not any(name.startswith("visual.") for name in trainable)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    loss = model(x).square().mean()
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)
               for name, p in model.named_parameters() if name.endswith(".lora_b"))
    optimizer.step()
    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        save_lora(model, tmp, config)
        saved = torch.load(Path(tmp) / "lora.pt", weights_only=True)
        assert saved.keys() == lora_state_dict(model).keys()
        merged = merge_lora_weights(original.state_dict(), saved, config)
        restored = Tiny()
        restored.load_state_dict(merged)
        restored.eval()
        assert torch.allclose(model(x), restored(x), atol=1e-5)
    print("LoRA freeze/update/save/merge: passed")


if __name__ == "__main__":
    test()
