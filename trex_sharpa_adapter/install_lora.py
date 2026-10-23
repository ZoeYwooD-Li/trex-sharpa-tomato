"""Add optional single-GPU LoRA support to an installed Sharpa T-Rex checkout."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    if source.count(old) != 1:
        raise RuntimeError(f"Unexpected upstream {label}; inspect before patching")
    return source.replace(old, new, 1)


def install(repo: Path) -> None:
    repo = repo.resolve()
    trainer = repo / "scripts" / "train.py"
    source = trainer.read_text(encoding="utf-8")
    if 'choices=["json", "lerobot", "sharpa"]' not in source:
        raise RuntimeError("Install the v3 Sharpa adapter before LoRA")
    backup = trainer.with_suffix(".py.pre_sharpa_lora")
    if not backup.exists():
        shutil.copy2(trainer, backup)

    source = replace_once(source,
        "from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds\n",
        "from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds\n"
        "from qwen_vla.sharpa_lora import inject_lora, save_lora\n",
        "LoRA import")
    source = replace_once(source,
        '        sd = accelerator.get_state_dict(model)\n'
        '        torch.save(sd, os.path.join(save_dir, "model.pt"))',
        '        if args.lora_rank > 0:\n'
        '            save_lora(accelerator.unwrap_model(model), save_dir, args.lora_config)\n'
        '        else:\n'
        '            sd = accelerator.get_state_dict(model)\n'
        '            torch.save(sd, os.path.join(save_dir, "model.pt"))',
        "checkpoint storage")
    source = replace_once(source,
        '                "model_path": args.model_path,',
        '                "model_path": args.model_path,\n'
        '                "lora_rank": args.lora_rank,\n'
        '                "lora_alpha": args.lora_alpha,\n'
        '                "lora_dropout": args.lora_dropout,',
        "checkpoint training args")
    source = replace_once(source,
        '        self.world_size   = dist.get_world_size()',
        '        self.world_size   = dist.get_world_size() if dist.is_initialized() else 1',
        "single-GPU metric size")
    source = replace_once(source,
        '        for t in [self.total_loss, self.action_loss, self.tactile_loss, self.flare_loss]:\n'
        '            dist.all_reduce(t, op=dist.ReduceOp.SUM)',
        '        for t in [self.total_loss, self.action_loss, self.tactile_loss, self.flare_loss]:\n'
        '            if dist.is_initialized():\n'
        '                dist.all_reduce(t, op=dist.ReduceOp.SUM)',
        "single-GPU metric reduction")
    source = replace_once(source,
        '    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu\n'
        '    accelerator.state.deepspeed_plugin.deepspeed_config["train_batch_size"] = (args.train_bsz_per_gpu * dist.get_world_size() * accelerator.gradient_accumulation_steps)',
        '    if accelerator.state.deepspeed_plugin is not None:\n'
        '        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu\n'
        '        accelerator.state.deepspeed_plugin.deepspeed_config["train_batch_size"] = (\n'
        '            args.train_bsz_per_gpu * accelerator.num_processes * accelerator.gradient_accumulation_steps)',
        "optional DeepSpeed")
    source = replace_once(source,
        '        // dist.get_world_size()',
        '        // accelerator.num_processes',
        "single-GPU step count")
    source = replace_once(source,
        '        else:\n'
        '            param.requires_grad = True\n\n'
        '    # Keep the embedded VQ-VAE',
        '        else:\n'
        '            param.requires_grad = True\n\n'
        '    if args.lora_rank > 0:\n'
        '        args.lora_config = inject_lora(\n'
        '            model, args.lora_rank, args.lora_alpha, args.lora_dropout,\n'
        '            gradient_checkpointing=bool(args.lora_gradient_checkpointing))\n'
        '        accelerator.print(f"LoRA: {len(args.lora_config[\'target_modules\'])} linear layers, "\n'
        '                          f"{args.lora_config[\'trainable_parameters\']:,} trainable parameters")\n\n'
        '    # Keep the embedded VQ-VAE',
        "LoRA injection")
    source = replace_once(source,
        '    parser.add_argument("--max_train_batches", type=int, default=0, help="Stop after this many training batches; 0 disables the short-run limit")',
        '    parser.add_argument("--max_train_batches", type=int, default=0, help="Stop after this many training batches; 0 disables the short-run limit")\n'
        '    parser.add_argument("--lora_rank", type=int, default=0, help="0: full fine-tuning; >0: train only LoRA matrices")\n'
        '    parser.add_argument("--lora_alpha", type=float, default=16.0)\n'
        '    parser.add_argument("--lora_dropout", type=float, default=0.05)\n'
        '    parser.add_argument("--lora_gradient_checkpointing", type=int, default=1)',
        "LoRA command-line options")

    shutil.copy2(Path(__file__).with_name("sharpa_lora.py"), repo / "qwen_vla" / "sharpa_lora.py")
    trainer.write_text(source, encoding="utf-8")
    print("Installed optional LoRA and single-GPU path; backup:", backup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    install(parser.parse_args().repo)
