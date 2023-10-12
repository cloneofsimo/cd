import os
from typing import List, Type, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

import time
from prepare_dataset import HOMAR_CONTEXT_EXAMPLE
from peft import get_peft_model, LoraConfig

import lightning as L


class LogitIndexDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.path = path
        self.tokenizer = tokenizer
        self.logit_index_data = torch.load(
            path
        )  # this should be all preprocessed logits and index

    def __len__(self):
        return len(self.logit_index_data)

    def __getitem__(self, index):
        logit, index, data = self.logit_index_data[index]

        # logit is [T-1, K], index is [T-1, K], data is [T]
        # explode it to [T-1, V]

        vocab_size = self.tokenizer.vocab_size

        # scatter logit at index
        logit_scatter = torch.zeros(logit.shape[0], vocab_size)
        logit_scatter.scatter_(1, index, logit)

        return logit_scatter, data


def train(
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf",
    path="./homar_simpson.pt",
    out_dir="./test_checkpoints/",
    log_wandb=True,
    # hyperparam
    eval_interval_step=10,
    save_interval_step=20,
    num_epochs=100,
    log_interval=1,
    learning_rate=3e-4,
    batch_size=16,
    micro_batch_size=2,  # gradient accumulation = batch_size / micro_batch_size
    weight_decay=0.0,
    lora_r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    warmup_iters=10,
    mse_coeff=0.5,
    ce_coeff=0.5,
    
):
    
    

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    
    PAD_TOKEN_ID = tokenizer.unk_token_id
    
    def collate_fn_with_padding(batches):
        # batches is a list of (logit, target)
        # logit is [T-1, V], target is [T]
        # pad logit to [T, V]
        # pad target to [T]
        # return logit, target
        logit, data = zip(*batches)
        logit = nn.utils.rnn.pad_sequence(logit, batch_first=True, padding_value=PAD_TOKEN_ID)
        data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=PAD_TOKEN_ID)
        return logit, data
    
    
    
    dataloader = DataLoader(
        LogitIndexDataset(path, tokenizer),
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_padding,
    )

    # setup everything here
    if log_wandb:
        import wandb

        wandb.init(project="qlora", entity="simo")

    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.config.use_cache = False

    # print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fabric.print(f"Number of trainable parameters: {num_params}")
    # print the number of total parameters
    num_params_all = sum(p.numel() for p in model.parameters())
    fabric.print(f"Number of total parameters: {num_params_all}")
    # ratio of trainable parameters
    fabric.print(
        f"Ratio of trainable parameters: {100 * num_params / num_params_all:.4f} %"
    )
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    model.train()
    
    gradient_accumulation_iters = batch_size // micro_batch_size
    
    
    step_count = 0

    for iter_num in range(num_epochs):
        for teacher_logits, data in dataloader:
            print(data.shape)
            if step_count <= warmup_iters:
                # linear warmup
                lr = learning_rate * step_count / warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            t0 = time.time()
            data = fabric.to_device(data)
            teacher_logits = fabric.to_device(teacher_logits)[:, :-1, :]
            
            with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
                
                logits = model(data[:, :-1]).logits.contiguous()
                
                target = data[:, 1:].reshape(-1).contiguous()
                
                vocab_size = logits.shape[-1]
                
                mask = (target != PAD_TOKEN_ID).float().reshape(-1, 1)
                
                elementwise_mse = (logits.reshape(-1, vocab_size) - teacher_logits.reshape(-1, vocab_size)).pow(2)
                
                mse_loss = (elementwise_mse * mask).mean()
                
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size),target,ignore_index = PAD_TOKEN_ID) * ce_coeff \
                    + mse_loss * mse_coeff
                
                
                fabric.backward(loss / gradient_accumulation_iters)

            if (iter_num + 1) % gradient_accumulation_iters == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1
                    
                # if step_count % eval_interval_step == 0:
                #     val_loss = validate(fabric, model, val_data, tokenizer_path)
                #     fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                #     fabric.barrier()

                if step_count % save_interval_step == 0:
                    print(f"Saving LoRA weights to {out_dir}")
                    
                    # We are only saving the LoRA weights
                    model.save_pretrained(os.path.join(out_dir, f"iter-{step_count:06d}-adaptor"))
                    # fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"), checkpoint)

            dt = time.time() - t0
            if iter_num % log_interval == 0:
                wandb.log({"train_loss": loss.item()})
                fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    
    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    path = "./homar_simpson.pt"
    #train()
    
    
    def collate_fn_with_padding(batches):
        # batches is a list of (logit, target)
        # logit is [T-1, V], target is [T]
        # pad logit to [T, V]
        # pad target to [T]
        # return logit, target
        logit, data = zip(*batches)
        logit = nn.utils.rnn.pad_sequence(logit, batch_first=True, padding_value=PAD_TOKEN_ID)
        data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=PAD_TOKEN_ID)
        return logit, data

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    PAD_TOKEN_ID = tokenizer.unk_token_id
    print(PAD_TOKEN_ID)
    dataloader = DataLoader(
        LogitIndexDataset(path, tokenizer),
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn_with_padding,
    )
    
    for teacher_logits, data in dataloader:
        print(data)
        break
    
    train()
    