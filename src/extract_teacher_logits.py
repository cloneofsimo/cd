from typing import List, Type, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM


from prepare_dataset import HOMAR_CONTEXT_EXAMPLE


class DistillationDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


@torch.no_grad()
def extract_logit_with_prefix(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_prompt_template: str = HOMAR_CONTEXT_EXAMPLE,  # "SYSTEMPROMPT : QUERY {}, ANSWER {}"
    input_sequence_pairs: List[Tuple[str, str]] = None,
    save_topk_logits_with_index: int = 100,
):
    def collate_fn(batch):
        return tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt", max_length=2048
        )

    padding_id = tokenizer.pad_token_id

    input_prompts = [
        input_prompt_template.format(queries, answers)
        for queries, answers in input_sequence_pairs
    ]

    len_responses = [
        len(tokenizer.encode(answers)) for queries, answers in input_sequence_pairs
    ]
    global_idx = 0

    dataset = DistillationDataset(input_prompts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    outs = []

    for batch in dataloader:
        batch["input_ids"] = batch["input_ids"].to(model.device)
        mask = batch["input_ids"] != padding_id
        output = model(**batch)
        logits = output.logits
        # split and remove padding
        logits = torch.split(logits, 1, dim=0)
        for idx, logit in enumerate(logits):
            print(logit.shape)
            logit = logit.squeeze(0)
            len_seq = mask[idx].sum().item()

            this_logit = logit[: len_seq - 1, :]
            topk_logits, topk_indices = torch.topk(
                this_logit, save_topk_logits_with_index
            )
            print(topk_logits.shape, topk_indices.shape)
            individual_index = batch["input_ids"][idx][:len_seq].cpu()

            len_response = len_responses[global_idx]
            global_idx += 1

            outs.append(
                (
                    topk_logits.cpu()[-len_response:,],
                    topk_indices.cpu()[-len_response:,],
                    individual_index[-len_response:,],
                )
            )

            # check if the saving range is correct
            print(tokenizer.decode(individual_index[-len_response:,]))

    return outs


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("./homar_simpson.csv")

    quries = df["query"].tolist()
    responses = df["response"].tolist()

    print(len(quries))

    model_name = "meta-llama/Llama-2-13b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    outs = extract_logit_with_prefix(
        model=model,
        tokenizer=tokenizer,
        input_sequence_pairs=list(zip(quries, responses)),
        save_topk_logits_with_index=100,
    )

    torch.save(outs, "homar_simpson.pt")
