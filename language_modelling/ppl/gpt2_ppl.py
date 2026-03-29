import numpy as np
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def conditional_perplexity(
    rows, model, tokenizer, device='cuda'
):
    """
    rows format:
    [
        {
            "context": "I have a",
            "generations": [
                " laptop",
                " meeting",
                ...
            ]
        },
        ...
    ]
    """
    perplexities = []
    goodperplexities = []
    total_nll = 0
    total_tokens = 0
    g = 0
    ct = 0

    # for every prompt
    for i, row in tqdm(
        enumerate(rows),
        total=len(rows),
        desc='Evaluating PPL',
    ):
        # prompt_input_ids = torch.LongTensor([row.prompt['tokens']]).to(device)
        prompt = row["context"]
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        if not (
            prompt_input_ids.shape[1] == 1
            and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id
        ) and not prompt_input_ids.shape[1] == 0:  # this means unconditional, prompt is BOS token (verify)
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (
                prompt_input_ids.shape[1] - 1
            )
            # print("in")
        else:
            prompt_loss = 0
            # print("out")
        # for every generation conditioned on the prompt
        generations = row["generations"]
        for gen in generations:
            full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(
                device
            )
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (
                full_input_ids.shape[1] - 1
            )
            loss = (full_loss - prompt_loss) / (
                full_input_ids.shape[1] - prompt_input_ids.shape[1]
            )

            ppl = np.exp(loss.item())
            if ppl < 100:  # for sanity
                goodperplexities.append(ppl)
                # perplexities.append(ppl)
                g += 1

            if ppl < 1e4:
                perplexities.append(ppl)
            else:
                print("ppl values are weirldly large. Check for errors")
                print(f"\n########\n{gen}\n########\n")

            total_nll += (full_loss - prompt_loss).item()
            total_tokens += full_input_ids.shape[1] - prompt_input_ids.shape[1]
            # print(full_input_ids[0], prompt_input_ids[0])
            # print(full_loss, prompt_loss)
            # input()

    return np.nanmean(perplexities), np.exp(total_nll / total_tokens)



def compute_perplexity(generations, device):
    """
    Compute perplexity of a batch of texts.
    """
    eval_modelname = 'gpt2-xl'
    print(f'computing {eval_modelname} ppl')
    eval_model = AutoModelForCausalLM.from_pretrained(
        eval_modelname
    ).to(device)
    eval_tokenizer = AutoTokenizer.from_pretrained(eval_modelname)
    torch.cuda.empty_cache()
    with torch.no_grad():
        ppl, total_ppl = conditional_perplexity(
            generations,
            eval_model,
            eval_tokenizer,
            device=device,
        )
    return ppl, total_ppl
