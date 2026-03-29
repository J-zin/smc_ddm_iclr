"""

Convert the output of the MDLM model to the format expected by the SSD-LM evaluation script.

"""

# Current:
# {"text": "<|endoftext|>\n\The last time ..."}

# Goal (for SSD-LM eval script):
# {
#   "context_len": 5,
#   "context": [],
#   "context_string": "",
#   "len": 0,
#   "tokens": [],
#   "string": [],
#   "gold_tokens": [],
#   "gold_string": ""
# }

import os
import glob
import json

import click

from transformers import AutoTokenizer


def get_possible_prompts(prompt_path):
    with open(prompt_path) as f:
        return [json.loads(line)["context_string"] for line in f]


def file_to_exp_info(file):
    parent_dir = os.path.dirname(file)
    info_path = os.path.join(parent_dir, 'info.json')

    with open(info_path) as f:
        relevant_config = json.load(f)['fk_steering']

    relevant_keys = [
        'potential_type',
        'k_particles',
        'lmbda',
        'reward_fn',
        'reward_label',
        'num_x0_samples',
    ]
    relevant_config = '_'.join([str(relevant_config[key]) for key in relevant_keys])

    return relevant_config


def load_texts(file):
    with open(file) as f:
        return [json.loads(line)["text"] for line in f]


def process_prompted_output(prompt_to_text, tokenizer, trim_len=50):
    prompt_to_data = {prompt: {} for prompt in prompt_to_text}

    for prompt, texts in prompt_to_text.items():
        cleaned_texts = []
        tokenized = []
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        prompt_to_data[prompt]["context_string"] = prompt
        prompt_to_data[prompt]["context_len"] = prompt_len
        prompt_to_data[prompt]["context"] = prompt_tokens

        for text in texts:
            tokenized_text = tokenizer.encode(text, add_special_tokens=False)[
                prompt_len : prompt_len + trim_len
            ]
            decoded_text = tokenizer.decode(tokenized_text)

            print('\t', decoded_text)

            cleaned_texts.append(decoded_text)
            tokenized.append(tokenized_text)

        prompt_to_data[prompt]["string"] = cleaned_texts
        prompt_to_data[prompt]["tokens"] = tokenized
        prompt_to_data[prompt]["len"] = len(tokenized[0])

    return prompt_to_data


def process_file(*, file, prompts, expected_per, tokenizer, max_len):
    # config_info = file_to_exp_info(file)
    config_info = "abc" # not needed, I think
    texts = load_texts(file)
    texts = [text.strip('<|endoftext|>') for text in texts]
    texts = ['\n\n' + text.strip() for text in texts]

    prompt_to_text = {prompt: [] for prompt in prompts}
    for text in texts:
        found_prompt = [prompt for prompt in prompts if text.startswith(prompt)]
        assert len(found_prompt) == 1
        found_prompt = found_prompt[0]
        prompt_to_text[found_prompt].append(text)

    # confirm that the number of samples per prompt is as expected
    for prompt, text in prompt_to_text.items():
        assert len(text) == expected_per

    prompt_to_data = process_prompted_output(prompt_to_text, tokenizer, max_len)
    return config_info, prompt_to_data


@click.command()
@click.option(
    '--glob_expression',
    default="../outputs/openwebtext-train/*/*/*/sample_evaluation/*/text_samples.jsonl",
    help='Glob pattern for input files.',
)
@click.option(
    '--prompt_path',
    default='pplm_discrim_prompts_orig.jsonl',
    help='Path to the prompt file.',
)
@click.option(
    '--max_len', default=50, type=int, help='Max length of generated text to consider.'
)
@click.option(
    '--expected_per',
    default=20,
    type=int,
    help='Expected number of samples per prompt.',
)
def main(glob_expression, prompt_path, max_len, expected_per):
    tokenizer_name = 'roberta-large'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    prompts = get_possible_prompts(prompt_path)
    print(prompts)

    files = list(glob.glob(glob_expression))
    print(files)
    assert len(files) > 0

    for file in files:
        print(file)
        config_info, prompt_to_data = process_file(
            file=file,
            prompts=prompts,
            expected_per=expected_per,
            tokenizer=tokenizer,
            max_len=max_len,
        )
        # get parent dir path
        s_path = os.path.join(os.path.dirname(file), config_info + '_ssdlm_gen.jsonl')

        with open(s_path, 'w') as f:
            for _, data in prompt_to_data.items():
                f.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    main()
