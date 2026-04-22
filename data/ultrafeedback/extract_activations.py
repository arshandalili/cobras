import argparse
from tqdm import trange

import pandas as pd
import torch

from odesteer.lm import HuggingFaceLM
from odesteer.utils import get_project_dir


@torch.no_grad()
def extract_base_activations(
    model: HuggingFaceLM,
    prompts: list[str],
    responses: list[str],
    layer_idx: int | list[int] | None,
):
    full_texts = [f'{p}\n{r}' for p, r in zip(prompts, responses)]
    return model.extract_prompt_eos_activations(full_texts, layer_idx)


@torch.no_grad()
def extract_chat_activations(
    model: HuggingFaceLM,
    prompts: list[str],
    responses: list[str],
    layer_idx: int | list[int] | None,
):
    messages = [[
        {"role": "user", "content": p},
        {"role": "assistant", "content": r}
    ] for p, r in zip(prompts, responses)]
    return model.extract_message_eos_activations(messages, layer_idx = layer_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type = str, default = 'Llama3.1-8B-Base')
    parser.add_argument('-l', '--layer_idx', type = int, default = -1)
    parser.add_argument('-b', '--batch_size', type = int, default = 10)
    args = parser.parse_args()

    data_dir = get_project_dir() / 'data' / 'ultrafeedback'
    texts_dir = data_dir / 'texts'
    activations_dir = data_dir / 'activations' / args.model
    activations_dir.mkdir(parents = True, exist_ok = True)

    model = HuggingFaceLM(args.model, device = "auto", dtype = torch.float32)
    layer_idx = None if args.layer_idx == -1 else args.layer_idx
    layer_tag = 'all' if layer_idx is None else str(layer_idx)
    
    if 'Base' in args.model and 'Qwen' not in args.model:
        extract_func = extract_base_activations
    else:
        extract_func = extract_chat_activations

    # Process train, val, test splits with pos/neg pairs
    for split in ['train', 'val', 'test']:
        for label in ['pos', 'neg']:
            file = texts_dir / f'{split}_{label}.jsonl'
            if not file.exists():
                print(f'Skipping {file.stem} - file not found')
                continue

            if layer_idx is None:
                print(f'Processing {file.stem} on all layers')
            else:
                print(f'Processing {file.stem} on layer {layer_idx}')
            if (activations_dir / f'{file.stem}_activations_layer{layer_idx}.pt').exists():
                print(f'Activations already exist for {file.stem} on layer {layer_idx}')
                continue

            df = pd.read_json(file, lines = True, orient = 'records')
            prompts, responses = df['prompt'].tolist(), df['response'].tolist()

            num_batches = (len(df) + args.batch_size - 1) // args.batch_size

            if layer_idx is None:
                num_layers = model.get_num_layers()
                missing_layers = [
                    i for i in range(num_layers)
                    if not (activations_dir / f'{file.stem}_activations_layer{i}.pt').exists()
                ]
                if len(missing_layers) == 0:
                    print(f'Activations already exist for {file.stem} on all layers')
                    continue
                activations_dict = {i: [] for i in missing_layers}
            else:
                save_path = activations_dir / f'{file.stem}_activations_layer{layer_idx}.pt'
                if save_path.exists():
                    print(f'Activations already exist for {file.stem} on layer {layer_idx}')
                    continue
                activations_lst = []

            for i in trange(num_batches):
                start_idx = i * args.batch_size
                end_idx = min((i + 1) * args.batch_size, len(df))
                batch_prompts = prompts[start_idx:end_idx]
                batch_responses = responses[start_idx:end_idx]
                # batch_activations = extract_func(model, batch_prompts, batch_responses, layer_idx = layer_idx)
                batch_activations = extract_func(
                    model,
                    batch_prompts,
                    batch_responses,
                    layer_idx=layer_idx,
                )

                if layer_idx is None:
                    for k in activations_dict:
                        activations_dict[k].append(batch_activations[k].cpu())
                else:
                    activations_lst.append(batch_activations.cpu())

                del batch_activations, batch_prompts, batch_responses
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            torch.save(df.idx.values, activations_dir / f'{file.stem}_question_idx.pt')

            if layer_idx is None:
                for k, v in activations_dict.items():
                    activations = torch.cat(v, dim=0)
                    torch.save(activations, activations_dir / f'{file.stem}_activations_layer{k}.pt')
            else:
                activations = torch.cat(activations_lst, dim=0)
                torch.save(activations, activations_dir / f'{file.stem}_activations_layer{layer_idx}.pt')