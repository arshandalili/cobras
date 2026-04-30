import argparse
from tqdm import trange

import pandas as pd
import torch

from odesteer.utils import get_project_dir
from odesteer.lm import HuggingFaceLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='Llama3.1-8B-Base')
    parser.add_argument('-l', '--layer_idx', type=int, default=-1)
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    args = parser.parse_args()

    data_dir = get_project_dir() / 'data' / 'toxicity'
    jigsaw_dir = data_dir / 'jigsaw'
    activations_dir = data_dir / 'activations' / args.model
    activations_dir.mkdir(parents=True, exist_ok=True)

    model = HuggingFaceLM(args.model, device="auto", dtype=torch.float32)
    layer_idx = None if args.layer_idx == -1 else args.layer_idx

    final_train_df = pd.read_json(jigsaw_dir / 'final_train.jsonl', lines=True, orient='records')
    num_batches = (len(final_train_df) + args.batch_size - 1) // args.batch_size

    if layer_idx is None:
        num_layers = model.get_num_layers()

        missing_pos_layers = [
            i for i in range(num_layers)
            if not (activations_dir / f'jigsaw_pos_activations_layer{i}.pt').exists()
        ]
        missing_neg_layers = [
            i for i in range(num_layers)
            if not (activations_dir / f'jigsaw_neg_activations_layer{i}.pt').exists()
        ]

        missing_layers = sorted(set(missing_pos_layers + missing_neg_layers))

        if len(missing_layers) == 0:
            print('Jigsaw activations already exist for all layers')
            raise SystemExit

        pos_activations_dict = {i: [] for i in missing_layers}
        neg_activations_dict = {i: [] for i in missing_layers}
    else:
        pos_save_path = activations_dir / f'jigsaw_pos_activations_layer{layer_idx}.pt'
        neg_save_path = activations_dir / f'jigsaw_neg_activations_layer{layer_idx}.pt'

        if pos_save_path.exists() and neg_save_path.exists():
            print(f'Jigsaw activations already exist for layer {layer_idx}')
            raise SystemExit

        pos_activations_lst = []
        neg_activations_lst = []

    for i in trange(num_batches):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(final_train_df))
        batch_data = final_train_df.iloc[start_idx:end_idx]

        batch_pos_texts, batch_neg_texts = [], []
        for idx, label in enumerate(final_train_df['label'].iloc[start_idx:end_idx]):
            if label <= 0.5:
                batch_pos_texts.append(batch_data['text'].iloc[idx])
            else:
                batch_neg_texts.append(batch_data['text'].iloc[idx])

        if len(batch_pos_texts) > 0:
            batch_activations = model.extract_prompt_eos_activations(
                batch_pos_texts,
                layer_idx=missing_layers if layer_idx is None else layer_idx,
            )

            if layer_idx is None:
                for k in pos_activations_dict:
                    pos_activations_dict[k].append(batch_activations[k].cpu())
            else:
                pos_activations_lst.append(batch_activations.cpu())

        if len(batch_neg_texts) > 0:
            batch_activations = model.extract_prompt_eos_activations(
                batch_neg_texts,
                layer_idx=missing_layers if layer_idx is None else layer_idx,
            )

            if layer_idx is None:
                for k in neg_activations_dict:
                    neg_activations_dict[k].append(batch_activations[k].cpu())
            else:
                neg_activations_lst.append(batch_activations.cpu())

    if layer_idx is None:
        for k, v in pos_activations_dict.items():
            if len(v) > 0:
                torch.save(
                    torch.cat(v, dim=0),
                    activations_dir / f'jigsaw_pos_activations_layer{k}.pt',
                )

        for k, v in neg_activations_dict.items():
            if len(v) > 0:
                torch.save(
                    torch.cat(v, dim=0),
                    activations_dir / f'jigsaw_neg_activations_layer{k}.pt',
                )
    else:
        if len(pos_activations_lst) > 0:
            torch.save(
                torch.cat(pos_activations_lst, dim=0),
                activations_dir / f'jigsaw_pos_activations_layer{layer_idx}.pt',
            )

        if len(neg_activations_lst) > 0:
            torch.save(
                torch.cat(neg_activations_lst, dim=0),
                activations_dir / f'jigsaw_neg_activations_layer{layer_idx}.pt',
            )