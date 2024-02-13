import argparse
import functools
import os
from os.path import join
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset.helpers import collate_fn
from dataset.dataset import MultimerDataset

from model.alphafold_finetune_multimer import AlphaFold_Multimer
import util
from model.openfold.feats import atom14_to_atom37
from dataset.openfold_util import residue_constants
from dataset.af2_util import protein_alt
import json
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_dataset(args):
    return MultimerDataset(args)

def val(args, model, val_dataloader):
    model.eval()
    confidence_list = {}
    for step, (batch, targets) in enumerate(val_dataloader):
        single_test = batch['test']['single']
        pair_test = batch['test']['pair']
        aatype_test = batch['test']['aatype']

        protein_test = batch['test']['protein']
        protein_test.update({'chain_idx': torch.tensor(batch['test']['chain_idx'])})
        protein_test_device = protein_test.copy()

        if args.cuda:
            single_test = single_test.cuda(args.device_id)
            pair_test = pair_test.cuda(args.device_id)
            aatype_test = aatype_test.cuda(args.device_id)
            for key in protein_test_device.keys():
                protein_test_device[key] = protein_test_device[key].cuda(args.device_id)
        
        protein_test_device.update({'target': targets})
        with torch.no_grad():            
            postition_full, confidence, docking_score = model(pair_test, single_test, aatype_test, None, None, None, protein_test=protein_test_device, training=False)

        confidence = confidence.cpu().numpy()
        plddt_b_factors = np.repeat(
            confidence[..., None], residue_constants.atom_type_num, axis=-1
        )
        final_pos = atom14_to_atom37(postition_full[-1].cpu(), protein_test)
        final_atom_mask = protein_test["atom37_atom_exists"]
        dist_per_residue = np.zeros_like(final_atom_mask.squeeze(0))
        prot_test = protein_alt.Protein(
            aatype=aatype_test.squeeze(0).cpu().numpy(),
            atom_positions=final_pos.squeeze(0).cpu().numpy(),
            atom_mask=final_atom_mask.squeeze(0).cpu().numpy(),
            residue_index=protein_test['residue_index'].squeeze(0).numpy(),
            b_factors=plddt_b_factors[0],
            chain_index=batch['test']['chain_idx']
        )
        pdb_lines = protein_alt.to_pdb(prot_test)
        #print(targets)
        output_dir_pred = os.path.join(args.output_dir, f"{targets}_pred_all_full.pdb")
        with open(output_dir_pred, 'w') as f:
            f.write(pdb_lines)
        #confidence_list.update({targets: docking_score.item()})
    # ranked_order = [m for m, _ in sorted(confidence_list.items(), key=lambda x: x[1], reverse=True)]
    # ranking_output_path = os.path.join(args.output_dir , 'ranking_debug.json')
    # with open(ranking_output_path, 'w') as f:
    #     label = 'iptm+ptm'
    #     f.write(json.dumps(
    #         {label: confidence_list, 'order': ranked_order}, indent=4))
    
    return "done"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="/train_chains", type=str,
        help="File of targets for training")
    parser.add_argument("--output_dir",default="test_run",type=str,
        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--model_dir",default="",type=str,
        help="model directory if load model from checkpoints")
    parser.add_argument("--embedding_dir",default="",type=str,
        help="The directory where pre-generated msa embeddings are stored.")

    parser.add_argument("--device_id", type=int, default=0, help="cude device id")
    parser.add_argument("--seed", type=int, default=999, help="random seed for initialization")

    parser.add_argument("--ipa_depth", type=int, default=8, help="depth of ipd block")
    parser.add_argument("--point_scale", type=int, default=1, help="point scale for translations")
    parser.add_argument("--contact", default=False, action="store_true", help='whether to contact prediction is in the model')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    args.device_id = 0
    # Set CUDA
    args.cuda = True if torch.cuda.is_available() else False
    args.n_gpu = 1 #Only use 1 gpu for now

    #Print and save args
    util.print_options(args)

    # Set seed
    set_seed(args)

    # Get datasets
    val_dataset = get_dataset(args)
    
    collate = functools.partial(collate_fn, args=args)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=3, collate_fn=collate)

    #Get model
    model = AlphaFold_Multimer(args)

    if args.model_dir:
        model.load_state_dict(
            torch.load(f'{args.model_dir}/model_state_dict.pt', map_location='cpu')
        )
        model.cuda(args.device_id)
        print(f'Checkpoints (model) loaded from {args.model_dir}')

    if args.cuda:
        model.cuda(args.device_id)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        #print(f'layer {name}: {param}')
        total_params+=param
    print('trainable parameters: ', total_params)
    # Training
    print(val(args, model, val_dataloader))

if __name__ == "__main__":
    main()

    
