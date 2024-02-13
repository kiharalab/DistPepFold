from torch.utils.data.dataset import Dataset
from os.path import join
import torch
import numpy as np
from dataset.openfold_util import residue_constants
from dataset.af2_util import all_atom

def parse_fasta(fasta_string: str):
  """Parses FASTA string and returns list of strings with amino-acid sequences.

  Arguments:
    fasta_string: The string contents of a FASTA file.

  Returns:
    A tuple of two lists:
    * A list of sequences.
    * A list of sequence descriptions taken from the comment lines. In the
      same order as the sequences.
  """
  sequences = []
  descriptions = []
  index = -1
  for line in fasta_string.splitlines():
    line = line.strip()
    if line.startswith('>'):
      index += 1
      descriptions.append(line[1:])  # Remove the '>' at the beginning.
      sequences.append('')
      continue
    elif not line:
      continue  # Skip blank lines.
    sequences[index] += line

  return sequences, descriptions

def get_aatype(sequences):
    aatype = []
    for seq in sequences:
        for aa in seq:
            #print(aa)
            #res_shortname = residue_constants.restype_3to1.get(aa, 'X')
            res_shortname = aa
            restype_idx = residue_constants.restype_order.get(res_shortname, residue_constants.restype_num)
            aatype.append(restype_idx)
    return np.array(aatype)

def get_chain_index(sequences):
    chain_index = []
    counter = 1
    for seq in sequences:
        chain_index.extend(len(seq) * [counter])
        counter += 1
    return np.array(chain_index)

class MultimerDataset(Dataset):

    def __init__(self, args):
        self.args = args
        
    def __len__(self):
        return 1

    def __getitem__(self, index):
        embedding_dir = self.args.embedding_dir
        fasta = join(embedding_dir, f'input.fasta')

        with open(fasta, 'r') as f:
            fasta_str = f.read()
        sequences, _ = parse_fasta(fasta_str)
        target_lengths = [len(a) for a in  sequences]
        target_seq_len = sum(target_lengths)
        data = {}
        data["target"] = 'test'
        data['seq_length'] = target_seq_len
        #loading the embeddings
        emd_dir = join(embedding_dir, f'model_1_multimer.npz')
        emd = np.load(emd_dir)
        single = torch.tensor(emd['single'])
        pair = torch.tensor(emd['pair'])
        single_test, pair_test = single.unsqueeze(0), pair.unsqueeze(0)

        aatype = get_aatype(sequences)
        #print(aatype)
        chain_idx = get_chain_index(sequences)
        protein_test = dict()
        protein_test['aatype'] = aatype
        protein_test['residue_index'] = np.array(range(target_seq_len), dtype=np.int32)
        protein_test = all_atom.make_atom14_positions(protein_test, training=False)
        for key in protein_test:
            protein_test[key] = torch.tensor(protein_test[key]).unsqueeze(0)
        
        data['test'] = {
            'aatype': torch.tensor(aatype).unsqueeze(0),
            'protein': protein_test,
            'chain_idx': chain_idx,
            'single': single_test,
            'pair': pair_test,
        }
       
        return data
