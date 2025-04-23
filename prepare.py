import os
import pickle as pkl
import numpy as np
import json
from tqdm import tqdm
def create_dir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)
		
pwd = '/net/kihara/home/ykagaya/Share/CAPRI55/AFrun/AF232/T231'
dest = '/home/kihara/zhan1797/Desktop/DPFold/examples'
target = 'T231'
fasta = os.path.join(pwd, target+'.fasta')
emb_dir = os.path.join(pwd, target)
ranking = os.path.join(emb_dir, 'ranking_debug.json')
confidence = json.load(open(ranking, 'r'))

rank = 0
write = []
for i in tqdm(range(len(confidence['order']))):
    model = confidence['order'][i]
    name = f'{target}_{model}_{rank}'
    out_dir = os.path.join(dest, name)
    create_dir(out_dir)
    # copy fasta
    os.system(f'cp {fasta} {out_dir}/{name}.fasta')
    emb_out = os.path.join(out_dir, name)
    create_dir(emb_out)

    pkl_file = os.path.join(emb_dir, 'result_' + model + '.pkl')
    with open(pkl_file, 'rb') as f:
        result = pkl.load(f)
    # save embedding
    np.savez(os.path.join(emb_out, 'model_1_multimer.npz'), 
                        single=result['representations']['single'],
                        pair=result['representations']['pair'])
    rank += 1
    write.append(name + '\n')

with open('T231_targets', 'w') as f:
    f.writelines(write)