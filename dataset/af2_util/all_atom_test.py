import all_atom
import jax
import jax.numpy as jnp
import numpy as np

import protein


TEST_PDB_PATH = "/net/kihara-fast-scratch/jain163/attentivedist2_data/renum_chain/11ASA.pdb"

def main():
	with open(TEST_PDB_PATH) as f:
		pdb_str = f.read()
	prot = protein.from_pdb_string(pdb_str)
	gt_frames_res = all_atom.atom37_to_frames(
							jnp.asarray(prot.aatype),
							jnp.asarray(prot.atom_positions),  # (..., 37, 3)
							jnp.asarray(prot.atom_mask),
							)
	gt_angles_res = all_atom.atom37_to_torsion_angles(
							jnp.asarray(np.expand_dims(prot.aatype,0)),
							jnp.asarray(np.expand_dims(prot.atom_positions,0)),  # (..., 37, 3)
							jnp.asarray(np.expand_dims(prot.atom_mask,0)),
							)

	print(gt_frames_res['rigidgroups_gt_frames'].shape)
	print(gt_angles_res['torsion_angles_sin_cos'].shape)

if __name__=="__main__":
	main()
