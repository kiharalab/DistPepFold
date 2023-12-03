import protein

TEST_PDB_PATH = "/net/kihara-fast-scratch/jain163/attentivedist2_data/renum_chain/11ASA.pdb"

def main():
	with open(TEST_PDB_PATH) as f:
		pdb_str = f.read()
	prot = protein.from_pdb_string(pdb_str)
	print(prot.atom_positions[30])

if __name__=="__main__":
	main()
