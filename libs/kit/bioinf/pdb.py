from Bio import PDB


def pdb_to_seqs(file_path):
    """ reads a PDB file and returns the sequence """
    parser = PDB.PDBParser()
    structure = parser.get_structure("X", file_path)
    seqs_mod = []
    for model in structure:
        seqs = {}
        for chain in model:
            seq = ""
            for residue in chain:
                if PDB.Polypeptide.is_aa(residue.get_resname()):
                    seq += PDB.Polypeptide.protein_letters_3to1[residue.get_resname()]

            seqs[chain.id] = seq
        seqs_mod.append(seqs)
    return seqs_mod
