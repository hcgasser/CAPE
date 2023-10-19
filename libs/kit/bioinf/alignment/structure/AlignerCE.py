import tempfile
from Bio.PDB import PDBParser
from Bio.PDB.cealign import CEAligner

import kit.data
from kit.bioinf import pdb_to_seq
from kit.bioinf.folding.esm import sharded_forward, convert_outputs_to_structure, convert_outputs_to_pdb


class AlignerCE:
    def __init__(self, ref_str=None, ref_pdbfile=None, save_pdbfile=None):
        if ref_str is None and ref_pdbfile is None:
            raise Exception("either seq_ref_str or seq_ref_file need to be set ")
        elif ref_str is not None:
            ref_str = ref_str.replace("*", "").replace("-", "")
            outputs = sharded_forward(ref_str)
            self.ref, pdb = convert_outputs_to_structure("reference", outputs)
            if save_pdbfile is not None:
                kit.data.str_to_file("\n".join(pdb), save_pdbfile)
        else:
            self.ref = PDBParser().get_structure("reference", ref_pdbfile)

        self.aligner = CEAligner()
        self.aligner.set_reference(self.ref)
        self.aligned_structure = None

    def calc_rmsd_to(self, src_str=None, src_pdbfile=None, save_pdbfile=None):
        if src_str is None and src_pdbfile is None:
            raise Exception("either seq_ref_str or seq_ref_file need to be set")
        elif src_str is not None:
            src_str = src_str.replace("*", "").replace("-", "")
            outputs = sharded_forward(src_str)
            src, pdb = convert_outputs_to_structure("source", outputs)
            if save_pdbfile is not None:
                kit.data.str_to_file("\n".join(pdb), save_pdbfile)
        else:
            src = PDBParser().get_structure("source", src_pdbfile)

        self.aligner.align(src)
        return self.aligner.rms
