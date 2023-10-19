import os
import shutil
import tempfile
import subprocess
import numpy as np
from copy import deepcopy

from Bio import Align, AlignIO
from Bio.Align import substitution_matrices
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

from kit.log import log_info
from kit.bioinf.sf import SequenceFrame
from kit.bioinf.fasta import write_fasta, read_fasta, seqs_to_fasta, fastas_to_seqs
from kit.utils import temp_working_directory


class PairwiseSequenceAligner:
    def __init__(self, substitution_matrix="BLOSUM62", open_gap_score=-2, extend_gap_score=-1, wildcard="-"):
        self.aligner = Align.PairwiseAligner()  # https://biopython.org/docs/1.75/api/Bio.Align.html#Bio.Align.PairwiseAligner
        self.aligner.substitution_matrix = substitution_matrices.load(substitution_matrix)
        self.aligner.open_gap_score = open_gap_score
        self.aligner.extend_gap_score = extend_gap_score
        self.aligner.wildcard = wildcard    

    def score_seq_to_seq(self, seq1, seq2, translate=('', '', '-*')):
        seq1 = seq1.translate(str.maketrans(*translate))
        seq2 = seq2.translate(str.maketrans(*translate))
        return self.aligner.score(seq1, seq2)

    def score_seq_to_seqs(self, seq, tgts, translate=('', '', '-*')):
        result = []
        for i, tgt in enumerate(tgts):
            s = self.score_seq_to_seq(seq, tgt, translate=translate)
            result.append(s)
        return np.array(result)

    def score_seqs_to_seqs(self, seqs, tgts, translate=('', '', '-*')):
        results = []
        for seq in seqs:
            results.append(self.score_seq_to_seqs(seq, tgts, translate=translate))
        return np.stack(results, axis=0)
    
    def get_seq_to_seqs_dissimilarity(self, seq, seqs):
        max_score = self.score_seq_to_seq(seq, seq)
        scores = self.score_seq_to_seqs(seq, seqs)
        return max_score - scores

    def align_seq_to_seq(self, seq1, seq2, translate=('', '', '-*')):
        seq1 = seq1.translate(str.maketrans(*translate))
        seq2 = seq2.translate(str.maketrans(*translate))
        return self.aligner.align(seq1, seq2)[0]

    def align_seq_to_seqs(self, seq, tgts, translate=('', '', '-*')):
        result = []
        for i, tgt in enumerate(tgts):
            best_alignment = self.align_seq_to_seq(seq, tgt)
            result.append(best_alignment)
        return result

    def get_alignmnet_to_closest(self, seq, tgts, translate=('', '', '-*')):
        best_alignments = self.align_seq_to_seqs(seq, tgts, translate=translate)
        best_scores = [a.score for a in best_alignments]
        best_idx = np.argmax(best_scores)
        return best_alignments[best_idx]
    

class MultipleSequenceAligner:
    def __init__(self, sf, distance_model_name='blosum62', phylogenetic_tree_model_name='nj'):
        self.sf = sf
        self.msa_file_path = None
        self.msa = None
        self.msa_annotations = None
        self.occupancy = None
        self.occupancy_threshold = None
        self.occupancy_indices_above_threshold = None
        self.distance_model_name = distance_model_name
        self.distance_matrix = None
        self.phylogenetic_tree_model_name = phylogenetic_tree_model_name

    def align(self, msa_file_path=None, calculate_distances=False):
        """Align sequences in self.sf using MUSCLE.
        
        Args:
            msa_file_path (str): path to the output MSA file. If None, the MSA file will be saved in a temporary directory.
               if the filename ends with '.phy', the MSA file will be saved in PHYLIP format otherwise in FASTA format.
            calculate_distances (bool): whether to calculate distances using the specified distance model.
        """

        self.msa_file_path = msa_file_path
        with temp_working_directory() as tmp_dir_path:
            input_file_path = os.path.join(tmp_dir_path, "input.fa")
            output_file_path = os.path.join(tmp_dir_path, "output.afa") if msa_file_path is None else msa_file_path
            
            self.sf.to_fasta(input_file_path, save_src=False)
            mode = '-super5' if len(self.sf) > 500 else '-align'
            command = ['muscle',
               mode, input_file_path,
               '-output', output_file_path
            ]

            log_info(f"Muscle: {' '.join(command)}", start=True)
            result = subprocess.run(command, capture_output=True)
            log_info(f"Muscle finished", stop=True)
            self.msa = AlignIO.read(output_file_path, 'fasta')

            sf_msa = SequenceFrame()
            if output_file_path.endswith('.phy'):
                self.write_phy(output_file_path)
                sf_msa.from_phy(output_file_path)
            else:
                sf_msa.from_fasta(output_file_path)
            assert len(self.sf) == len(sf_msa)

            if calculate_distances:
                self.calculate_distances()

            self.sf.df_src['seq_msa'] = ''
            for (idx_pre, row_pre), (idx_post, row_post) in zip(self.sf.df.iterrows(), sf_msa.df_src.iterrows()):
                seq_msa = row_post.seq_src
                seq_hash = row_pre.seq_hash

                indices = row_pre['index']
                for index in indices:
                    assert seq_hash == self.sf.df_src.at[index, 'seq_hash']
                    self.sf.df_src.at[index, 'seq_msa'] = seq_msa

            self.sf.set_view()

    def write_phy(self, file_path):
        if self.msa is None:
            raise ValueError("MSA is not calculated yet")
        if self.msa_annotations is not None:
            msa = deepcopy(self.msa)
            for seq in msa:
                if seq.id in self.msa_annotations:
                    seq.id += ('_' + self.msa_annotations[seq.id])
        else:
            msa = self.msa

        AlignIO.write(msa, file_path, "phylip-relaxed")
        
    def calculate_distances(self):
        if self.msa is not None:
            log_info(f"Calculating distances using {self.distance_model_name}")
            calculator = DistanceCalculator(self.distance_model_name)
            self.distance_matrix = calculator.get_distance(self.msa)
            log_info(f"Finished calculating distances")
            return self.distance_matrix
        else:
            raise ValueError("MSA is not calculated yet")

    def construct_phylogenetic_tree(self, method='ascii'):
        """ Construct a phylogenetic tree using the specified method.
        
        Args:
            method (str): 'ascii' or 'plot'
        """
        if self.msa is not None:
            if self.distance_matrix is None and self.phylogenetic_tree_model_name != 'raxml':
                self.calculate_distances()

            tree = None
            constructor = DistanceTreeConstructor()
            if self.phylogenetic_tree_model_name == 'nj':
                tree = constructor.nj(self.distance_matrix)
            elif self.phylogenetic_tree_model_name == 'upgma':
                tree = constructor.upgma(self.distance_matrix)
            elif self.phylogenetic_tree_model_name == 'raxml':
                
                if self.msa_file_path is None or not self.msa_file_path.endswith('.phy'):
                    raise ValueError("a '.phy' file must be specified as 'msa_file_path' in 'align' for raxml")
                else:
                    output_dir_path = os.path.dirname(self.msa_file_path)
                    with temp_working_directory(output_dir_path) as tmp_dir_path:
                        # Remove files starting with 'RAxML_' in the output directory
                        directory_path = os.path.dirname(self.msa_file_path)
                        file_list = os.listdir(directory_path)
                        for file_name in file_list:
                            if file_name.startswith('RAxML_'):
                                file_path = os.path.join(directory_path, file_name)
                                os.remove(file_path)

                        # raxmlHPC -s output.phy -n my_tree -m PROTGAMMAGTR -p 42
                        command = ['raxmlHPC', '-s', os.path.basename(self.msa_file_path), '-n', 'my_tree', '-m', 'PROTGAMMAGTR', '-p', '42']
                        log_info(f"Raxml: {' '.join(command)}", start=True)
                        result = subprocess.run(command, capture_output=True)
                        log_info(f"Raxml finished", stop=True)
            else:
                raise ValueError(f"Unknown phylogenetic tree model name: {self.phylogenetic_tree_model_name}")
            
            if self.phylogenetic_tree_model_name != 'raxml':
                if method == 'ascii':
                    Phylo.draw_ascii(tree)
                elif method == 'plot':
                    Phylo.draw(tree)
            return tree
        else:
            raise ValueError("MSA is not calculated yet")
        
    def calculate_occupancy(self):
        if self.msa is not None:
            alignment_length = self.msa.get_alignment_length()

            self.occupancy = []
            for i in range(alignment_length):
                column = self.msa[:, i]
                self.occupancy.append(column.count('-') / len(column))
        else:
            raise ValueError("MSA is not calculated yet")
        
    def set_occupancy_threshold(self, threshold):
        if self.occupancy is None:
            self.calculate_occupancy()
        self.occupancy_threshold = threshold

        self.occupancy_indices_above_threshold = []
        for idx, value in enumerate(self.occupancy):
            if value >= threshold:
                self.occupancy_indices_above_threshold.append(idx)

        self.sf.df_src[f'seq_msa_{threshold}'] = self.sf.df_src.seq_msa.apply(lambda seq_msa: ''.join([seq_msa[idx] for idx in self.occupancy_indices_above_threshold]))
        self.sf.set_view()

    @staticmethod
    def seq_idx_to_msa_idx(seq_msa):
        msa_idx_list = []
        msa_idx = 0
        while msa_idx < len(seq_msa):
            if seq_msa[msa_idx] != '-':
                msa_idx_list.append(msa_idx)
            msa_idx += 1

        return msa_idx_list

    @staticmethod
    def msa_idx_to_seq_idx(seq_msa):
        idx_list = []
        msa_idx = 0
        idx = 0
        while msa_idx < len(seq_msa):
            if seq_msa[msa_idx] == '-':
                idx_list.append(None)
            else:
                idx_list.append(idx)
                idx += 1
            msa_idx += 1

        return idx_list
