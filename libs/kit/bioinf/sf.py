import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from kit.loch.utils import get_seq_hash


class SequenceFrame():
    """ Structure to store sequences and their metadata

    The following columns are at least present:
        - accession: the accession of the entry
        - seq_hash: the sequence hash of the entry (seq and seq_src)
        - seq_src: the original sequence in the source
        - seq: the current view of the original sequence
        - seq_hash: the sequence hash of the current view
        - complete: whether the current view is complete
        - c_accession: the number of accessions of the current view
        - c_seq_src: the number of original sequences of the current view
    """

    def __init__(self, translate=('', '', '-'), final_stop_token=False):
        self.df_src = pd.DataFrame(columns=['pack', 'accession', 'seq_hash', 'seq_src'])
        self.df = None
        self.set_view(translate, final_stop_token)

    def _convert_seq(self, seq_src):
        trans = str.maketrans(*self.translate)
        seq = ''
        seq_src = seq_src.translate(trans)
        if len(seq_src) > 0:
            if self.final_stop_token:
                seq = seq_src if seq_src[-1] == '*' else (seq_src + '*')
            else:
                seq = seq_src[:-1] if seq_src[-1] == '*' else seq_src
        return seq

    def set_view(self, translate=None, final_stop_token=None):
        if translate != None:
            self.translate = translate
        if final_stop_token != None:
            self.final_stop_token = final_stop_token
        
        if len(self.df_src) > 0:
            self.df = self.df_src.copy()
            self.df = self.df.reset_index(drop=False)
            self.df['seq'] = None
            for idx, row in self.df_src.iterrows():
                seq = self._convert_seq(row.seq_src)
                self.df.at[idx, 'seq'] = seq
                assert self.df.at[idx, 'seq_hash'] == get_seq_hash(seq)  # check that the seq_hash of the converted sequence is the same as the original seq_hash

            self.df = self.df.groupby(['seq_hash', 'seq']).agg(set).reset_index()
            self.df['c_accession'] = self.df.accession.apply(len)
            self.df['c_seq_src'] = self.df.seq_src.apply(len)

    def __getitem__(self, key):
        return self.df[key]
    
    def __getattr__(self, attr):
        return getattr(self.df, attr)
    
    def __len__(self):
        return len(self.df)

    def from_seqs(self, seqs, packs=None, accessions=None, append=False):
        df_tmp = pd.DataFrame({
            'pack': packs,
            'accession': accessions, 
            'seq_src': seqs,  
        })
        df_tmp['seq_hash'] = df_tmp.seq_src.apply(get_seq_hash)
        if accessions is None:
            df_tmp['accession'] = df_tmp.seq_hash

        self.df_src = pd.concat([self.df_src, df_tmp]) if append else df_tmp
        self.set_view()
         
    def from_fasta(self, fasta_file_path, encoding=None, append=False):
        self.from_file(fasta_file_path, "fasta", encoding=encoding, append=append)

    def from_phy(self, phy_file_path, encoding=None, append=False):
        self.from_file(phy_file_path, "phylip-relaxed", encoding=encoding, append=append)

    def from_file(self, file_path, format, encoding=None, append=False):
        file = open(file_path, "r", encoding=encoding)

        seqs, accessions = [], []
        for seq_record in SeqIO.parse(file, format):
            seqs.append(str(seq_record.seq))
            accessions.append(seq_record.description)

        self.from_seqs(seqs, accessions=accessions, append=append)

    def to_fasta(self, fasta_file_path, save_src=False):
        seqs = []
        df = self.df_src if save_src else self.df
        for _, row in self.df.iterrows():
            seq = row.seq_src if save_src else row.seq
            accession = row.accession if save_src else get_seq_hash(seq) 
            seqs.append(
                SeqRecord(
                    Seq(seq),
                    id=accession,
                    description=""
                )
            )

        SeqIO.write(seqs, fasta_file_path, format="fasta")
