import os
from kit.path import join
from kit.bioinf.immuno.utils import to_mhc_prefix
from kit.bioinf.immuno.mhc_1 import Mhc1Predictor
from kit.data import str_to_file
from mhcflurry import Class1PresentationPredictor


class MHCflurryPredictor(Mhc1Predictor):
    def __init__(self, data_dir_path=None, limit=0.02):
        super().__init__("MHCflurry", data_dir_path, limit)
        self.factor = 0.01
        self.predictor = Class1PresentationPredictor.load()

    def predict_peptides(self, peptides, allele):
        df = self.predictor.predict(peptides=peptides, alleles=[allele], verbose=0)

        for _, row in df.iterrows():
            self.percentiles[allele][row.peptide] = row.presentation_percentile
        self.updated_percentiles = True


def write_mhcflurry_script(data_dir_path, mhc_1_kmers, mhc_1_alleles):
    mhc = "MHC_I"

    mhc_flurry_dir_path = join(data_dir_path, mhc, "MHCFlurry")
    text = f"folder={mhc_flurry_dir_path}\n"

    for allele in mhc_1_alleles:
        text += (
            f"mhcflurry-predict $folder/peptides.pep -alleles HLA-{allele} "
            + f"--out $folder/output/peptides_{to_mhc_prefix(allele)}.csv\n"
        )

    str_to_file(
        "peptide" + "\n".join(list(mhc_1_kmers)),
        join(mhc_flurry_dir_path, "peptides.pep"),
    )
    str_to_file(text, os.path.join(mhc_flurry_dir_path, "script.sh"))
