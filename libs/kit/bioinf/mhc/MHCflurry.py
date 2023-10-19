import os
from kit.path import join
from kit.bioinf.mhc import to_HLA_prefix, MHC1Predictor
from kit.data import str_to_file
from mhcflurry import Class1PresentationPredictor


class MHCflurryPredictor(MHC1Predictor):
    def __init__(self, folder=None, limit=0.02):
        super().__init__("MHCflurry", folder, limit)
        self.factor = 0.01
        self.predictor = Class1PresentationPredictor.load()

    def predict_peptides(self, peptides, allele):
        df = self.predictor.predict(
            peptides=peptides,
            alleles=[allele],
            verbose=0)

        for _, row in df.iterrows():
            self.percentiles[allele][row.peptide] = row.presentation_percentile
        self.updated_percentiles = True


def write_MHCflurry_script(folder, kmers_MHC_I, alleles_mhc_I):
    mhc = "MHC_I"

    FOLDER_MHC = join(folder, mhc, 'MHCFlurry')
    text = f"folder={FOLDER_MHC}\n"

    for allele in alleles_mhc_I:
        text += f"mhcflurry-predict $folder/peptides.pep -alleles HLA-{allele} --out $folder/output/peptides_{to_HLA_prefix(allele)}.csv\n"

    str_to_file("peptide" + "\n".join(list(kmers_MHC_I)), join(FOLDER_MHC, "peptides.pep"))
    str_to_file(text, os.path.join(FOLDER_MHC, "script.sh"))
