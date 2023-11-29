import os
import tempfile
import warnings
import subprocess

import torch

from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from kit.nn import move_dict_to_device
from kit.bioinf.fasta import read_fasta
from kit.path import get_entries, join
from kit.log import log_info
from kit.data import str_to_file

esm_model, esm_tokenizer = None, None
warnings.simplefilter("ignore", PDBConstructionWarning)


def fasta_to_pdb(path, regex=None, online=False):
    if regex is None:
        fasta_files = [path]
    else:
        fasta_files = get_entries(path, regex, returndict=False)

    cnt = 0
    for fasta_file in fasta_files:
        filename = fasta_file.split(os.sep)
        dirname = os.sep.join(filename[:-1])
        parent_folder = filename[-2]
        filename = filename[-1]
        filename_root = filename.removesuffix(".fasta")
        pdb_filename = os.path.join(dirname, filename_root, filename_root + "_ESM.pdb")
        if (
            not os.path.exists(pdb_filename) or os.path.getsize(pdb_filename) < 1000
        ) and parent_folder != filename_root:
            cnt += 1
            log_info(f"converting {filename} to PDB with ESMFold")
            df = read_fasta(fasta_file, return_df=True)
            seq = df.index[0].translate(str.maketrans("", "", "*-"))
            if online:
                command = [
                    "curl",
                    "-X",
                    "POST",
                    "--data",
                    seq,
                    "https://api.esmatlas.com/foldSequence/v1/pdb/",
                ]
                result = subprocess.run(command, capture_output=True, check=False)
                pdb = result.stdout.decode("utf-8")
                str_to_file(
                    pdb, join(dirname, filename_root, filename_root + "_ESM.pdb")
                )
            else:
                outputs = sharded_forward(seq)
                pdb = convert_outputs_to_pdb(filename_root, outputs)
                str_to_file(
                    "\n".join(pdb),
                    join(dirname, filename_root, filename_root + "_ESM.pdb"),
                )

    return cnt


# adjusted from transformers.models.esm.modeling_esmfold.EsmForProteinFolding.forward
def sharded_forward(
    seq, attention_mask=None, position_ids=None, masking_pattern=None, num_recycles=None
):
    global esm_model, esm_tokenizer

    from transformers import AutoTokenizer, EsmForProteinFolding
    from transformers.models.esm.openfold_utils import (
        compute_predicted_aligned_error,
        compute_tm,
        make_atom14_masks,
    )
    from transformers.models.esm.modeling_esmfold import (
        categorical_lddt,
        EsmForProteinFoldingOutput,
    )

    if esm_tokenizer is None:
        esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    if esm_model is None:
        esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        esm_model.esm = esm_model.esm.half()

    input_ids = esm_tokenizer([seq], return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].cuda()
    with torch.no_grad():
        cfg = esm_model.config.esmfold_config

        aa = input_ids  # B x L
        batch_size = aa.shape[0]
        token_length = aa.shape[1]
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(aa, device=device)
        if position_ids is None:
            position_ids = torch.arange(token_length, device=device).expand_as(
                input_ids
            )

        #
        # move ESM model to GPU
        #
        esm_model.esm.cuda()

        # === ESM ===
        esmaa = esm_model.af2_idx_to_esm_idx(aa, attention_mask)

        if masking_pattern is not None:
            masked_aa, esmaa, mlm_targets = esm_model.bert_mask(
                aa, esmaa, attention_mask, masking_pattern
            )
        else:
            masked_aa = aa
            mlm_targets = None

        # We get sequence and pair representations from whatever version of ESM /
        # configuration we are using. The sequence representation esm_s is always
        # present. The pair embedding esm_z may be present depending on the
        # configuration of the model. If esm_z is not used by the model then it
        # is returned as None here.
        esm_s = esm_model.compute_language_model_representations(esmaa)

        # Convert esm_s and esm_z, if present, to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        esm_s = esm_s.to(esm_model.esm_s_combine.dtype)

        if cfg.esm_ablate_sequence:
            esm_s = esm_s * 0

        esm_s = esm_s.detach()

        #
        # move ESM model back to the CPU
        #
        esm_model.esm.cpu()

        #
        # move remaining model to the GPU
        #
        for part in list(esm_model.children())[1:]:
            part.cuda()
        esm_model.esm_s_combine = torch.nn.Parameter(esm_model.esm_s_combine.cuda())

        # === preprocessing ===
        esm_s = (esm_model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = esm_model.esm_s_mlp(esm_s)

        s_z_0 = s_s_0.new_zeros(
            batch_size, token_length, token_length, cfg.trunk.pairwise_state_dim
        )

        if esm_model.config.esmfold_config.embed_aa:
            s_s_0 += esm_model.embedding(masked_aa)

        structure: dict = esm_model.trunk(
            s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles
        )
        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
            in [
                "s_z",
                "s_s",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
                "states",
            ]
        }

        # Add BERT mask for the loss to use, if available.
        if mlm_targets:
            structure["mlm_targets"] = mlm_targets

        disto_logits = esm_model.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        lm_logits = esm_model.lm_head(structure["s_s"])
        structure["lm_logits"] = lm_logits

        structure["aatype"] = aa
        make_atom14_masks(structure)
        # Of course, this doesn't respect the true mask because it doesn't know about it...
        # We're not going to properly mask change of index tensors:
        #    "residx_atom14_to_atom37",
        #    "residx_atom37_to_atom14",
        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= attention_mask.unsqueeze(-1)

        structure["residue_index"] = position_ids

        lddt_head = esm_model.lddt_head(structure["states"]).reshape(
            structure["states"].shape[0],
            batch_size,
            token_length,
            -1,
            esm_model.lddt_bins,
        )
        structure["lddt_head"] = lddt_head
        plddt = categorical_lddt(lddt_head[-1], bins=esm_model.lddt_bins)
        structure["plddt"] = plddt

        ptm_logits = esm_model.ptm_head(structure["s_z"])
        structure["ptm_logits"] = ptm_logits
        structure["ptm"] = compute_tm(
            ptm_logits, max_bin=31, no_bins=esm_model.distogram_bins
        )
        structure.update(
            compute_predicted_aligned_error(
                ptm_logits, max_bin=31, no_bins=esm_model.distogram_bins
            )
        )

        #
        # move remaining network back to the CPU
        #
        esm_model.cpu()
        move_dict_to_device(structure, "cpu")

        output = EsmForProteinFoldingOutput(**structure)

        return output


def convert_outputs_to_pdb(outputs):
    from transformers.models.esm.openfold_utils import (
        to_pdb,
        OFProtein,
        atom14_to_atom37,
    )

    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def convert_outputs_to_structure(name, outputs):
    pdb = convert_outputs_to_pdb(outputs)
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_filename = temp_file.name
        # Write the string data to the temporary file
        temp_file.write("\n".join(pdb))
    parser = PDBParser()
    return parser.get_structure(name, temp_filename), pdb
