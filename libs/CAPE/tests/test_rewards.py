import os

import numpy as np
from numpy.testing import assert_allclose

from kit.bioinf.immuno.mhc_1 import Mhc1Predictor
import kit.globals as G
from kit.data import DD
from CAPE.RL.reward import rewards_seqs, d_visible_natural
from CAPE.profiles import Profile


MHC_1_ALLELEs = [
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-B*07:02",
    "HLA-B*39:01",
    "HLA-C*07:01",
    "HLA-C*16:01",
]

PERCENTILES_DIR_PATH = os.path.join(
    os.environ["PF"], "artefacts", "immunology", "netMHCpan", "percentile"
)

PREDICTOR_MHC_1 = Mhc1Predictor.get_predictor("Mhc1PredictorNetMhcPan")(
    data_dir_path=PERCENTILES_DIR_PATH, limit=0.02, mhc_1_alleles_to_load=MHC_1_ALLELEs
)

SEQ = (
    "MGGKWSKSSIVGWPQIRERIRRAPVAAEGVGAEGQADDVGGVSKHSAVTGANTN"
    "SANSQDEEAVAEEGEGEVPEPVMRPVPQKGPGGLGKFGGLLDGDDYSGKGDGID"
    "DLQNYQFQGVNDDWTGYTPGPLDDPPNYPGWCPPLCPLDPDWVEPVPEDDEPCD"
    "TNNNKQSSMSGQGQEDQEREDDEWGRDDAIARDSRADRERDQERTHPKDCCCCC*"
)

MHC_1_PEPTIDE_LENGTHS = [8, 9, 10]

EXPECTED_RESULT = -np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        2,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
).astype("float")

RWD_PREMATURE_STOP = None
RWD_MISSING_STOP = None
RWD_IMPOSSIBLE_TOKEN = -1


def test_rewards_normal_seq():
    seq_test = SEQ
    expected_rewards = EXPECTED_RESULT.copy()
    rewards = rewards_seqs(
        PREDICTOR_MHC_1,
        [seq_test],
        MHC_1_ALLELEs,
        Profile.VIS_DOWN,
        MHC_1_PEPTIDE_LENGTHS,
        rwd_premature_stop=RWD_PREMATURE_STOP,
        rwd_missing_stop=RWD_MISSING_STOP,
        rwd_impossible_token=RWD_IMPOSSIBLE_TOKEN,
    )
    assert_allclose(rewards.squeeze(), expected_rewards)


d_visible_natural[tuple(MHC_1_ALLELEs + MHC_1_PEPTIDE_LENGTHS)] = set(
    ["GQADDVGGV", "VPQKGPGGL"]
)


def test_rewards_incnat_seq():
    G.ENV.INPUT = os.path.join(os.environ["PF"], "libs", "CAPE", "tests")
    G.DHPARAMS = DD()
    G.DHPARAMS["FOLDER"] = "test_data"

    seq_test = SEQ
    expected_rewards = EXPECTED_RESULT.copy()
    expected_rewards[41] = 1
    expected_rewards[87] = 2
    rewards = rewards_seqs(
        PREDICTOR_MHC_1,
        [seq_test],
        MHC_1_ALLELEs,
        Profile.VIS_UP_NAT,
        MHC_1_PEPTIDE_LENGTHS,
        rwd_premature_stop=RWD_PREMATURE_STOP,
        rwd_missing_stop=RWD_MISSING_STOP,
        rwd_impossible_token=RWD_IMPOSSIBLE_TOKEN,
    )
    assert_allclose(rewards.squeeze(), expected_rewards)


def test_rewards_premature_stop():
    seq_test = SEQ[:3] + "*" + SEQ[3:]
    expected_rewards = np.concatenate([EXPECTED_RESULT[:4], [-18], EXPECTED_RESULT[4:]])
    rewards = rewards_seqs(
        PREDICTOR_MHC_1,
        [seq_test],
        MHC_1_ALLELEs,
        Profile.VIS_DOWN,
        MHC_1_PEPTIDE_LENGTHS,
        rwd_premature_stop=RWD_PREMATURE_STOP,
        rwd_missing_stop=RWD_MISSING_STOP,
        rwd_impossible_token=RWD_IMPOSSIBLE_TOKEN,
    )
    assert_allclose(rewards.squeeze(), expected_rewards)


def test_rewards_missing_stop():
    seq_test = SEQ[:-1]
    expected_rewards = EXPECTED_RESULT.copy()[:-1]
    expected_rewards[-1] += -18
    rewards = rewards_seqs(
        PREDICTOR_MHC_1,
        [seq_test],
        MHC_1_ALLELEs,
        Profile.VIS_DOWN,
        MHC_1_PEPTIDE_LENGTHS,
        rwd_premature_stop=RWD_PREMATURE_STOP,
        rwd_missing_stop=RWD_MISSING_STOP,
        rwd_impossible_token=RWD_IMPOSSIBLE_TOKEN,
    )
    assert_allclose(rewards.squeeze(), expected_rewards)


def test_rewards_impossible_token():
    seq_test = SEQ[:20] + "X" + SEQ[20:]
    expected_rewards = np.concatenate(
        [EXPECTED_RESULT[:20], [RWD_IMPOSSIBLE_TOKEN], EXPECTED_RESULT[20:]]
    )
    rewards = rewards_seqs(
        PREDICTOR_MHC_1,
        [seq_test],
        MHC_1_ALLELEs,
        Profile.VIS_DOWN,
        MHC_1_PEPTIDE_LENGTHS,
        rwd_premature_stop=RWD_PREMATURE_STOP,
        rwd_missing_stop=RWD_MISSING_STOP,
        rwd_impossible_token=RWD_IMPOSSIBLE_TOKEN,
    )
    assert_allclose(rewards.squeeze(), expected_rewards)


def test_rewards_multiple_mistakes():
    seq_test = SEQ
    expected_rewards = EXPECTED_RESULT.copy()
    _, _, _, checked_peptides_orig = rewards_seqs(
        PREDICTOR_MHC_1,
        [seq_test],
        MHC_1_ALLELEs,
        Profile.VIS_UP,
        MHC_1_PEPTIDE_LENGTHS,
        rwd_premature_stop=RWD_PREMATURE_STOP,
        rwd_missing_stop=RWD_MISSING_STOP,
        rwd_impossible_token=RWD_IMPOSSIBLE_TOKEN,
        return_details=True,
    )

    seq_test = (
        SEQ[:20]
        + "X"
        + SEQ[20:22]
        + "*"
        + SEQ[22:30]
        + "-"
        + SEQ[30:50]
        + "*"
        + SEQ[50:60]
        + "?"
        + SEQ[60:]
    )
    expected_rewards = -np.concatenate(
        [
            EXPECTED_RESULT[:20],
            [-RWD_IMPOSSIBLE_TOKEN],
            EXPECTED_RESULT[20:23],
            [18],
            EXPECTED_RESULT[23:30],
            [-RWD_IMPOSSIBLE_TOKEN],
            EXPECTED_RESULT[30:51],
            [18],
            EXPECTED_RESULT[51:60],
            [-RWD_IMPOSSIBLE_TOKEN],
            EXPECTED_RESULT[60:],
        ]
    )
    rewards, _, _, checked_peptides = rewards_seqs(
        PREDICTOR_MHC_1,
        [seq_test],
        MHC_1_ALLELEs,
        Profile.VIS_UP,
        MHC_1_PEPTIDE_LENGTHS,
        rwd_premature_stop=RWD_PREMATURE_STOP,
        rwd_missing_stop=RWD_MISSING_STOP,
        rwd_impossible_token=RWD_IMPOSSIBLE_TOKEN,
        return_details=True,
    )
    assert_allclose(rewards.squeeze(), expected_rewards)

    assert checked_peptides_orig == checked_peptides
