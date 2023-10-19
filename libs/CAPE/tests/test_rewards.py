from CAPE.RL.reward import rewards_seqs

import numpy as np
from numpy.testing import assert_allclose
import os

from kit.bioinf.mhc import get_predictor


MHCs = ['HLA-A*02:01','HLA-A*24:02',
        'HLA-B*07:02','HLA-B*39:01',
        'HLA-C*07:01','HLA-C*16:01']


folder = os.path.join(os.environ['PF'], 'artefacts', 'immunology', 'netMHCpan', 'percentile')
print(folder)

predictor_MHC_I = get_predictor('netMHCpan')( 
    folder=folder, limit=0.02, MHC_I_alleles_to_load=MHCs
)

seq =   'MGGKWSKSSIVGWPQIRERIRRAPVAAEGVGAEGQADDVGGVSKHSAVTGANTN' \
        'SANSQDEEAVAEEGEGEVPEPVMRPVPQKGPGGLGKFGGLLDGDDYSGKGDGID' \
        'DLQNYQFQGVNDDWTGYTPGPLDDPPNYPGWCPPLCPLDPDWVEPVPEDDEPCD' \
        'TNNNKQSSMSGQGQEDQEREDDEWGRDDAIARDSRADRERDQERTHPKDCCCCC*'

expected_result = -np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0,
         0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float')

rwd_premature_stop = None
rwd_missing_stop = None
rwd_impossible_token = -1

def test_rewards_normal_seq():
    seq_test = seq
    expected_rewards = expected_result.copy()
    rewards = rewards_seqs(predictor_MHC_I, [seq_test], MHCs, 'deimmunize', lengths=[8,9,10], 
                        rwd_premature_stop=rwd_premature_stop, 
                        rwd_missing_stop=rwd_missing_stop, 
                        rwd_impossible_token=rwd_impossible_token)
    assert_allclose(rewards.squeeze(), expected_rewards)


def test_rewards_premature_stop():
    seq_test = seq[:3] + '*' + seq[3:]
    expected_rewards = np.concatenate([expected_result[:4], [-18], expected_result[4:]])
    rewards = rewards_seqs(predictor_MHC_I, [seq_test], MHCs, 'deimmunize', lengths=[8,9,10], 
                        rwd_premature_stop=rwd_premature_stop, 
                        rwd_missing_stop=rwd_missing_stop, 
                        rwd_impossible_token=rwd_impossible_token)
    assert_allclose(rewards.squeeze(), expected_rewards)


def test_rewards_missing_stop():
    seq_test = seq[:-1]
    expected_rewards = expected_result.copy()[:-1]
    expected_rewards[-1] += -18
    rewards = rewards_seqs(predictor_MHC_I, [seq_test], MHCs, 'deimmunize', lengths=[8,9,10], 
                        rwd_premature_stop=rwd_premature_stop, 
                        rwd_missing_stop=rwd_missing_stop, 
                        rwd_impossible_token=rwd_impossible_token)
    assert_allclose(rewards.squeeze(), expected_rewards)


def test_rewards_impossible_token():
    seq_test = seq[:20] + 'X' + seq[20:]
    expected_rewards = np.concatenate([expected_result[:20], [rwd_impossible_token], expected_result[20:]])
    rewards = rewards_seqs(predictor_MHC_I, [seq_test], MHCs, 'deimmunize', lengths=[8,9,10], 
                        rwd_premature_stop=rwd_premature_stop, 
                        rwd_missing_stop=rwd_missing_stop, 
                        rwd_impossible_token=rwd_impossible_token)
    assert_allclose(rewards.squeeze(), expected_rewards)


def test_rewards_multiple_mistakes():
    seq_test = seq
    expected_rewards = expected_result.copy()
    rewards_orig, checked_peptides_orig = rewards_seqs(predictor_MHC_I, [seq_test], MHCs, 'immunize', lengths=[8,9,10],
                        rwd_premature_stop=rwd_premature_stop, 
                        rwd_missing_stop=rwd_missing_stop, 
                        rwd_impossible_token=rwd_impossible_token,
                        return_checked_peptides=True)
                
    seq_test = seq[:20] + 'X' + seq[20:22] + '*' + seq[22:30] + '-' + seq[30:50] + '*' + seq[50:60] + '?' + seq[60:]
    expected_rewards = -np.concatenate([
        expected_result[:20], [-rwd_impossible_token], 
        expected_result[20:23], [18], 
        expected_result[23:30], [-rwd_impossible_token], 
        expected_result[30:51], [18], 
        expected_result[51:60], [-rwd_impossible_token], 
        expected_result[60:]])
    rewards, checked_peptides = rewards_seqs(predictor_MHC_I, [seq_test], MHCs, 'immunize', lengths=[8,9,10],
                        rwd_premature_stop=rwd_premature_stop, 
                        rwd_missing_stop=rwd_missing_stop, 
                        rwd_impossible_token=rwd_impossible_token,
                        return_checked_peptides=True)
    assert_allclose(rewards.squeeze(), expected_rewards)

    assert checked_peptides_orig == checked_peptides
