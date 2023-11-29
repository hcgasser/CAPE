from kit.loch.checks import check_seq_hashes_in_loch


def test_loch_integrity():
    fasta_mismatches, pdb_mismatches = check_seq_hashes_in_loch()
    assert len(fasta_mismatches) == 0
    assert len(pdb_mismatches) == 0
