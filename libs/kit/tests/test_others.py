from kit.bioinf import get_kmers
from kit.data.utils import remove_lines_between, remove_lines


def test_get_kmers():
    seq = "ABCDEFGHIJ"
    lengths = [4, 3, 2]

    # Test 1
    expected_result_aa = [
        "CD",
        "CDE",
        "CDEF",
        "DE",
        "DEF",
        "DEFG",
        "EF",
        "EFG",
        "EFGH",
        "FG",
        "FGH",
        "FGHI",
        "GH",
        "GHI",
        "HI",
    ]
    result = get_kmers(seq, lengths, check_aa=True)
    assert set(result) == set(expected_result_aa)

    # Test 2 - ignoring amino acid constraint
    expected_result = expected_result_aa + [
        "AB",
        "ABC",
        "ABCD",
        "BC",
        "BCD",
        "BCDE",
        "GHIJ",
        "HIJ",
        "IJ",
    ]
    result = get_kmers(seq, lengths, check_aa=False)
    assert set(result) == set(expected_result)


def test_remove_lines_between():
    lines = ["These", "are", "only", "some", "lines", "to", "test", "with"]

    # Test 1
    expected_result = ["These", "with"]
    result = remove_lines_between(
        lines, "These", "with", remove_start=False, remove_end=False
    )
    assert result == expected_result

    # Test 2
    expected_result = ["These", "with"]
    result = remove_lines_between(
        lines, "are", "with", remove_start=True, remove_end=False
    )
    assert result == expected_result

    # Test 3
    expected_result = ["These", "are", "only", "with"]
    result = remove_lines_between(
        lines, "only", "test", remove_start=False, remove_end=True
    )
    assert result == expected_result

    # Test 4
    expected_result = ["These", "are", "only", "test", "with"]
    result = remove_lines_between(
        lines, "only", "test", remove_start=False, remove_end=False
    )

    # Test 5
    expected_result = ["These", "are"]
    result = remove_lines_between(lines, "only", remove_start=True, remove_end=True)


def test_remove_lines():
    lines = ["These", "are", "only", "some", "lines", "to", "test", "with"]

    # Test 1
    except_result = ["These", "are", "some", "to", "test", "with"]
    result = remove_lines(lines, [".+nl.+", ".+ne.+"])
    assert result == except_result
