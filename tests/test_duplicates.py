import pandas as pd

from src.check_duplicates import flag_duplicates, normalize


def test_normalize_strips_and_lowercases():
    assert normalize(pd.Series(["  Hello World  "])).iloc[0] == "hello world"


def test_exact_copy_is_flagged():
    train = pd.DataFrame({"narrative": ["my payment was lost"]})
    test = pd.DataFrame({"narrative": ["my payment was lost"]})
    assert flag_duplicates(train, test).tolist() == [True]


def test_case_and_edge_spaces_are_ignored():
    train = pd.DataFrame({"narrative": ["my payment was lost"]})
    test = pd.DataFrame({"narrative": ["  My Payment Was LOST "]})
    assert flag_duplicates(train, test).tolist() == [True]


def test_new_text_is_not_flagged():
    train = pd.DataFrame({"narrative": ["my payment was lost"]})
    test = pd.DataFrame({"narrative": ["a different complaint entirely"]})
    assert flag_duplicates(train, test).tolist() == [False]


def test_mixed_rows_keep_the_test_index():
    train = pd.DataFrame({"narrative": ["a b c", "d e f"]})
    test = pd.DataFrame({"narrative": ["a b c", "x y z", "D E F"]}, index=[10, 11, 12])
    flags = flag_duplicates(train, test)
    assert flags.tolist() == [True, False, True]
    assert flags.index.tolist() == [10, 11, 12]