import pandas as pd

from src import data_cleaning
from src.data_cleaning import CATEGORY_CONSOLIDATION, load_clean_data
from src.predict import labels

LONG_TEXT = "this complaint has more than ten words in it so it should always be kept"


def write_csv(tmp_path, rows):
    path = tmp_path / "complaints.csv"
    pd.DataFrame(rows, columns=["product", "narrative"]).to_csv(path, index=False)
    return str(path)


def test_old_category_names_are_merged(tmp_path, monkeypatch):
    monkeypatch.setattr(data_cleaning, "MIN_CATEGORY_SIZE", 1)
    rows = [
        ("Credit card", LONG_TEXT),
        ("Prepaid card", LONG_TEXT),
        ("Credit card or prepaid card", LONG_TEXT),
    ]
    df = load_clean_data(write_csv(tmp_path, rows))
    assert set(df["product"]) == {"Credit card or prepaid card"}
    assert len(df) == 3


def test_rare_categories_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(data_cleaning, "MIN_CATEGORY_SIZE", 2)
    rows = [("Mortgage", LONG_TEXT)] * 3 + [("Student loan", LONG_TEXT)]
    df = load_clean_data(write_csv(tmp_path, rows))
    assert set(df["product"]) == {"Mortgage"}


def test_category_exactly_at_the_minimum_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(data_cleaning, "MIN_CATEGORY_SIZE", 2)
    rows = [("Mortgage", LONG_TEXT)] * 2
    df = load_clean_data(write_csv(tmp_path, rows))
    assert len(df) == 2


def test_narratives_under_the_word_minimum_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(data_cleaning, "MIN_CATEGORY_SIZE", 1)
    too_short = " ".join(["word"] * (data_cleaning.MIN_WORD_COUNT - 1))
    just_enough = " ".join(["word"] * data_cleaning.MIN_WORD_COUNT)
    rows = [("Mortgage", too_short), ("Mortgage", just_enough)]
    df = load_clean_data(write_csv(tmp_path, rows))
    assert len(df) == 1
    assert df.iloc[0]["narrative"] == just_enough


def test_merged_names_exist_in_the_model_labels():
    assert set(CATEGORY_CONSOLIDATION.values()) <= set(labels)