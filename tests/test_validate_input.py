import pytest

from src.predict import MIN_INPUT_WORDS, validate_input

LONG_ENOUGH = "My mortgage servicer applied my payment to the wrong month."


def test_valid_text_returns_none():
    assert validate_input(LONG_ENOUGH) is None


@pytest.mark.parametrize("bad_value", [12345, None, ["a list"], 3.5])
def test_non_text_is_rejected(bad_value):
    assert validate_input(bad_value) == "Input must be text."


@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_or_blank_is_rejected(empty):
    assert validate_input(empty) == "Input cannot be empty."


@pytest.mark.parametrize("short", ["yes", "one two three four"])
def test_too_short_is_rejected(short):
    message = validate_input(short)
    assert message is not None
    assert f"minimum {MIN_INPUT_WORDS} words" in message


def test_exactly_the_minimum_words_is_accepted():
    text = " ".join(["word"] * MIN_INPUT_WORDS)
    assert validate_input(text) is None


def test_padding_spaces_do_not_count_as_words():
    assert validate_input("  one two three four  ") is not None