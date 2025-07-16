import sys
from pathlib import Path
import pytest

# Path Fix 
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer

from identifier.utils import (
    normalize_text, 
    generate_char_ngram_profile, 
    generate_subword_profile
)

# Test Cases for normalize_text 

def test_normalize_lowercase():
    assert normalize_text("This Is A Test") == "this is a test"

def test_normalize_removes_punctuation():
    assert normalize_text("Hello, world! 123.") == "hello world"

def test_normalize_collapses_whitespace():
    assert normalize_text("  Extra   spaces  ") == "extra spaces"

def test_normalize_empty_string():
    assert normalize_text("") == ""


# Test Cases for generate_char_ngram_profile

def test_ngram_profile_generation():
    text = "ababab"
    profile = generate_char_ngram_profile(text, n=2, profile_size=2)
    assert profile == ["ab", "ba"]

def test_ngram_profile_size_limit():
    text = "abcde"
    profile = generate_char_ngram_profile(text, n=2, profile_size=3)
    assert len(profile) == 3

def test_ngram_text_too_short():
    profile = generate_char_ngram_profile("hi", n=3, profile_size=10)
    assert profile == []


# Test Cases for generate_subword_profile 

@pytest.fixture(scope="module")
def tokenizer():
    """Fixture to load the tokenizer once for all subword tests."""
    return AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def test_subword_profile_generation(tokenizer):
    """
    Tests that the subword profile correctly identifies the most frequent tokens.
    This test is designed to be robust to tokenizer version changes.
    """
    # We construct a text where we know the token frequencies.
    # 'test' appears 3 times, 'example' appears 2 times, 'string' appears 1 time.
    text = "test test test example example string"

    # Dynamically get the token IDs for our test words.
    test_id = tokenizer.encode("test", add_special_tokens=False)[0]
    example_id = tokenizer.encode("example", add_special_tokens=False)[0]
    string_id = tokenizer.encode("string", add_special_tokens=False)[0]

    # Generate the profile for the text.
    profile = generate_subword_profile(text, tokenizer, profile_size=3)

    # Assert that the generated profile has the correct length.
    assert len(profile) == 3

    # Assert that the tokens are in the correct order of frequency.
    assert profile[0] == test_id    # Most frequent
    assert profile[1] == example_id # Second most frequent
    assert profile[2] == string_id  # Third most frequent

def test_subword_profile_empty_text(tokenizer):
    profile = generate_subword_profile("", tokenizer, profile_size=10)
    assert profile == []