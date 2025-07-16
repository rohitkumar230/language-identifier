import sys
from pathlib import Path
import pytest

# Path Fix 
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from identifier.core import LanguageIdentifier
from identifier.advanced import HybridIdentifier

# Test Fixtures 
@pytest.fixture(scope="module")
def profiles_path():
    """A fixture to provide the path to the profiles directory and check its existence."""
    path = project_root / "profiles"
    if not path.exists() or not any(path.glob("*.json")):
        pytest.fail(
            f"Profiles directory not found or empty at {path}.\n"
            "Please run 'python build_profiles.py' first."
        )
    return path

@pytest.fixture(scope="module")
def simple_identifier(profiles_path):
    """Fixture to create an instance of the simple LanguageIdentifier."""
    return LanguageIdentifier(profile_dir=profiles_path)

@pytest.fixture(scope="module")
def advanced_identifier(profiles_path):
    """Fixture to create an instance of the advanced HybridIdentifier."""
    return HybridIdentifier(profile_dir=profiles_path)


# Test Cases 

def test_simple_identification(simple_identifier):
    """Test basic identification with the simple model."""
    result = simple_identifier.identify("This is a test sentence written in English.")
    assert result['prediction'] == 'en'
    
    result = simple_identifier.identify("Das ist ein Test, der auf Deutsch geschrieben wurde.")
    assert result['prediction'] == 'de'

def test_advanced_identification(advanced_identifier):
    """Test basic identification with the advanced model."""
    result = advanced_identifier.identify("Questo è un test per il modello avanzato in italiano.")
    assert result['prediction'] == 'it'
    
    result = advanced_identifier.identify("Esta es una prueba para el modelo avanzado en español.")
    assert result['prediction'] == 'es'

def test_response_structure(simple_identifier):
    """Verify that the response dictionary has the correct keys."""
    result = simple_identifier.identify("A valid sentence.")
    assert "prediction" in result
    assert "distribution" in result
    assert "top_features" in result
    assert isinstance(result['distribution'], list)
    assert isinstance(result['top_features'], list)

@pytest.mark.parametrize("text, expected_lang", [
    ("Le chat est sur la branche de l'arbre.", "fr"),
])

def test_multiple_languages_advanced(advanced_identifier, text, expected_lang):
    """Use parametrize to test multiple languages efficiently."""
    result = advanced_identifier.identify(text)
    assert result['prediction'] == expected_lang

def test_short_text_handling(simple_identifier, advanced_identifier):
    """Test that the models handle very short text gracefully by returning an error dict."""
    result = simple_identifier.identify("a")
    assert "error" in result
    assert "short" in result['error']

    result = advanced_identifier.identify("b")
    assert "error" in result
    assert "short" in result['error']

def test_non_alphabetic_text(simple_identifier):
    """Test text with only numbers and punctuation."""
    result = simple_identifier.identify("12345 !@#$%^&*()_+")
    assert "error" in result
    assert "lacks valid characters" in result['error']