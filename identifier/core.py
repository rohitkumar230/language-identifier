import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from .utils import normalize_text, generate_char_ngram_profile

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LanguageIdentifier:
    """
    Identifies the language of a given text using the n-gram profile matching method.

    This class loads a set of pre-computed language profiles and compares an
    input text's generated profile against them to find the best match.
    """
    def __init__(self, profile_dir: Path, n_gram_size: int = 3, profile_size: int = 300):
        """
        Initializes the identifier by loading language profiles from a directory.

        Args:
            profile_dir: The path to the directory containing language profile files
                         (e.g., 'en_chars.json', 'fr_chars.json').
            n_gram_size: The character n-gram size to use for text profiling (e.g., 3 for trigrams).
            profile_size: The number of top n-grams to keep in a profile.
        """
        self.profiles: Dict[str, List[str]] = {}
        self.n_gram_size = n_gram_size
        self.profile_size = profile_size
        
        # This finds all the character-based profiles in the given directory.
        profile_paths = list(profile_dir.glob("*_chars.json"))
        
        if not profile_paths:
            raise FileNotFoundError(f"Whoops! No language profiles (*_chars.json) found in {profile_dir}.")

        for profile_path in profile_paths:
            lang_code = profile_path.stem.split('_')[0]
            try:
                with profile_path.open('r', encoding='utf-8') as f:
                    self.profiles[lang_code] = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Could not read or parse {profile_path}, skipping.")
            except Exception as e:
                logging.warning(f"Error loading {profile_path}: {e}")

        loaded_langs = list(self.profiles.keys())
        logging.info(f"LanguageIdentifier is ready. Loaded {len(loaded_langs)} languages: {', '.join(loaded_langs)}")

    def _calculate_distance(self, text_profile: List[str], lang_profile: List[str]) -> int:
        """
        Calculates the 'out-of-place' distance between two profiles.
        
        The distance is the sum of rank differences for shared n-grams.
        If an n-gram from the text profile is not in the language profile,
        a maximum penalty is applied. Lower distance means a better match.

        Args:
            text_profile: The generated n-gram profile for the input text.
            lang_profile: A pre-loaded n-gram profile for a specific language.

        Returns:
            An integer representing the distance between the two profiles.
        """
        # Creates a quick lookup map of n-gram to its rank for the language profile.
        lang_profile_ranks = {ngram: i for i, ngram in enumerate(lang_profile)}
        
        total_distance = 0
        for i, ngram in enumerate(text_profile):
            if ngram in lang_profile_ranks:
                # Calculates how far out of place is this n-gram?
                rank_in_lang = lang_profile_ranks[ngram]
                total_distance += abs(i - rank_in_lang)
            else:
                # This n-gram is very important in our text, but not in the language profile. Penalize it heavily.
                total_distance += self.profile_size
                
        return total_distance

    def identify(self, text: str, top_n: int = 3) -> Dict[str, Any]:
        """
        Identifies the language of a text and returns a detailed analysis.

        Args:
            text: The input text to analyze.
            top_n: The number of top language matches to return in the distribution.

        Returns:
            A dictionary containing the prediction, score distribution, and top features.
        """
        if not self.profiles:
            return {"error": "Identifier has no profiles loaded."}

        # Clean up and profile the input text.
        normalized_text = normalize_text(text)
        text_profile = generate_char_ngram_profile(
            normalized_text, self.n_gram_size, self.profile_size
        )
        
        if not text_profile:
            # This happens if the text is too short or contains no usable characters.
            return {"error": "Text too short or lacks valid characters to analyze."}

        # Score the text against every language we know.
        scores = {
            lang_code: self._calculate_distance(text_profile, lang_profile)
            for lang_code, lang_profile in self.profiles.items()
        }
            
        # Sort the results to find the best match (lowest score wins).
        sorted_scores = sorted(scores.items(), key=lambda item: item[1])
        
        best_match_lang, _ = sorted_scores[0]
        
        # Find the n-grams that were most influential for the top prediction.
        # These are n-grams from our text that are also highly ranked in the winning language.
        best_lang_profile = self.profiles[best_match_lang]
        best_lang_profile_ranks = {ngram: i for i, ngram in enumerate(best_lang_profile)}
        
        top_features = sorted(
            [ngram for ngram in text_profile if ngram in best_lang_profile_ranks],
            key=lambda ngram: best_lang_profile_ranks[ngram]
        )[:5]

        # Package it all up in a nice, clean dictionary for the user/API.
        return {
            "prediction": best_match_lang,
            "distribution": [{"lang": lang, "score": score} for lang, score in sorted_scores[:top_n]],
            "top_features": top_features
        }