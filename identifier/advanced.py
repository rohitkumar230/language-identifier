import json
import logging
from pathlib import Path
from typing import Dict, Any

from transformers import AutoTokenizer, PreTrainedTokenizer

from .core import LanguageIdentifier
from .utils import normalize_text, generate_char_ngram_profile, generate_subword_profile

class HybridIdentifier(LanguageIdentifier):
    """
    Identifies language using a hybrid approach, combining character n-gram
    and subword token profiles for enhanced accuracy.

    This class inherits from LanguageIdentifier but uses its own logic to
    load two types of profiles and combine their distance scores.
    """
    def __init__(self, profile_dir: Path, n_gram_size: int = 3, profile_size: int = 300, alpha: float = 0.5):
        """
        Initializes the hybrid identifier.

        Args:
            profile_dir: Path to the directory with both char and subword profiles.
            n_gram_size: The size of character n-grams.
            profile_size: The number of top features to keep in each profile.
            alpha: The weighting factor. 0.0 gives all weight to subwords,
                   1.0 gives all weight to n-grams. 0.5 is a balanced mix.
        """
        # We don't call super().__init__() because our loading logic is different.
        # We need to load two separate sets of profiles.
        self.n_gram_size = n_gram_size
        self.profile_size = profile_size
        self.alpha = alpha
        
        # This tokenizer is our 'expert' on subword structure, it's a great general-purpose choice for multilingual text.
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        self.char_profiles: Dict[str, list] = {}
        self.subword_profiles: Dict[str, list] = {}
        
        # Load both profile types from the directory.
        for profile_path in profile_dir.glob("*.json"):
            lang_code = profile_path.stem.split('_')[0]
            if "_chars" in profile_path.name:
                with profile_path.open('r', encoding='utf-8') as f:
                    self.char_profiles[lang_code] = json.load(f)
            elif "_subwords" in profile_path.name:
                 with profile_path.open('r', encoding='utf-8') as f:
                    self.subword_profiles[lang_code] = json.load(f)

        # Check that we have a matching set of profiles for each model type.
        char_langs = set(self.char_profiles.keys())
        subword_langs = set(self.subword_profiles.keys())
        
        if not char_langs or char_langs != subword_langs:
            logging.warning("Mismatch or missing profiles between char and subword models. The identifier might not work as expected.")
        
        logging.info(f"HybridIdentifier is ready. Using {len(char_langs)} languages. Alpha set to {self.alpha}.")

    def identify(self, text: str, top_n: int = 3) -> Dict[str, Any]:
        """
        Identifies language by blending n-gram and subword scores.

        This method overrides the parent's identify() method to implement
        the hybrid scoring logic, polymorphism.

        Args:
            text: The input text to analyze.
            top_n: The number of top language matches to return.

        Returns:
            A dictionary with the prediction, distribution, and top features.
        """
        # Generate both types of profiles for the input text.
        text_char_profile = generate_char_ngram_profile(
            normalize_text(text), self.n_gram_size, self.profile_size
        )
        text_subword_profile = generate_subword_profile(text, self.tokenizer, self.profile_size)

        if not text_char_profile or not text_subword_profile:
            return {"error": "Text too short to generate a reliable profile."}

        # Calculate a final, blended score for each language.
        final_scores = {}
        # We assume char_profiles holds the master list of supported languages.
        for lang_code, char_profile in self.char_profiles.items():
            # If a subword profile is missing for some reason, we can't score it.
            if lang_code not in self.subword_profiles:
                continue
            
            subword_profile = self.subword_profiles[lang_code]
            
            # Calculate both distances.
            char_dist = self._calculate_distance(text_char_profile, char_profile)
            subword_dist = self._calculate_distance(text_subword_profile, subword_profile)
            
            # Combine both the distances using our weight (alpha), this is a simple linear interpolation.
            final_scores[lang_code] = (self.alpha * char_dist) + ((1 - self.alpha) * subword_dist)
            
        if not final_scores:
            return {"error": "Could not score text against any language. Check profiles."}

        # Sort by the final blended score.
        sorted_scores = sorted(final_scores.items(), key=lambda item: item[1])
        best_match_code, _ = sorted_scores[0]
        
        # Format the output. For simplicity, we'll still show char n-grams as the top features, as they are more human-readable than token IDs.
        distribution = [{"lang": lang, "score": f"{score:.2f}"} for lang, score in sorted_scores[:top_n]]
        
        best_char_profile = self.char_profiles[best_match_code]
        best_char_profile_ranks = {ngram: i for i, ngram in enumerate(best_char_profile)}
        top_features = sorted(
            [ngram for ngram in text_char_profile if ngram in best_char_profile_ranks],
            key=lambda ngram: best_char_profile_ranks[ngram]
        )[:5]

        return {
            "prediction": best_match_code,
            "distribution": distribution,
            "top_features": top_features
        }