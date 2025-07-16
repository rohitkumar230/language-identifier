import re
import json
import logging
from pathlib import Path
from collections import Counter
from itertools import islice

from datasets import load_dataset
from transformers import AutoTokenizer

# Configuration
LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese"
}
PROFILE_SIZE = 300
N_GRAM_SIZE = 3
MAX_SAMPLES_PER_LANG = 25_000  # Using an underscore for readability
TOKENIZER_MODEL = "bert-base-multilingual-cased"

# Setup Logging and Paths
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths relative to this script's location, this makes the script runnable from anywhere.
ROOT_DIR = Path(__file__).resolve().parent
PROFILES_DIR = ROOT_DIR / "profiles"
PROFILES_DIR.mkdir(exist_ok=True)


# Core Helper Functions, these functions are duplicated from 'identifier/utils.py'.

def normalize_text(text: str) -> str:
    """Cleans text for character n-gram analysis."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_profile(items: list, profile_size: int) -> list:
    """Generic function to get the most common items from a list."""
    if not items:
        return []
    counts = Counter(items)
    return [item for item, count in counts.most_common(profile_size)]


def main():
    """
    Main function to orchestrate the profile generation process.
    """
    logging.info("Starting Language Profile Generation")
    
    logging.info(f"Loading tokenizer: {TOKENIZER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    logging.info("Tokenizer loaded successfully.")
    
    for lang_code, lang_name in LANGUAGES.items():
        logging.info(f"\nProcessing {lang_name} ({lang_code})...")
        
        try:
            # Load Data, using streaming=True is memory-efficient for large datasets.
            # We load the data as an iterable and take only what we need.
            logging.info(f"  > Loading dataset 'wikiann' for '{lang_code}'.")
            dataset = load_dataset("wikiann", lang_code, split="train", streaming=True)
            
            # Aggregate tokens from the specified number of samples. 'islice' is a memory-friendly way to take N items from an iterator.
            samples = islice(dataset, MAX_SAMPLES_PER_LANG)
            raw_text = " ".join([" ".join(sample['tokens']) for sample in samples])
            
            if not raw_text:
                logging.warning(f"  > No text found for {lang_name}. Skipping.")
                continue

            # Generate and Save Character N-Gram Profile
            logging.info("  > Generating character n-gram profile.")
            normalized_for_ngrams = normalize_text(raw_text)
            char_ngrams = (
                normalized_for_ngrams[i:i+N_GRAM_SIZE] 
                for i in range(len(normalized_for_ngrams) - N_GRAM_SIZE + 1)
            )
            char_profile = generate_profile(list(char_ngrams), PROFILE_SIZE)
            
            char_profile_path = PROFILES_DIR / f"{lang_code}_chars.json"
            with char_profile_path.open('w', encoding='utf-8') as f:
                json.dump(char_profile, f, indent=2)
            logging.info(f"  > Saved character profile to {char_profile_path}")

            # Generate and Save Subword Token Profile
            logging.info("  > Generating subword token profile.")
            token_ids = tokenizer.encode(raw_text, add_special_tokens=False)
            subword_profile = generate_profile(token_ids, PROFILE_SIZE)
            
            subword_profile_path = PROFILES_DIR / f"{lang_code}_subwords.json"
            with subword_profile_path.open('w', encoding='utf-8') as f:
                json.dump(subword_profile, f, indent=2)
            logging.info(f"  > Saved subword profile to {subword_profile_path}")

        except Exception as e:
            logging.error(f"  > FAILED to process {lang_name}. Error: {e}")

    logging.info("\n All language profiles generated successfully!")


if __name__ == "__main__":
    main()