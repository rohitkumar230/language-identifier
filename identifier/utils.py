import re
from collections import Counter
from transformers import AutoTokenizer, PreTrainedTokenizer

def normalize_text(text: str) -> str:
    """
    Prepares text for analysis by standardizing it.
    
    This involves converting to lowercase and removing everything that isn't a
    letter or a space. It's a simple but effective way to reduce noise.
    
    Args:
        text: The raw input string.
        
    Returns:
        A cleaned and normalized version of the text.
    """
    # revisit this later, as this might not be best for non-latin languages.
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces.
    text = re.sub(r'\s+', ' ', text).strip() # Collapse multiple spaces into one.
    return text

def generate_char_ngram_profile(text: str, n: int, profile_size: int) -> list[str]:
    """
    Creates a language "fingerprint" based on character n-grams.
    
    It scans the text, counts all character sequences of length 'n',
    and returns a list of the most frequent ones, in order.
    
    Args:
        text: The normalized input text.
        n: The size of the n-gram (e.g., 3 for trigrams).
        profile_size: The number of top n-grams to include in the profile.
        
    Returns:
        A list of the most common n-grams, or an empty list if text is too short.
    """
    # Can't create n-grams if the text is shorter than n.
    if len(text) < n:
        return []
    
    # Slide a window of size 'n' across the text to get all n-grams.
    ngrams = (text[i:i+n] for i in range(len(text) - n + 1))
    
    # Count the occurrences of each n-gram and get the most common ones.
    ngram_counts = Counter(ngrams)
    most_common = ngram_counts.most_common(profile_size)
    
    # We only care about the n-grams themselves, not their counts.
    # The order (from most to least common) is the crucial part.
    return [ngram for ngram, count in most_common]

def generate_subword_profile(text: str, tokenizer: PreTrainedTokenizer, profile_size: int) -> list[int]:
    """
    Creates a language fingerprint using subword tokens from a transformer model.

    Instead of characters, this uses a pre-trained tokenizer to break text
    into meaningful pieces (subwords) and finds the most common ones.

    Args:
        text: The raw input text (no normalization needed for most tokenizers).
        tokenizer: An initialized tokenizer (e.g., from Hugging Face).
        profile_size: The number of top subword tokens to include.

    Returns:
        A list of the most common subword token IDs, ordered by frequency.
    """
    # Tokenize the text to get a list of subword ID numbers.
    # We skip special tokens like [CLS] or [SEP] as they don't represent the language itself.
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    if not token_ids:
        return []
    
    # Count the token IDs and get the most frequent ones.
    token_counts = Counter(token_ids)
    most_common = token_counts.most_common(profile_size)
    
    # Just like with n-grams, we return the ordered list of token IDs.
    return [token_id for token_id, count in most_common]