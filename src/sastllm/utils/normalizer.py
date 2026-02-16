import spacy

from sastllm.configs import get_logger

logger = get_logger()


class Normalizer:
    """
    A lightweight NLP-based text normalizer that uses spaCy to extract the core
    semantic content from reasoning or explanatory sentences. It performs:
    
    - Lowercasing
    - Lemmatization
    - Stop word removal
    - Punctuation filtering
    - Part-of-speech filtering (keeps NOUN, VERB, ADJ, PROPN, NUM)

    Useful for:
    - Normalizing LLM outputs
    - Deduplication
    - Semantic similarity analysis
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """
        Initializes the spaCy language model and sets the desired POS tags
        to retain in the normalized output.
        
        Args:
            model_name (str): The model spaCy model.
                              Default is 'en_core_web_sm'.
        """
        logger.debug("Initializing Normalizer with model: %s", model_name)
        try:
            self.nlp = spacy.load(model_name)
        except Exception as e:
            logger.error("Failed to initialize NLP model %s: %s", model_name, e)
            raise ValueError(f"Failed to initialize NLP model {model_name}: {e}") from e
    
        self.keep_pos = {"NOUN", "VERB", "ADJ", "PROPN", "NUM"}
        
        logger.debug("Normalizer initialized with model: %s", model_name)


    def normalize_text(self, text: str) -> str:
        """
        Normalize a single sentence by removing noise and extracting meaningful tokens.

        Args:
            text (str): The raw functionality string.

        Returns:
            str: Normalized, lemmatized version of the input string.
        """
        logger.debug("Normalizing text: %s", text)
        
        doc = self.nlp(text.lower())
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.pos_ in self.keep_pos
        ]
        
        logger.debug("Normalized text: %s", " ".join(tokens))
        return " ".join(tokens)
