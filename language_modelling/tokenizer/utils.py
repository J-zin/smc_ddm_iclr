def create_token_ids_translation_map(tokenizer1, tokenizer2, synonyms={}):
    """
    Create a mapping from token IDs in tokenizer1 to token IDs in tokenizer2.

    Args:
        tokenizer1: First HuggingFace tokenizer.
        tokenizer2: Second HuggingFace tokenizer.

    Returns:
        A dictionary mapping token IDs from tokenizer1 to tokenizer2.
    """
    vocab1 = tokenizer1.get_vocab()
    vocab2 = tokenizer2.get_vocab()

    translation_map = {}

    for token, id1 in vocab1.items():
        if token in vocab2:
            id2 = vocab2[token]
            translation_map[id1] = id2
        elif token in synonyms and synonyms[token] in vocab2:
            id2 = vocab2[synonyms[token]]
            translation_map[id1] = id2
        else:
            raise AssertionError(f"{token}: {id1} is not present in tokenizer2")
    return translation_map
