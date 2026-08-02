def build_vocab(corpus, special_tokens):
    """Build a character-level vocab; specials get the lowest ids, then sorted unique chars"""
    id_to_token = special_tokens.copy()
    chars = sorted(set("".join(corpus)))
    for char in chars:
        if char not in id_to_token:
            id_to_token.append(char)
    token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
    return {'token_to_id': token_to_id, 'id_to_token': id_to_token}

def encode_prompt(text, vocab, add_bos=True):
    """Encode text into token ids using vocab, optionally prepending <bos>"""
    ids = []
    if add_bos is True and '<bos>' in vocab['token_to_id']:
        ids.append(vocab['token_to_id'].get('<bos>'))
    for c in text:
        if c in vocab['token_to_id']:
            ids.append(vocab['token_to_id'].get(c))
        elif '<unk>' in vocab['token_to_id']:
            ids.append(vocab['token_to_id'].get('<unk>'))
        else:
            continue
    return ids

def decode_tokens(token_ids, vocab, skip_special=True):
    """Convert token ids back into a string using vocab['id_to_token'], optionally skipping specials"""
    tokens = []
    for id in token_ids:
        if skip_special is True and vocab['id_to_token'][id].startswith('<') and vocab['id_to_token'][id].endswith('>'):
            continue
        else:
            tokens.append(vocab['id_to_token'][int(id)])
    result = ''.join(tokens)
    return result
