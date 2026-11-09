import hashlib

PAD = 0


def conv_char_to_code_and_truncate(data_text, conv_dict, msg_len, from_end=False, random_crop=False):
    def encode(text):
        if random_crop and len(text) > msg_len:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            start = int.from_bytes(digest[:8], "big") % (len(text) - msg_len + 1)
        elif from_end:
            start = max(0, len(text) - msg_len)
        else:
            start = 0

        segment = text[start:start + msg_len]

        return (
            [conv_dict.get(char, 1) for char in segment]
            + [PAD] * (msg_len - len(segment))
        )

    if isinstance(data_text, str):
        return encode(data_text)

    return data_text.apply(encode)


def conv_char_to_code(data_text, conv_dict):
    def encode(text):
        if isinstance(text, str):
            return [conv_dict.get(char, 1) for char in text]
        return [encode(line) for line in text]

    if isinstance(data_text, (str, list)):
        return encode(data_text)
    return data_text.apply(encode)


def conv_code_to_char(coded_data, conv_dict):
    def decode(codes):
        return "".join(conv_dict[code] for code in codes)

    if isinstance(coded_data, list):
        return decode(coded_data)
    return coded_data.apply(decode)
