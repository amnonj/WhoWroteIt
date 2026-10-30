PAD = 0

def conv_char_to_code_and_truncate(data_text, conv_dict, msg_len):
    def encode(text):
        return [conv_dict.get(char, 1) for char in text[0:msg_len]]+[PAD]*(msg_len-len(text))

    if isinstance(data_text, str):
        return encode(data_text)
    return data_text.apply(encode)


def conv_char_to_code(data_text, conv_dict):
    def encode(text):
        return [conv_dict.get(char, 1) for char in text]

    if isinstance(data_text, str):
        return encode(data_text)
    return data_text.apply(encode)


def conv_code_to_char(coded_data, conv_dict):
    def decode(codes):
        return "".join(conv_dict[code] for code in codes)

    if isinstance(coded_data, list):
        return decode(coded_data)
    return coded_data.apply(decode)
