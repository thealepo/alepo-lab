import pytest
import tiktoken
import os

from base import Tokenizer

test_strings = [
    '',
    '?',
    'Hello World!',
    'I am Alex, welcome to my code!',
]

def unpack(text):
    # print contents of file
    if text.startswith('FILE:'):
        dirname = os.path.dirname(os.path.abspath(__file__))
        asap_file = os.path.join(dirname, text[5:])
        contents = open(asap_file , 'r' , encoding='utf-8').read()
        return contents
    else:
        return text

def encode_decode(text):
    text = unpack(text)
    tokenizer = Tokenizer()
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    return text == decoded

def text_example(tokenizer: Tokenizer):
    text = 'aaabdaaabac'
    tokenizer.train(text , 256+3)
    ids = tokenizer.encode(text)
    assert ids == [286, 473, 5412, 4435, 12340]
    assert tokenizer.decode(tokenizer.encode(text)) == text