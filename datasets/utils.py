import string
from torch.utils.data import random_split
from itertools import takewhile

def get_printable(character):
    if character == PAD_TOKEN: return "🔴"
    if character == START_TOKEN: return "🚦"
    if character == END_TOKEN: return "🤚"
    return character

def split_data(dataset, ratio):
    training_size = int(ratio * len(dataset))
    validation_size = int(len(dataset) - training_size)
    return random_split(dataset, [training_size, validation_size])

LETTERS = [letter for letter in string.ascii_letters]
NUMBERS = [number for number in string.digits]
SYMBOLS = ["#", "+", "/", "*", ")", "(", '"', "-", "!", "?", ",", ".", ":", ";", "'", "&"]
START_TOKEN = '\N{Start of Text}'
END_TOKEN = '\N{End of Text}'
PAD_TOKEN = '\N{Substitute}'
CHARS = ["\x00", PAD_TOKEN, START_TOKEN, END_TOKEN] + SYMBOLS + [" "] + LETTERS + NUMBERS
int_to_printable = [get_printable(c) for _, c in enumerate(CHARS)]
char_to_index = {c:i for i, c in enumerate(CHARS)}
characters_to_ints = lambda text: [char_to_index[c] for c in text]
ints_to_characters = lambda tensor: "".join([int_to_printable[int(each_token)] for each_token in takewhile(lambda x: x != char_to_index[PAD_TOKEN], tensor)])
