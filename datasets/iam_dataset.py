import os
import cupy
import cv2 as cv
from datasets.utils import characters_to_ints
from datasets.data_processing import image_array_processing
from datasets.utils import START_TOKEN, PAD_TOKEN, END_TOKEN

def image_to_array(folder, image_path, size):
    split_image_path = image_path.split('-')
    image_outer_folder = split_image_path[0]
    image_inner_folder = "-".join(split_image_path[:2])
    complete_image_path = os.path.join(folder, image_outer_folder, image_inner_folder, image_path)
    image_array = cv.imread(complete_image_path)
    if image_array is not None:
        image_array_as_grayscale = cv.cvtColor(image_array, cv.COLOR_RGB2GRAY)
        return image_array_processing(image_array_as_grayscale, size)
    else: return None

def word_to_token_array(text, max_length=19):
    number_of_pad_tokens = max_length - len(text)
    word_as_char_tokens = characters_to_ints(START_TOKEN) + characters_to_ints(text) + characters_to_ints(END_TOKEN)
    if number_of_pad_tokens != 0: word_as_char_tokens.extend(characters_to_ints(PAD_TOKEN) * number_of_pad_tokens)
    return cupy.array(word_as_char_tokens, dtype=cupy.uint8)

def iam_dataset(folder, txt_file, image_size):
    # a01-000u-06-05 ok 159 1910 1839 458 63 NP Exchange - Example of each line data
    image_arrays = []
    word_token_arrays = []
    dataset_text_line = open(os.path.join(folder, txt_file)).read().splitlines()
    for line in dataset_text_line:
        if line[0] == '#':
            continue
        split_line_data = line.split()
        data_not_corrupted = split_line_data[1] == 'ok'
        if data_not_corrupted:
            image_path = split_line_data[0] + '.png'
            word_written_in_image = ' '.join(split_line_data[8:])
            image_array = image_to_array(folder, image_path, image_size)
            if image_array is None: continue
            if len(word_written_in_image) == 1: continue
            word_as_character_tokens = word_to_token_array(word_written_in_image)
            image_arrays.append(cupy.array(image_array))
            word_token_arrays.append(cupy.array(word_as_character_tokens))
    return image_arrays, word_token_arrays

def iam_dataloader(folder, text_file, image_size, batch_size=128):
    image_arrays, word_token_arrays = iam_dataset(folder, text_file, image_size)
    num_samples = len(image_arrays)
    i = 0
    start_index = 0
    end_index = batch_size
    batched_image_arrays = []
    batched_word_token_arrays = []
    while True:
        if num_samples == 0: break
        batch_image_array = cupy.stack(image_arrays[start_index:end_index], axis=0)
        batch_word_token_array = cupy.stack(word_token_arrays[start_index:end_index], axis=0)
        batched_image_arrays.append(batch_image_array)
        batched_word_token_arrays.append(batch_word_token_array)
        i += 1 
        start_index = i * batch_size
        end_index = start_index + batch_size
        num_samples -= end_index
        if num_samples < 0: num_samples = 0
    return batched_image_arrays, batched_word_token_arrays