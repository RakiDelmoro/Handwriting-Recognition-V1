import os
import numpy
import cv2 as cv
from datasets.utils import characters_to_ints
from datasets.data_processing import image_array_processing
from datasets.utils import START_TOKEN, PAD_TOKEN, END_TOKEN

def image_to_array(folder, image_path, size, patch_width):
    split_image_path = image_path.split('-')
    image_outer_folder = split_image_path[0]
    image_inner_folder = "-".join(split_image_path[:2])
    complete_image_path = os.path.join(folder, image_outer_folder, image_inner_folder, image_path)
    image_array = cv.imread(complete_image_path)
    if image_array is not None:
        image_array_as_grayscale = cv.cvtColor(image_array, cv.COLOR_RGB2GRAY)
        return image_array_processing(image_array_as_grayscale, size, patch_width)
    else: return None

def word_to_token_array(text, max_length=19):
    number_of_pad_tokens = max_length - len(text)
    word_as_char_tokens = characters_to_ints(START_TOKEN) + characters_to_ints(text) + characters_to_ints(END_TOKEN)
    if number_of_pad_tokens != 0: word_as_char_tokens.extend(characters_to_ints(PAD_TOKEN) * number_of_pad_tokens)
    return numpy.array(word_as_char_tokens, dtype=numpy.uint8)

def iam_dataset(folder, txt_file, image_size, patch_width):
    # a01-000u-06-05 ok 159 1910 1839 458 63 NP Exchange - Example of each line data
    extracted_data = []
    dataset_text_line = open(os.path.join(folder, txt_file)).read().splitlines()
    for line in dataset_text_line:
        if line[0] == '#':
            continue
        line_data_as_list = line.split()
        data_not_corrupted = line_data_as_list[1] == 'ok'
        if data_not_corrupted:
            image_path = line_data_as_list[0] + '.png'
            word_written_in_image = ' '.join(line_data_as_list[8:])
            image_array = image_to_array(folder, image_path, image_size, patch_width)
            if image_array is None: continue
            if len(word_written_in_image) == 1: continue
            word_as_character_tokens = word_to_token_array(word_written_in_image)
            extracted_data.append([image_array, word_as_character_tokens])
    return extracted_data

def iam_dataloader(dataset, batch_size=128):
    remaining_samples = len(dataset) % batch_size
    batched_iterator = len(dataset) // batch_size
    dataloader = []
    for i in range(batched_iterator):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        pulled_data = dataset[start_idx:end_idx]
        batched_image_array = numpy.stack([image_arr for image_arr, _ in pulled_data])
        batched_word_token = numpy.stack([word_token for _, word_token in pulled_data])
        dataloader.append([batched_image_array, batched_word_token])
    if remaining_samples != 0:
        # pulled the last remaining data
        pulled_remaining_data = dataset[-remaining_samples:]
        batched_image_array = numpy.stack([image_arr for image_arr, _ in pulled_remaining_data])
        batched_word_token = numpy.stack([word_token for _, word_token in pulled_remaining_data])
        dataloader.append([batched_image_array, batched_word_token])
    return dataloader
