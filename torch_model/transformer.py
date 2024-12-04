import math
import torch
import random
import torch.nn as nn
from einops.layers.torch import Rearrange
from Model.utils import unpadded_length_tokens
from datasets.utils import ints_to_characters, char_to_index, PAD_TOKEN
from Model.configurations import NETWORK_FEATURE_SIZE, ATTENTION_FEATURE_SIZE, NUM_ATTENTION_HEADS,  MLP_FEATURE_SIZE, NUM_LAYERS, NUMBER_OF_CLASSES, MAX_PATCHES_LENGTH, DEVICE

class CNNFeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(7,7), stride=1, device=DEVICE), nn.MaxPool2d(kernel_size=(2,2), stride=1))
        self.second_layer_conv = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3), stride=1, device=DEVICE), nn.MaxPool2d(kernel_size=(2,2), stride=1))
        self.third_layer_conv = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=1, device=DEVICE), nn.MaxPool2d(kernel_size=(2,2), stride=1))
        self.fourth_layer_conv = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, device=DEVICE), nn.MaxPool2d(kernel_size=(2,2), stride=2))
        self.fifth_layer_conv = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, device=DEVICE), nn.MaxPool2d(kernel_size=(2,2), stride=2))
        self.sixth_layer_conv = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, device=DEVICE), nn.MaxPool2d(kernel_size=(2,2), stride=2))
        self.activation_function = nn.ReLU()
        self.layer_output = nn.Linear(240, NETWORK_FEATURE_SIZE, device=DEVICE) #TODO: Avoid hard coded in features

    def forward(self, batched_image):
        conv1_output = self.activation_function(self.first_layer_conv(batched_image))
        conv2_output = self.activation_function(self.second_layer_conv(conv1_output))
        conv3_output = self.activation_function(self.third_layer_conv(conv2_output))
        conv4_output = self.activation_function(self.fourth_layer_conv(conv3_output))
        conv5_output = self.activation_function(self.fifth_layer_conv(conv4_output))
        conv6_output = self.activation_function(self.sixth_layer_conv(conv5_output))
        batch, channels, _, _ = conv6_output.shape
        # Flatten H, W 
        flattened_conv6_output = conv6_output.reshape(batch, channels, -1)
        return self.layer_output(flattened_conv6_output)

class PositionalEncoding(nn.Module):
    """ Use cos-sin wave for position of each patches in sequence """
    def __init__(self):
        super().__init__()
        position = torch.arange(MAX_PATCHES_LENGTH, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, NETWORK_FEATURE_SIZE, 2, device=DEVICE) * (-math.log(10000.0) / NETWORK_FEATURE_SIZE))
        pe = torch.zeros(MAX_PATCHES_LENGTH, 1, NETWORK_FEATURE_SIZE, device=DEVICE)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Arguments: x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``"""
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        return x.transpose(0, 1)

class ImageEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_extraction = CNNFeatureExtraction()
        self.position_embeddings = PositionalEncoding()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, NETWORK_FEATURE_SIZE, device=DEVICE))

    def masked_patches(self, image_patches, span_masked_ratio, max_span_length):
        batch, patches, _ = image_patches.shape
        mask_patches = torch.ones(batch, patches, 1, device=DEVICE)
        span_patches_length = int(patches * span_masked_ratio)
        total_span = span_patches_length // max_span_length
        for _ in range(total_span):
            idx = random.randint(0, patches-10) 
            mask_patches[:, idx:idx + max_span_length, :] = 0
        image_patches_with_random_masked = image_patches * mask_patches + (1 - mask_patches) * self.mask_token
        return image_patches_with_random_masked

    def forward(self, batched_image_array: torch.Tensor):
        feature_extracted = self.cnn_extraction(batched_image_array)
        masked_feature_patches = self.masked_patches(image_patches=feature_extracted, span_masked_ratio=0.2, max_span_length=2)
        tokens_with_positional_embedding = self.position_embeddings(masked_feature_patches)
        return tokens_with_positional_embedding

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_feature_size = ATTENTION_FEATURE_SIZE
        self.num_heads = NUM_ATTENTION_HEADS
        self.total_attention_feature_size = NUM_ATTENTION_HEADS*ATTENTION_FEATURE_SIZE
        self.query = nn.Linear(NETWORK_FEATURE_SIZE, self.total_attention_feature_size, device=DEVICE)
        self.key = nn.Linear(NETWORK_FEATURE_SIZE, self.total_attention_feature_size, device=DEVICE)
        self.value = nn.Linear(NETWORK_FEATURE_SIZE, self.total_attention_feature_size, device=DEVICE)
        self.attn_output_layer = nn.Linear(self.total_attention_feature_size, NETWORK_FEATURE_SIZE, device=DEVICE)

    def transpose_for_attn_scores(self, x: torch.Tensor) -> torch.Tensor:
        # batch | patches | attn_heads | attention_dimension
        new_x_shape = x.shape[:-1] + (self.num_heads, self.attention_feature_size)
        x = x.view(new_x_shape)
        # batch | attn_heads | patches | attention_dimension
        return x.permute(0, 2, 1, 3)

    def forward(self, input_embeddings: torch.Tensor):
        # batch | attention heads | patches | attention feature size
        query_layer = self.transpose_for_attn_scores(self.query(input_embeddings))
        key_layer = self.transpose_for_attn_scores(self.key(input_embeddings))
        value_layer = self.transpose_for_attn_scores(self.value(input_embeddings))
        # Take the dot product of "query" and "key" to the raw attention scores.
        attention_scores = (torch.matmul(query_layer, key_layer.transpose(-2, -1))) / math.sqrt(self.attention_feature_size)
        # normalize the attention scores to probabilities.
        attention_axons = nn.functional.softmax(attention_scores, dim=-1)
        # batch | attention heads | patches | attention feature size
        input_embeddings_context = torch.matmul(attention_axons, value_layer)
        # batch | patches | attention heads | attention feature size
        input_embeddings_context = input_embeddings_context.permute(0, 2, 1, 3).contiguous()
        # batch | patches | total attention feature size
        attention_output_shape = input_embeddings_context.shape[:-2] + (self.total_attention_feature_size,)
        attention_output = self.attn_output_layer(input_embeddings_context.view(attention_output_shape))
        return attention_output

class EncoderMultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_1 = nn.Linear(NETWORK_FEATURE_SIZE, MLP_FEATURE_SIZE, device=DEVICE)
        self.linear_layer_2 = nn.Linear(MLP_FEATURE_SIZE, NETWORK_FEATURE_SIZE, device=DEVICE)
        self.activation_function = nn.GELU()

    def forward(self, attention_output: torch.Tensor):
        layer_1_output = self.linear_layer_1(attention_output)
        activation_function_output = self.activation_function(layer_1_output)
        layer_2_output = self.linear_layer_2(activation_function_output)
        return layer_2_output

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_attetion = MultiHeadAttention()
        self.mlp_encoder = EncoderMultiLayerPerceptron()
        # Layer norm keeps the layer activation bounded for all samples in a batch
        self.attention_output_activation_normalization = nn.LayerNorm(NETWORK_FEATURE_SIZE, device=DEVICE)
        self.mlp_output_activation_normalization = nn.LayerNorm(NETWORK_FEATURE_SIZE, device=DEVICE)

    def forward(self, input_embeddings: torch.Tensor):
        attention_output = self.attention_output_activation_normalization(self.multi_head_attetion(input_embeddings))
        residual_connection_1 = attention_output + input_embeddings # first residual connection
        mlp_output = self.mlp_output_activation_normalization(self.mlp_encoder(residual_connection_1))
        residual_connection_2 = mlp_output + residual_connection_1 # second residual connection
        return residual_connection_2

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Layer()
        self.encoder_layers = nn.ModuleList([self.layer for _ in range(NUM_LAYERS)])

    def forward(self, input_embeddings: torch.Tensor):
        layer_output = input_embeddings
        for layer in self.encoder_layers: layer_output = layer(layer_output)
        return layer_output

class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_1 = nn.Linear(NETWORK_FEATURE_SIZE, MLP_FEATURE_SIZE, device=DEVICE)
        self.linear_layer_2 = nn.Linear(MLP_FEATURE_SIZE, NETWORK_FEATURE_SIZE, device=DEVICE)
        self.activation_function = nn.ReLU()

    def forward(self, encoder_output: torch.Tensor):
        layer_1_output = self.linear_layer_1(encoder_output)
        activation_function_output = self.activation_function(layer_1_output)
        layer_2_output = self.linear_layer_2(activation_function_output)
        return layer_2_output

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_embeddings = ImageEmbeddings()
        self.encoder_layers = EncoderLayer()
        # self.multi_layer_perceptron = MultiLayerPerceptron()
        self.model_output_prediction = nn.Linear(NETWORK_FEATURE_SIZE, NUMBER_OF_CLASSES, device=DEVICE)

    def forward(self, batched_image_array):
        input_embeddings = self.image_embeddings(batched_image_array)
        encoder_output = self.encoder_layers(input_embeddings)
        model_prediction = self.model_output_prediction(encoder_output)
        return model_prediction

    def get_stress_and_update_parameters(self, model_prediction, expected_prediction, optimizer, learning_rate):
        optimizer = optimizer(self.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss(ignore_index=char_to_index[PAD_TOKEN])
        model_prediction = model_prediction.view(-1, NUMBER_OF_CLASSES)
        loss = loss_func(model_prediction, expected_prediction.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def prediction_as_indices(self, model_prediction):
        model_prediction_as_probabilities = torch.nn.Softmax(dim=-1).forward(model_prediction)
        model_prediction = model_prediction_as_probabilities.data
        model_prediction_indices = model_prediction.topk(1)[1].squeeze(-1).cpu().numpy()
        return model_prediction_indices

    def prediction_as_str(self, model_prediction_in_indies):
        return ints_to_characters(model_prediction_in_indies)
