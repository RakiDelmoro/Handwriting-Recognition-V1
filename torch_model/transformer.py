import math
import torch
import torch.nn as nn
from Model.utils import unpadded_length_tokens
from einops.layers.torch import Rearrange
from Model.configurations import PATCH_SIZE, NETWORK_FEATURE_SIZE, ATTENTION_FEATURE_SIZE, NUM_ATTENTION_HEADS, NUM_PATCHES, MLP_FEATURE_SIZE, NUM_LAYERS, NUMBER_OF_CLASSES

class ImageEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        patch_h, patch_w = PATCH_SIZE
        num_patches = NUM_PATCHES
        # Rearrange from: batch | height | width -> batch | patches | height | patch width
        self.to_patch_embedding = nn.Sequential(Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1=patch_h, p2=patch_w), nn.Linear(patch_h*patch_w, NETWORK_FEATURE_SIZE))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, NETWORK_FEATURE_SIZE))

    def forward(self, batched_image_array: torch.Tensor):
        input_embeddings = self.to_patch_embedding(batched_image_array)
        embeddings = input_embeddings + self.position_embeddings
        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_feature_size = ATTENTION_FEATURE_SIZE
        self.num_heads = NUM_ATTENTION_HEADS
        self.total_attention_feature_size = NUM_ATTENTION_HEADS*ATTENTION_FEATURE_SIZE
        self.query = nn.Linear(NETWORK_FEATURE_SIZE, self.total_attention_feature_size)
        self.key = nn.Linear(NETWORK_FEATURE_SIZE, self.total_attention_feature_size)
        self.value = nn.Linear(NETWORK_FEATURE_SIZE, self.total_attention_feature_size)
        self.attn_output_layer = nn.Linear(self.total_attention_feature_size, NETWORK_FEATURE_SIZE)
    
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
        self.linear_layer_1 = nn.Linear(NETWORK_FEATURE_SIZE, MLP_FEATURE_SIZE)
        self.linear_layer_2 = nn.Linear(MLP_FEATURE_SIZE, NETWORK_FEATURE_SIZE)
        self.activation_function = nn.ReLU()

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
        self.attention_output_activation_normalization = nn.LayerNorm(NETWORK_FEATURE_SIZE)
        self.mlp_output_activation_normalization = nn.LayerNorm(NETWORK_FEATURE_SIZE)

    def forward(self, input_embeddings: torch.Tensor):
        attention_output = self.attention_output_activation_normalization(self.multi_head_attetion(input_embeddings))
        # first residual connection
        residual_connection_1 = attention_output + input_embeddings
        mlp_output = self.mlp_output_activation_normalization(self.mlp_encoder(residual_connection_1))
        # second residual connection
        residual_connection_2 = mlp_output + residual_connection_1
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
        self.linear_layer_1 = nn.Linear(NETWORK_FEATURE_SIZE, MLP_FEATURE_SIZE)
        self.linear_layer_2 = nn.Linear(MLP_FEATURE_SIZE, NETWORK_FEATURE_SIZE)
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
        self.multi_layer_perceptron = MultiLayerPerceptron()
        self.model_output_prediction = nn.Linear(NETWORK_FEATURE_SIZE, NUMBER_OF_CLASSES)
        self.output_activation = nn.Softmax(dim=-1)

    def forward(self, batched_image_array):
        input_embeddings = self.image_embeddings(batched_image_array)
        encoder_output = self.encoder_layers(input_embeddings)
        mlp_output = self.multi_layer_perceptron(encoder_output)
        model_prediction = self.output_activation(self.model_output_prediction(mlp_output))
        return model_prediction

    def get_stress_and_update_parameters(self, model_prediction, expected_prediction, optimizer, learning_rate):
        optimizer = optimizer(self.parameters(), lr=learning_rate)
        _, _, number_of_classes = model_prediction.shape
        model_prediction = model_prediction.reshape(-1, number_of_classes)
        expected_prediction = expected_prediction.reshape(-1)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(model_prediction, expected_prediction)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
