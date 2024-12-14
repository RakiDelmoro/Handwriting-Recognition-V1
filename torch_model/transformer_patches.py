import math
import torch
import random
import torch.nn as nn
from einops.layers.torch import Rearrange
from Model.utils import unpadded_length_tokens
from datasets.utils import ints_to_characters, char_to_index, PAD_TOKEN
from Model.configurations import NETWORK_FEATURE_SIZE, ATTENTION_FEATURE_SIZE, NUM_ATTENTION_HEADS,  MLP_FEATURE_SIZE, NUM_LAYERS, MAX_PATCHES_LENGTH, IMAGE_SIZE, PATCH_SIZE, DEVICE
    
class ImageEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        image_h, image_w = IMAGE_SIZE
        patch_h, patch_w = PATCH_SIZE
        num_patches = (image_h // patch_h) * (image_w // patch_w)
        patch_feature_size = patch_h * patch_w
        self.to_patches_neurons = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1=patch_h, p2=patch_w),
            nn.Linear(patch_feature_size, NETWORK_FEATURE_SIZE, device=DEVICE))
        self.classification_token = nn.Parameter(torch.zeros(1, 1, NETWORK_FEATURE_SIZE, device=DEVICE))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, NETWORK_FEATURE_SIZE, device=DEVICE))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, NETWORK_FEATURE_SIZE, device=DEVICE))

    def forward(self, batched_image):
        patches_activations = self.to_patches_neurons(batched_image)
        batch, _, _ = patches_activations.shape
        cls_tokens = self.classification_token.expand(batch, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch, -1, -1)
        patches_with_cls_and_dist_tokens = torch.cat((cls_tokens, patches_activations, distillation_tokens), dim=1)
        patches_activations = patches_with_cls_and_dist_tokens + self.position_embeddings
        return patches_activations

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_feature_size = ATTENTION_FEATURE_SIZE
        self.num_heads = NUM_ATTENTION_HEADS
        self.total_attention_feature_size = NUM_ATTENTION_HEADS*ATTENTION_FEATURE_SIZE
        self.query = nn.Linear(NETWORK_FEATURE_SIZE, self.total_attention_feature_size, device=DEVICE)
        self.key = nn.Linear(NETWORK_FEATURE_SIZE, self.total_attention_feature_size, device=DEVICE)
        self.value = nn.Linear(NETWORK_FEATURE_SIZE, self.total_attention_feature_size, device=DEVICE)

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
        return input_embeddings_context.view(attention_output_shape)

class EncoderMultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_1 = nn.Linear(NETWORK_FEATURE_SIZE, MLP_FEATURE_SIZE, device=DEVICE)
        self.linear_layer_2 = nn.Linear(MLP_FEATURE_SIZE, NETWORK_FEATURE_SIZE, device=DEVICE)
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

    def forward(self, input_embeddings: torch.Tensor):
        attention_output = self.multi_head_attetion(input_embeddings)
        residual_connection_1 = attention_output + input_embeddings # first residual connection
        mlp_output = self.mlp_encoder(residual_connection_1)
        residual_connection_2 = mlp_output + residual_connection_1 # second residual connection
        return residual_connection_2

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Layer()
        self.encoder_layers = nn.ModuleList([self.layer for _ in range(NUM_LAYERS)])

    def forward(self, input_embeddings: torch.Tensor):
        layer_output = input_embeddings
        for layer in self.encoder_layers:
            layer_output = layer(layer_output)
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
        self.multi_layer_perceptron = MultiLayerPerceptron()
        self.model_output_prediction = nn.Linear(NETWORK_FEATURE_SIZE, 10, device=DEVICE)

    def forward(self, batched_image_array):
        input_embeddings = self.image_embeddings(batched_image_array)
        encoder_output = self.encoder_layers(input_embeddings)
        mlp_output = self.multi_layer_perceptron(encoder_output)
        model_prediction = self.model_output_prediction(mlp_output)[:, 0, :]
        return model_prediction

    def get_stress_and_update_parameters(self, model_prediction, expected_prediction, optimizer, learning_rate):
        optimizer = optimizer(self.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(model_prediction, expected_prediction)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, optimizer.state_dict()

    def prediction_as_indices(self, model_prediction):
        model_prediction_as_probabilities = torch.nn.Softmax(dim=-1).forward(model_prediction)
        model_prediction = model_prediction_as_probabilities.data
        model_prediction_indices = model_prediction.topk(1)[1].squeeze(-1).cpu().numpy()
        return model_prediction_indices

    def prediction_as_str(self, model_prediction_in_indies):
        return ints_to_characters(model_prediction_in_indies)
    
    def model_parameters_count(self):
        print(sum(p.numel() for p in self.parameters()))
