import cupy
from utils import axons_and_dentrites_initialization, softmax

def transformer_model(network_feature_size, num_attn_heads, attention_feature_size):

    def image_patches_embeddings(image_patches, parameters=None):
        batch_size = image_patches.shape[0]
        num_patches = image_patches.shape[1]
        input_feature = image_patches.shape[0]
        output_feature = network_feature_size
        if parameters is None: axons, dentrites = axons_and_dentrites_initialization(output_feature, input_feature)
        else: axons, dentrites = parameters
        # This array is to provide information of image patches position. Unlike RNN which process data sequentially. Transformer work on all tokens/patches simultaneuously.
        trainable_positional_embedding = cupy.zeros([batch_size, num_patches, output_feature])
        image_projection = cupy.dot(image_patches, axons) + dentrites
        # batch size | patches | image patched feature size
        return image_projection + trainable_positional_embedding

    def self_attention(image_patches, parameters=None):
        batch_size = image_patches.shape[0]
        num_patches = image_patches.shape[1]
        total_attention_feature_size = num_attn_heads * attention_feature_size

        total_attn_feature_size = num_attn_heads * attention_feature_size
        if parameters is None: 
            query = axons_and_dentrites_initialization(network_feature_size, total_attn_feature_size)
            key = axons_and_dentrites_initialization(network_feature_size, total_attn_feature_size)
            value = axons_and_dentrites_initialization(network_feature_size, total_attn_feature_size)
        else: query = parameters[0], key = parameters[1], value = parameters[2]
        # batch | attention heads | patches | attention feature size
        image_patches_query = (cupy.dot(image_patches, query[0]) + query[1]).reshape(batch_size, num_attn_heads, num_patches, attention_feature_size)
        image_patches_key = (cupy.dot(image_patches, key[0]) + key[1]).reshape(batch_size, num_attn_heads, num_patches, attention_feature_size)
        image_patches_value = (cupy.dot(image_patches, value[0]) + value[1]).reshape(batch_size, num_attn_heads, num_patches, attention_feature_size)
        # attention scores -> batch | attention heads | patches | patches
        attention_scores = cupy.matmul(image_patches_query, image_patches_key.transpose(0, 1, 3, 2))
        # attention scores as probabilities
        attentions_probability = softmax(attention_scores)
        # image_patches_context -> batch | patches | attention heads | attention feature size
        image_patches_context = cupy.matmul(attentions_probability, image_patches_value).reshape(batch_size, num_patches, num_attn_heads, attention_feature_size)
        # batch | patches | attention heads * attention feature size
        return image_patches_context.reshape(batch_size, num_patches, total_attention_feature_size)

    return self_attention

x = cupy.random.randn(1, 14, 256)    
transformer_model(network_feature_size=256, num_attn_heads=8, attention_feature_size=128)(x)