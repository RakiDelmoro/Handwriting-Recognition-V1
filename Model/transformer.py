import cupy
from Model.utils import axons_and_dentrites_initialization, softmax

def transformer_model(network_feature_size, num_attn_heads, attention_feature_size, number_of_classes, padding_token):

    def image_patches_embeddings(image_patches, parameters=None):
        batch_size = image_patches.shape[0]
        num_patches = image_patches.shape[1]
        input_feature = image_patches.shape[-1]
        output_feature = network_feature_size
        if parameters is None: axons, dentrites = axons_and_dentrites_initialization(input_feature, output_feature)
        else: axons, dentrites = parameters
        # This array is to provide information of image patches position. Unlike RNN which process data sequentially. Transformer work on all tokens/patches simultaneuously.
        trainable_positional_embedding = cupy.zeros([batch_size, num_patches, output_feature])
        image_projection = cupy.dot(image_patches, axons) + dentrites
        # batch size | patches | image patched feature size
        return image_projection + trainable_positional_embedding
    
    def word_tokens_embeddings(word_tokens, padding_token=padding_token, parameters=None):
        if parameters is None: character_tokens_axons = axons_and_dentrites_initialization(number_of_classes, network_feature_size)[0]
        else: character_tokens_axons = parameters
        # Assign no connection for padding index
        character_tokens_axons[padding_token] = cupy.zeros(network_feature_size)
        # batch | tokens length | network feature size
        tokens_embeddings = character_tokens_axons[word_tokens]
        trainable_positional_embedding = cupy.zeros(tokens_embeddings.shape)
        return tokens_embeddings + trainable_positional_embedding
    
    def self_attention(input_tokens, parameters=None):
        batch_size = input_tokens.shape[0]
        num_tokens = input_tokens.shape[1]
        total_attn_feature_size = num_attn_heads * attention_feature_size
        if parameters is None: 
            axons_for_query, dentrites_for_query = axons_and_dentrites_initialization(network_feature_size, total_attn_feature_size)
            axons_for_key, dentrites_for_key = axons_and_dentrites_initialization(network_feature_size, total_attn_feature_size)
            axons_for_value, dentrites_for_value = axons_and_dentrites_initialization(network_feature_size, total_attn_feature_size)
            axons_for_attention, dentrites_for_attention = axons_and_dentrites_initialization(total_attn_feature_size, network_feature_size)
        else:
            axons_for_query, dentrites_for_query = parameters[0]
            axons_for_key, dentrites_for_key = parameters[1]
            axons_for_value, dentrites_for_value = parameters[2]
            axons_for_attention, dentrites_for_attention = parameters[3]
        # batch | attention heads | tokens | attention feature size
        image_patches_query = (cupy.dot(input_tokens, axons_for_query) + dentrites_for_query).reshape(batch_size, num_attn_heads, num_tokens, attention_feature_size)
        image_patches_key = (cupy.dot(input_tokens, axons_for_key) + dentrites_for_key).reshape(batch_size, num_attn_heads, num_tokens, attention_feature_size)
        image_patches_value = (cupy.dot(input_tokens, axons_for_value) + dentrites_for_value).reshape(batch_size, num_attn_heads, num_tokens, attention_feature_size)
        # attention scores -> batch | attention heads | tokens | tokens
        attention_scores = cupy.matmul(image_patches_query, image_patches_key.transpose(0, 1, 3, 2))
        # attention scores as probabilities
        attentions_probabilities = softmax(attention_scores)
        # image_patches_context -> batch | patches | attention heads | attention feature size
        image_patches_context = cupy.matmul(attentions_probabilities, image_patches_value).reshape(batch_size, num_tokens, num_attn_heads, attention_feature_size)
        # batch | tokens | attention heads * attention feature size
        attention_output = image_patches_context.reshape(batch_size, num_tokens, total_attn_feature_size)
        # batch | tokens | network feature size
        attention_output = cupy.dot(attention_output, axons_for_attention) + dentrites_for_attention
        return attention_output

    def test_runner(image_patches, word_tokens):
        image_embeddings = image_patches_embeddings(image_patches)
        attention_output = self_attention(image_patches)

    return test_runner
