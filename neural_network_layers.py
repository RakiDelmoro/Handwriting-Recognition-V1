import cupy
import torch

def linear_neurons(input_neurons, axons, dentrites, activation_function):
    return activation_function(cupy.matmul(input_neurons, axons) + dentrites)

def convolution_neurons(input_image, axons, step_of_patch_window):
    ''' input_neurons shape -> batch | img_channels (I will think of it as a patches) | height | width
        axons -> output_neurons_patches| input_neurons_patches | patch_window_height | patch_window_width 
        step_of_patch_window -> int '''
    batch, _, height, width = input_image.shape
    output_patches, _, patch_window_h, patch_window_w = axons.shape
    height_output = cupy.floor(1 + (height - patch_window_h) / step_of_patch_window).astype(int).item()
    width_output = cupy.floor(1 + (width - patch_window_w) / step_of_patch_window).astype(int).item()
    image_featured_extracted = cupy.zeros((batch, output_patches, height_output, width_output), dtype=cupy.float32)
    input_image_expanded = cupy.expand_dims(input_image, axis=1)
    axons_expanded = cupy.expand_dims(axons, axis=0)
    for i in range(height_output):
        for j in range(width_output):
            vertical_pixel_start = i*step_of_patch_window
            vertical_pixel_end = i*step_of_patch_window+patch_window_h
            horizontal_pixel_start = j*step_of_patch_window
            horizontal_pixel_end = j*step_of_patch_window+patch_window_w
            image_windows = input_image_expanded[:, :, :, vertical_pixel_start:vertical_pixel_end, horizontal_pixel_start:horizontal_pixel_end]
            extracted_feature_based_on_window_size = image_windows * axons_expanded
            aggregate_extracted_feature = cupy.sum(extracted_feature_based_on_window_size, axis=(2,3,4))
            image_featured_extracted[:, :, i, j] = aggregate_extracted_feature
    return image_featured_extracted

def attention_mechanism_neurons():
    pass

def multi_layer_neurons(input_neurons, depth, layers_axons, layers_dentrites):
    activation = input_neurons
    neurons_activations = [activation]
    for each in range(depth):
        axons = layers_axons[each]
        dentrites = layers_dentrites[each]
        activation = activation(cupy.matmul(activation, axons) + dentrites)
        neurons_activations.append(activation)
    return neurons_activations

def torch_convolution_neurons(input_image, weights, stride):
    image_featured_extracted = torch.nn.functional.conv2d(input_image, weights, stride=stride)
    return image_featured_extracted

B, C, H, W = 1, 1, 28, 28
OUT_C, IN_C, H_K, H_W = 8, C, 3, 3

# batch | patches | height | width
image_test = cupy.random.randn(B, C, H, W)
# batch | patches | patch_window_h | patch_window_w
axons = cupy.random.randn(OUT_C, IN_C, H_K, H_W)

torch_image_test = torch.tensor(image_test)
torch_axons = torch.tensor(axons)

output_cupy = convolution_neurons(image_test, axons, step_of_patch_window=1)
output_torch = torch_convolution_neurons(torch_image_test, torch_axons, stride=1)

# Use allclose function since the two different operation may be different in terms of floating-point precisions.
print(cupy.allclose(output_cupy, cupy.array(output_torch))) # Check if two operations are equal
