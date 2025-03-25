import torch

def model_norm(model: torch.nn.Module, include_last: bool=False):
    """
    takes any pytorch module or network and does model normalisation across all layers.
    by default, the last layer is not normalised.
    """
    param_list = [param for param in model.parameters() if param.requires_grad]
    param_geny = iter(param_list)  # reset generator

    param_count = len(param_list)
    layers = param_count if include_last else param_count - 1
    layer_scales = []

    for _ in range(layers):
        neurons = next(param_geny)
        max_pos_in = 0

        if neurons.dim() >= 2:  # if layer is linear/conv

            for neuron in neurons:    
                input_sum = torch.sum(torch.clamp(neuron, min=0)) # sum all positive parameters
                max_pos_in = max(max_pos_in, input_sum.item())

        if max_pos_in > 0:
            neurons.data /= max_pos_in  # without .data this becomes out-of-place for some reason

        layer_scales.append(max_pos_in)

    return layer_scales

# algorithm for data normalisation
def data_norm(model: torch.nn.Module, activations: list[torch.Tensor], include_last: bool=True):
    """
    takes a pytorch module or network and does data normalisation.
    requires the list of maximum activations from each layer.
    unlike model normalisation, also normalises the last layer by default.
    """
    param_list = [param for param in model.parameters() if param.requires_grad]
    param_geny = iter(param_list)

    param_count = len(param_list)
    layers = param_count if include_last else param_count - 1
    layer_scales = []

    previous_factor = 1

    for i in range(layers):
        neurons = next(param_geny)
        max_weight = 0

        if neurons.dim() >= 2:

            for neuron in neurons:
                # grab maximum single weight across input connections
                max_weight = max(max_weight, torch.max(neuron))

        if max_weight > 0:
            scale_factor = max(max_weight, activations[i])
            applied_factor = scale_factor / previous_factor

        # rescale all weights wrt applied factor
        neurons.data = neurons / applied_factor # without .data this becomes out-of-place for some reason
        previous_factor = scale_factor
        layer_scales.append(applied_factor)
    
    return layer_scales