import torch
import numpy as np

def get_decoder_params(encoder_model, 
                       latent_output_length, 
                       input_shape, 
                       encoder_block_params,
                       decoder_output_channels,
                       right_part,
                       implicit_latent=False,
                       verbose=True
                      ):
    
    encoder_lengths = get_decoder_target_lengths(encoder_model, input_shape, latent_output_length)

    # Commented because the last encoder length was being ignored
    if implicit_latent:
        start_length = encoder_lengths[0]
        target_lengths = encoder_lengths[1:]
    else:
        start_length = latent_output_length    
        target_lengths = encoder_lengths
        
    decoder_params = _search_decoder_params(start_length, target_lengths, encoder_block_params, verbose=verbose)
    
    hidden_params_decoder = _fill_params(decoder_params, decoder_output_channels, right_part)

    return hidden_params_decoder
    
def get_decoder_target_lengths(encoder_model, input_shape, latent_length=-1):
    x = torch.rand(input_shape)
    input_length = input_shape[-1]
    
    # If implicit latent space, only returns the output length of the encoder 
    if latent_length == -1:
        encoder_output_length = encoder_model.layers(x).shape[-1]
        return encoder_output_length
    
    encoder_lengths = []
    for layers in encoder_model.layers:
        for inner_layer in layers:
            x = inner_layer(x)
            inner_layer_output_length = x.shape[-1]
            encoder_lengths.append(inner_layer_output_length)
    encoder_lengths = encoder_lengths + [input_length]

    return sorted(set(encoder_lengths), reverse=False)

def _fill_params(last_params, output_channels, common_part):
    complete_params = []
    for i, (params, channels) in enumerate(zip(last_params, output_channels)):
        complete_params.append((channels, *params, *common_part))
    
    return complete_params

def _search_decoder_params(start_length, target_lengths, target_block, verbose=False):
    """
    Returns the proper number of decoder parameters which match the 
    output length from the encoder opposite layer.
    Note. A latent to decoder layer is first built before the other decoder layers.
    """
    params = []
    last_length = start_length
    target_params = (len(target_lengths) - 1) * target_block
    #target_params = encoder_params
    
    if verbose:
        print(f"Targets: {target_lengths}")
    
    for target_length, (k_enc, s_enc, p_enc) in zip(target_lengths, target_params):
        predicted_length = _conv_transpose1d_output_length(last_length, k_enc, s_enc, p_enc)
        length_direction = predicted_length - target_length
        k_dec = k_enc
        s_dec = s_enc
        pad_dec = p_enc
        out_pad = 0

        if verbose:
            print(f"\nNext parameters: {k_enc, s_enc, p_enc}")
            print(f"\tinitial_length {last_length}; predicted_length: {predicted_length}; target: {target_length}")
        
        if length_direction < 0:
            if s_dec > abs(length_direction):
                out_pad = abs(length_direction)
            else:
                k_dec += abs(length_direction)
                
            if verbose:
                print(f"(Adjusting out padding to {out_pad})")
        elif length_direction > 0:
            pad_dec += abs(length_direction) // 2
            
            if length_direction % 2 != 0:
                k_dec += 1
            
            if verbose:
                print(f"(Adjusting padding to {pad_dec})")
        
        predicted_length = _conv_transpose1d_output_length(last_length, k_dec, s_dec, pad_dec, out_pad)
        
        params.append((k_dec, s_dec, pad_dec, out_pad))
        last_length = predicted_length
        
        if verbose:
            print(f"\tinitial_length {last_length}; predicted_length: {predicted_length}; target: {target_length}")
            print(f"\tSaved parameters {k_dec, s_dec, pad_dec, out_pad}")

    return params

def _conv_transpose1d_output_length(L_in, kernel_size, stride=1, padding=0, output_padding=0, dilation=1):
    """
    Calculate the output length of a ConvTranspose1d layer.

    Parameters:
    L_in (int): Length of the input.
    kernel_size (int): Size of the convolving kernel.
    stride (int): Stride of the convolution. Default is 1.
    padding (int): Amount of padding added to both sides of the input. Default is 0.
    output_padding (int): Amount of padding added to one side of the output. Default is 0.
    dilation (int): Spacing between kernel elements. Default is 1.

    Returns:
    int: Calculated length of the output.
    """
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return L_out

###############
# Deprecated #
##############

def entropy(*ps):
    if not isinstance(ps, np.ndarray):
        ps = np.array(ps)
    ps = ps.squeeze()
    #print(ps, np.log(ps), -ps * np.log(ps), -np.sum(ps * np.log(ps)))
    return -ps.dot(np.log(ps))
    
def cross_entropy(p_target, p_candidate):
    if not isinstance(p_target, np.ndarray):
        p_target = np.array(p_target)
    if not isinstance(p_candidate, np.ndarray):
        p_candidate = np.array(p_candidate)
    
    p_target = p_target.squeeze()
    p_candidate = p_candidate.squeeze()
    
    #print(ps, np.log(ps), -ps * np.log(ps), -np.sum(ps * np.log(ps)))
    return -p_target.dot(np.log(p_candidate))