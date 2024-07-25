import torch
from torch import nn

##################################
#### NN architecture modeling ####
##################################

class AutoencoderLayerBuilder(nn.Module):
    """
    AutoencoderLayerBuilder is a custom neural network module that facilitates the addition of various layers 
    such as fully connected layers, convolutional layers, transposed convolutional layers, and more. 
    It provides a convenient interface to build complex neural network architectures dynamically, specifically 
    for autoencoders.
    """

    def __init__(self):
        super(AutoencoderLayerBuilder, self).__init__()
        self.layers = nn.Sequential()

        self.last_input_length = 0
        self.last_input_channel = 0

    def add_fc_layer(self, params: list, on_flatten=False, start_dim=1):
        """
        Adds a fully connected (dense) layer to the model.
        
        Args:
            params (list): A list containing layer parameters [units, activation, bias, p_dropout, has_batch_norm].
        """
        layers = nn.Sequential()
        units, activation, bias, p_dropout, has_batch_norm = params
        
        # Add postprocessing layers if possible
        if on_flatten:
            layers.append(nn.Flatten(start_dim))
        
        layers.append(nn.Linear(self.last_input_length, units, bias=bias))
        self.last_input_length = units
        
        #print(self.last_input_length)
        
        if has_batch_norm:
            layers.append(self.add_batch_norm(units))
        if activation != "linear" and activation is not None:
            layers.append(self.add_activation(activation))
        if 0.0 < p_dropout < 1.0:
            layers.append(self.add_dropout(p_dropout))
        
        self.layers.append(layers)
        
    def add_fc_layers(self, fc_params: list, on_flatten_first=False):
        """
        Adds multiple fully connected (dense) layers to the model.
        
        Args:
            fc_params (list): A list of parameter lists for each fully connected layer.
        """
        for params in fc_params:
            self.add_fc_layer(params, on_flatten_first)
            
            if on_flatten_first:
                on_flatten_first = False
                
    def add_conv_layer(self, params):
        """
        Adds a 1D convolutional layer to the model.
        
        Args:
            params (list): A list containing layer parameters [output_channels, kernel_length, stride, padding, activation, pool, bias, p_dropout, has_batch_norm].
        """
        layers = nn.Sequential()
        output_channels, kernel_length, stride, padding, activation, pool, bias, p_dropout, has_batch_norm = params
        
        conv1d_layer = nn.Conv1d(self.last_input_channel, 
                                 output_channels,
                                 kernel_length,
                                 stride,
                                 padding,
                                 bias=bias)
        self.last_input_channel = output_channels
        layers.append(conv1d_layer)

        # Add layers if possible
        if pool is not None and isinstance(pool, (tuple, list)):
            layers.append(self.add_pooling(pool))
        if activation != "linear" and activation is not None:
            layers.append(self.add_activation(activation))
        if has_batch_norm:
            layers.append(self.add_batch_norm(output_channels))
        if 0.0 < p_dropout < 1.0:
            layers.append(self.add_dropout(p_dropout))

        self.layers.append(layers)

    def add_conv_layers(self, conv_params: list):
        """
        Adds multiple 1D convolutional layers to the model.
        
        Args:
            conv_params (list): A list of parameter lists for each convolutional layer.
        """
        for params in conv_params:
            self.add_conv_layer(params)
    
    def add_transp_conv_layer(self, params):
        """
        Adds a 1D transposed convolutional layer to the model.
        
        Args:
            params (list): A list containing layer parameters [output_channels, kernel_length, stride, padding, out_padding, activation, bias, p_dropout, has_batch_norm].
        """
        
        layers = nn.Sequential()
        output_channels, kernel_length, stride, padding, out_padding, activation, bias, p_dropout, has_batch_norm = params
        
        conv_transp = nn.ConvTranspose1d(self.last_input_channel, 
                                         output_channels, 
                                         kernel_length,
                                         stride, 
                                         padding=padding,
                                         output_padding=out_padding,
                                         bias=bias)
        self.last_input_channel = output_channels
        
        layers.append(conv_transp)

        # Add layers if possible
        if activation != "linear" and activation is not None:
            layers.append(self.add_activation(activation))
        if has_batch_norm:
            layers.append(self.add_batch_norm(output_channels))
        if 0.0 < p_dropout < 1.0:
            layers.append(self.add_dropout(p_dropout))

        self.layers.append(layers)

    def add_transp_conv_layers(self, transp_conv_params: list):
        """
        Adds multiple 1D transposed convolutional layers to the model.
        
        Args:
            transp_conv_params (list): A list of parameter lists for each transposed convolutional layer.
        """
        for params in transp_conv_params:
            self.add_transp_conv_layer(params)
    
    def add_batch_norm(self, n: int=1):
        """
        Returns a batch normalization layer.
        
        Args:
            n (int, optional): The number of features for batch normalization. Defaults to 1.
        
        Returns:
            torch.nn.Module: Batch normalization layer.
        """
        return nn.BatchNorm1d(n)

    def add_activation(self, name: str, alpha: float=0.25):
        """
        Returns an activation function layer.
        
        Args:
            name (str): The name of the activation function.
            alpha (float, optional): The alpha value for PReLU. Defaults to 0.25.
        
        Returns:
            torch.nn.Module: Activation function layer.
        
        Raises:
            ValueError: If the specified activation function is not supported.
        """
        if name == 'relu':
            return nn.ReLU()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'prelu':
            return nn.PReLU(init=alpha)
        elif name == 'leaky_relu':
            return nn.LeakyReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'softmax':
            return nn.Softmax(dim=1)
        else:
            raise ValueError(f"Activation function '{name}' is not supported.")

    def add_pooling(self, pool: tuple):
        """
        Returns a pooling layer.
        
        Args:
            pool (tuple): A tuple containing pool parameters [name, size, stride, padding] or 
            for [name, scale_factor].
        
        Returns:
            torch.nn.Module: Pooling layer.
        
        Raises:
            ValueError: If the specified pooling type is not supported.
        """
        name, *params = pool
        if name == 'max_pool':
            size, stride, padding = params
            return nn.MaxPool1d(size, stride, padding)
        elif name == 'avg_pool':
            size, stride, padding = params
            return nn.AvgPool1d(size, stride, padding)
        elif name == 'upsample':
            scale_factor = params
            return nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=True)
        else:
            raise ValueError(f"Pooling class '{name}' is not supported.")
    
    def add_adaptive_pooling(self, name: str, output_length: int):
        """
        Returns a pooling layer.
        
        Args:
            name (str): The name of the aggregation function.
            output_length (int): The length of the output.
        
        Returns:
            torch.nn.Module: Adaptive Pooling layer.
        
        Raises:
            ValueError: If the specified pooling type is not supported.
        """
        if name == 'max':
            return nn.AdaptiveMaxPool1d(output_length)
        elif name == 'avg':
            return nn.AdaptiveAvgPool1d(output_length)
        else:
            raise ValueError(f"Pooling class '{name}' is not supported.")

    def add_dropout(self, p: float):
        """
        Returns a dropout layer.
        
        Args:
            p (float): The dropout probability.
        
        Returns:
            torch.nn.Module: Dropout layer.
        """
        return nn.Dropout(p)
    
    def summarize_model(self, shape, *args, **kwargs):
        """
        Prints a summary of the model architecture.
        
        Args:
            shape (tuple): The shape of the input tensor.
            *args: Additional arguments for the summary function.
            **kwargs: Additional keyword arguments for the summary function.
        """
        from torchinfo import summary
        print(summary(self, input_size=shape, *args, **kwargs))

    def summarize_weights(self):
        """
        Prints the shape of the weights of each layer in the model.
        """
        for weights in self.parameters():
            print(f"Layer {weights.shape}")

    def unfreeze_layers(self, model):
        """
        Unfreezes all layers in the given model, allowing their parameters to be updated during training.
        
        Args:
            model (torch.nn.Module): The model whose layers are to be unfrozen.
        """
        for layer in model:
            self._unfreeze_parameters(layer, True)
            
    def _unfreeze_parameters(self, layer, requires_grad):
        """
        Sets the requires_grad attribute of all parameters in a given layer.
        
        Args:
            layer (torch.nn.Module): The layer whose parameters' requires_grad attribute is to be set.
            requires_grad (bool): Whether the parameters require gradients.
        """
        for parameters in layer.parameters():
            parameters.requires_grad = requires_grad
            
    def show_frozen_parameters_status(self, model):
        """
        Prints the frozen status of the parameters in the given model.
        
        Args:
            model (torch.nn.Module): The model whose parameter status is to be shown.
        """
        _on_unfrozen_found = False
        requires_grad_to_is_frozen = {False: "frozen", True: "not frozen"}
        
        for i, parameters in enumerate(model.parameters()):
            if not _on_unfrozen_found and parameters.requires_grad:
                _on_unfrozen_found = True
                print(40 * "*")
            print(f"Parameters_{i}, size {tuple(parameters.shape)} : {requires_grad_to_is_frozen[parameters.requires_grad]}")

    def set_ae_requires_grad(self, next_requires_grad=True):
        self.encoder.requires_grad_(next_requires_grad)
        self.latent.requires_grad_(next_requires_grad)
        self.decoder.requires_grad_(next_requires_grad)

class CategoricalEncoder(AutoencoderLayerBuilder):
    """
    CategoricalEncoder is a custom neural network module designed to encode categorical variables
    into a dense representation. It includes a fully connected layer, an optional activation function,
    and optional batch normalization.

    Args:
        input_length (int): The number of unique categories (input dimension).
        output_length (int): The length of the encoded representation (output dimension).
        bias (bool, optional): Whether to include a bias term in the linear layer. Defaults to True.
        activation (str, optional): The activation function to use. Defaults to "relu".
        has_batch_norm (bool, optional): Whether to include batch normalization. Defaults to True.
    """

    def __init__(self, input_length, output_length, activation="relu", bias=True, has_batch_norm=True, is_conv=True):
        super(CategoricalEncoder, self).__init__()

        self.input_length = input_length
        self.output_length = output_length
        self.bias = bias
        self.activation = activation
        self.has_batchnorm = has_batch_norm
        self.is_conv = is_conv
        
        self.last_input_length = input_length
        self.layers = nn.Sequential()
        
        self.add_fc_layer([output_length, activation, bias, 0.0, has_batch_norm])
    
    def forward(self, x):
        x = self.layers(x)
        if self.is_conv:
            x = x.unsqueeze(1)
        return x

class FullyConnectedEncoder(AutoencoderLayerBuilder):
    """
    FullyConnectedEncoder is a custom neural network module designed to encode input data
    through fully connected layers with customizable specifications.
    """

    def __init__(self, input_length, layer_specifications: list, **kwargs):
        """
        Parameters:
            input_length (int): The number of input features.
            layer_specifications (list): List of tuples defining the layers. Each tuple contains specifications for a layer,
                                including units, activation function, bias, dropout probability, and batch normalization.
            **kwargs: Additional keyword arguments.
        """
        if not isinstance(layer_specifications, list):
            raise Exception("Input error. 'layer_specifications' should be a list")
        if len(layer_specifications) < 1:
            raise Exception("Input error. 'layer_specifications' length is 0")
        
        super(FullyConnectedEncoder, self).__init__()

        # Encoder attributes
        self.input_length = input_length
        self.last_input_length = input_length
        self.layer_specifications = layer_specifications
        self.layers = nn.Sequential()
        
        self.add_fc_layers(layer_specifications)

    def forward(self, x):
        return self.layers(x)

class FullyConnectedDecoder(AutoencoderLayerBuilder):
    """
    FullyConnectedDecoder is a custom neural network module designed to decode input data
    through fully connected layers with customizable specifications.
    """

    def __init__(self, input_length: int, layer_specifications: list, **kwargs):
        """
        Parameters:
            input_length (int): The number of input features.
            layer_specifications (list): List of tuples defining the layers. Each tuple contains specifications for a layer,
                                including units, activation function, bias, dropout probability, and batch normalization.
            **kwargs: Additional keyword arguments.
        """
        if not isinstance(layer_specifications, list):
            raise Exception("Input error. 'layer_specifications' should be a list of tuples")
        if len(layer_specifications) < 1:
            raise Exception("Input error. 'layer_specifications' length is 0")
        
        super(FullyConnectedDecoder, self).__init__()
        
        self.input_length = input_length
        self.last_input_length = input_length
        self.layer_specifications = layer_specifications
        self.layers = nn.Sequential()
        
        # Hidden layer
        self.add_fc_layers(layer_specifications)
        
    def forward(self, x):
        return self.layers(x)

class ConvEncoder(AutoencoderLayerBuilder):
    """
    ConvEncoder is a custom neural network module designed to encode input data
    through convolutional layers with customizable specifications.
    """

    def __init__(self, input_length: int, input_channel: int, layer_specifications: list, on_global_pool=True):
        """
        Parameters:
            input_length (int): The length of the input data.
            input_channel (int): The number of input channels.
            layer_specifications (list): List of tuples defining the layers. Each tuple contains specifications for a layer,
                                including output channels, kernel length, stride, padding, activation function,
                                pooling, bias, dropout probability, and batch normalization.
            **kwargs: Additional keyword arguments.
        """
        if not isinstance(layer_specifications, list):
            raise Exception("Input error. 'layer_specifications' should be a list")
        if len(layer_specifications) < 1:
            raise Exception("Input error. 'layer_specifications' length is 0")
        
        super(ConvEncoder, self).__init__()
        
        # Encoder attributes
        self.last_input_length = input_length
        self.last_input_channel = input_channel
        self.layer_specifications = layer_specifications
        self.layers = nn.Sequential()
        self.on_global_pool = on_global_pool
        
        # Layers
        self.add_conv_layers(layer_specifications)

        if self.on_global_pool:
            self.layers.append(self.add_adaptive_pooling("max", 1))
    
    def forward(self, x):
        x = self.layers(x)
        return x

class ConvDecoder(AutoencoderLayerBuilder):
    """
    Convolutional Decoder module.

    Args:
        input_channel (int): Number of input channels.
        layer_specs (list): List of specifications for each layer.
    """
    def __init__(self, input_channel, layer_specs: list, on_transpose_conv=True):
        if not isinstance(layer_specs, list):
            raise ValueError("Input error. 'layer_specs' should be a list")
        
        super(ConvDecoder, self).__init__()
        
        # Decoder attributes
        self.layer_specs = layer_specs
        self.last_input_channel = input_channel
        self.on_transpose_conv = on_transpose_conv
        self.layers = nn.Sequential()
        
        # Hidden layers
        if on_transpose_conv:
            self.add_transp_conv_layers(layer_specs)
        else:
            self.add_transp_conv_layers(layer_specs)
    
    def forward(self, x):
        return self.layers(x)

class ConvAutoencoderImplicit(AutoencoderLayerBuilder):
    def __init__(
        self, 
        input_length: int, 
        input_channel: int, 
        encoder_specs: list, 
        decoder_specs: list,
        on_global_pool=False,
        on_transpose_conv=True,
        n_categories=0
    ):
        super(ConvAutoencoderImplicit, self).__init__()
        
        # Attributes
        self.input_length = input_length
        self.input_channel = input_channel
        self.latent_output_channel = 1
        self.output_length = input_length
        self.n_categories = n_categories
        
        # Encoder
        self.encoder = ConvEncoder(input_length, input_channel, encoder_specs, on_global_pool=on_global_pool)
        
        # Encoder output length: number_channels * length_last_conv_output
        latent_input_length = self.encoder.last_input_channel * self.get_encoder_output_length([1, input_channel, input_length])
        
        # Category encoding for optional inputs
        if self.n_categories > 0:
            # When categorical data is used, latent output channel is 2
            self.latent_output_channel += 1
            self.category_encoder = CategoricalEncoder(n_categories, latent_input_length)
        
        # Decoder
        self.decoder = ConvDecoder(self.latent_output_channel, decoder_specs, on_transpose_conv=on_transpose_conv)
    
    def forward(self, x, x_category=None):
        x = self.encoder(x)
        
        if not x_category is None:
            x_category_encoding = self.category_encoder(x_category)#.unsqueeze(1)
            
            inputs_concat = [x, x_category_encoding]
            x = torch.cat(inputs_concat, dim=1)
            
        x = self.decoder(x)
        return x

    def get_encoder_output_length(self, input_shape):
        from pyae.utils import get_decoder_target_lengths
        return get_decoder_target_lengths(self.encoder, input_shape)

class ConvAutoencoderLatentFC1(AutoencoderLayerBuilder):
    def __init__(
        self, 
        input_length: int, 
        input_channel: int, 
        latent_length: int,
        encoder_specs: list, 
        latent_specs: list,
        decoder_specs: list,
        n_categories=0,
        pad=0
    ):
        super(ConvAutoencoderLatentFC1, self).__init__()
        
        # Attributes
        self.input_length = input_length
        self.input_channel = input_channel
        self.latent_length = latent_length
        self.latent_output_channel = 1
        self.output_length = input_length
        self.n_categories = n_categories
        self.pad = pad
        
        # Encoder
        self.encoder = ConvEncoder(input_length, input_channel, encoder_specs)
        
        # Encoder output length: number_channels * length_last_conv_output
        latent_input_length = self.encoder.last_input_channel * self.get_encoder_output_length([1, input_channel, input_length])
        
        # Latent layer
        self.latent = LatentFC1(latent_input_length, latent_length, latent_specs, pad=pad)
        
        # Category encoding for optional inputs
        if self.n_categories > 0:
            # When categorical data is used, latent output channel is 2
            self.latent_output_channel += 1
            self.category_encoder = CategoricalEncoder(n_categories, latent_length)
        
        # Decoder
        self.decoder = ConvDecoder(self.latent_output_channel, decoder_specs)
    
    def forward(self, x, x_category=None):
        x = self.encoder(x)
        x = self.latent(x)
        
        if not x_category is None:
            x_category_encoding = self.category_encoder(x_category)#.unsqueeze(1)
            
            inputs_concat = [x, x_category_encoding]
            x = torch.cat(inputs_concat, dim=1)
            
        x = self.decoder(x)
        return x

    def get_encoder_output_length(self, input_shape):
        from pyae.utils import get_decoder_target_lengths
        return get_decoder_target_lengths(self.encoder, input_shape)

class VariationalLatent(AutoencoderLayerBuilder):
    """
    VariationalBuilds a a custom neural network module designed to parameterize
    the mean and log-variance of a latent space in a variational autoencoder (VAE).
    """

    def __init__(self, input_length, latent_length, layer_specifications):
        """
        Parameters:
            input_length (int): The size of the input data.
            latent_length (int): The size of the latent space.
            layer_specifications (tuple): Specifications for the layers, including activation function,
                                 bias, and whether batch normalization is applied.
        """
        super(VariationalLatent, self).__init__()

        # Encoder attributes
        self.last_input_length = input_length
        self.layer_specifications = layer_specifications
        self.layers = nn.Sequential()
        
        # Layers for parameters of a normal distribution N(mu, sigma)
        activation, bias, on_batch_norm = layer_specifications
        params = (latent_length, activation, bias, 0.0, on_batch_norm)
        
        # layer_mean, stored in self.layers[0]
        self.add_fc_layer(params)
        # layer_log_variance, stored in self.layers[1]
        self.last_input_length = input_length
        self.add_fc_layer(params)

    def forward(self, x):
        mean, log_var = self._forward_distribution_params(x)
        z = self._sample_latent(mean, log_var)
        return z, mean, log_var
    
    def _forward_distribution_params(self, x):
        """
        Computes the mean and log-variance of the latent space.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple: Tuple containing the mean and log-variance.
        """
        mean = self.layers[0](x)
        log_var = self.layers[1](x)
        return mean, log_var

    def _sample_latent(self, mean, log_var):
        """
        Samples the latent variable from the parameterized distribution.

        Args:
            mean (torch.Tensor): Mean of the distribution.
            log_var (torch.Tensor): Log-variance of the distribution.

        Returns:
            torch.Tensor: Sampled latent variable.
        """
        # sigma/std = exp { log (std ** 2) / 2 } -> exp { 2 * log (std) / 2 } -> var
        std = torch.exp(log_var / 2)
        eps = torch.rand_like(std)
        z = mean + std + eps
        return z

class LatentFC1(AutoencoderLayerBuilder):
    """
    Builds a a custom neural network with only 1 fully-connected layer module representing 
    the latent space in an autoencoder.
    """

    def __init__(self, 
                 input_length,
                 latent_length,
                 layer_specifications,
                 on_conv_ae=True, 
                 pad=1,
                 mode="replicate"
                ):
        """
        Parameters:
            input_length (int): The size of the input data.
            latent_length (int): The size of the latent data.
            layer_specifications (list): Specifications for the layers, including activation function,
                                bias, and whether batch normalization is applied.
        """
        super(LatentFC1, self).__init__()

        # Encoder attributes
        self.input_length = input_length
        self.latent_length = latent_length
        self.layer_specifications = layer_specifications
        self.on_conv_ae = on_conv_ae
        self.pad = pad
        self.mode = mode
        self.last_input_length = input_length
        self.output_reshape = (-1, 1, latent_length)
        self.layers = nn.Sequential()
        
        latent_specifications = [latent_length, *layer_specifications]
        self.add_fc_layer(latent_specifications, on_flatten=True)
    
    def forward(self, x):
        x = self.layers(x)
        if self.on_conv_ae:
            x = x.view(self.output_reshape)  # Shape (batch, channel, length)
        if self.pad > 0:
            x = self.set_pad_to_outputs(x)
        return x
    
    def set_pad_to_outputs(self, inputs):
        return torch.nn.functional.pad(inputs, pad=(self.pad, self.pad), mode=self.mode)

class LatentFC3(AutoencoderLayerBuilder):
    """
    Builds a a custom neural network with only 3 fully-connected layer module representing 
    the latent space in an autoencoder.
    """

    def __init__(self,
                 input_length, 
                 latent_length, 
                 input_channel, 
                 layer_specifications, 
                 on_conv_ae=True, 
                ):
        """
        Parameters:
            input_length (int): The size of the input data.
            latent_length (int): The size of the latent data.
            input_channel (int): The number of input channels from the last encoder layer.
            layer_specifications (list): Specifications for the layers, including activation function,
                                bias, and whether batch normalization is applied.
        """
        super(LatentFC3, self).__init__()

        # Encoder attributes
        self.input_length = input_length
        self.latent_length = latent_length
        self.input_channel = input_channel
        self.layer_specifications = layer_specifications
        self.on_conv_ae = on_conv_ae
        self.last_input_length = input_length
        self.output_reshape = (-1, input_channel, input_length // input_channel)
        self.layers = nn.Sequential()

        
        encoder_to_latent_specifications = [input_length, *layer_specifications]
        latent_specifications = [latent_length, *layer_specifications]
        latent_to_decoder_specifications = [input_length, *layer_specifications]

        self.add_fc_layers([encoder_to_latent_specifications, 
                            latent_specifications, 
                            latent_to_decoder_specifications
                           ],
                           on_flatten_first=True
                           )

    def forward(self, x):
        x = self.layers(x)
        if self.on_conv_ae:
            return x.view(self.output_reshape)  # Shape (batch, channel, length)
        return x
        
class DCECLatentFC3(AutoencoderLayerBuilder):
    """
    Builds a a custom neural network with only 3 fully-connected layer module representing 
    the latent space in an autoencoder for Deep Convolutional Embedded Clustering (DCEC).
    """

    def __init__(self, input_length, input_channel, layer_specifications, on_conv_ae=True):
        """
        Parameters:
            input_length (int): The size of the input data.
            input_channel (int): The number of input channels from the last encoder layer.
            layer_specifications (list): Specifications for the layers, including activation function,
                                bias, and whether batch normalization is applied.
        """
        super(DCECLatentFC3, self).__init__()

        # Encoder attributes
        self.input_length = input_length
        self.input_channel = input_channel
        self.layer_specifications = layer_specifications
        self.last_input_length = input_length
        self.output_reshape = (-1, input_channel, input_length // input_channel)
        self.layers = nn.Sequential()

        self.on_conv_ae = on_conv_ae
        
        encoder_to_latent_specifications = [self.input_length, *layer_specifications[0]]
        latent_specifications = layer_specifications[1]
        latent_to_decoder_specifications = [self.input_length, *layer_specifications[-1]]

        self.add_fc_layers([encoder_to_latent_specifications, 
                            latent_specifications, 
                            latent_to_decoder_specifications],
                           on_flatten_first=True
                          )

    def forward(self, x):
        # Compute on input layer
        x = self.layers[0](x)
        # Compute on latent layer
        z = self.layers[1](x)
        # Compute on output layer
        x = self.layers[2](z)
        
        if self.on_conv_ae:
            return x.view(self.output_reshape), z  # Shape (batch, channel, length)
        return x, z
        
class EnsembleLatentFC3(AutoencoderLayerBuilder):
    """
    Builds a a custom neural network with only 3 fully-connected layer module representing 
    the latent space in an autoencoder.
    """

    def __init__(self, latent_length, input_lengths, layer_specifications, increment_output_units=2, on_conv_ae=True):
        """
        Parameters:
            latent_length (int): The sizes of the input data.
            input_lengths (int): The sizes of the input data.
            layer_specifications (list): Specifications for the layers, including activation function,
                                bias, and whether batch normalization is applied.
            on_conv_ae (bool)
        """
        super(EnsembleLatentFC3, self).__init__()
        
        # Encoder attributes
        self.input_lengths = input_lengths
        self.increment_output_units = increment_output_units
        self.layer_specifications = layer_specifications
        self.latent_length = latent_length
        self.concat_length = len(input_lengths) * latent_length
        # Deleted to use latent layer as output
        #self.output_reshape = (-1, 1, len(input_lengths) * latent_length)
        self.output_reshape = (-1, 1, latent_length)
        self.layers = nn.Sequential()

        self.on_conv_ae = on_conv_ae
        
        # Encoders to concatenation block
        on_flatten_first = True
        for input_length in self.input_lengths:
            self.last_input_length = input_length

            # Deleted to use latent layer as output
            #encoder_to_concat_specifications = [self.latent_length, *layer_specifications[0]]
            encoder_to_concat_specifications = [self.latent_length, *layer_specifications]
            self.add_fc_layer(encoder_to_concat_specifications, on_flatten=True)
        
            if on_flatten_first:
                on_flatten_first = False

        self.last_input_length = self.concat_length
        # Concatenation to latent block
        # Deleted to use latent layer as output
        #concat_to_latent_specifications = [self.concat_length, *layer_specifications[1]]
        concat_to_latent_specifications = [self.concat_length, *layer_specifications]
        self.add_fc_layer(concat_to_latent_specifications)
        
        # pre-latent to latent block
        # Deleted to use latent layer as output
        #prelatent_to_latent_specifications = [self.latent_length, *layer_specifications[2]]
        prelatent_to_latent_specifications = [self.latent_length, *layer_specifications]
        self.add_fc_layer(concat_to_latent_specifications)
        
        # Latent blocks
        # Deleted to use latent layer as output
        #latent_specifications = layer_specifications[3]
        latent_specifications = [latent_length, *layer_specifications]
        self.add_fc_layer(latent_specifications)
        
        # latent to decoder block
        #latent_to_decoder_specifications = [increment_output_units * self.latent_length, *layer_specifications[3]]
        latent_to_decoder_output_length = self.concat_length
        # Deleted to use latent layer as output
        #latent_to_decoder_specifications = [latent_to_decoder_output_length, *layer_specifications[4]]
        latent_to_decoder_specifications = [latent_to_decoder_output_length, *layer_specifications]
        self.add_fc_layer(latent_to_decoder_specifications)

    def forward(self, xs):
        concat_inputs = []
        # Compute on encoder to concat
        for i, x in enumerate(xs):
            x_encoder = self.layers[i](x)
            #print(x.shape, x_encoder.shape)
            concat_inputs.append(x_encoder)
        
        # Concatenate along the feature dimension (dim=1)
        concatenated_latent = torch.cat(concat_inputs, dim=1)
        #print(f"concat: {concatenated_latent.shape}")
        # Compute on prelatent to latent
        x = self.layers[-4](concatenated_latent)
        # Compute on concat to latent
        x = self.layers[-3](x)
        # Compute on latent
        x = self.layers[-2](x)
        # Compute on latent to decoder
        #x = self.layers[-1](x)
        
        return x.view(self.output_reshape)  # Shape (batch, channel, length)

class ClusteringLayer(AutoencoderLayerBuilder):
    """
    ClusteringLayer is a custom neural network module representing the clustering centers branch
    from the autoencoder latent layer.
    """

    def __init__(self, n_clusters, features_length, pretrain_cluster_weights=None):
        """
        Parameters:
            n_clusters (int): The number of clusters.
            features_length (int): The size of the latent data.
            pretrain_cluster_weights ()
        """
        super(ClusteringLayer, self).__init__()
        
        # Encoder attributes
        self.features_length = features_length
        self.n_clusters = n_clusters
        # Pretrain weights
        self.pretrain_cluster_weights = pretrain_cluster_weights
        self.layers = nn.Sequential()

        # Initialize clustering weights
        self.initialize_weights(pretrain_cluster_weights)

    def forward(self, z):
        return self._t_distribution(z)

    def _t_distribution(self, z, dim_norm=2):
        """
        Computes a soft labels on z by Student's T-distribution.
        """
        # Reshape z to add centers dimension (batch, 1, features) to broadcast
        z_expanded = z.unsqueeze(1)
        
        # Reshape z to add batch dimension (1, centers, features) to broadcast
        centers_expanded = self.cluster_weights.unsqueeze(0)
        
        # Compute MSE loss from shape (batch, centers, features) to (batch, centers)
        mse_loss = torch.sqrt(
            torch.sum(
                (z_expanded - centers_expanded) ** 2, 
                dim=dim_norm
            )
        )
        # Apply the Student's t-distribution
        q_raw = 1.0 / (1.0 + mse_loss)
        # Normalize the distribution
        q = q_raw / q_raw.sum(dim=1).unsqueeze(1)
        return q

    def initialize_weights(self, weights=None):
        if weights is None:
            self.cluster_weights = nn.Parameter(torch.empty(self.n_clusters, self.features_length), requires_grad=True)
            nn.init.xavier_uniform_(self.cluster_weights) # Xavier initialization
        else:
            self.cluster_weights = nn.Parameter(weights, requires_grad=True)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super(BottleneckBlock, self).__init__()
        filter1, filter2, filter3 = filters
        
        self.conv1 = nn.Conv1d(in_channels, filter1, kernel_length=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm1d(filter1)
        
        self.conv2 = nn.Conv1d(filter1, filter2, kernel_length=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(filter2)
        
        self.conv3 = nn.Conv1d(filter2, filter3, kernel_length=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(filter3)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filter3:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, filter3, kernel_length=1, stride=stride, padding=0),
                nn.BatchNorm1d(filter3)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock1D, self).__init__()
        
        # Branch 1: 1x1 Convolution
        self.branch1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # Branch 2: 1x1 Convolution followed by 3x3 Convolution
        self.branch3x3_1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=1)
        self.branch3x3_2 = nn.Conv1d(out_channels // 2, out_channels, kernel_size=3, padding=1)
        
        # Branch 3: 1x1 Convolution followed by two 3x3 Convolutions (approximating 5x5)
        self.branch5x5_1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=1)
        self.branch5x5_2 = nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.branch5x5_3 = nn.Conv1d(out_channels // 2, out_channels, kernel_size=3, padding=1)
        
        # Branch 4: 1x1 Convolution followed by three 3x3 Convolutions (approximating 7x7)
        self.branch7x7_1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=1)
        self.branch7x7_2 = nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.branch7x7_3 = nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.branch7x7_4 = nn.Conv1d(out_channels // 2, out_channels, kernel_size=3, padding=1)
        
        # Branch 5: 3x3 Max pooling followed by 1x1 Convolution
        self.branch_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = F.relu(self.branch3x3_2(branch3x3))
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = F.relu(self.branch5x5_2(branch5x5))
        branch5x5 = F.relu(self.branch5x5_3(branch5x5))
        
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = F.relu(self.branch7x7_2(branch7x7))
        branch7x7 = F.relu(self.branch7x7_3(branch7x7))
        branch7x7 = F.relu(self.branch7x7_4(branch7x7))
        
        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)
        
        # Concatenate branches along the channel axis (dimension 1)
        output = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7, branch_pool], dim=1)
        
        return output

class InceptionAutoencoder(nn.Module):
    def __init__(self, in_channels, encoding_channels, bottleneck_channels):
        super(InceptionAutoencoder, self).__init__()
        
        # Encoder
        self.enc_conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.enc_inception1 = InceptionBlock1D(64, encoding_channels)
        self.enc_conv2 = nn.Conv1d(encoding_channels * 5, bottleneck_channels, kernel_size=3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=2, padding=1)
        
        # Decoder
        self.dec_conv1 = nn.ConvTranspose1d(bottleneck_channels, encoding_channels * 5, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_inception1 = InceptionBlock1D(encoding_channels * 5, encoding_channels)
        self.dec_conv2 = nn.ConvTranspose1d(encoding_channels * 5, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose1d(64, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.enc_conv1(x))
        x = self.enc_inception1(x)
        x = F.relu(self.enc_conv2(x))
        
        # Bottleneck
        x = F.relu(self.bottleneck(x))
        
        # Decoder
        x = F.relu(self.dec_conv1(x))
        x = self.dec_inception1(x)
        x = F.relu(self.dec_conv2(x))
        x = torch.sigmoid(self.dec_conv3(x))  # Using sigmoid for normalized outputs
        
        return x

# Example usage
in_channels = 1
encoding_channels = 32
bottleneck_channels = 64
autoencoder = InceptionAutoencoder(in_channels, encoding_channels, bottleneck_channels)
input_tensor = torch.randn(1, in_channels, 256)  # Example input tensor with shape (batch_size, channels, length)
output_tensor = autoencoder(input_tensor)
print(output_tensor.shape)  # Expected output shape: (batch_size, in_channels, length)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_length=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_length=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_length=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out