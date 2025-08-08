import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

from utils.miscellaneous import get_decoder_target_lengths

##################################
#### NN architecture modeling ####
##################################

class AutoencoderNetworkBuilder(nn.Module):
    """
    AutoencoderNetworkBuilder is a custom neural network module that facilitates the addition of various layers 
    such as dense layers, convolutional layers, transposed convolutional layers, and more. 
    It provides a convenient interface to build complex neural network architectures dynamically, specifically 
    for autoencoders.
    """

    def __init__(self):
        super(AutoencoderNetworkBuilder, self).__init__()
        self.layers = nn.Sequential()

        self.last_input_length = 0
        self.last_input_channel = 0

    def add_dense_block(self, params: list, on_flatten=False, start_dim=1):
        """
        Adds a dense (dense) layer to the model.
        
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
        
    def add_dense_blocks(self, dense_params: list, on_flatten_first=False):
        """
        Adds multiple dense (dense) layers to the model.
        
        Args:
            dense_params (list): A list of parameter lists for each dense layer.
        """
        for params in dense_params:
            self.add_dense_block(params, on_flatten_first)
            
            if on_flatten_first:
                on_flatten_first = False
                
    def add_conv_block(self, params):
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

    def add_conv_blocks(self, conv_params: list):
        """
        Adds multiple 1D convolutional layers to the model.
        
        Args:
            conv_params (list): A list of parameter lists for each convolutional layer.
        """
        for params in conv_params:
            self.add_conv_block(params)
    
    def add_transp_conv_block(self, params):
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

    def add_transp_conv_blocks(self, transp_conv_params: list):
        """
        Adds multiple 1D transposed convolutional layers to the model.
        
        Args:
            transp_conv_params (list): A list of parameter lists for each transposed convolutional layer.
        """
        for params in transp_conv_params:
            self.add_transp_conv_block(params)
    
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
            size, scale_factor = params # One of size or scale_factor should be None, but not both.
            return nn.Upsample(size=size, scale_factor=scale_factor, mode="linear", align_corners=True)
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

class CategoricalEncoder(AutoencoderNetworkBuilder):
    """
    CategoricalEncoder is a custom neural network module designed to encode categorical variables
    into a dense representation. It includes a dense layer, an optional activation function,
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
        
        self.add_dense_block([output_length, activation, bias, 0.0, has_batch_norm])
    
    def forward(self, x):
        x = self.layers(x)
        if self.is_conv:
            return x.unsqueeze(1)
        return x


@dataclass
class FCLayerConfig:
    units: int
    activation: Optional[str] = None
    bias: bool = True
    dropout: float = 0.0
    batch_norm: bool = False


@dataclass
class Conv1DLayerConfig:
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    activation: Optional[str] = None
    pool: Optional[Tuple[str, ...]] = None
    bias: bool = True
    dropout: float = 0.0
    batch_norm: bool = False

@dataclass
class Upsample1DLayerConfig:
    scale_factor: int
    mode: str = "linear"
    align_corners: bool = True


@dataclass
class TransposedConv1DLayerConfig:
    out_channels: int
    kernel_size: int
    stride: int = 1  # always 1
    padding: int = 1
    output_padding: int = 0  # unused now
    activation: Optional[str] = None
    bias: bool = True
    dropout: float = 0.0
    batch_norm: bool = False


class LayerFactory:
    @staticmethod
    def activation(name: str, alpha: float = 0.25) -> nn.Module:
        match name:
            case 'relu': return nn.ReLU()
            case 'elu': return nn.ELU()
            case 'prelu': return nn.PReLU(init=alpha)
            case 'leaky_relu': return nn.LeakyReLU()
            case 'sigmoid': return nn.Sigmoid()
            case 'tanh': return nn.Tanh()
            case 'softmax': return nn.Softmax(dim=1)
            case _: raise ValueError(f"Unsupported activation: {name}")

    @staticmethod
    def dense(input_dim: int, cfg: FCLayerConfig) -> nn.Sequential:
        layers = [nn.Linear(input_dim, cfg.units, bias=cfg.bias)]
        if cfg.batch_norm:
            layers.append(nn.BatchNorm1d(cfg.units))
        if cfg.activation:
            layers.append(LayerFactory.activation(cfg.activation))
        if 0.0 < cfg.dropout < 1.0:
            layers.append(nn.Dropout(cfg.dropout))
        return nn.Sequential(*layers)

    @staticmethod
    def conv1d(in_channels: int, cfg: Conv1DLayerConfig) -> nn.Sequential:
        layers = [
            nn.Conv1d(in_channels, cfg.out_channels, cfg.kernel_size, cfg.stride, cfg.padding, bias=cfg.bias)
        ]
        if cfg.pool:
            name, *params = cfg.pool
            if name == 'max_pool':
                layers.append(nn.MaxPool1d(*params))
            elif name == 'avg_pool':
                layers.append(nn.AvgPool1d(*params))
            elif name == 'upsample':
                layers.append(nn.Upsample(*params, mode='linear', align_corners=True))
            else:
                raise ValueError(f"Unknown pooling type: {name}")
        if cfg.activation:
            layers.append(LayerFactory.activation(cfg.activation))
        if cfg.batch_norm:
            layers.append(nn.BatchNorm1d(cfg.out_channels))
        if 0.0 < cfg.dropout < 1.0:
            layers.append(nn.Dropout(cfg.dropout))
        return nn.Sequential(*layers)

    @staticmethod
    def transposed_conv1d(in_channels: int, cfg: TransposedConv1DLayerConfig) -> nn.Sequential:
        layers = [
            nn.ConvTranspose1d(
                in_channels, cfg.out_channels, cfg.kernel_size, cfg.stride,
                cfg.padding, output_padding=cfg.output_padding, bias=cfg.bias
            )
        ]
        if cfg.activation:
            layers.append(LayerFactory.activation(cfg.activation))
        if cfg.batch_norm:
            layers.append(nn.BatchNorm1d(cfg.out_channels))
        if 0.0 < cfg.dropout < 1.0:
            layers.append(nn.Dropout(cfg.dropout))
        return nn.Sequential(*layers)
    
    @staticmethod
    def upsample1d(cfg: Upsample1DLayerConfig) -> nn.Module:
        if cfg.scale_factor is None and cfg.size is None:
            raise ValueError("Upsample1DLayerConfig must specify either scale_factor or size.")
        return nn.Upsample(scale_factor=cfg.scale_factor, size=cfg.size, mode=cfg.mode, align_corners=cfg.align_corners)



class NetworkBuilder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers: List[nn.Module] = []
        self.last_input_length = 0
        self.last_input_channel = 0

    def add_dense_block(self, cfg: FCLayerConfig, flatten: bool = False, start_dim: int = 1):
        if flatten:
            self.layers.append(nn.Flatten(start_dim=start_dim))
        layer = LayerFactory.dense(self.last_input_length, cfg)
        self.last_input_length = cfg.units
        self.layers.append(layer)

    def add_dense_blocks(self, cfgs: List[FCLayerConfig], flatten_first: bool = False):
        for cfg in cfgs:
            self.add_dense_block(cfg, flatten=flatten_first)
            flatten_first = False

    def add_conv_block(self, cfg: Conv1DLayerConfig):
        layer = LayerFactory.conv1d(self.last_input_channel, cfg)
        self.last_input_channel = cfg.out_channels
        self.layers.append(layer)

    def add_conv_blocks(self, cfgs: List[Conv1DLayerConfig]):
        for cfg in cfgs:
            self.add_conv_block(cfg)

    def add_transposed_conv_layer(self, cfg: TransposedConv1DLayerConfig):
        layer = LayerFactory.transposed_conv1d(self.last_input_channel, cfg)
        self.last_input_channel = cfg.out_channels
        self.layers.append(layer)

    def add_transposed_conv_layers(self, cfgs: List[TransposedConv1DLayerConfig]):
        for cfg in cfgs:
            self.add_transposed_conv_layer(cfg)
    
    def add_upsample_layer(self, cfg: Upsample1DLayerConfig):
        self.layers.append(LayerFactory.upsample1d(cfg))

    def add_upsample_and_conv_layers(self, cfgs: List[Union[Upsample1DLayerConfig, TransposedConv1DLayerConfig]]):
        for cfg in cfgs:
            if isinstance(cfg, Upsample1DLayerConfig):
                self.add_upsample_layer(cfg)
            elif isinstance(cfg, TransposedConv1DLayerConfig):
                self.add_transposed_conv_layer(cfg)
            else:
                raise ValueError(f"Unsupported config type: {type(cfg)}")


    def build(self) -> nn.Sequential:
        return nn.Sequential(*self.layers)

    def forward(self, x):
        return self.build()(x)

    def summarize(self, input_shape):
        from torchinfo import summary
        model = self.build()
        print(summary(model, input_size=input_shape))

    def show_parameters(self):
        for i, p in enumerate(self.parameters()):
            print(f"Param {i}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}")


class DenseEncoder(AutoencoderNetworkBuilder):
    """
    DenseEncoder is a custom neural network module designed to encode input data
    through dense layers with customizable specifications.
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
        
        super().__init__()

        # Encoder attributes
        self.input_length = input_length
        self.last_input_length = input_length
        self.layer_specifications = layer_specifications
        self.layers = nn.Sequential()
        
        self.add_dense_blocks(layer_specifications)

    def forward(self, x):
        return self.layers(x)

class DenseDecoder(AutoencoderNetworkBuilder):
    """
    DenseDecoder is a custom neural network module designed to decode input data
    through dense layers with customizable specifications.
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
        
        super().__init__()
        
        self.input_length = input_length
        self.last_input_length = input_length
        self.layer_specifications = layer_specifications
        self.layers = nn.Sequential()
        
        # Hidden layer
        self.add_dense_blocks(layer_specifications)
        
    def forward(self, x):
        return self.layers(x)

class ConvEncoder(AutoencoderNetworkBuilder):
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
        self.add_conv_blocks(layer_specifications)

        if self.on_global_pool:
            self.layers.append(self.add_adaptive_pooling("max", 1))
    
    def forward(self, x):
        return self.layers(x)

class ConvDecoder(AutoencoderNetworkBuilder):
    """
    Convolutional Decoder module.

    Args:
        input_channel (int): Number of input channels.
        layer_specs (list): List of specifications for each layer.
    """
    def __init__(self, input_channel, layer_specs: list, on_transpose_conv=True):
        if not isinstance(layer_specs, list):
            raise ValueError("Input error. 'layer_specs' should be a list")
        
        super().__init__()
        
        # Decoder attributes
        self.layer_specs = layer_specs
        self.last_input_channel = input_channel
        self.on_transpose_conv = on_transpose_conv
        self.layers = nn.Sequential()
        
        # Hidden layers
        if on_transpose_conv:
            self.add_transp_conv_blocks(layer_specs)
        else:
            self.add_conv_blocks(layer_specs)
    
    def forward(self, x):
        return self.layers(x)

class ConvAutoencoderImplicit(AutoencoderNetworkBuilder):
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
        super().__init__()
        
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
    
    def forward(self, x, *x_categories):
        x = self.encoder(x)
        
        if self.n_categories > 0 and len(x_categories) > 0:
            x_categories_encoding = self.category_encoder(x_categories)#.unsqueeze(1)
            
            inputs_concat = [x, x_categories_encoding]
            x = torch.cat(inputs_concat, dim=1)
            
        return self.decoder(x)

    def get_encoder_output_length(self, input_shape):
        return get_decoder_target_lengths(self.encoder, input_shape)

class ConvAutoencoderLatentFC1(AutoencoderNetworkBuilder):
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
        super().__init__()
        
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
    
    def forward(self, x, *x_categories):
        x = self.encoder(x)
        x = self.latent(x)
        
        if self.n_categories > 0 and len(x_categories) > 0:
            x_categories_encoding = self.category_encoder(x_categories)#.unsqueeze(1)
            
            inputs_concat = [x, x_categories_encoding]
            x = torch.cat(inputs_concat, dim=1)
            
        return self.decoder(x)

    def get_encoder_output_length(self, input_shape):
        return get_decoder_target_lengths(self.encoder, input_shape)
class LatentFC1(AutoencoderNetworkBuilder):
    """
    Builds a a custom neural network with only 1 dense layer module representing 
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
        self.add_dense_block(latent_specifications, on_flatten=True)
    
    def forward(self, x):
        x = self.layers(x)
        if self.on_conv_ae:
            x = x.view(self.output_reshape)  # Shape (batch, channel, length)
        if self.pad > 0:
            x = self.set_pad_to_outputs(x)
        return x
    
    def set_pad_to_outputs(self, inputs):
        return torch.nn.functional.pad(inputs, pad=(self.pad, self.pad), mode=self.mode)

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, branch_channels=64):
        super(InceptionBlock1D, self).__init__()
        
        # Branch 1: 1x1 Convolution
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
        
        # Branch 2: 1x1 Convolution followed by 3x3 Convolution
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Branch 3: 1x1 Convolution followed by two 3x3 Convolutions (approximating 5x5)
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Branch 4: 1x1 Convolution followed by three 3x3 Convolutions (approximating 7x7)
        self.branch7x7 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Branch 5: 3x3 Max pooling followed by 1x1 Convolution
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        branch_pool = self.branch_pool(x)
        
        # Concatenate branches along the channel axis (dimension 1)
        output = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7, branch_pool], dim=1)
        return output

class InceptionBlock1DWithUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, branch_channels=64, upsample_scale=2, size=None):
        super(InceptionBlock1DWithUpsampling, self).__init__()
        
        self.upsample_scale = upsample_scale
        self.size = size
        
        if size is not None:
            on_scale_factor = False
        elif isinstance(self.upsample_scale, int):
            on_scale_factor = True
        else:
            raise "Inform either 'upsample_scale' or 'size'."

        # Branch 1: 1x1 Convolution
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
        
        # Branch 2: 1x1 Convolution followed by 3x3 Convolution
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Branch 3: 1x1 Convolution followed by two 3x3 Convolutions (approximating 5x5)
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Branch 4: 1x1 Convolution followed by three 3x3 Convolutions (approximating 7x7)
        self.branch7x7 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Branch 5: 3x3 Max pooling followed by 1x1 Convolution
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
        
        # Upsampling layer to act as transpose convolution
        if on_scale_factor:
            self.upsample = nn.Upsample(scale_factor=upsample_scale, mode='linear', align_corners=True)
        else:
            self.upsample = nn.Upsample(size=self.size, mode='linear', align_corners=True)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        branch_pool = self.branch_pool(x)
        
        # Concatenate branches along the channel axis (dimension 1)
        output = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7, branch_pool], dim=1)
        
        # Apply upsampling to the concatenated output
        output = self.upsample(output)
        return output

# Example Encoder-Decoder structure
class InceptionAutoencoder1D(nn.Module):
    def __init__(self, units, output_size):
        super(InceptionAutoencoder1D, self).__init__()
        self.units = units
        self.output_size = output_size
        
        self.encoder = nn.Sequential(
            InceptionBlock1D(1, self.units),
            nn.MaxPool1d(2),  # Downsampling
            InceptionBlock1D(5 * self.units, 2 * self.units),
            nn.MaxPool1d(2),  # Downsampling
            InceptionBlock1D(5 * 2 * self.units, 4 * self.units),
            nn.MaxPool1d(2),  # Downsampling
            InceptionBlock1D(5 * 4 * self.units, 6 * self.units),
            nn.MaxPool1d(2),  # Downsampling
        )
        self.decoder = nn.Sequential(
            InceptionBlock1DWithUpsampling(5 * 6 * self.units, 4 * self.units, upsample_scale=2),
            InceptionBlock1DWithUpsampling(5 * 4 * self.units, 2 * self.units, upsample_scale=2),
            InceptionBlock1DWithUpsampling(5 * 2 * self.units, self.units, upsample_scale=2),
            InceptionBlock1DWithUpsampling(5 * self.units, 1, size=self.output_size),
        )
        self.channel_adapter = nn.Conv1d(5, 1, kernel_size=1)

    def forward(self, x, *args):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self.channel_adapter(decoded)

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