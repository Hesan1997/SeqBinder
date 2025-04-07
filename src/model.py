import torch
import torch.nn as nn
from typing import List, Optional


class Trunk(nn.Module):
    """
    A fusion module that combines outputs from a protein branch and a compound
    branch, then processes the merged features through fully connected layers
    to produce a scalar prediction.

    Each branch must have a `return_shape` attribute (an integer specifying the
    number of output features). The outputs are concatenated and passed through
    fully connected layers with ReLU activation and dropout (0.3).
    """

    def __init__(self, p_branch: nn.Module, c_branch: nn.Module, 
                 neurons: List[int] = None):
        """
        Initialize the Trunk module with pre-instantiated branches and fully
        connected layers.

        :param p_branch: Protein branch module with a `return_shape` attribute (int).
        :param c_branch: Compound branch module with a `return_shape` attribute (int).
        :param neurons: List of integers specifying the neuron counts for each
                        fully connected layer. Defaults to [1024, 512] if not provided.
        """
        super(Trunk, self).__init__()
        self.p_branch = p_branch
        self.c_branch = c_branch

        if neurons is None:
            neurons = [1024, 512]

        # Compute total feature dimension from both branches.
        in_features = self.p_branch.return_shape + self.c_branch.return_shape

        # Build dense layers with ReLU and dropout.
        dense_layers = []
        for n in neurons:
            dense_layers.append(nn.Linear(in_features, n))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(0.3))
            in_features = n
        dense_layers.append(nn.Linear(in_features, 1))
        self.dense_layers = nn.Sequential(*dense_layers)

    def forward(self, x_p: torch.Tensor, x_c: torch.Tensor) -> torch.Tensor:
        """
        Process inputs from the protein and compound branches, fuse their outputs,
        and compute a scalar prediction.

        :param x_p: Input tensor for the protein branch (e.g., shape: [batch_size,
                    sequence_length]).
        :param x_c: Input tensor for the compound branch (e.g., shape: [batch_size,
                    sequence_length]).
        :return: Tensor of scalar predictions (shape: [batch_size,]).
        """
        # Get branch representations.
        p_repr = self.p_branch(x_p)
        c_repr = self.c_branch(x_c)

        # Concatenate features and compute prediction.
        x = torch.cat((p_repr, c_repr), dim=1)
        x = self.dense_layers(x)
        return x.squeeze(1)


class ConvBranch(nn.Module):
    """
    A convolutional branch for processing tokenized sequences.

    This module first embeds the input tokens, then applies a series of 1D
    convolutional layers with ReLU activations. Finally, global max pooling is
    used to produce a fixed-size output representation. The final number of
    channels is exposed via the `return_shape` attribute.
    """

    def __init__(self, vocabulary_size: int, base_channels: int, 
                 kernel_size: int, num_layers: int, embedding_dim: int = 64):
        """
        Initialize the ConvBranch.

        :param vocabulary_size: The size of the vocabulary (number of unique tokens).
        :param base_channels: The base number of channels for the convolutional layers.
        :param kernel_size: The kernel size used in all convolutional layers.
        :param num_layers: The number of convolutional layers to stack.
        :param embedding_dim: The dimension of the token embeddings. Defaults to 64.
        """
        super(ConvBranch, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)

        # Build convolutional layers dynamically based on the number of layers.
        conv_layers = []
        for i in range(num_layers):
            in_channels = (embedding_dim if i == 0 else 
                           base_channels * (2 ** (i - 1)))
            out_channels = base_channels * (2 ** i)
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                         stride=1, padding=0))
            conv_layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*conv_layers)

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # The final output dimension is determined by the last conv layer's channels.
        self.return_shape = base_channels * (2 ** (num_layers - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the convolutional branch.

        :param x: Input tensor with shape (batch_size, sequence_length), 
                  containing token indices.
        :return: Output tensor with shape (batch_size, return_shape) representing
                 the encoded features.
        """
        # Embed input tokens and rearrange dimensions for convolution.
        x = self.embedding(x)     # (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1)      # (batch_size, embedding_dim, sequence_length)
        
        # Apply convolutional layers and global max pooling.
        x = self.conv_layers(x)
        x = self.max_pool(x)      # (batch_size, final_channels, 1)
        return x.squeeze(-1)      # (batch_size, final_channels)


class LSTMBranch(nn.Module):
    """
    A bidirectional LSTM branch for processing tokenized sequences.

    This module applies an embedding layer followed by a bidirectional LSTM and uses
    global max pooling to produce a fixed-size output representation. The output
    dimensionality is exposed via the `return_shape` attribute.
    """

    def __init__(self, vocabulary_size: int, hidden_size: int, num_layers: int,
                 embedding_dim: int = 64):
        """
        Initialize the LSTMBranch.

        :param vocabulary_size: The size of the vocabulary (number of unique tokens).
        :param hidden_size: Hidden size for the LSTM.
        :param num_layers: Number of LSTM layers.
        :param embedding_dim: Dimension of the token embeddings. Defaults to 64.
        """
        super(LSTMBranch, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # The output dimension after global max pooling is 2 * hidden_size.
        self.return_shape = 2 * hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass for the LSTM branch.

        :param x: Input tensor of shape (batch_size, sequence_length).
        :return: Tensor of shape (batch_size, 2 * hidden_size) after global max pooling.
        """
        x = self.embedding(x)          # (batch_size, sequence_length, embedding_dim)
        x, _ = self.lstm(x)            # (batch_size, sequence_length, 2 * hidden_size)
        x = x.max(dim=1).values        # Global max pooling over sequence length
        return x


if __name__ == '__main__':
    pass
