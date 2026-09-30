# RNNs

This project covers recurrent neural networks and their use with sequential
and temporal data.

## Learning Objectives

By the end of this project, I should be able to explain:

- What a Recurrent Neural Network (RNN) is
- What a Long Short-Term Memory network (LSTM) is
- What a Gated Recurrent Unit (GRU) is
- What a Bidirectional Recurrent Neural Network (BRNN) is
- What the exploding gradient problem is and when it occurs
- What the vanishing gradient problem is and when it occurs
- How LSTMs and GRUs help reduce the vanishing gradient problem

## Task 0 - RNN Cell

File: `0-rnn_cell.py`

The `RNNCell` class represents one time step of a simple recurrent neural
network.

The constructor initializes:

- `Wh`: weights for the concatenated previous hidden state and current input
- `Wy`: weights for the output
- `bh`: hidden-state bias
- `by`: output bias

The `forward` method:

1. Concatenates the previous hidden state with the current input.
2. Computes the next hidden state using the `tanh` activation function.
3. Computes the output logits from the new hidden state.
4. Applies softmax to obtain output probabilities.

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/RNNs`
