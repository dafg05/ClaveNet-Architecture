# ClaveNet-Architecture

This a repo containing a module that houses ClaveNet's architecure and loss function, both of which are based on [Monotonic Groove Transformer](https://github.com/behzadhaki/MonotonicGrooveTransformer) (MGT). 

## Requirements

1. ` $ pip install -r requirements.txt`

Note that some packages are hosted on github repos.

## Groove Transformer

The MGT architecture. Largely based on a transformer encoder, but adapted to input and output torch.tensor versions of Hits-Velocities and Offsets (HVO) Sequences.

## HVO Loss

A function that calculates the loss between model-predicted and target HVO Sequences. A sum of the binary cross entropy for Hits and the mean squared error for both velocities and offsets.
