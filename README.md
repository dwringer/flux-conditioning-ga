# flux-conditioning-ga

**Repository Name:** FLUX Conditioning Genetic Algorithm

**Author:** dwringer

**License:** MIT

**Requirements:**
- invokeai>=4

## Introduction
This node takes a collection of prompt Conditioning objects for FLUX and applies a selected genetic algorithm operation to their embedding tensors, with an optional mutation that perturbs elements of the prompt embeddings at random.

The Differential Evolution Scale and Num Crossover Points inputs are only used when applying the differential evolution or n-point splice crossover methods, respectively.

### Installation:

To install these nodes, simply place the folder containing this
repository's code (or just clone the repository yourself) into your
`invokeai/nodes` folder.
