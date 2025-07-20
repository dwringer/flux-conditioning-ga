# flux-conditioning-ga

**Repository Name:** FLUX Conditioning Genetic Algorithm

**Author:** dwringer

**License:** MIT

**Requirements:**
- invokeai>=4

## Introduction
This node takes a collection of prompt Conditioning objects for FLUX and applies a selected genetic algorithm operation to their embedding tensors, with an optional mutation that perturbs elements of the prompt embeddings at random.

Certain input fields are used only with certain selected crossover modes. Please see the tooltips/inputs documentation for more details.

### Installation:

To install these nodes, simply place the folder containing this
repository's code (or just clone the repository yourself) into your
`invokeai/nodes` folder.

## Overview
### Nodes
- [Flux Conditioning Genetic Algorithm](#flux-conditioning-genetic-algorithm) - Applies a genetic algorithm to a list of FLUX Conditionings,

<details>
<summary>

### Output Definitions

</summary>

- `SelectedFluxConditioningOutput` - Output definition with 1 fields
</details>

## Nodes
### Flux Conditioning Genetic Algorithm
**ID:** `flux_conditioning_genetic_algorithm`

**Category:** conditioning

**Tags:** conditioning, flux, genetic, algorithm, evolution

**Version:** 1.3.0

**Description:** Applies a genetic algorithm to a list of FLUX Conditionings,

generating a new population through crossover and mutation.
    Includes options for different crossover strategies and mutation noise types.
    Handles T5 embedding size mismatches during crossover.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `candidates` | `list[FluxConditioningField]` | List of FLUX Conditioning candidates for the genetic algorithm. | None |
| `population_size` | `int` | Desired size of the new conditioning population. | 10 |
| `crossover_method` | `Literal[(Differential Evolution, BLX-alpha, N-Point Splice)]` | Method to use for combining parent embeddings. | Differential Evolution |
| `de_base_candidate` | `Optional[FluxConditioningField]` | Optional FLUX Conditioning to use as the base (A) for Differential Evolution crossover. | None |
| `num_crossover_points` | `int` | Number of crossover points for N-Point Splice. (Only used with N-Point Splice method) | 3 |
| `differential_evolution_scale` | `float` | Scale factor for differential evolution crossover. (Only used with Differential Evolution method) | 0.7 |
| `crossover_rate` | `float` | Crossover rate; probability of change for each tensor element. (Only used with DE and BLX-alpha methods) | 1.0 |
| `blx_alpha` | `float` | Alpha parameter for BLX-alpha crossover. (Only used with BLX-alpha method) | 0.5 |
| `expand_t5_embeddings` | `bool` | If true, smaller T5 embeddings are expanded by concatenating clones if their size along dimension 1 is an exact multiple of the larger embedding's size. Otherwise, the larger embedding is cut down. This only applies during crossover when sizes are mismatched. | True |
| `mutation_rate` | `float` | Rate of mutation per tensor element (0.0 to 1.0). | 0.1 |
| `mutation_strength` | `float` | Strength of mutation (noise multiplier). | 0.05 |
| `use_gaussian_mutation` | `bool` | If true, uses Gaussian noise for mutation; otherwise, uses uniform noise. | True |
| `seed` | `int` | Random seed for deterministic behavior. | 0 |
| `selected_member_index` | `int` | Index of the new population member to output as a single conditioning. | 0 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `SelectedFluxConditioningOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `conditioning` | `FluxConditioningField` | The selected Flux Conditioning |


</details>

---

## Footnotes
For questions/comments/concerns/etc, use github or drop into the InvokeAI discord where you'll probably find someone who can help.
