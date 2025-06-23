import torch
import random
import json
from typing import List, Optional, Union, Literal
from invokeai.invocation_api import (
    BaseInvocation,
    InputField,
    InvocationContext,
    invocation,
    OutputField,
    invocation_output,
    BaseInvocationOutput,
    StringOutput,
    StringCollectionOutput,
)
from invokeai.app.invocations.fields import(
    FluxConditioningField,
)
from invokeai.app.invocations.primitives import (
    FluxConditioningOutput,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    FLUXConditioningInfo,
    ConditioningFieldData,
)
from invokeai.backend.util.logging import info, warning, error

# Define a custom output class for a single selected Flux Conditioning
@invocation_output("selected_flux_conditioning_output")
class SelectedFluxConditioningOutput(BaseInvocationOutput):
    """Output for a single selected Flux Conditioning from the genetic algorithm."""
    conditioning: FluxConditioningField = OutputField(description="The selected Flux Conditioning")


@invocation(
    "flux_conditioning_genetic_algorithm",
    title="Flux Conditioning Genetic Algorithm",
    tags=["conditioning", "flux", "genetic", "algorithm", "evolution"],
    category="conditioning",
    version="1.0.7",  # Incrementing version for feature change
)
class FluxConditioningGeneticAlgorithmInvocation(BaseInvocation):
    """
    Applies a genetic algorithm to a list of FLUX Conditionings,
    generating a new population through crossover and mutation.
    Includes options for different crossover strategies.
    """

    candidates: list[FluxConditioningField] = InputField(
        description="List of FLUX Conditioning candidates for the genetic algorithm.",
        ui_order=0,
    )
    population_size: int = InputField(
        default=10,
        gt=0,
        description="Desired size of the new conditioning population.",
        ui_order=1, # Re-ordered due to removal of mask_vector_pairs_json and initialize_random_masks_if_none
    )
    crossover_method: Literal["Splice", "N-Point Splice", "Differential Evolution"] = InputField(
        default="Splice",
        description="Method to use for combining parent embeddings.",
        ui_order=2, # Re-ordered
        ui_choice_labels={
            "Splice": "Splice (Single-point crossover)",
            "N-Point Splice": "N-Point Splice (Multiple random points)",
            "Differential Evolution": "Differential Evolution (3-parent crossover)"
        }
    )
    num_crossover_points: int = InputField(
        default=3,
        ge=2,
        description="Number of crossover points for N-Point Splice. (Only used with N-Point Splice method)",
        ui_order=3, # Re-ordered
    )
    differential_evolution_scale: float = InputField(
        default=0.7,
        ge=0.0,
        description="Scale factor for differential evolution crossover. (Only used with Differential Evolution method)",
        ui_order=4, # Re-ordered
    )
    mutation_rate: float = InputField(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Rate of mutation per tensor element (0.0 to 1.0).",
        ui_order=5, # Re-ordered
    )
    mutation_strength: float = InputField(
        default=0.05,
        ge=0.0,
        description="Strength of mutation (Gaussian noise multiplier).",
        ui_order=6, # Re-ordered
    )
    seed: int = InputField(
        default=0,
        description="Random seed for deterministic behavior.",
        ui_order=7, # Re-ordered
    )
    selected_member_index: int = InputField(
        default=0,
        ge=0,
        description="Index of the new population member to output as a single conditioning.",
        ui_order=8, # Re-ordered
    )

    def _load_conditioning(
        self, context: InvocationContext, field: FluxConditioningField
    ) -> Union[
        FLUXConditioningInfo,
        None,
    ]:
        """Loads a conditioning object from the context."""
        if field is None:
            return None
        else:
            try:
                loaded_cond = context.conditioning.load(field.conditioning_name).conditionings[0]
                return loaded_cond.to("cpu")
            except Exception as e:
                error(f"Failed to load conditioning {field.conditioning_name}: {e}")
                return None

    def _clip_embeds(
        self,
        context: InvocationContext,
        field: FluxConditioningField | None,
        like: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Accesses CLIP embeddings from a FluxConditioningField."""
        cond = self._load_conditioning(context, field)
        if cond and isinstance(cond, FLUXConditioningInfo):
            return cond.clip_embeds
        if like is not None:
            warning(f"Could not load CLIP embeddings for {field.conditioning_name if field else 'None'}. Returning zeros_like.")
            return torch.zeros_like(like)
        warning(f"Could not load CLIP embeddings for {field.conditioning_name if field else 'None'}.")
        return None

    def _t5_embeds(
        self,
        context: InvocationContext,
        field: FluxConditioningField | None,
        like: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Accesses T5 embeddings from a FluxConditioningField."""
        cond = self._load_conditioning(context, field)
        if cond and isinstance(cond, FLUXConditioningInfo):
            return cond.t5_embeds
        if like is not None:
            warning(f"Could not load T5 embeddings for {field.conditioning_name if field else 'None'}. Returning zeros_like.")
            return torch.zeros_like(like)
        warning(f"Could not load T5 embeddings for {field.conditioning_name if field else 'None'}.")
        return None

    def _n_point_splice_crossover(self, t1: torch.Tensor, t2: torch.Tensor, num_points: int, py_rng: random.Random) -> torch.Tensor:
        """
        Performs N-point splice crossover between two tensors.
        """
        if t1.shape != t2.shape:
            error("Tensors must have the same shape for N-point splice crossover. Returning clone of first tensor.")
            return t1.clone()

        flat1 = t1.flatten()
        flat2 = t2.flatten()
        child_flat = torch.empty_like(flat1)

        if flat1.numel() < 2:
            return flat1.clone().reshape(t1.shape)

        # Standard N-point splice crossover
        if num_points >= flat1.numel():
            warning(f"Number of crossover points ({num_points}) is greater than or equal to tensor size ({flat1.numel()}). Performing direct copy from t2.")
            return t2.clone().reshape(t1.shape)
        
        crossover_points = sorted(py_rng.sample(range(1, flat1.numel()), num_points))
        
        current_segment_start = 0
        use_t1 = True # Start with t1
        for point in crossover_points:
            if use_t1:
                child_flat[current_segment_start:point] = flat1[current_segment_start:point]
            else:
                child_flat[current_segment_start:point] = flat2[current_segment_start:point]
            use_t1 = not use_t1
            current_segment_start = point
        
        # Add the last segment
        if use_t1:
            child_flat[current_segment_start:] = flat1[current_segment_start:]
        else:
            child_flat[current_segment_start:] = flat2[current_segment_start:]
                
        return child_flat.reshape(t1.shape)


    def invoke(
        self, context: InvocationContext
    ) -> SelectedFluxConditioningOutput:
        """
        Main invocation method for the genetic algorithm.
        Generates a new population of FLUX Conditionings and outputs them.
        Optionally outputs a single selected member.
        """
        if not self.candidates:
            warning("No conditioning candidates provided. Returning empty list.")
            return SelectedFluxConditioningOutput(
                conditioning=FluxConditioningField(conditioning_name=""),
            )

        # Initialize random number generators
        py_rng = random.Random(self.seed)
        torch_rng = (
            torch.Generator().manual_seed(self.seed)
        )

        # Load CLIP and T5 embeddings from candidates
        clip_candidates = [self._clip_embeds(context, c) for c in self.candidates]
        t5_candidates = [self._t5_embeds(context, c) for c in self.candidates]

        # Filter out any None values if loading failed
        clip_candidates = [c for c in clip_candidates if c is not None]
        t5_candidates = [t for t in t5_candidates if t is not None]

        if not clip_candidates or not t5_candidates:
            error("Failed to load sufficient CLIP or T5 embeddings from candidates. Check input conditionings.")
            return SelectedFluxConditioningOutput(
                conditioning=FluxConditioningField(conditioning_name=""),
            )

        # Ensure all tensors in the filtered lists have the same shape.
        first_clip_shape = clip_candidates[0].shape
        first_t5_shape = t5_candidates[0].shape

        if not all(c.shape == first_clip_shape for c in clip_candidates):
            warning("Inconsistent CLIP embedding shapes among candidates. Operations might fail.")
        if not all(t.shape == first_t5_shape for t in t5_candidates):
            warning("Inconsistent T5 embedding shapes among candidates. Operations might fail.")


        def splice_crossover(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
            """
            Performs single-point crossover between two tensors.
            """
            if t1.shape != t2.shape:
                error("Tensors must have the same shape for splice crossover. Returning clone of first tensor.")
                return t1.clone()

            flat1 = t1.flatten()
            flat2 = t2.flatten()
            
            if flat1.numel() < 2:
                return flat1.clone().reshape(t1.shape)

            # Standard splice crossover
            split = py_rng.randint(1, flat1.numel() - 1)
            child_flat = torch.cat((flat1[:split], flat2[split:]), dim=0)
                
            return child_flat.reshape(t1.shape)

        def differential_evolution_crossover(
            t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, scale_factor: float
        ) -> torch.Tensor:
            """
            Performs differential evolution crossover: child = t1 + scale_factor * (t2 - t3).
            Requires t1, t2, t3 to have the same shape.
            """
            if not (t1.shape == t2.shape == t3.shape):
                error("All three tensors must have the same shape for differential evolution. Returning clone of t1.")
                return t1.clone()

            t1_float = t1.to(torch.float32)
            t2_float = t2.to(torch.float32)
            t3_float = t3.to(torch.float32)

            # Standard differential evolution
            difference = t2_float - t3_float
            child_float = t1_float + scale_factor * difference
            
            return child_float.to(t1.dtype)


        def mutate_tensor(
            tensor: torch.Tensor, rate: float, strength: float
        ) -> torch.Tensor:
            """Applies Gaussian noise mutation to a tensor."""
            if rate == 0.0:
                return tensor.clone()

            tensor = tensor.clone()
            mask = torch.rand(tensor.size(), generator=torch_rng) < rate
            noise = torch.randn(tensor.size(), generator=torch_rng) * strength
            tensor[mask] += noise[mask]
            return tensor

        new_population: List[FluxConditioningOutput] = []

        # Generate the new population
        for i in range(self.population_size):
            clip_child = None
            t5_child = None

            # Select parents
            num_parents = 3 if self.crossover_method == "Differential Evolution" else 2
            p_indices = py_rng.sample(range(len(self.candidates)), num_parents)
            p1_idx = p_indices[0]
            p2_idx = p_indices[1]
            p3_idx = p_indices[2] if num_parents == 3 else None

            if self.crossover_method == "Splice":
                clip_child = mutate_tensor(
                    splice_crossover(clip_candidates[p1_idx], clip_candidates[p2_idx]),
                    self.mutation_rate,
                    self.mutation_strength,
                )
                t5_child = mutate_tensor(
                    splice_crossover(t5_candidates[p1_idx], t5_candidates[p2_idx]),
                    self.mutation_rate,
                    self.mutation_strength,
                )
            elif self.crossover_method == "N-Point Splice":
                clip_child = mutate_tensor(
                    self._n_point_splice_crossover(clip_candidates[p1_idx], clip_candidates[p2_idx], self.num_crossover_points, py_rng),
                    self.mutation_rate,
                    self.mutation_strength,
                )
                t5_child = mutate_tensor(
                    self._n_point_splice_crossover(t5_candidates[p1_idx], t5_candidates[p2_idx], self.num_crossover_points, py_rng),
                    self.mutation_rate,
                    self.mutation_strength,
                )
            elif self.crossover_method == "Differential Evolution":
                if len(clip_candidates) < 3:
                    error("Not enough candidates for Differential Evolution crossover (need at least 3). Falling back to cloning parent.")
                    clip_child = mutate_tensor(clip_candidates[0].clone(), self.mutation_rate, self.mutation_strength)
                    t5_child = mutate_tensor(t5_candidates[0].clone(), self.mutation_rate, self.mutation_strength)
                else:
                    clip_child = mutate_tensor(
                        differential_evolution_crossover(
                            clip_candidates[p1_idx], clip_candidates[p2_idx], clip_candidates[p3_idx], 
                            self.differential_evolution_scale
                        ),
                        self.mutation_rate,
                        self.mutation_strength,
                    )
                    t5_child = mutate_tensor(
                        differential_evolution_crossover(
                            t5_candidates[p1_idx], t5_candidates[p2_idx], t5_candidates[p3_idx], 
                            self.differential_evolution_scale
                        ),
                        self.mutation_rate,
                        self.mutation_strength,
                    )

            if clip_child is None or t5_child is None:
                error("Crossover failed to produce child embeddings. Skipping this member.")
                continue

            conditioning_info = FLUXConditioningInfo(clip_embeds=clip_child, t5_embeds=t5_child)
            conditioning_data = ConditioningFieldData(conditionings=[conditioning_info])
            conditioning_name = context.conditioning.save(conditioning_data)
            
            new_population.append(
                FluxConditioningOutput(
                    conditioning=FluxConditioningField(conditioning_name=conditioning_name)
                )
            )
            info(f"Generated new conditioning member {i+1}/{self.population_size}")

        if not new_population:
            warning("No new population members were generated. Returning an empty conditioning.")
            return SelectedFluxConditioningOutput(
                conditioning=FluxConditioningField(conditioning_name=""),
            )

        selected_idx = self.selected_member_index
        if not (0 <= selected_idx < len(new_population)):
            warning(
                f"Selected member index {self.selected_member_index} is out of bounds for "
                f"population size {len(new_population)}. Returning a random member."
            )
            selected_idx = py_rng.randint(0, len(new_population) - 1)
            
        selected_conditioning_output = new_population[selected_idx]

        return SelectedFluxConditioningOutput(
            conditioning=selected_conditioning_output.conditioning,
        )
