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
    version="1.1.0",  # Increment version for every feature change
)
class FluxConditioningGeneticAlgorithmInvocation(BaseInvocation):
    """
    Applies a genetic algorithm to a list of FLUX Conditionings,
    generating a new population through crossover and mutation.
    Includes options for different crossover strategies and mutation noise types.
    """

    candidates: list[FluxConditioningField] = InputField(
        description="List of FLUX Conditioning candidates for the genetic algorithm.",
        ui_order=0,
    )
    population_size: int = InputField(
        default=10,
        gt=0,
        description="Desired size of the new conditioning population.",
        ui_order=1,
    )
    crossover_method: Literal["Differential Evolution", "BLX-alpha", "N-Point Splice"] = InputField(
        default="Differential Evolution",
        description="Method to use for combining parent embeddings.",
        ui_order=2,
        ui_choice_labels={
            "Differential Evolution": "Differential Evolution (3-parent crossover)",
            "BLX-alpha": "BLX-alpha (Blend Crossover Alpha)", # Added new crossover method
            "N-Point Splice": "N-Point Splice (Multiple random points)"
        }
    )
    num_crossover_points: int = InputField(
        default=3,
        ge=1,
        description="Number of crossover points for N-Point Splice. (Only used with N-Point Splice method)",
        ui_order=3,
    )
    differential_evolution_scale: float = InputField(
        default=0.7,
        ge=0.0,
        description="Scale factor for differential evolution crossover. (Only used with Differential Evolution method)",
        ui_order=4,
    )
    crossover_rate: float = InputField(
        default=1.0,
        ge=0.0,
        le=1.0, # Rate should be between 0 and 1 for probability
        description="Crossover rate; probability of change for each tensor element. (Only used with DE and BLX-alpha methods)",
        ui_order=5,
    )
    blx_alpha: float = InputField( # New input field for BLX-alpha
        default=0.5,
        ge=0.0,
        description="Alpha parameter for BLX-alpha crossover. (Only used with BLX-alpha method)",
        ui_order=6, # Placed after DE rate but before mutation params
    )
    mutation_rate: float = InputField(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Rate of mutation per tensor element (0.0 to 1.0).",
        ui_order=7,
    )
    mutation_strength: float = InputField(
        default=0.05,
        ge=0.0,
        description="Strength of mutation (noise multiplier).",
        ui_order=8,
    )
    use_gaussian_mutation: bool = InputField( # New input field for mutation noise type
        default=True,
        description="If true, uses Gaussian noise for mutation; otherwise, uses uniform noise.",
        ui_order=9,
    )
    seed: int = InputField(
        default=0,
        description="Random seed for deterministic behavior.",
        ui_order=10,
    )
    selected_member_index: int = InputField(
        default=0,
        ge=0,
        description="Index of the new population member to output as a single conditioning.",
        ui_order=11,
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

    def differential_evolution_crossover(
        self, t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, scale_factor: float, rate: float, torch_rng: torch.Generator
    ) -> torch.Tensor:
        """
        Performs differential evolution crossover: child = t1 + scale_factor * (t2 - t3).
        Applies a binary mask based on 'rate' for element-wise application.
        Requires t1, t2, t3 to have the same shape.
        """
        if not (t1.shape == t2.shape == t3.shape):
            error("All three tensors must have the same shape for differential evolution. Returning clone of t1.")
            return t1.clone()

        t1_float = t1.to(torch.float32)
        t2_float = t2.to(torch.float32)
        t3_float = t3.to(torch.float32)

        # Standard differential evolution calculation
        difference = t2_float - t3_float
        mutant = t1_float + scale_factor * difference

        if rate < 1.0:
            # Apply crossover rate: binary mask over the tensor
            crossover_mask = torch.rand(t1.size(), generator=torch_rng) < rate

            # Child is mutant where mask is True, otherwise it's original t1
            child_float = torch.where(crossover_mask, mutant, t1_float)
        else:
            child_float = mutant

        return child_float.to(t1.dtype)


    def blx_alpha_crossover(self, t1: torch.Tensor, t2: torch.Tensor, alpha: float, rate: float, torch_rng: torch.Generator) -> torch.Tensor:
        """
        Performs BLX-alpha crossover between two tensors.
        For each element, child is chosen uniformly from [min - alpha * I, max + alpha * I].
        """
        if t1.shape != t2.shape:
            error("Tensors must have the same shape for BLX-alpha crossover. Returning clone of first tensor.")
            return t1.clone()

        t1_float = t1.to(torch.float32)
        t2_float = t2.to(torch.float32)

        min_val = torch.min(t1_float, t2_float)
        max_val = torch.max(t1_float, t2_float)
        
        I = max_val - min_val # Interval length
        
        lower_bound = min_val - alpha * I
        upper_bound = max_val + alpha * I
        
        # Generate random numbers uniformly between lower_bound and upper_bound
        # This requires scaling a [0,1) uniform random tensor
        uniform_random = torch.rand(t1.size(), generator=torch_rng)
        mutant = lower_bound + uniform_random * (upper_bound - lower_bound)

        if rate < 1.0:
            # Apply crossover rate: binary mask over the tensor
            crossover_mask = torch.rand(t1.size(), generator=torch_rng) < rate
        
            # Child is mutant where mask is True, otherwise it's original t1
            child_float = torch.where(crossover_mask, mutant, t1_float)
        else:
            child_float = mutant

        return child_float.to(t1.dtype)


    def n_point_splice_crossover(self, t1: torch.Tensor, t2: torch.Tensor, num_points: int, py_rng: random.Random) -> torch.Tensor:
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


    def mutate_tensor(
        self, tensor: torch.Tensor, rate: float, strength: float, use_gaussian: bool, torch_rng: torch.Generator
    ) -> torch.Tensor:
        """
        Applies mutation (Gaussian or uniform noise) to a tensor.
        """
        if rate == 0.0:
            return tensor.clone()

        tensor = tensor.clone()
        mask = torch.rand(tensor.size(), generator=torch_rng) < rate
        
        if use_gaussian:
            noise = torch.randn(tensor.size(), generator=torch_rng) * strength
        else:
            # Generate uniform noise in [-strength, strength]
            noise = (torch.rand(tensor.size(), generator=torch_rng) * 2 - 1) * strength
            
        tensor[mask] += noise[mask]
        return tensor


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
        # Pass a 'like' tensor to ensure zeros_like returns a tensor with the correct shape/dtype
        # if a conditioning fails to load. This assumes all candidates *should* have the same shape.
        initial_clip_like = self._clip_embeds(context, self.candidates[0])
        initial_t5_like = self._t5_embeds(context, self.candidates[0])

        if initial_clip_like is None or initial_t5_like is None:
            error("Could not load initial conditioning to determine embedding shapes. Exiting.")
            return SelectedFluxConditioningOutput(
                conditioning=FluxConditioningField(conditioning_name=""),
            )

        clip_candidates = [self._clip_embeds(context, c, like=initial_clip_like) for c in self.candidates]
        t5_candidates = [self._t5_embeds(context, c, like=initial_t5_like) for c in self.candidates]

        # Filter out any None values if loading failed (though _clip_embeds/_t5_embeds should return zeros_like)
        clip_candidates = [c for c in clip_candidates if c is not None]
        t5_candidates = [t for t in t5_candidates if t is not None]

        if not clip_candidates or not t5_candidates:
            error("Failed to load sufficient CLIP or T5 embeddings from candidates. Check input conditionings.")
            return SelectedFluxConditioningOutput(
                conditioning=FluxConditioningField(conditioning_name=""),
            )

        # Ensure all tensors in the filtered lists have the same shape.
        # This check is still good practice, even with zeros_like fallback
        first_clip_shape = clip_candidates[0].shape
        first_t5_shape = t5_candidates[0].shape

        if not all(c.shape == first_clip_shape for c in clip_candidates):
            warning("Inconsistent CLIP embedding shapes among candidates. Operations might fail.")
        if not all(t.shape == first_t5_shape for t in t5_candidates):
            warning("Inconsistent T5 embedding shapes among candidates. Operations might fail.")

        new_population: List[FluxConditioningInfo] = []

        # Generate the new population
        for i in range(self.population_size):
            clip_child = None
            t5_child = None

            num_available_embeds = len(clip_candidates) # Assuming clip_candidates and t5_candidates have same length

            # Handle insufficient candidates for crossover
            if num_available_embeds == 1:
                # Case 1: Only one candidate, apply mutation directly
                info("Only one candidate available. Applying mutation directly to generate new member.")
                clip_child = self.mutate_tensor(
                    clip_candidates[0].clone(), # Clone to ensure the original candidate's tensor is not modified
                    self.mutation_rate,
                    self.mutation_strength,
                    self.use_gaussian_mutation,
                    torch_rng,
                )
                t5_child = self.mutate_tensor(
                    t5_candidates[0].clone(), # Clone to ensure the original candidate's tensor is not modified
                    self.mutation_rate,
                    self.mutation_strength,
                    self.use_gaussian_mutation,
                    torch_rng,
                )
            else:
                # Prepare candidate lists for parent selection and crossover
                # Use these `current_*_candidates` for sampling parents, as they might be augmented.
                current_clip_candidates = list(clip_candidates) # Create a mutable copy
                current_t5_candidates = list(t5_candidates) # Create a mutable copy

                # Determine required number of parents for the chosen crossover method
                required_parents = 3 if self.crossover_method == "Differential Evolution" else 2

                # If there are insufficient candidates for the chosen crossover method, handle specific cases
                if num_available_embeds < required_parents:
                    if self.crossover_method == "Differential Evolution" and num_available_embeds == 2:
                        # Case 2: Two candidates for Differential Evolution (which needs 3 parents)
                        # Deterministically clone the first candidate to fulfill the requirement
                        info("Two candidates available for Differential Evolution. Cloning the first candidate to meet the 3-parent requirement.")
                        current_clip_candidates.append(clip_candidates[0].clone())
                        current_t5_candidates.append(t5_candidates[0].clone())
                    else:
                        # For other cases where candidates are still insufficient (e.g., DE with only 1 candidate,
                        # or 2-parent methods with less than 2 candidates, which are now covered by num_available_embeds == 1)
                        error(f"Insufficient candidates ({num_available_embeds}) for {self.crossover_method} (requires {required_parents}). Skipping this member generation.")
                        continue # Skip this population member if parents cannot be selected

                try:
                    # Select parents from the (potentially augmented) current_clip_candidates/current_t5_candidates
                    if self.crossover_method == "Differential Evolution":
                        # Differential Evolution needs 3 distinct parents for t1, t2, t3
                        p_indices = py_rng.sample(range(len(current_clip_candidates)), 3)
                        p1_idx, p2_idx, p3_idx = p_indices
                    else: # For N-Point Splice, BLX-alpha, we need 2 parents
                        p_indices = py_rng.sample(range(len(current_clip_candidates)), 2)
                        p1_idx, p2_idx = p_indices
                        p3_idx = None # Not used for these methods

                    if self.crossover_method == "N-Point Splice":
                        clip_child = self.mutate_tensor(
                            self.n_point_splice_crossover(current_clip_candidates[p1_idx], current_clip_candidates[p2_idx], self.num_crossover_points, py_rng),
                            self.mutation_rate,
                            self.mutation_strength,
                            self.use_gaussian_mutation,
                            torch_rng,
                        )
                        t5_child = self.mutate_tensor(
                            self.n_point_splice_crossover(current_t5_candidates[p1_idx], current_t5_candidates[p2_idx], self.num_crossover_points, py_rng),
                            self.mutation_rate,
                            self.mutation_strength,
                            self.use_gaussian_mutation,
                            torch_rng,
                        )
                    elif self.crossover_method == "BLX-alpha": # New BLX-alpha crossover
                        clip_child = self.mutate_tensor(
                            self.blx_alpha_crossover(current_clip_candidates[p1_idx], current_clip_candidates[p2_idx], self.blx_alpha, self.crossover_rate, torch_rng),
                            self.mutation_rate,
                            self.mutation_strength,
                            self.use_gaussian_mutation,
                            torch_rng,
                        )
                        t5_child = self.mutate_tensor(
                            self.blx_alpha_crossover(current_t5_candidates[p1_idx], current_t5_candidates[p2_idx], self.blx_alpha, self.crossover_rate, torch_rng),
                            self.mutation_rate,
                            self.mutation_strength,
                            self.use_gaussian_mutation,
                            torch_rng,
                        )
                    elif self.crossover_method == "Differential Evolution":
                        # p1_idx, p2_idx, p3_idx are already set for DE
                        clip_child = self.mutate_tensor(
                            self.differential_evolution_crossover(
                                current_clip_candidates[p1_idx], current_clip_candidates[p2_idx], current_clip_candidates[p3_idx],
                                self.differential_evolution_scale, self.crossover_rate, torch_rng # Pass rate and torch_rng
                            ),
                            self.mutation_rate,
                            self.mutation_strength,
                            self.use_gaussian_mutation,
                            torch_rng,
                        )
                        t5_child = self.mutate_tensor(
                            self.differential_evolution_crossover(
                                current_t5_candidates[p1_idx], current_t5_candidates[p2_idx], current_t5_candidates[p3_idx],
                                self.differential_evolution_scale, self.crossover_rate, torch_rng # Pass rate and torch_rng
                            ),
                            self.mutation_rate,
                            self.mutation_strength,
                            self.use_gaussian_mutation,
                            torch_rng,
                        )
                    else: # Fallback in case of unexpected crossover method
                        warning(f"Unknown crossover method: {self.crossover_method}. Cloning parent and mutating.")
                        clip_child = self.mutate_tensor(current_clip_candidates[p1_idx].clone(), self.mutation_rate, self.mutation_strength, self.use_gaussian_mutation, torch_rng)
                        t5_child = self.mutate_tensor(current_t5_candidates[p1_idx].clone(), self.mutation_rate, self.mutation_strength, self.use_gaussian_mutation, torch_rng)

                except Exception as e:
                    error(f"Error during crossover or mutation for member {i+1}: {e}. Skipping this member.")
                    continue


            if clip_child is None or t5_child is None:
                error("Child embeddings were not produced. Skipping this member.")
                continue

            conditioning_info = FLUXConditioningInfo(clip_embeds=clip_child, t5_embeds=t5_child)            
            new_population.append(conditioning_info)
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
            
        selected_conditioning_info = new_population[selected_idx]
        selected_conditioning_data = ConditioningFieldData(conditionings=[selected_conditioning_info])
        selected_conditioning_name = context.conditioning.save(selected_conditioning_data)
        selected_conditioning_output = FluxConditioningOutput(
            conditioning=FluxConditioningField(conditioning_name=selected_conditioning_name)
        )

        return SelectedFluxConditioningOutput(
            conditioning=selected_conditioning_output.conditioning,
        )
