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
    version="1.2.1",  # Increment version for T5 embedding size handling optimization
)
class FluxConditioningGeneticAlgorithmInvocation(BaseInvocation):
    """
    Applies a genetic algorithm to a list of FLUX Conditionings,
    generating a new population through crossover and mutation.
    Includes options for different crossover strategies and mutation noise types.
    Handles T5 embedding size mismatches during crossover.
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
            "BLX-alpha": "BLX-alpha (Blend Crossover Alpha)",
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
    blx_alpha: float = InputField(
        default=0.5,
        ge=0.0,
        description="Alpha parameter for BLX-alpha crossover. (Only used with BLX-alpha method)",
        ui_order=6,
    )
    expand_t5_embeddings: bool = InputField( # New input field for T5 embedding size handling
        default=True,
        description=(
            "If true, smaller T5 embeddings are expanded by concatenating clones if their size along dimension 1 "
            "is an exact multiple of the larger embedding's size. Otherwise, the larger embedding is cut down. "
            "This only applies during crossover when sizes are mismatched."
        ),
        ui_order=7,
    )
    mutation_rate: float = InputField(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Rate of mutation per tensor element (0.0 to 1.0).",
        ui_order=8,
    )
    mutation_strength: float = InputField(
        default=0.05,
        ge=0.0,
        description="Strength of mutation (noise multiplier).",
        ui_order=9,
    )
    use_gaussian_mutation: bool = InputField(
        default=True,
        description="If true, uses Gaussian noise for mutation; otherwise, uses uniform noise.",
        ui_order=10,
    )
    seed: int = InputField(
        default=0,
        description="Random seed for deterministic behavior.",
        ui_order=11,
    )
    selected_member_index: int = InputField(
        default=0,
        ge=0,
        description="Index of the new population member to output as a single conditioning.",
        ui_order=12,
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

    def _handle_t5_mismatch(self, t_a: torch.Tensor, t_b: torch.Tensor, expand: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Handles size mismatch for T5 embedding tensors along dimension 1.
        If 'expand' is True, attempts to expand the smaller tensor by concatenating clones
        if its size is an exact multiple of the larger embedding's size along dimension 1.
        Otherwise, it cuts down the larger tensor to match the smaller one.
        Returns the two (potentially modified) tensors with matching dimensions.
        """
        # This function is now only called if a mismatch is confirmed, so no initial check needed here.

        len_a = t_a.shape[1]
        len_b = t_b.shape[1]

        # Ensure tensors are on CPU for manipulation, if needed, before cloning or slicing
        t_a_proc = t_a.to("cpu")
        t_b_proc = t_b.to("cpu")

        if expand:
            # Try to expand the smaller tensor
            if len_a < len_b and len_b % len_a == 0:
                reps = len_b // len_a
                info(f"Expanding T5 tensor A (size {len_a}) to match B (size {len_b}) by concatenating {reps} clones.")
                t_a_expanded = torch.cat([t_a_proc.clone() for _ in range(reps)], dim=1)
                return t_a_expanded, t_b_proc
            elif len_b < len_a and len_a % len_b == 0:
                reps = len_a // len_b
                info(f"Expanding T5 tensor B (size {len_b}) to match A (size {len_a}) by concatenating {reps} clones.")
                t_b_expanded = torch.cat([t_b_proc.clone() for _ in range(reps)], dim=1)
                return t_a_proc, t_b_expanded
            else:
                warning(
                    f"T5 embedding sizes ({len_a} and {len_b}) are mismatched and not exact multiples. "
                    "Falling back to cutting down the larger embedding as expansion is not possible."
                )
                # Fallback to cutting down if expansion is not possible or not an exact multiple
                min_len = min(len_a, len_b)
                return t_a_proc[:, :min_len], t_b_proc[:, :min_len]
        else:
            # Cut down the larger tensor
            min_len = min(len_a, len_b)
            info(f"Cutting down larger T5 embedding to size {min_len} due to `expand_t5_embeddings` being False.")
            return t_a_proc[:, :min_len], t_b_proc[:, :min_len]


    def differential_evolution_crossover(
        self, t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, scale_factor: float, rate: float, torch_rng: torch.Generator
    ) -> torch.Tensor:
        """
        Performs differential evolution crossover: child = t1 + scale_factor * (t2 - t3).
        Applies a binary mask based on 'rate' for element-wise application.
        Assumes t1, t2, t3 have the same shape, which should be ensured by the caller for T5 embeddings.
        """
        # All three tensors are expected to have the same shape at this point for T5,
        # handled by the pairwise alignment in the invoke method.
        if not (t1.shape == t2.shape == t3.shape):
            error("Tensors passed to differential_evolution_crossover do not have consistent shapes. This should be handled before calling.")
            # Attempt to proceed by using the shape of t1 for error handling if shapes are truly inconsistent.
            # This is a fallback and indicates a logic error in the caller if it happens.
            min_dim1 = min(t1.shape[1], t2.shape[1], t3.shape[1])
            t1 = t1[:, :min_dim1]
            t2 = t2[:, :min_dim1]
            t3 = t3[:, :min_dim1]


        t1_float = t1.to(torch.float32)
        t2_float = t2.to(torch.float32)
        t3_float = t3.to(torch.float32)

        # Standard differential evolution calculation
        difference = t2_float - t3_float
        mutant = t1_float + scale_factor * difference

        if rate < 1.0:
            # Apply crossover rate: binary mask over the tensor
            crossover_mask = torch.rand(t1.size(), generator=torch_rng, device=t1.device) < rate

            # Child is mutant where mask is True, otherwise it's original t1
            child_float = torch.where(crossover_mask, mutant, t1_float)
        else:
            child_float = mutant

        return child_float.to(t1.dtype)


    def blx_alpha_crossover(self, t1: torch.Tensor, t2: torch.Tensor, alpha: float, rate: float, torch_rng: torch.Generator) -> torch.Tensor:
        """
        Performs BLX-alpha crossover between two tensors.
        For each element, child is chosen uniformly from [min - alpha * I, max + alpha * I].
        Assumes t1 and t2 have the same shape, handled by the caller for T5 embeddings.
        """
        if t1.shape != t2.shape:
            error("Tensors passed to blx_alpha_crossover do not have consistent shapes. This should be handled before calling.")
            # Fallback similar to DE, if this error occurs
            min_dim1 = min(t1.shape[1], t2.shape[1])
            t1 = t1[:, :min_dim1]
            t2 = t2[:, :min_dim1]


        t1_float = t1.to(torch.float32)
        t2_float = t2.to(torch.float32)

        min_val = torch.min(t1_float, t2_float)
        max_val = torch.max(t1_float, t2_float)
        
        I = max_val - min_val # Interval length
        
        lower_bound = min_val - alpha * I
        upper_bound = max_val + alpha * I
        
        # Generate random numbers uniformly between lower_bound and upper_bound
        uniform_random = torch.rand(t1_float.size(), generator=torch_rng, device=t1.device)
        mutant = lower_bound + uniform_random * (upper_bound - lower_bound)

        if rate < 1.0:
            # Apply crossover rate: binary mask over the tensor
            crossover_mask = torch.rand(t1_float.size(), generator=torch_rng, device=t1.device) < rate
        
            # Child is mutant where mask is True, otherwise it's original t1
            child_float = torch.where(crossover_mask, mutant, t1_float)
        else:
            child_float = mutant

        return child_float.to(t1.dtype)


    def n_point_splice_crossover(self, t1: torch.Tensor, t2: torch.Tensor, num_points: int, py_rng: random.Random) -> torch.Tensor:
        """
        Performs N-point splice crossover between two tensors.
        Assumes t1 and t2 have the same shape, handled by the caller for T5 embeddings.
        """
        if t1.shape != t2.shape:
            error("Tensors passed to n_point_splice_crossover do not have consistent shapes. This should be handled before calling.")
            # Fallback similar to DE, if this error occurs
            min_dim1 = min(t1.shape[1], t2.shape[1])
            t1 = t1[:, :min_dim1]
            t2 = t2[:, :min_dim1]

        # Flatten the tensors for easier point selection
        flat1 = t1.flatten()
        flat2 = t2.flatten()
        child_flat = torch.empty_like(flat1, device=t1.device) # Ensure child is on the same device

        if flat1.numel() < 2:
            # If the tensor has less than 2 elements, direct copy to avoid index errors
            return flat1.clone().reshape(t1.shape)

        # Standard N-point splice crossover
        if num_points >= flat1.numel():
            warning(f"Number of crossover points ({num_points}) is greater than or equal to tensor size ({flat1.numel()}). Performing direct copy from t2.")
            return t2.clone().reshape(t1.shape)
        
        # Generate and sort unique crossover points
        crossover_points = sorted(py_rng.sample(range(1, flat1.numel()), num_points))
        
        current_segment_start = 0
        use_t1 = True # Start with t1's segment
        for point in crossover_points:
            if use_t1:
                child_flat[current_segment_start:point] = flat1[current_segment_start:point]
            else:
                child_flat[current_segment_start:point] = flat2[current_segment_start:point]
            use_t1 = not use_t1 # Toggle for the next segment
            current_segment_start = point
        
        # Add the last segment after the last crossover point
        if use_t1:
            child_flat[current_segment_start:] = flat1[current_segment_start:]
        else:
            child_flat[current_segment_start:] = flat2[current_segment_start:]
                
        return child_flat.reshape(t1.shape) # Reshape back to original tensor dimensions


    def mutate_tensor(
        self, tensor: torch.Tensor, rate: float, strength: float, use_gaussian: bool, torch_rng: torch.Generator
    ) -> torch.Tensor:
        """
        Applies mutation (Gaussian or uniform noise) to a tensor.
        """
        if rate == 0.0:
            return tensor.clone()

        # Clone the tensor to avoid modifying the original in-place
        mutated_tensor = tensor.clone()
        # Generate a mask for elements to be mutated
        mask = torch.rand(mutated_tensor.size(), generator=torch_rng, device=mutated_tensor.device) < rate
        
        if use_gaussian:
            # Generate Gaussian noise
            noise = torch.randn(mutated_tensor.size(), generator=torch_rng, device=mutated_tensor.device) * strength
        else:
            # Generate uniform noise in [-strength, strength]
            noise = (torch.rand(mutated_tensor.size(), generator=torch_rng, device=mutated_tensor.device) * 2 - 1) * strength
            
        # Apply noise only to masked elements
        mutated_tensor[mask] += noise[mask]
        return mutated_tensor


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
        torch_rng = torch.Generator().manual_seed(self.seed)

        # Load CLIP and T5 embeddings from candidates
        # Use the first candidate's embeddings as a 'like' tensor for shape/dtype fallback
        initial_clip_like = self._clip_embeds(context, self.candidates[0])
        initial_t5_like = self._t5_embeds(context, self.candidates[0])

        if initial_clip_like is None or initial_t5_like is None:
            error("Could not load initial conditioning to determine embedding shapes. Exiting.")
            return SelectedFluxConditioningOutput(
                conditioning=FluxConditioningField(conditioning_name=""),
            )

        # Load all candidates, using zeros_like as fallback for failed loads
        clip_candidates = [self._clip_embeds(context, c, like=initial_clip_like) for c in self.candidates]
        t5_candidates = [self._t5_embeds(context, c, like=initial_t5_like) for c in self.candidates]

        # Filter out any None values (though _clip_embeds/_t5_embeds should return zeros_like)
        clip_candidates = [c for c in clip_candidates if c is not None]
        t5_candidates = [t for t in t5_candidates if t is not None]

        if not clip_candidates or not t5_candidates:
            error("Failed to load sufficient CLIP or T5 embeddings from candidates. Check input conditionings.")
            return SelectedFluxConditioningOutput(
                conditioning=FluxConditioningField(conditioning_name=""),
            )

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
                    clip_candidates[0].clone(),
                    self.mutation_rate,
                    self.mutation_strength,
                    self.use_gaussian_mutation,
                    torch_rng,
                )
                t5_child = self.mutate_tensor(
                    t5_candidates[0].clone(),
                    self.mutation_rate,
                    self.mutation_strength,
                    self.use_gaussian_mutation,
                    torch_rng,
                )
            else:
                # Prepare candidate lists for parent selection and crossover
                current_clip_candidates = list(clip_candidates)
                current_t5_candidates = list(t5_candidates)

                # Determine required number of parents for the chosen crossover method
                required_parents = 3 if self.crossover_method == "Differential Evolution" else 2

                # If there are insufficient candidates for the chosen crossover method, handle specific cases
                if num_available_embeds < required_parents:
                    if self.crossover_method == "Differential Evolution" and num_available_embeds == 2:
                        # Case 2: Two candidates for Differential Evolution (needs 3 parents)
                        # Deterministically clone the first candidate to fulfill the requirement
                        info("Two candidates available for Differential Evolution. Cloning the first candidate to meet the 3-parent requirement.")
                        current_clip_candidates.append(clip_candidates[0].clone())
                        current_t5_candidates.append(t5_candidates[0].clone())
                    else:
                        error(f"Insufficient candidates ({num_available_embeds}) for {self.crossover_method} (requires {required_parents}). Skipping this member generation.")
                        continue # Skip this population member if parents cannot be selected

                try:
                    # Select parents from the (potentially augmented) current_clip_candidates/current_t5_candidates
                    if self.crossover_method == "Differential Evolution":
                        # Differential Evolution needs 3 distinct parents for t1, t2, t3
                        p_indices = py_rng.sample(range(len(current_clip_candidates)), 3)
                        p1_idx, p2_idx, p3_idx = p_indices

                        # Get the original T5 tensors
                        t5_p1_original = current_t5_candidates[p1_idx]
                        t5_p2_original = current_t5_candidates[p2_idx]
                        t5_p3_original = current_t5_candidates[p3_idx]

                        # Check for any T5 mismatch among the three parents
                        t5_mismatch_detected = False
                        if not (t5_p1_original.shape[1] == t5_p2_original.shape[1] == t5_p3_original.shape[1]):
                            t5_mismatch_detected = True
                        
                        t5_p1_final, t5_p2_final, t5_p3_aligned = t5_p1_original, t5_p2_original, t5_p3_original

                        if t5_mismatch_detected:
                            info("T5 embedding size mismatch detected for Differential Evolution parents. Handling mismatch.")
                            # Step 1: Align t5_p1 and t5_p2
                            if t5_p1_original.shape[1] != t5_p2_original.shape[1]:
                                t5_p1_aligned, t5_p2_aligned = self._handle_t5_mismatch(t5_p1_original.clone(), t5_p2_original.clone(), self.expand_t5_embeddings)

                            # Step 2: Align the (now potentially aligned) t5_p1 with t5_p3
                            if t5_p1_aligned.shape[1] != t5_p3_original.shape[1]:
                                t5_p1_final, t5_p3_aligned = self._handle_t5_mismatch(t5_p1_aligned.clone(), t5_p3_original.clone(), self.expand_t5_embeddings)

                            # Step 3: Make sure that t5_p1 is finalized
                            if t5_p1_final.shape[1] != t5_p1_aligned.shape[1]:
                                t5_p1_final = t5_p1_aligned

                            # Step 4: Ensure t5_p2_aligned matches the final size of t5_p1_final.
                            # This covers cases where t5_p1_aligned might have been further trimmed by t5_p3_original.
                            if t5_p1_final.shape[1] != t5_p2_aligned.shape[1]:
                                t5_p2_final, _ = self._handle_t5_mismatch(t5_p2_aligned.clone(), t5_p1_final.clone(), self.expand_t5_embeddings)

                            # Step 5: Make sure t5_p2 is finalized
                            if t5_p2_final.shape[1] != t5_p2_aligned.shape[1]:
                                t5_p2_final = t5_p2_aligned
                        else:
                            info("T5 embedding sizes are consistent for Differential Evolution parents. No mismatch handling required.")
                        
                        clip_child = self.mutate_tensor(
                            self.differential_evolution_crossover(
                                current_clip_candidates[p1_idx], current_clip_candidates[p2_idx], current_clip_candidates[p3_idx],
                                self.differential_evolution_scale, self.crossover_rate, torch_rng
                            ),
                            self.mutation_rate,
                            self.mutation_strength,
                            self.use_gaussian_mutation,
                            torch_rng,
                        )
                        t5_child = self.mutate_tensor(
                            self.differential_evolution_crossover(
                                t5_p1_final, t5_p2_final, t5_p3_aligned, # Use the aligned T5 tensors
                                self.differential_evolution_scale, self.crossover_rate, torch_rng
                            ),
                            self.mutation_rate,
                            self.mutation_strength,
                            self.use_gaussian_mutation,
                            torch_rng,
                        )
                    else: # For N-Point Splice, BLX-alpha, we need 2 parents
                        p_indices = py_rng.sample(range(len(current_clip_candidates)), 2)
                        p1_idx, p2_idx = p_indices

                        t5_p1 = current_t5_candidates[p1_idx]
                        t5_p2 = current_t5_candidates[p2_idx]

                        # Only call _handle_t5_mismatch if there's an actual mismatch
                        if t5_p1.shape[1] != t5_p2.shape[1]:
                            info("T5 embedding size mismatch detected for 2-parent crossover. Handling mismatch.")
                            t5_p1, t5_p2 = self._handle_t5_mismatch(
                                t5_p1.clone(), # Clone before passing to helper
                                t5_p2.clone(), # Clone before passing to helper
                                self.expand_t5_embeddings
                            )
                        else:
                            info("T5 embedding sizes are consistent for 2-parent crossover. No mismatch handling required.")

                        if self.crossover_method == "N-Point Splice":
                            clip_child = self.mutate_tensor(
                                self.n_point_splice_crossover(current_clip_candidates[p1_idx], current_clip_candidates[p2_idx], self.num_crossover_points, py_rng),
                                self.mutation_rate,
                                self.mutation_strength,
                                self.use_gaussian_mutation,
                                torch_rng,
                            )
                            t5_child = self.mutate_tensor(
                                self.n_point_splice_crossover(t5_p1, t5_p2, self.num_crossover_points, py_rng),
                                self.mutation_rate,
                                self.mutation_strength,
                                self.use_gaussian_mutation,
                                torch_rng,
                            )
                        elif self.crossover_method == "BLX-alpha":
                            clip_child = self.mutate_tensor(
                                self.blx_alpha_crossover(current_clip_candidates[p1_idx], current_clip_candidates[p2_idx], self.blx_alpha, self.crossover_rate, torch_rng),
                                self.mutation_rate,
                                self.mutation_strength,
                                self.use_gaussian_mutation,
                                torch_rng,
                            )
                            t5_child = self.mutate_tensor(
                                self.blx_alpha_crossover(t5_p1, t5_p2, self.blx_alpha, self.crossover_rate, torch_rng),
                                self.mutation_rate,
                                self.mutation_strength,
                                self.use_gaussian_mutation,
                                torch_rng,
                            )
                        else: # Fallback in case of unexpected crossover method (should be caught earlier)
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
