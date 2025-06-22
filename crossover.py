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
    output_mask_vector_pair: str = OutputField(
        description="JSON representation of the T5 and CLIP mask vector pair for the selected output.",
        ui_order=0
    )

# A) New node: KVListToJSONString
@invocation(
    "kv_list_to_json_string",
    title="Key-Value List to JSON String",
    tags=["json", "string", "utility"],
    category="utility",
    version="1.0.0",
)
class KVListToJSONString(BaseInvocation):
    """
    Takes four strings (two key-value pairs) and outputs them as a JSON object string.
    """
    key1: str = InputField(description="First key", ui_order=0)
    value1: str = InputField(description="First value", ui_order=1)
    key2: str = InputField(description="Second key", ui_order=2)
    value2: str = InputField(description="Second value", ui_order=3)

    def invoke(self, context: InvocationContext) -> StringOutput:
        output_dict = {
            self.key1: self.value1,
            self.key2: self.value2
        }
        json_string = json.dumps(output_dict)
        info(f"Generated JSON string: {json_string}")
        return StringOutput(value=json_string)


# B) New node: DecodeMaskVectorPairs
@invocation_output("decoded_mask_vector_pairs_output")
class DecodedMaskVectorPairsOutput(BaseInvocationOutput):
    """Output for decoded raw_conditioning and mask_vector_pair lists."""
    raw_conditionings: StringCollectionOutput = OutputField(description="List of raw conditioning strings.")
    mask_vector_pairs: StringCollectionOutput = OutputField(description="List of mask vector pair JSON strings.")

@invocation(
    "decode_mask_vector_pairs",
    title="Decode Mask Vector Pairs",
    tags=["json", "list", "utility", "mask"],
    category="utility",
    version="1.0.0",
)
class DecodeMaskVectorPairs(BaseInvocation):
    """
    Reads a list of JSON strings, extracts "raw_conditioning" and "mask_vector_pair",
    and outputs them as two parallel lists.
    """
    json_list_input: List[str] = InputField(description="List of JSON strings to decode.", ui_order=0)

    def invoke(self, context: InvocationContext) -> DecodedMaskVectorPairsOutput:
        raw_conditionings_list = []
        mask_vector_pairs_list = []

        for json_str in self.json_list_input:
            try:
                data = json.loads(json_str)
                raw_conditioning = data.get("raw_conditioning")
                mask_vector_pair = data.get("mask_vector_pair")

                if raw_conditioning is not None:
                    raw_conditionings_list.append(raw_conditioning)
                else:
                    warning(f"JSON object missing 'raw_conditioning' key: {json_str}")

                if mask_vector_pair is not None:
                    mask_vector_pairs_list.append(mask_vector_pair)
                else:
                    warning(f"JSON object missing 'mask_vector_pair' key: {json_str}")

            except json.JSONDecodeError as e:
                error(f"Failed to decode JSON string: {json_str}. Error: {e}")
            except Exception as e:
                error(f"An unexpected error occurred while processing JSON string: {json_str}. Error: {e}")

        info(f"Decoded {len(raw_conditionings_list)} raw conditionings and {len(mask_vector_pairs_list)} mask vector pairs.")
        return DecodedMaskVectorPairsOutput(
            raw_conditionings=StringCollectionOutput(collection=raw_conditionings_list),
            mask_vector_pairs=StringCollectionOutput(collection=mask_vector_pairs_list)
        )


@invocation(
    "flux_conditioning_genetic_algorithm",
    title="Flux Conditioning Genetic Algorithm",
    tags=["conditioning", "flux", "genetic", "algorithm", "evolution"],
    category="conditioning",
    version="1.0.5", # Incrementing version due to optimized mask crossover
)
class FluxConditioningGeneticAlgorithmInvocation(BaseInvocation):
    """
    Applies a genetic algorithm to a list of FLUX Conditionings,
    generating a new population through crossover and mutation.
    Includes options for different crossover strategies and mask vector handling.
    """

    candidates: list[FluxConditioningField] = InputField(
        description="List of FLUX Conditioning candidates for the genetic algorithm.",
        ui_order=0,
    )
    mask_vector_pairs_json: str | list[str] | None = InputField(
        default=None,
        description="Optional: List of JSON strings, each a [T5_mask, CLIP_mask] pair. Used for masked crossover.",
        ui_order=1,
    )
    initialize_random_masks_if_none: bool = InputField(
        default=False,
        description="If no masks are provided, initialize random binary masks for crossover.",
        ui_order=2,
    )
    population_size: int = InputField(
        default=10,
        gt=0,
        description="Desired size of the new conditioning population.",
        ui_order=3,
    )
    crossover_method: Literal["Splice", "Differential Evolution"] = InputField(
        default="Splice",
        description="Method to use for combining parent embeddings.",
        ui_order=4,
        ui_choice_labels={
            "Splice": "Splice (Single-point crossover)",
            "Differential Evolution": "Differential Evolution (3-parent crossover)"
        }
    )
    differential_evolution_scale: float = InputField(
        default=0.7,
        ge=0.0,
        description="Scale factor for differential evolution crossover. (Only used with Differential Evolution method)",
        ui_order=5,
    )
    mutation_rate: float = InputField(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Rate of mutation per tensor element (0.0 to 1.0).",
        ui_order=6,
    )
    mutation_strength: float = InputField(
        default=0.05,
        ge=0.0,
        description="Strength of mutation (Gaussian noise multiplier).",
        ui_order=7,
    )
    seed: Optional[int] = InputField(
        default=None,
        description="Random seed for deterministic behavior.",
        ui_order=8,
    )
    selected_member_index: int = InputField(
        default=0,
        ge=0,
        description="Index of the new population member to output as a single conditioning.",
        ui_order=9,
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

    def _splice_mask_crossover(self, mask1: torch.Tensor, mask2: torch.Tensor, py_rng: random.Random) -> torch.Tensor:
        """
        Performs single-point splice crossover between two binary masks.
        """
        if mask1.shape != mask2.shape:
            error("Masks must have the same shape for splice mask crossover. Returning clone of first mask.")
            return mask1.clone()

        flat1 = mask1.flatten()
        flat2 = mask2.flatten()

        if flat1.numel() < 2:
            return flat1.clone().reshape(mask1.shape)

        split = py_rng.randint(1, flat1.numel() - 1)
        child_flat = torch.cat((flat1[:split], flat2[split:]), dim=0)
        return child_flat.reshape(mask1.shape)


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
                output_mask_vector_pair=json.dumps([[], []])
            )

        # Initialize random number generators
        py_rng = random.Random(self.seed)
        torch_rng = (
            torch.Generator().manual_seed(self.seed) if self.seed is not None else torch.default_generator
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
                output_mask_vector_pair=json.dumps([[], []])
            )

        # Ensure all tensors in the filtered lists have the same shape.
        first_clip_shape = clip_candidates[0].shape
        first_t5_shape = t5_candidates[0].shape

        if not all(c.shape == first_clip_shape for c in clip_candidates):
            warning("Inconsistent CLIP embedding shapes among candidates. Operations might fail.")
        if not all(t.shape == first_t5_shape for t in t5_candidates):
            warning("Inconsistent T5 embedding shapes among candidates. Operations might fail.")

        # --- Mask Vector Handling ---
        initial_mask_vectors: Optional[List[List[torch.Tensor]]] = None
        
        # Attempt to load masks if provided
        if self.mask_vector_pairs_json:
            if not list == type(self.mask_vector_pairs_json):
                self.mask_vector_pairs_json = [self.mask_vector_pairs_json for i in range(self.population_size)]
            initial_mask_vectors = []
            for mask_json_str in self.mask_vector_pairs_json:
                try:
                    decoded_masks = json.loads(mask_json_str)
                    if isinstance(decoded_masks, list) and len(decoded_masks) == 2 and \
                       isinstance(decoded_masks[0], list) and isinstance(decoded_masks[1], list):
                        t5_mask_list = decoded_masks[0]
                        clip_mask_list = decoded_masks[1]
                        
                        t5_mask = torch.tensor(t5_mask_list, dtype=torch.bool).reshape(first_t5_shape)
                        clip_mask = torch.tensor(clip_mask_list, dtype=torch.bool).reshape(first_clip_shape)
                        initial_mask_vectors.append([t5_mask, clip_mask])
                    else:
                        warning(f"Invalid mask vector pair JSON structure: {mask_json_str}. Will not use provided masks.")
                        initial_mask_vectors = None
                        break
                except json.JSONDecodeError as e:
                    warning(f"Failed to decode mask vector JSON: {mask_json_str}. Error: {e}. Will not use provided masks.")
                    initial_mask_vectors = None
                    break
                except Exception as e:
                    warning(f"Error processing mask vector JSON: {mask_json_str}. Error: {e}. Will not use provided masks.")
                    initial_mask_vectors = None
                    break
            
            if initial_mask_vectors and len(initial_mask_vectors) != len(self.candidates):
                warning(f"Number of mask vector pairs ({len(initial_mask_vectors)}) does not match number of candidates ({len(self.candidates)}). Will not use provided masks.")
                initial_mask_vectors = None

        # If no masks were provided or loading failed, and initialization is requested
        if initial_mask_vectors is None and self.initialize_random_masks_if_none:
            info("No masks provided, but 'initialize_random_masks_if_none' is True. Generating random masks.")
            initial_mask_vectors = []
            for _ in range(len(self.candidates)):
                random_t5_mask = (torch.rand(first_t5_shape, generator=torch_rng) < 0.5).to(torch.bool)
                random_clip_mask = (torch.rand(first_clip_shape, generator=torch_rng) < 0.5).to(torch.bool)
                initial_mask_vectors.append([random_t5_mask, random_clip_mask])
            info(f"Generated {len(initial_mask_vectors)} random mask pairs.")
        elif initial_mask_vectors is None:
             info("No masks provided and 'initialize_random_masks_if_none' is False. Crossover will be unmasked.")


        def splice_crossover(t1: torch.Tensor, t2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Performs single-point crossover between two tensors, optionally applying a mask.
            If a mask is provided, elements where mask is False (0) are copied from t1,
            and elements where mask is True (1) are spliced from t2 at a random point.
            """
            if t1.shape != t2.shape:
                error("Tensors must have the same shape for splice crossover. Returning clone of first tensor.")
                return t1.clone()

            if mask is not None and mask.shape != t1.shape:
                error("Mask shape must match tensor shape for masked splice crossover. Performing unmasked splice.")
                mask = None

            flat1 = t1.flatten()
            flat2 = t2.flatten()
            child_flat = torch.empty_like(flat1)

            if flat1.numel() < 2:
                return flat1.clone().reshape(t1.shape)

            if mask is None:
                # Standard splice crossover
                split = py_rng.randint(1, flat1.numel() - 1)
                child_flat = torch.cat((flat1[:split], flat2[split:]), dim=0)
            else:
                # Masked splice crossover
                flat_mask = mask.flatten()
                
                # Elements where mask is False (0) are taken from t1
                child_flat[~flat_mask] = flat1[~flat_mask]

                # For elements where mask is True (1), perform splice
                masked_indices = torch.where(flat_mask)[0]
                if masked_indices.numel() > 0:
                    masked_flat1 = flat1[masked_indices]
                    masked_flat2 = flat2[masked_indices]

                    if masked_flat1.numel() < 2:
                         child_flat[masked_indices] = masked_flat1
                    else:
                        split_masked = py_rng.randint(1, masked_flat1.numel() - 1)
                        spliced_masked_part = torch.cat((masked_flat1[:split_masked], masked_flat2[split_masked:]), dim=0)
                        child_flat[masked_indices] = spliced_masked_part
                else:
                    child_flat = flat1.clone()
            
            return child_flat.reshape(t1.shape)

        def differential_evolution_crossover(
            t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, scale_factor: float, mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Performs differential evolution crossover: child = t1 + scale_factor * (t2 - t3).
            Optionally applies a mask: only elements where mask is True (1) are affected by DE.
            Requires t1, t2, t3 to have the same shape.
            """
            if not (t1.shape == t2.shape == t3.shape):
                error("All three tensors must have the same shape for differential evolution. Returning clone of t1.")
                return t1.clone()

            if mask is not None and mask.shape != t1.shape:
                error("Mask shape must match tensor shape for masked differential evolution. Performing unmasked DE.")
                mask = None

            t1_float = t1.to(torch.float32)
            t2_float = t2.to(torch.float32)
            t3_float = t3.to(torch.float32)

            child_float = t1_float.clone()

            if mask is None:
                # Standard differential evolution
                difference = t2_float - t3_float
                child_float = t1_float + scale_factor * difference
            else:
                # Masked differential evolution
                difference = t2_float - t3_float
                child_float[mask] = t1_float[mask] + scale_factor * difference[mask]
            
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
        generated_mask_pairs: List[str] = []

        # Generate the new population
        for i in range(self.population_size):
            clip_child = None
            t5_child = None
            current_t5_mask: Optional[torch.Tensor] = None
            current_clip_mask: Optional[torch.Tensor] = None

            # Select parents
            p_indices = py_rng.sample(range(len(self.candidates)), 3 if self.crossover_method == "Differential Evolution" else 2)
            p1_idx = p_indices[0]
            p2_idx = p_indices[1]
            p3_idx = p_indices[2] if self.crossover_method == "Differential Evolution" else None

            # --- Determine which masks to use for this child's crossover ---
            if initial_mask_vectors:
                if len(initial_mask_vectors) > 1:
                    parent1_t5_mask, parent1_clip_mask = initial_mask_vectors[p1_idx]
                    parent2_t5_mask, parent2_clip_mask = initial_mask_vectors[p2_idx]
                    
                    # Use splice crossover for masks
                    current_t5_mask = self._splice_mask_crossover(parent1_t5_mask, parent2_t5_mask, py_rng)
                    current_clip_mask = self._splice_mask_crossover(parent1_clip_mask, parent2_clip_mask, py_rng)
                    info(f"Masks for child {i+1} generated via splice crossover from provided/random masks.")
                elif len(initial_mask_vectors) == 1:
                    current_t5_mask, current_clip_mask = initial_mask_vectors[0]
                    info(f"Using the single available mask for child {i+1}.")
                else:
                    info(f"No valid masks to apply for child {i+1}. Crossover will be unmasked.")
            else:
                info(f"No masks available for child {i+1}. Crossover will be unmasked.")


            if self.crossover_method == "Splice":
                clip_child = mutate_tensor(
                    splice_crossover(clip_candidates[p1_idx], clip_candidates[p2_idx], mask=current_clip_mask),
                    self.mutation_rate,
                    self.mutation_strength,
                )
                t5_child = mutate_tensor(
                    splice_crossover(t5_candidates[p1_idx], t5_candidates[p2_idx], mask=current_t5_mask),
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
                            self.differential_evolution_scale, mask=current_clip_mask
                        ),
                        self.mutation_rate,
                        self.mutation_strength,
                    )
                    t5_child = mutate_tensor(
                        differential_evolution_crossover(
                            t5_candidates[p1_idx], t5_candidates[p2_idx], t5_candidates[p3_idx], 
                            self.differential_evolution_scale, mask=current_t5_mask
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

            if current_t5_mask is not None and current_clip_mask is not None:
                output_t5_mask_list = current_t5_mask.flatten().tolist()
                output_clip_mask_list = current_clip_mask.flatten().tolist()
                mask_pair_json_str = json.dumps([output_t5_mask_list, output_clip_mask_list])
            else:
                mask_pair_json_str = json.dumps([[], []])
            
            generated_mask_pairs.append(mask_pair_json_str)

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
                output_mask_vector_pair=json.dumps([[], []])
            )

        selected_idx = self.selected_member_index
        if not (0 <= selected_idx < len(new_population)):
            warning(
                f"Selected member index {self.selected_member_index} is out of bounds for "
                f"population size {len(new_population)}. Returning a random member."
            )
            selected_idx = py_rng.randint(0, len(new_population) - 1)
        
        selected_conditioning_output = new_population[selected_idx]
        selected_mask_pair_json = generated_mask_pairs[selected_idx]

        return SelectedFluxConditioningOutput(
            conditioning=selected_conditioning_output.conditioning,
            output_mask_vector_pair=selected_mask_pair_json
        )
