import logging
import torch

from pathlib import Path
from typing import List, Optional, Callable, Dict

from comfy import model_management
from diffusers import KandinskyV22PriorPipeline
from diffusers.pipelines.kandinsky import KandinskyPriorPipelineOutput
from diffusers.utils import numpy_to_pil
from diffusers.utils.torch_utils import randn_tensor

from .utils import get_vanilla_callback


logger = logging.getLogger()

def _encode_prompt(
        tokenizer,
        text_encoder,
        prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
):
    device = text_encoder.device
    num_images_per_prompt = 1
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    # get prompt text embeddings
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    text_mask = text_inputs.attention_mask.bool().to(device)

    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )
        text_input_ids = text_input_ids[:, : tokenizer.model_max_length]

    text_encoder_output = text_encoder(text_input_ids.to(device))

    prompt_embeds = text_encoder_output.text_embeds
    text_encoder_hidden_states = text_encoder_output.last_hidden_state

    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
    text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

    if do_classifier_free_guidance:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_text_mask = uncond_input.attention_mask.bool().to(device)
        negative_prompt_embeds_text_encoder_output = text_encoder(uncond_input.input_ids.to(device))

        negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.text_embeds
        uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output.last_hidden_state

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

        seq_len = uncond_text_encoder_hidden_states.shape[1]
        uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
        uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )
        uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        # done duplicates

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

        text_mask = torch.cat([uncond_text_mask, text_mask])

    return prompt_embeds, text_encoder_hidden_states, text_mask


def prepare_latents(shape, dtype, device, generator, scheduler):
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma
    return latents


def get_zero_embed(image_encoder, device):
    zero_img = torch.zeros(1, 3, image_encoder.config.image_size, image_encoder.config.image_size).to(
        device=device, dtype=image_encoder.dtype
    )
    return image_encoder(zero_img)["image_embeds"]


def text_prior_inference(
        encoder,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 25,
        generator: Optional[torch.Generator] = None,
        guidance_scale: float = 4.0,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    ):
        tokenizer, text_encoder, prior, scheduler, zero_embeds = encoder
        device = text_encoder.device
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt = [prompt]

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif negative_prompt is not None:
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        # if the negative prompt is defined we double the batch size to
        # directly retrieve the negative prompt embedding
        if negative_prompt is not None:
            prompt = prompt + negative_prompt
            negative_prompt = 2 * negative_prompt

        batch_size = len(prompt)

        prompt_embeds, text_encoder_hidden_states, text_mask = _encode_prompt(
            tokenizer, text_encoder, prompt, do_classifier_free_guidance, negative_prompt
        )

        # prior
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        embedding_dim = prior.config.embedding_dim

        latents = prepare_latents(
            (batch_size, embedding_dim),
            prompt_embeds.dtype,
            device,
            generator,
            scheduler,
        )
        _num_timesteps = len(timesteps)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            predicted_image_embedding = prior(
                latent_model_input,
                timestep=t,
                proj_embedding=prompt_embeds,
                encoder_hidden_states=text_encoder_hidden_states,
                attention_mask=text_mask,
            ).predicted_image_embedding

            if do_classifier_free_guidance:
                predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                predicted_image_embedding = predicted_image_embedding_uncond + guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )

            if i + 1 == timesteps.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = timesteps[i + 1]

            latents = scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=latents,
                generator=generator,
                prev_timestep=prev_timestep,
            ).prev_sample

            if callback_on_step_end is not None:
                callback_on_step_end(i, num_inference_steps)

        latents = prior.post_process_latents(latents)

        image_embeddings = latents

        # if negative prompt has been defined, we retrieve split the image embedding into two
        if negative_prompt is not None:
            image_embeddings, zero_embeds = image_embeddings.chunk(2)

        # TODO: check it
        # self.maybe_free_model_hooks()
        return KandinskyPriorPipelineOutput(image_embeds=image_embeddings, negative_image_embeds=zero_embeds)


def encode_text(encoder, num_inference_steps, guidance_scale, seed, prompt, negative_prior):
    device: torch.device = model_management.get_torch_device()
    offload_device: torch.device = model_management.intermediate_device()
    generator = torch.Generator().manual_seed(seed)

    tokenizer, text_encoder, prior, scheduler, zero_embeds = encoder
    text_encoder.to(device)
    prior.to(device)

    result = text_prior_inference(
        encoder,
        prompt,
        negative_prompt=negative_prior,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale,
        callback_on_step_end=get_vanilla_callback(num_inference_steps),
    )
    text_encoder.to(offload_device)
    prior.to(offload_device)

    return result.image_embeds, result.negative_image_embeds


def encode_image(encoder, image):
    device: torch.device = model_management.get_torch_device()
    offload_device: torch.device = model_management.intermediate_device()

    image_processor, image_encoder = encoder
    image_encoder.to(device)

    image = numpy_to_pil(image.numpy())[0]
    image = (
        image_processor(image, return_tensors="pt")
        .pixel_values[0]
        .unsqueeze(0)
        .to(dtype=image_encoder.dtype, device=device)
    )
    image_emb = image_encoder(image)["image_embeds"]
    # TODO: offload effectively
    image_encoder.to(offload_device)

    return image_emb


def load_prior_kandinsky22(path: Path):
    pipeline: KandinskyV22PriorPipeline = KandinskyV22PriorPipeline.from_pretrained(
        path, torch_dtype=torch.float16
    )

    tokenizer = pipeline.components['tokenizer']
    text_encoder = pipeline.components['text_encoder']

    image_encoder = pipeline.components['image_encoder']
    image_processor = pipeline.components['image_processor']

    scheduler = pipeline.components['scheduler']
    prior = pipeline.components['prior']

    device: torch.device = model_management.get_torch_device()
    offload_device: torch.device = model_management.intermediate_device()

    zero_embeds = get_zero_embed(image_encoder.to(device), device).to(offload_device)
    image_encoder.to(offload_device)

    return \
        (image_processor, image_encoder), \
        (tokenizer, text_encoder, prior, scheduler, zero_embeds)
