import torch
from typing import Optional, List

from diffusers.utils import numpy_to_pil

import comfy.model_management


@torch.no_grad()
def process_embeds(embeddings, attention_mask, cut_context):
    if cut_context:
        embeddings[attention_mask == 0] = torch.zeros_like(embeddings[attention_mask == 0])
        max_seq_length = attention_mask.sum(-1).max() + 1
        embeddings = embeddings[:, :max_seq_length]
        attention_mask = attention_mask[:, :max_seq_length]
    return embeddings, attention_mask


@torch.no_grad()
def encode_prompt_(
        text_tokenizer,
        text_encoder,
        prompt,
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
        device=None,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        _cut_context=False,
        attention_mask: Optional[torch.FloatTensor] = None,
        negative_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask. Must provide if passing `prompt_embeds` directly.
            negative_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated negative attention mask. Must provide if passing `negative_prompt_embeds` directly.
        """
        if prompt is not None and negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        max_length = 128

        if prompt_embeds is None:
            text_inputs = text_tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds = text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
            prompt_embeds, attention_mask = process_embeds(prompt_embeds, attention_mask, _cut_context)
            prompt_embeds = prompt_embeds * attention_mask.unsqueeze(2)

        if text_encoder is not None:
            dtype = text_encoder.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        attention_mask = attention_mask.repeat(num_images_per_prompt, 1)
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]

            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
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
            if negative_prompt is not None:
                uncond_input = text_tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=128,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                text_input_ids = uncond_input.input_ids.to(device)
                negative_attention_mask = uncond_input.attention_mask.to(device)

                negative_prompt_embeds = text_encoder(
                    text_input_ids,
                    attention_mask=negative_attention_mask,
                )
                negative_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds[:, : prompt_embeds.shape[1]]
                negative_attention_mask = negative_attention_mask[:, : prompt_embeds.shape[1]]
                negative_prompt_embeds = negative_prompt_embeds * negative_attention_mask.unsqueeze(2)

            else:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_attention_mask = torch.zeros_like(attention_mask)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            if negative_prompt_embeds.shape != prompt_embeds.shape:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
                negative_attention_mask = negative_attention_mask.repeat(num_images_per_prompt, 1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
        else:
            negative_prompt_embeds = None
            negative_attention_mask = None
        return prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask


@torch.no_grad()
def encode_single_prompt(
        text_tokenizer,
        text_encoder,
        prompt: str,
        num_images_per_prompt: int = 1,
        cut_context: bool = True):

    device = comfy.model_management.get_torch_device()
    interim_device = comfy.model_management.intermediate_device()
    text_encoder.to(device)

    max_length = 128

    text_inputs = text_tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]
    prompt_embeds, attention_mask = process_embeds(prompt_embeds, attention_mask, cut_context)
    prompt_embeds = prompt_embeds * attention_mask.unsqueeze(2)

    print(f"INPUT IDS: {text_input_ids.shape}")
    print(f"ATTENTION_MASK: {attention_mask.shape}")
    print(f"PROMPT EMBEDS: {prompt_embeds.shape}")
    print()

    if text_encoder is not None:
        dtype = text_encoder.dtype
    else:
        raise ValueError("Text encoder must be defined")

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
    attention_mask = attention_mask.repeat(num_images_per_prompt, 1)

    text_encoder.to(interim_device)

    return prompt_embeds.to(interim_device), attention_mask.to(interim_device)


def expand_prompt_embed(prompt_embeds, attention_mask, num_images_per_prompt):
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(num_images_per_prompt, seq_len, -1)
    attention_mask = attention_mask.repeat(num_images_per_prompt, 1)
    return prompt_embeds, attention_mask


def expand(embed, attention, total_seq_len):
    bs, seq_len, dim = embed.shape
    ret_embed = torch.zeros((bs, total_seq_len, dim), dtype=embed.dtype, device=embed.device)
    ret_embed[:, : seq_len, :] = embed

    bs, seq_len = attention.shape
    ret_attention = torch.zeros((bs, total_seq_len), dtype=attention.dtype, device=attention.device)
    ret_attention[:, : seq_len] = attention

    return ret_embed, ret_attention


def combine_prompt_embeds(
        prompt_embeds, attention_mask, negative_prompt_embeds, negative_attention_mask,
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1):

    if num_images_per_prompt > 1:
        prompt_embeds, attention_mask = \
            expand_prompt_embed(prompt_embeds, attention_mask, num_images_per_prompt)

    if not do_classifier_free_guidance:
        return prompt_embeds, attention_mask.bool()

    bs_embed, seq_len, _ = prompt_embeds.shape
    neg_seq_len = negative_prompt_embeds.shape[1]
    total_seq_len = max(seq_len, neg_seq_len)


    uncond_tokens: List[str]

    if negative_prompt_embeds is None:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_attention_mask = torch.zeros_like(attention_mask)
    else:
        negative_prompt_embeds, negative_attention_mask = expand(negative_prompt_embeds, negative_attention_mask, total_seq_len)
        prompt_embeds, attention_mask = expand(prompt_embeds, attention_mask, total_seq_len)

    if num_images_per_prompt > 1:
        negative_prompt_embeds, negative_attention_mask = \
            expand_prompt_embed(negative_prompt_embeds, negative_attention_mask, num_images_per_prompt)

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    attention_mask = torch.cat([negative_attention_mask, attention_mask]).bool()
    return prompt_embeds, attention_mask


@torch.no_grad()
def encode_double_prompt(
        tokenizer,
        text_encoder,
        prompt: str,
        neg_prompt: str,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        cut_context: bool = True,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        negative_attention_mask: Optional[torch.FloatTensor] = None):

    device = None # TODO: how to get one???
    prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask = encode_prompt_(
        tokenizer,
        text_encoder,
        prompt,
        do_classifier_free_guidance,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        negative_prompt=neg_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        _cut_context=cut_context,
        attention_mask=attention_mask,
        negative_attention_mask=negative_attention_mask,
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        attention_mask = torch.cat([negative_attention_mask, attention_mask]).bool()

    return prompt_embeds, attention_mask


def movq_decode(latents, movq):
    image = movq.decode(latents, force_not_quantize=True)['sample']

    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return numpy_to_pil(image)
