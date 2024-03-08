from pathlib import Path

import torch
from typing import Optional, List

from diffusers import Kandinsky3Pipeline
from diffusers.utils import numpy_to_pil

from comfy import model_management
import comfy


def load_kandinsky3(ckpt_pth: Path):
    pipeline: Kandinsky3Pipeline = Kandinsky3Pipeline.from_pretrained(
        ckpt_pth, variant="fp16", torch_dtype=torch.float16
    )

    tokenizer = pipeline.components['tokenizer']
    text_encoder = pipeline.components['text_encoder']
    movq = pipeline.components['movq']
    unet = comfy.model_patcher.ModelPatcher(
        pipeline.components['unet'],
        load_device=model_management.get_torch_device(),
        offload_device=model_management.intermediate_device(),
    )

    return unet, (tokenizer, text_encoder), movq


def prepare_noise(latent_image, seed):
    """
    creates random noise given a latent image and a seed
    """
    generator = torch.manual_seed(seed)
    return torch.randn(
        latent_image.size(),
        dtype=latent_image.dtype,
        layout=latent_image.layout,
        generator=generator,
        device="cpu"
    )


def convert_cond(cond):
    out = []
    for c in cond:
        # c = (embed, meta_dict)
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            model_conds["c_crossattn"] = comfy.conds.CONDCrossAttn(c[0])  # TODO: remove
            temp["cross_attn"] = c[0]
        temp["model_conds"] = model_conds
        out.append(temp)
    '''
    [
        {
            "model_conds": {"c_cross_attn": CONDCrossAttn(c[0])},
            "cross_attn": c[0],
            **c[1]
        }
        for c in cond
    ]
    '''
    return out


def prepare_mask(noise_mask, shape, device):
    """ensures noise mask is of proper dimensions"""
    noise_mask = torch.nn.functional.interpolate(noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
    noise_mask = torch.cat([noise_mask] * shape[1], dim=1)
    noise_mask = comfy.utils.repeat_to_batch_size(noise_mask, shape[0])
    noise_mask = noise_mask.to(device)
    return noise_mask


def get_models_from_cond(cond, model_type):
    models = []
    for c in cond:
        if model_type in c:
            models += [c[model_type]]
    return models


def get_additional_models(positive, negative, dtype):
    """loads additional models in positive and negative conditioning"""
    control_nets = set(get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control"))

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = get_models_from_cond(positive, "gligen") + get_models_from_cond(negative, "gligen")
    gligen = [x[1] for x in gligen]
    models = control_models + gligen
    return models, inference_memory


def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        if hasattr(m, 'cleanup'):
            m.cleanup()


def prepare_sampling(model, noise_shape, positive, negative, noise_mask):
    device = model.load_device
    positive_prompt = convert_cond(positive)
    negative_prompt = convert_cond(negative)

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise_shape, device)

    real_model = None
    models, inference_memory = get_additional_models(positive_prompt, negative_prompt, model.model_dtype())
    comfy.model_management.load_models_gpu([model] + models, model.memory_required([noise_shape[0] * 2] + list(noise_shape[1:])) + inference_memory)
    real_model = model.model

    return real_model, positive_prompt, negative_prompt, noise_mask, models


def sample(
        model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
        denoise=1.0, disable_noise=False, start_step=None, last_step=None,
        force_full_denoise=False, noise_mask=None, sigmas=None, callback=None,
        disable_pbar=False, seed=None):
    real_model, positive_copy, negative_copy, noise_mask, models = prepare_sampling(
        model, noise.shape, positive, negative, noise_mask
    )

    noise = noise.to(model.load_device)
    latent_image = latent_image.to(model.load_device)

    sampler = comfy.samplers.KSampler(real_model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(comfy.model_management.intermediate_device())

    cleanup_additional_models(models)
    cleanup_additional_models(
        set(
            get_models_from_cond(prompt_copy, "control")
        )
    )
    return samples


def get_callback(steps):
    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        pbar.update_absolute(step + 1, total_steps)
    return callback


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                    latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                    force_full_denoise=False):
    latent_image = latent["samples"]
    if 'batch_index' in latent:
        raise NotImplementedError('Batch Index is not supported')
    noise = prepare_noise(latent_image, seed)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = get_callback(steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = sample(
        model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
        denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
        force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback,
        disable_pbar=disable_pbar, seed=seed
    )
    out = latent.copy()
    out["samples"] = samples
    return (out, )


def common_ksampler_(model, seed, steps, cfg, sampler_name, scheduler, prompt_embed, latent_image, denoise=None):
    # TODO: finish?
    '''
    latents = self.prepare_latents(
        (batch_size * num_images_per_prompt, 4, height, width),
        prompt_embeds.dtype,
        device,
        generator,
        latents,
        self.scheduler,
    )

    if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
        self.text_encoder_offload_hook.offload()

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attention_mask,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                noise_pred = (guidance_scale + 1.0) * noise_pred_text - guidance_scale * noise_pred_uncond
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            ).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                attention_mask = callback_outputs.pop("attention_mask", attention_mask)
                negative_attention_mask = callback_outputs.pop("negative_attention_mask", negative_attention_mask)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    pass
    '''