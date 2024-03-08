from typing import Optional, Tuple, Union

import torch
import comfy
from diffusers import DDPMScheduler

from comfy import model_management
from comfy.model_patcher import ModelPatcher

from .autoencode import combine_prompt_embeds


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


def get_callback(steps):
    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, total_steps):
        pbar.update_absolute(step + 1, total_steps)

    return callback


def sample(
        model: ModelPatcher,
        latents: Union[torch.Tensor, torch.cuda.FloatTensor],
        num_inference_steps: int,
        scheduler: DDPMScheduler,
        guidance_scale: float,
        callback,
        positive: Tuple,
        negative: Tuple,
        seed: Optional[int] = None,
        steps: Optional[int] = None):
    device: torch.device = model_management.get_torch_device()
    offload_device: torch.device = model_management.intermediate_device()

    generator = torch.manual_seed(seed)
    # 4. Prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # 5. Prepare latents
    latents = latents * scheduler.init_noise_sigma

    do_classifier_free_guidance = guidance_scale >= 1.0

    num_images_per_prompt = latents.shape[0]

    prompt_embeds, attention_mask = combine_prompt_embeds(
        positive[0],
        positive[1]["attention_mask"],
        negative[0],
        negative[1]["attention_mask"],
        do_classifier_free_guidance,
        num_images_per_prompt
    )

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
    _num_timesteps = len(timesteps)
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        # predict the noise residual
        noise_pred = model.model(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=attention_mask,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = (guidance_scale + 1.0) * noise_pred_text - guidance_scale * noise_pred_uncond
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(
            noise_pred,
            t,
            latents,
            generator=generator,
        ).prev_sample

        callback(i, steps)

    return latents



def simple_sampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                    latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                    force_full_denoise=False):
    print(f'POSITIVE: {positive}')
    latent_image = latent["samples"]
    if 'batch_index' in latent:
        raise NotImplementedError('Batch Index is not supported')
    noise = prepare_noise(latent_image, seed)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = get_callback(steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    scheduler = DDPMScheduler()
    num_inference_steps = steps

    samples = sample(
        model=model,
        latents=noise,
        num_inference_steps=num_inference_steps,
        scheduler=scheduler,
        guidance_scale=cfg,
        callback=callback,
        positive=positive,
        negative=negative,
        seed=seed,
        steps=steps,
    )


    out = latent.copy()
    out["samples"] = samples
    return (out, )