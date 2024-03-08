import folder_paths
import comfy
from pathlib import Path
from typing import List

import torch
from diffusers import Kandinsky3Pipeline

from .kandinsky3simple import simple_sampler
from .kandinsky3 import common_ksampler, load_kandinsky3
from .autoencode import encode_double_prompt, encode_single_prompt, movq_decode

MANIFEST = {
    "name": "Kandinsky Nodes",
    "version": (0,0,1),
    "author": "Seva Oparin",
    "project": "https://github.com/vsevolod-oparin/",
    "description": "Help nodes to use kandinsky",
}


def get_checkpoint_folder(subtoken: str, checkpoint_dir: str = "checkpoints") -> List[str]:
    folder_list = folder_paths.get_filename_list_(checkpoint_dir)[1]
    return [
        folder for folder in folder_list
        if subtoken in Path(folder).name and Path(folder).parent.name == checkpoint_dir
    ]


class Kandinsky3ModelLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt": (get_checkpoint_folder("kandinsky"), ),
            }
        }

    RETURN_TYPES = ("MODEL", "T5", "MOVQ")

    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt, output_vae=True, output_clip=True):
        ckpt_pth = Path(ckpt)
        # if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
        return load_kandinsky3(ckpt_pth)

    '''
    def DEBUG_CODE(self, ckpt):
        
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        # return out[:3]
        # return "!"
        
        res = transforms.PILToTensor()(Image.open(Path(ckpt) / 'assets' / 'photo_2.jpg')) / 255.0
        res = res.permute((1, 2, 0)).unsqueeze(0)
        print(f'loader -> {res.shape}')
        return (res, )
    '''


class T5TextEncoder:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "t5": ("T5", )
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, t5, prompt):
        tokenizer, text_encoder = t5
        cond, attention_mask = encode_single_prompt(tokenizer, text_encoder, prompt)
        return (cond, {"attention_mask": attention_mask}),


class T5DoubleTextEncoder:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True}),
                "t5": ("T5", )
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, t5, positive, negative):
        tokenizer, text_encoder = t5
        cond, attention_mask = encode_double_prompt(tokenizer, text_encoder, positive, negative)
        return [[cond, {"attention_mask": attention_mask}]],


'''
class KandisnkyKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {class KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, prompt_embed, latent_image, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, prompt_embed, latent_image, denoise=denoise)
'''


MAX_RESOLUTION = 8192

class EmptyLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples":latent}, )


class KandinskyKSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return simple_sampler(
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
            latent_image, denoise=denoise
        )


class KandinskyKSamplerComplicated:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
            latent_image, denoise=denoise
        )


class KandinskyMovqDecoder:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", ),
                "movq": ("MOVQ", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"

    def decode(self, samples, movq):
        return movq_decode(samples["samples"], movq),


NODE_CLASS_MAPPINGS = {
    "comfy-kandinsky3-loader": Kandinsky3ModelLoader,
    "comfy-kandinsky3-encoder": T5TextEncoder,
    "comfy-kandinsky3-double-encoder": T5DoubleTextEncoder,
    "comfy-kandinsky3-movq-decoder": KandinskyMovqDecoder,
    "comfy-kandinsky3-ksampler": KandinskyKSampler,
    "confy-kandinsky3-empty-latent": EmptyLatentImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "comfy-kandinsky3-loader": "Kandinsky3 Loader",
    "comfy-kandinsky3-encoder": "Kandinsky3 T5 Encoder",
    "comfy-kandinsky3-double-encoder": "Kandinsky3 T5 Double Encoder",
    "comfy-kandinsky3-movq-decoder": "Kandinsky3 MovQ Decoder",
    "comfy-kandinsky3-ksampler": "Kandinsky3 KSampler",
    "confy-kandinsky3-empty-latent": "Kandinsky3 Latent Image",
}