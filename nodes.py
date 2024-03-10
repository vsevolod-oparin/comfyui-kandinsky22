import folder_paths
from pathlib import Path
from typing import List

import torch

from .logic.kandinsky22decoder import load_decoder_kandinsky22, prepare_latents, movq_decode, unet_decode, \
    prepare_latents_on_img, unet_img2img_decode, unet_hint_decode, combine_hint_latents
from .logic.kandinsky22prior import load_prior_kandinsky22, encode_image, encode_text

MANIFEST = {
    "name": "Kandinsky 2.2 Nodes",
    "version": (0,0,1),
    "author": "Seva Oparin",
    "project": "https://github.com/vsevolod-oparin/",
    "description": "Help nodes to use Kandinsky 2.2",
}


def get_checkpoint_folder(subtoken: str, checkpoint_dir: str = "checkpoints") -> List[str]:
    folder_list = folder_paths.get_filename_list_(checkpoint_dir)[1]
    return [
        folder for folder in folder_list
        if subtoken in Path(folder).name and Path(folder).parent.name == checkpoint_dir
    ]


class Kandinsky22PriorLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt": (get_checkpoint_folder("kandinsky"), ),
            }
        }

    RETURN_TYPES = ("IMAGE_ENCODER", "TEXT_ENCODER")

    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt):
        ckpt_pth = Path(ckpt)
        return load_prior_kandinsky22(ckpt_pth)


class Kandinsky22DecoderLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt": (get_checkpoint_folder("kandinsky"), ),
            }
        }

    RETURN_TYPES = ("MOVQ", "DECODER", "LATENT_INFO")

    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt):
        ckpt_pth = Path(ckpt)
        return load_decoder_kandinsky22(ckpt_pth)


class Kandinsky22Latents:
    MAX_RESOLUTION = 8192

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lat_info": ("LATENT_INFO", ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": Kandinsky22Latents.MAX_RESOLUTION, "step": 8}),
                "width": ("INT", {"default": 512, "min": 64, "max": Kandinsky22Latents.MAX_RESOLUTION, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT", )

    FUNCTION = "prepare_latents"
    CATEGORY = "latents"

    def prepare_latents(self, lat_info, batch_size, height, width, seed):
        shape = batch_size, height, width
        return prepare_latents(shape, lat_info, seed),


class Kandinsky22ImgLatents:
    MAX_RESOLUTION = 8192

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lat_info": ("LATENT_INFO", ),
                "image": ("IMAGE", ),
                "movq": ("MOVQ", ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": Kandinsky22ImgLatents.MAX_RESOLUTION, "step": 8}),
                "width": ("INT", {"default": 512, "min": 64, "max": Kandinsky22ImgLatents.MAX_RESOLUTION, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT", )

    FUNCTION = "prepare_latents"
    CATEGORY = "latents"

    def prepare_latents(self, image, movq, lat_info, batch_size, height, width, seed):
        shape = batch_size, height, width
        return prepare_latents_on_img(image, movq, shape, lat_info, seed),


class Kandinsky22HintCombiner:
    MAX_RESOLUTION = 8192

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hint": ("IMAGE",),
                "latents": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", )

    FUNCTION = "prepare_latents"
    CATEGORY = "latents"

    def prepare_latents(self, hint, latents):
        return combine_hint_latents(hint, latents),


class Kandinsky22UnetDecoder:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder": ("DECODER", ),
                "latents": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.0001}),
                "image_embeds": ("PRIOR_LATENT", ),
                "negative_image_embeds": ("PRIOR_LATENT", ),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "guidance_scale": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT", )

    FUNCTION = "decode"
    CATEGORY = "decoder"

    def decode(self, decoder, image_embeds, negative_image_embeds, latents, num_inference_steps, guidance_scale, seed, strength):
        return unet_decode(
            decoder, image_embeds, negative_image_embeds, latents,
            seed, num_inference_steps, guidance_scale, strength
        ),


class Kandinsky22HintUnetDecoder:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder": ("DECODER", ),
                "hint": ("IMAGE", ),
                "latents": ("LATENT",),
                "image_embeds": ("PRIOR_LATENT", ),
                "negative_image_embeds": ("PRIOR_LATENT", ),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "guidance_scale": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT", )

    FUNCTION = "decode"
    CATEGORY = "decoder"

    def decode(self, decoder, image_embeds, negative_image_embeds, latents, num_inference_steps, guidance_scale, seed, hint):
        return unet_hint_decode(
            decoder, image_embeds, negative_image_embeds, latents,
            seed, hint, num_inference_steps, guidance_scale
        ),


class Kandinsky22ImgUnetDecoder:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder": ("DECODER", ),
                "latents": ("LATENT",),
                "strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.0001}),
                "image_embeds": ("PRIOR_LATENT", ),
                "negative_image_embeds": ("PRIOR_LATENT", ),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "guidance_scale": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT", )

    FUNCTION = "decode"
    CATEGORY = "decoder"

    def decode(self, decoder, image_embeds, negative_image_embeds, latents, num_inference_steps, guidance_scale, seed, strength):
        return unet_img2img_decode(
            decoder, image_embeds, negative_image_embeds, latents,
            seed, num_inference_steps, guidance_scale, strength
        ),


class Kandinsky22MovqDecoder:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "movq": ("MOVQ",),
                "latents": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )

    FUNCTION = "decode"
    CATEGORY = "decoder"

    def decode(self, latents, movq):
        return movq_decode(latents, movq),


class Kandinsky22ImageEncoder:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoder": ("IMAGE_ENCODER", ),
                "image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("PRIOR_LATENT", )

    FUNCTION = "image_encode"
    CATEGORY = "conditioning"

    def image_encode(self, encoder, image):
        return encode_image(encoder, image),


class Kandinsky22PriorAveraging2:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "in1": ("PRIOR_LATENT", ),
                "w1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "in2": ("PRIOR_LATENT", ),
                "w2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("PRIOR_LATENT", )

    FUNCTION = "weight"
    CATEGORY = "conditioning"

    def weight(self,
               in1, w1,
               in2, w2):
        return torch.cat([
            in1.unsqueeze(0) * w1,
            in2.unsqueeze(0) * w2,
        ]).sum(dim=0),


class Kandinsky22PriorAveraging3:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "in1": ("PRIOR_LATENT", ),
                "w1": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "in2": ("PRIOR_LATENT", ),
                "w2": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "in3": ("PRIOR_LATENT", ),
                "w3": ("FLOAT", {"default": 0.34, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("PRIOR_LATENT", )

    FUNCTION = "weight"
    CATEGORY = "conditioning"

    def weight(self,
               in1, w1,
               in2, w2,
               in3, w3):
        return torch.cat([
            in1.unsqueeze(0) * w1,
            in2.unsqueeze(0) * w2,
            in3.unsqueeze(0) * w3,
        ]).sum(dim=0),


class Kandinsky22PriorAveraging4:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "in1": ("PRIOR_LATENT", ),
                "w1": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "in2": ("PRIOR_LATENT", ),
                "w2": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "in3": ("PRIOR_LATENT", ),
                "w3": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "in4": ("PRIOR_LATENT", ),
                "w4": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("PRIOR_LATENT", )

    FUNCTION = "weight"
    CATEGORY = "conditioning"

    def weight(self,
               in1, w1,
               in2, w2,
               in3, w3,
               in4, w4):
        return torch.cat([
            in1.unsqueeze(0) * w1,
            in2.unsqueeze(0) * w2,
            in3.unsqueeze(0) * w3,
            in4.unsqueeze(0) * w4,
        ]).sum(dim=0),


class Kandinsky22TextEncoder:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("TEXT_ENCODER",),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "guidance_scale": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prior": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("PRIOR_LATENT", "PRIOR_LATENT")
    RETURN_NAMES = ("image_embeds", "negative_image_embeds")

    FUNCTION = "text_encode"
    CATEGORY = "conditioning"

    def text_encode(self, text_encoder, num_inference_steps, guidance_scale, seed, prompt, negative_prior):
        return encode_text(
            text_encoder, num_inference_steps, guidance_scale, seed, prompt, negative_prior
        )


NODE_CLASS_MAPPINGS = {
    "comfy-kandinsky22-prior-loader": Kandinsky22PriorLoader,
    "comfy-kandinsky22-decoder-loader": Kandinsky22DecoderLoader,
    "comfy-kandinsky22-image-encoder": Kandinsky22ImageEncoder,
    "comfy-kandinsky22-text-encoder": Kandinsky22TextEncoder,
    "comfy-kandinsky22-latents": Kandinsky22Latents,
    "comfy-kandinsky22-hint-combiner": Kandinsky22HintCombiner,
    "comfy-kandinsky22-img-latents": Kandinsky22ImgLatents,
    "comfy-kandinsky22-movq-decoder": Kandinsky22MovqDecoder,
    "comfy-kandinsky22-unet-decoder": Kandinsky22UnetDecoder,
    "comfy-kandinsky22-hint-unet-decoder": Kandinsky22HintUnetDecoder,
    "comfy-kandinsky22-img-unet-decoder": Kandinsky22ImgUnetDecoder,
    "comfy-kandinsky22-prior-averaging-2": Kandinsky22PriorAveraging2,
    "comfy-kandinsky22-prior-averaging-3": Kandinsky22PriorAveraging3,
    "comfy-kandinsky22-prior-averaging-4": Kandinsky22PriorAveraging4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "comfy-kandinsky22-prior-loader": "Kandinsky2.2 Prior Loader",
    "comfy-kandinsky22-decoder-loader": "Kandinsky2.2 Decoder Loader",
    "comfy-kandinsky22-image-encoder": "Kandinsky2.2 Image Encoder",
    "comfy-kandinsky22-text-encoder": "Kandinsky2.2 Text Encoder",
    "comfy-kandinsky22-latents": "Kandinsky2.2 Latents",
    "comfy-kandinsky22-hint-combiner": "Kandinsky2.2 Hint Combiner",
    "comfy-kandinsky22-img-latents": "Kandinsky2.2 Image Latents",
    "comfy-kandinsky22-movq-decoder": "Kandinsky2.2 MovQ Decoder",
    "comfy-kandinsky22-unet-decoder": "Kandinsky2.2 Unet Decoder",
    "comfy-kandinsky22-hint-unet-decoder": "Kandinsky2.2 Hint Unet Decoder",
    "comfy-kandinsky22-img-unet-decoder": "Kandinsky2.2 Img2Img Unet Decoder",
    "comfy-kandinsky22-prior-averaging-2": "Kandinsky2.2 Prior 2-Averaging",
    "comfy-kandinsky22-prior-averaging-3": "Kandinsky2.2 Prior 3-Averaging",
    "comfy-kandinsky22-prior-averaging-4": "Kandinsky2.2 Prior 4-Averaging",
}