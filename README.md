# Kandinsky 2.2 ComfUI Plugin

Use the models of Kandinsky 2.2 published on 🤗 HuggingFace.

## Installation

For the easiest install experience, install the [Comfyui Manager](https://github.com/ltdrdata/ComfyUI-Manager) and 
use that to automate the installation process.

Otherwise, to manually install, simply clone the repo into the custom_nodes directory with this command:
```
git clone https://github.com/vsevolod-oparin/comfyui-kandinsky22
```
and install the requirements using:
```
python -s -m pip install -r requirements.txt
```
If you are using a venv, make sure you have it activated before installation and use:
```
pip install -r requirements.txt
```

### Download Models

Follow to `models/checkpoints` in ComfyUI directory and run the command
```
git clone --depth 1 <HF repository>
```
E.g. to download default pipeline run the following
```
git clone --depth 1 https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder
git clone --depth 1 https://huggingface.co/kandinsky-community/kandinsky-2-2-prior
```

**Note 1**: Git won't show much of the progress. You'll need to wait till the model will be downloaded.
**Note 2**: `--depth` argument can be skipped, but you're risking to download a lot of unnecessary data.


## Examples and Workflows

Here will be workflow examples

## Acknowledgments

**A special thanks to:**

-The developers of [Deforum](https://github.com/deforum-art/sd-webui-deforum) for providing code for these nodes and being overall awesome people!

-Comfyanonamous and the rest of the [ComfyUI](https://github.com/comfyanonymous/ComfyUI/tree/master) contributors for a fantastic UI!

