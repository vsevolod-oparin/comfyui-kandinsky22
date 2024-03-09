import comfy.utils


def get_vanilla_callback(total_steps):
    pbar = comfy.utils.ProgressBar(total_steps)

    def callback(step: int, total_steps: int):
        pbar.update_absolute(step + 1, total_steps)

    return callback
