# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# with the following modifications:
# - It uses the patched version of `ddim_step_with_logprob` from `ddim_with_logprob.py`. As such, it only supports the
#   `ddim` scheduler.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.

from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from .ddim_with_logprob import ddim_step_with_logprob


@torch.no_grad()
def pipeline_with_logprob(
    model: torch.nn.Module,
    cond: Optional[torch.FloatTensor] = None,
    scheduler: Any = None,
    device: Optional[torch.device] = None,
    batch_size: int = 1,
    num_3d_points: int = 4656,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
    
    latents: Optional[torch.FloatTensor] = None,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        model (`torch.nn.Module`): The model used for generation.
        cond (`torch.FloatTensor`, *optional*):
            The conditional values (alpha, Ma, Cl, Cd) to generate aircrafts.
        scheduler (`Any`): The scheduler used for generation.
        device (`torch.device`, *optional*): The device to use for inference.
        batch_size (`int`, *optional*, defaults to 1): The batch size to use for inference.
        num_3d_points (`int`, *optional*, defaults to 4656): The number of 3D points to generate.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """




    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0


    # 1. Prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps


    latents = torch.randn((batch_size, 3, num_3d_points), device = device)

    # 2. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
    all_latents = [latents]
    all_log_probs = []
    # with self.progress_bar(total=num_inference_steps) as progress_bar:
    
    for i, t in enumerate(timesteps):

        # predict the noise residual

        batched_times = torch.full((batch_size,), t, device = device, dtype = torch.long)
        noise_pred = model(latents, cond, batched_times)



        # compute the previous noisy sample x_t -> x_t-1
        latents, log_prob = ddim_step_with_logprob(
            scheduler, noise_pred, t, latents, eta
        )

        all_latents.append(latents)
        all_log_probs.append(log_prob)

        # call the callback, if provided
        if i == len(timesteps) - 1 or (
            (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
        ):
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)


    image = latents
    has_nsfw_concept = None


    # Offload last model to CPU
    # if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
    #     self.final_offload_hook.offload()

    return image, has_nsfw_concept, all_latents, all_log_probs
