from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from model.transformer_3d import PointDiffusionTransformer
import torch
import wandb
import pickle
from functools import partial
import tqdm
import tempfile

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def load_model(config_pretrained):
    """
    Load the pretrained model.
    Args:
        config_pretrained: the configuration of the pretrained model.
    Returns:    
        model: the pretrained model.
    """
    # modol for 3d wings design
    model = PointDiffusionTransformer(
        dtype=torch.float32,
        n_ctx = config_pretrained.num_3d_points, # number of points in the 3D point cloud
        heads = config_pretrained.n_head, # number of heads in transformer
        time_token_cond = True # condition on time token
    )

    # load the pretrained model and change keys in the state dict to match the current model by removing the prefix 'model.' if it exists
    state_dict = torch.load(config_pretrained.model_path)["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("model.", "")
        new_state_dict[k] = v
    # remove the keys that do not exist in the current model
    state_dict = {k: v for k, v in new_state_dict.items() if k in model.state_dict()}
    model.load_state_dict(state_dict)
    
    return model


def normalize(x, min_max, key):
    """
    Normalize the input data x to [-1, 1] using the min and max values of the key.
    """
    return 2 * (x - min_max[key + "_min"]) / (min_max[key + "_max"] - min_max[key + "_min"]) - 1

def prepare_condition(config_conditions, batch_size):
    """
    Prepare the conditions for the aircraft design.
    Args:
        config_conditions: the configuration of the conditions.
        batch_size: the batch size of the conditions.
    Returns:
        batch_normalized_cond: the normalized conditions.
    
    """
    normdict = pickle.load(open(config_conditions.normdict_path, "rb"))
    normalized_alpha = normalize(config_conditions.alpha, normdict, "alpha")
    normalized_mach = normalize(config_conditions.Ma, normdict, "Ma")
    normalized_label_condition = torch.tensor([normalized_alpha, normalized_mach], dtype = torch.float32)            
    batch_normalized_cond = torch.tile(normalized_label_condition, [batch_size, 1])
    if config_conditions.cond_force:
        normalized_lift = normalize(config_conditions.Cl, normdict, "Cl")
        normalized_drag = normalize(config_conditions.Cd, normdict, "Cd")
        normalized_force = torch.tensor([normalized_lift, normalized_drag], dtype = torch.float32)
        batch_normalized_force_cond = torch.tile(normalized_force, [batch_size, 1])
        batch_normalized_cond = torch.cat([batch_normalized_cond, batch_normalized_force_cond], dim = 1)
        
    return batch_normalized_cond

def reward_fn(images):
    """
    Reward function to compute lift / drag of the aircraft.
    Args:
        images: the generated images of the aircraft.
    Returns:
        rewards: the rewards of the aircraft.
    """
    # TODO: implement reward function for aircraft design
    import random
    batch_size = images.shape[0]
    
    return [random.random() for _ in range(batch_size)]

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # if config.resume_from:
    #     config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
    #     if "checkpoint_" not in os.path.basename(config.resume_from):
    #         # get the most recent checkpoint in this directory
    #         checkpoints = list(
    #             filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
    #         )
    #         if len(checkpoints) == 0:
    #             raise ValueError(f"No checkpoints found in {config.resume_from}")
    #         config.resume_from = os.path.join(
    #             config.resume_from,
    #             sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
    #         )
    model = load_model(config.pretrained)

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
        * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    
    # freeze parameters of models to save more memory
    model.requires_grad_(not config.use_lora)

    # switch to DDIM scheduler
    scheduler = DDIMScheduler()
    scheduler.num_inference_steps = config.sample.num_steps

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move model to device and cast to inference_dtype
    if config.use_lora:
        model.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # TODO: set correct lora layers
        # Set correct lora layers
        lora_attn_procs = {}
        for name in model.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else model.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = model.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(model.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = model.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        model.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return model(*args, **kwargs)

        model = _Wrapper(model.attn_processors)

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            model.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # model.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model,
                revision=config.pretrained.revision,
                subfolder="unet",
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    model, optimizer = accelerator.prepare(model, optimizer)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = (
        config.sample.batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        model.eval()
        samples = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate conditions
            cond = prepare_condition(config.conditions, config.sample.batch_size).to(accelerator.device)

            # sample
            with autocast():
                images, _, latents, log_probs = pipeline_with_logprob(
                    model,
                    cond=cond,
                    scheduler=scheduler,
                    device=accelerator.device,
                    batch_size = config.sample.batch_size,
                    num_3d_points = config.pretrained.num_3d_points,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = scheduler.timesteps.repeat(
                config.sample.batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            # TODO: run the reward function asynchronously
            # rewards = executor.submit(reward_fn, images)
            rewards = reward_fn(images)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "cond": cond,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            sample["rewards"] = torch.as_tensor(sample["rewards"], device=accelerator.device)
        
        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        
        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        # log rewards and images
        accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        del samples["rewards"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == config.sample.batch_size * config.sample.num_batches_per_epoch
        )
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            model.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                cond = prepare_condition(config.conditions, config.sample.batch_size).to(accelerator.device)
                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(model):
                        with autocast():
                            noise_pred = model(
                                sample["latents"][:, j],
                                cond,
                                sample["timesteps"][:, j]
                            )

                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(
                                scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                model.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()


if __name__ == "__main__":
    app.run(main)
