from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch

from meissonic.pipeline import _prepare_latent_image_ids


@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Optional[Union[List[str], str]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    softmax_temperature: float = 1.0,
    num_inference_steps: int = 48,
    guidance_scale: float = 9.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.IntTensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_encoder_hidden_states: Optional[torch.Tensor] = None,
    output_type="pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    micro_conditioning_aesthetic_score: int = 6,
    micro_conditioning_crop_coord: Tuple[int, int] = (0, 0),
    temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),

    batch_size = None, dtype=None,
    device = None,
    return_tfoutput = False,
):
    """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.transformer.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 16):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.IntTensor`, *optional*):
                Pre-generated tokens representing latent vectors in `self.vqvae`, to be used as inputs for image
                gneration. If not provided, the starting latents will be completely masked.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument. A single vector from the
                pooled and projected final hidden states.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                Pre-generated penultimate hidden states from the text encoder providing additional text conditioning.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            negative_encoder_hidden_states (`torch.Tensor`, *optional*):
                Analogous to `encoder_hidden_states` for the positive prompt.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            micro_conditioning_aesthetic_score (`int`, *optional*, defaults to 6):
                The targeted aesthetic score according to the laion aesthetic classifier. See
                https://laion.ai/blog/laion-aesthetics/ and the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            micro_conditioning_crop_coord (`Tuple[int]`, *optional*, defaults to (0, 0)):
                The targeted height, width crop coordinates. See the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            temperature (`Union[int, Tuple[int, int], List[int]]`, *optional*, defaults to (2, 0)):
                Configures the temperature scheduler on `self.scheduler` see `Scheduler#set_timesteps`.

        Examples:

        Returns:
            [`~pipelines.pipeline_utils.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.pipeline_utils.ImagePipelineOutput`] is returned, otherwise a
                `tuple` is returned where the first element is a list with the generated images.
        """
    
    if (prompt_embeds is not None and encoder_hidden_states is None) or (
        prompt_embeds is None and encoder_hidden_states is not None
    ):
        raise ValueError("pass either both `prompt_embeds` and `encoder_hidden_states` or neither")

    if (negative_prompt_embeds is not None and negative_encoder_hidden_states is None) or (
        negative_prompt_embeds is None and negative_encoder_hidden_states is not None
    ):
        raise ValueError(
            "pass either both `negatve_prompt_embeds` and `negative_encoder_hidden_states` or neither"
        )

    if (prompt is None and prompt_embeds is None) or (prompt is not None and prompt_embeds is not None):
        raise ValueError("pass only one of `prompt` or `prompt_embeds`")

    if isinstance(prompt, str):
        prompt = [prompt]

    if batch_size is None:
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
    batch_size = batch_size * num_images_per_prompt

    if height is None:
        height = self.transformer.config.sample_size * self.vae_scale_factor
    if width is None:
        width = self.transformer.config.sample_size * self.vae_scale_factor

    if device is None:
        device = self._execution_device

    if prompt_embeds is None:
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length, # 77
        ).input_ids.to(device)
        outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
        prompt_embeds = outputs.text_embeds
        encoder_hidden_states = outputs.hidden_states[-2]
    prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1)
    encoder_hidden_states = encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

    if guidance_scale > 1.0:
        if negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = [""] * len(prompt)

            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]

            input_ids = self.tokenizer(
                negative_prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length, # 77
            ).input_ids.to(self._execution_device)
            outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
            negative_prompt_embeds = outputs.text_embeds
            negative_encoder_hidden_states = outputs.hidden_states[-2]
        negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1)
        negative_encoder_hidden_states = negative_encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

        prompt_embeds = torch.concat([negative_prompt_embeds, prompt_embeds])
        encoder_hidden_states = torch.concat([negative_encoder_hidden_states, encoder_hidden_states])


    # Note that the micro conditionings _do_ flip the order of width, height for the original size
    # and the crop coordinates. This is how it was done in the original code base
    micro_conds = torch.tensor(
        [
            width,
            height,
            micro_conditioning_crop_coord[0],
            micro_conditioning_crop_coord[1],
            micro_conditioning_aesthetic_score,
        ],
        device=device,
        dtype=encoder_hidden_states.dtype,
    )
    micro_conds = micro_conds.unsqueeze(0)
    micro_conds = micro_conds.expand(2 * batch_size if guidance_scale > 1.0 else batch_size, -1)

    shape = (batch_size, height // self.vae_scale_factor, width // self.vae_scale_factor)
    if latents is None:
        latents = torch.full(
            shape, self.scheduler.config.mask_token_id, dtype=torch.long, device=device
        )

    self.scheduler.set_timesteps(num_inference_steps, temperature, device)
    timesteps = self.scheduler.timesteps

    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

    all_latents = [latents]
    tf_outputs = []
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if guidance_scale > 1.0:
                model_input = torch.cat([latents] * 2)
            else:
                model_input = latents

            if height == 1024: #args.resolution == 1024:
                img_ids = _prepare_latent_image_ids(model_input.shape[0], model_input.shape[-2], model_input.shape[-1], model_input.device, model_input.dtype)
            else:
                img_ids = _prepare_latent_image_ids(model_input.shape[0], 2*model_input.shape[-2], 2*model_input.shape[-1],model_input.device, model_input.dtype)
            txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            
            model_output = self.transformer(
                hidden_states = model_input,
                micro_conds=micro_conds,
                pooled_projections=prompt_embeds,
                encoder_hidden_states=encoder_hidden_states,
                img_ids = img_ids,
                txt_ids = txt_ids,
                timestep = torch.tensor([t], device=model_input.device, dtype=torch.long),
            )

            if guidance_scale > 1.0:
                uncond_logits, cond_logits = model_output.chunk(2)
                model_output = uncond_logits + guidance_scale * (cond_logits - uncond_logits)

            if return_tfoutput:
                tf_outputs.append(model_output.detach().cpu())

            latents = self.scheduler.step(
                model_output=model_output * softmax_temperature,    # rescale for better exploration
                timestep=t,
                sample=latents,
                generator=generator,
            ).prev_sample

            all_latents.append(latents)

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    if output_type == "latent":
        output = latents
    else:
        needs_upcasting = self.vqvae.dtype == torch.float16 and self.vqvae.config.force_upcast

        if needs_upcasting: self.vqvae.float()
        image = self.vqvae.decode(
            latents,
            force_not_quantize=True,
            shape=(
                batch_size,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
                self.vqvae.config.latent_channels,
            ),
        ).sample.clip(0, 1)
        image = self.image_processor.postprocess(image, output_type)
        if needs_upcasting: self.vqvae.half()

    if return_tfoutput:
        return image, all_latents, tf_outputs

    return image, all_latents