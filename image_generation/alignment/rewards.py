# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
from PIL import Image
import io
import numpy as np
import time
import requests

import torch
import torch.distributed as dist

from utils.distributed import get_local_rank


short_names = {
    "jpeg_incompressibility": "incomp",
    "jpeg_compressibility": "comp",
    "aesthetic_score": "aes",
    "imagereward": "imgr",
    "llava_strict_satisfaction": "llava_strict",
    "llava_bertscore": "llava",
}
use_prompt = {
    "jpeg_incompressibility": False,
    "jpeg_compressibility": False,
    "aesthetic_score": False,
    "imagereward": True,
}

def jpeg_incompressibility(dtype=torch.float32, device="cuda"):
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        sizes = np.array(sizes)
        return torch.from_numpy(sizes).cuda(), {}

    return _fn


def jpeg_compressibility(dtype=torch.float32, device="cuda"):
    jpeg_fn = jpeg_incompressibility(dtype, device)

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score(dtype=torch.float32, device="cuda", distributed=True):
    from alignment.aesthetic_scorer import AestheticScorer
    # why cuda() doesn't cause a bug?
    scorer = AestheticScorer(dtype=torch.float32, distributed=distributed).cuda() # ignore type;

    # @torch.no_grad() # original AestheticScorer already has no_grad()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def differentiable_aesthetic_score(dtype=torch.float32, device="cuda", distributed=True):
    from alignment.aesthetic_scorer import DifferentiableAestheticScorer
    # why cuda() doesn't cause a bug?
    scorer = DifferentiableAestheticScorer(dtype=torch.float32, distributed=distributed).cuda() # ignore type;

    # @torch.no_grad() # original AestheticScorer already has no_grad()
    def _fn(images, prompts, metadata):
        assert isinstance(images, torch.Tensor)
        if images.min() < 0: # normalize unnormalized images
            images = ((images / 2) + 0.5).clamp(0, 1)
        scores = scorer(images)
        return scores, {}

    return _fn

# For ImageReward
import ImageReward as RM
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def imagereward(dtype=torch.float32, device="cuda"):
    # aesthetic = RM.load_score("Aesthetic", device=device)
    if get_local_rank() == 0:  # only download once
        reward_model = RM.load("ImageReward-v1.0")
    dist.barrier()
    reward_model = RM.load("ImageReward-v1.0")
    reward_model.to(dtype).to(device)

    rm_preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _fn(images, prompts, metadata):
        dic = reward_model.blip.tokenizer(prompts,
                padding='max_length', truncation=True,  return_tensors="pt",
                max_length=reward_model.blip.tokenizer.model_max_length) # max_length=512
        device = images.device
        input_ids, attention_mask = dic.input_ids.to(device), dic.attention_mask.to(device)
        reward = reward_model.score_gard(input_ids, attention_mask, rm_preprocess(images))
        return reward.reshape(images.shape[0]).float(), {} # bf16 -> f32

    return _fn


def differentiable_imagereward(dtype=torch.float32, device="cuda"):
    # aesthetic = RM.load_score("Aesthetic", device=device)
    if get_local_rank() == 0:  # only download once
        reward_model = RM.load("ImageReward-v1.0")
    dist.barrier()
    reward_model = RM.load("ImageReward-v1.0")
    reward_model.to(dtype).to(device)

    rm_preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _fn(images, prompts, metadata):
        dic = reward_model.blip.tokenizer(prompts,
                padding='max_length', truncation=True,  return_tensors="pt",
                max_length=reward_model.blip.tokenizer.model_max_length) # max_length=512
        device = images.device
        input_ids, attention_mask = dic.input_ids.to(device), dic.attention_mask.to(device)
        reward = reward_model.score_gard(input_ids, attention_mask, rm_preprocess(images))
        return reward.reshape(images.shape[0]).float(), {} # bf16 -> f32

    return _fn

# For HPSv2 reward
# https://github.com/tgxs002/HPSv2/blob/master/hpsv2/img_score.py
def hpscore(dtype=torch.float32, device=torch.device('cuda')):
    import huggingface_hub
    import torchvision.transforms.functional as F
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.utils import root_path, hps_version_map
    from hpsv2.src.open_clip.transform import MaskAwareNormalize, ResizeMaxSize

    hps_version = "v2.1"
    model_dict = {}
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            # device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

    # initialize_model()
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()

    def _fn(images, prompts, metadata):
        image_size = model.visual.image_size[0]
        transforms = Compose([
            ResizeMaxSize(image_size, fill=0), # resize to 224x224
            MaskAwareNormalize(mean=model.visual.image_mean, std=model.visual.image_std),
        ])

        # these are not numerically identical, because
        # F.to_tensor(F.to_pil_image(img)) != img
        # due to RGB round up (in PIL it is 0~255 integer)

        # images = torch.stack([preprocess_val(F.to_pil_image(img)) for img in images.float()]).to(device)
        images = torch.stack([transforms(img) for img in images])
        texts = tokenizer(prompts).to(device=device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=dtype):
            outputs = model(images, texts)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T # (bs, bs)
            hps_score = torch.diagonal(logits_per_image) # (bs,)

        return hps_score, {}

    return _fn

differentiable_hpscore = hpscore

def llava_strict_satisfaction(dtype=torch.float32, device="cuda"):
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore(dtype=torch.float32, device="cuda"):
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn
