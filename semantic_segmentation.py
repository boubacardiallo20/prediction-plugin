"""Zero Shot Semantic Segmentation.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from importlib.util import find_spec
from PIL import Image
import torch

import fiftyone as fo
from fiftyone.core.models import Model

import os
import sys
import numpy as np
from pathlib import Path
from torchvision import transforms


class EXRSegModel(Model):
    def __init__(self, config):
        self.candidate_labels = config.get("categories", None)

        root_dir_path = str(Path(__file__).resolve().parents[2])
        if root_dir_path not in sys.path:
            sys.path.append(root_dir_path)

        # from models.model_factory import load_model

        # path_m = "assets/model_exr.pth"
        # weight_path = config.get("weight_path", path_m)
        # model_type = config.get("model_type", "bisenetv1")

        # # Step 1: Initialize the model
        # if os.path.isdir(weight_path):
        #     cfg_model_file = os.path.join(weight_path, "bisenetv1_config.yaml")
        # else:
        #     cfg_model_file = os.path.join(os.path.dirname(weight_path), "bisenetv1_config.yaml")

        # self.net = load_model(model_type)(cfg_model_file, mode="pred", weight_path=weight_path).net

        
        # self.preprocess = transforms.Compose([
        #     transforms.Resize((1024, 1024)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        # self.processor = EXRSegProcessor.from_pretrained(
        #     "CIDAS/clipseg-rd64-refined"
        # )
        # self.model = EXRSegForImageSegmentation.from_pretrained(
        #     "CIDAS/clipseg-rd64-refined"
        # )

    @property
    def media_type(self):
        return "image"

    def _predict(self, image):

        # inputs = self.processor(
        #     text=self.candidate_labels,
        #     images=[image] * len(self.candidate_labels),
        #     padding="max_length",
        #     return_tensors="pt",
        # )

        # input_tensor = self.preprocess(image).unsqueeze(0).cuda()
        # with torch.no_grad():
        #     outputs = self.net(input_tensor).cpu().detach().numpy()
        
        # with torch.no_grad():
        #     outputs = self.model(**inputs)

        # preds = outputs.logits.unsqueeze(1)

        # pylint: disable=no-member
        # mask = torch.argmax(preds, dim=0).squeeze().numpy()
        # print('mask', mask.shape)

        zero_mask = Image.new("1", image.size, 0)
        mask = np.array(zero_mask)

        return fo.Segmentation(mask=mask)

    def predict(self, args):

        image = Image.fromarray(args)
        predictions = self._predict(image)
        return predictions

    def predict_all(self, samples, args):
        pass


def EXRSeg_activator():
    return True


class CLIPSegZeroShotModel(Model):
    def __init__(self, config):
        self.candidate_labels = config.get("categories", None)

        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        self.processor = CLIPSegProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )

    @property
    def media_type(self):
        return "image"

    def _predict(self, image):
        inputs = self.processor(
            text=self.candidate_labels,
            images=[image] * len(self.candidate_labels),
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = outputs.logits.unsqueeze(1)
        # pylint: disable=no-member
        mask = torch.argmax(preds, dim=0).squeeze().numpy()
        return fo.Segmentation(mask=mask)

    def predict(self, args):
        image = Image.fromarray(args)
        predictions = self._predict(image)
        return predictions

    def predict_all(self, samples, args):
        pass


def CLIPSeg_activator():
    return find_spec("transformers") is not None


class GroupViTZeroShotModel(Model):
    def __init__(self, config):
        cats = config.get("categories", None)
        self.candidate_labels = [f"a photo of a {cat}" for cat in cats]

        from transformers import AutoProcessor, GroupViTModel

        self.processor = AutoProcessor.from_pretrained(
            "nvidia/groupvit-gccyfcc"
        )
        self.model = GroupViTModel.from_pretrained("nvidia/groupvit-gccyfcc")

    @property
    def media_type(self):
        return "image"

    def _predict(self, image):
        inputs = self.processor(
            text=self.candidate_labels,
            images=image,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**inputs, output_segmentation=True)
        preds = outputs.segmentation_logits.squeeze()
        # pylint: disable=no-member
        mask = torch.argmax(preds, dim=0).numpy()
        return fo.Segmentation(mask=mask)

    def predict(self, args):
        image = Image.fromarray(args)
        image = image.resize((224, 224))
        predictions = self._predict(image)
        return predictions

    def predict_all(self, samples, args):
        pass


def GroupViT_activator():
    return find_spec("transformers") is not None


SEMANTIC_SEGMENTATION_MODELS = {
    "CLIPSeg": {
        "activator": CLIPSeg_activator,
        "model": CLIPSegZeroShotModel,
        "name": "CLIPSeg",
    },
    "GroupViT": {
        "activator": GroupViT_activator,
        "model": GroupViTZeroShotModel,
        "name": "GroupViT",
    },
    "EXRSeg": {
        "activator": EXRSeg_activator,
        "model": EXRSegModel,
        "name": "EXRSeg",
    },
}


def _get_model(model_name, config):
    return SEMANTIC_SEGMENTATION_MODELS[model_name]["model"](config)


def run_zero_shot_semantic_segmentation(
    dataset, model_name, label_field, categories, **kwargs
):
    if "other" not in categories:
        categories.append("other")
    config = {"categories": categories}
    model = _get_model(model_name, config)
    dataset.apply_model(model, label_field=label_field)

    dataset.mask_targets[label_field] = {
        i: label for i, label in enumerate(categories)
    }
    dataset.save()
