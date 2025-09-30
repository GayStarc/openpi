import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy."""
    return {
        "image_front": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "image_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "image_wrist_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "target_keyframe_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(14),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaDualInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0
    # Optional prompt image keys that should be populated from the dataset.
    prompt_image_keys: tuple[str, ...] = ("target_keyframe_image")

    def __call__(self, data: dict) -> dict:
        state = data["state"]
        state = transforms.pad_to_dim(state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["image_front"])
        wrist_image = _parse_image(data["image_wrist"])
        right_image = _parse_image(data["image_wrist_right"])

        match self.model_type:
            case _model.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, right_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, right_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if self.prompt_image_keys:
            prompt_images: dict[str, np.ndarray] = {}
            prompt_image_masks: dict[str, np.bool_] = {}
            raw_prompt_images = data.pop("target_keyframe_image", None)
            if raw_prompt_images is not None:
                raw_prompt_images = np.asarray(raw_prompt_images)
                if raw_prompt_images.ndim == 3:
                    raw_prompt_images = raw_prompt_images[np.newaxis, ...]
                elif raw_prompt_images.ndim > 4:
                    raw_prompt_images = raw_prompt_images.reshape((-1,) + raw_prompt_images.shape[-3:])
            blank_image = np.zeros_like(base_image)
            for idx, key in enumerate(self.prompt_image_keys):
                image_source = None
                if raw_prompt_images is not None and idx < len(raw_prompt_images):
                    image_source = raw_prompt_images[idx]
                elif key in data:
                    image_source = data.pop(key)
                if image_source is not None:
                    prompt_images[key] = _parse_image(image_source)
                    prompt_image_masks[key] = np.True_
                else:
                    prompt_images[key] = blank_image
                    prompt_image_masks[key] = np.False_
            inputs["prompt_image"] = prompt_images
            inputs["prompt_image_mask"] = prompt_image_masks

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaDualOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :14])}
