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
