"""
Sanity checks.
"""

from typing import Tuple


def _check_shear_input(angle_deg: float) -> None:
    if not abs(angle_deg) < 90.0:
        raise ValueError(
            "The shear angle_deg has to be inside the open range (-90.0, 90.0)."
        )


def _check_translate_input(
        image_shape: Tuple[int, int, int],
        translate_height: int,
        translate_width: int
) -> None:
    height_cond = -image_shape[0] < translate_height < image_shape[0]
    width_cond = -image_shape[1] < translate_width < image_shape[1]
    if not height_cond:
        raise ValueError(
            "The translate_height has to be inside the open range "
            "(-image_height, image_height). In this case that means "
            f"{-image_shape[0]} < translate_height < {image_shape[0]}."
        )
    if not width_cond:
        raise ValueError(
            "The translate_width has to be inside the open range "
            "(-image_width, image_width). In this case that means "
            f"{-image_shape[1]} < translate_width < {image_shape[1]}."
        )


def _check_crop_input(
        image_shape: Tuple[int, int, int],
        offset_height: int,
        offset_width: int
) -> None:
    height_cond = 0 <= offset_height < image_shape[0]
    width_cond = 0 <= offset_width < image_shape[1]
    if not height_cond:
        raise ValueError(
            "The offset_height has to be inside the half-open range "
            "[0, image_height). In this case that means "
            f"0 <= offset_height < {image_shape[0]}."
        )
    if not width_cond:
        raise ValueError(
            "The translate_width has to be inside the half-open range "
            "[0, image_width). In this case that means "
            f"0 <= offset_width < {image_shape[1]}."
        )


def _check_fraction_range(
        fraction_range: Tuple[float, float],
        min_value: float,
        max_value: float,
        name: str
) -> None:
    if not min_value < fraction_range[0] < fraction_range[1] < max_value:
        raise ValueError(
            f"The {name} should be provided as (min_fraction, max_fraction), "
            f"where {min_value} < min_fraction < max_fraction < {max_value}."
        )
