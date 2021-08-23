"""
Sanity checks.
"""

from typing import Optional, Tuple


def _check_shear_input(angle_deg: float) -> None:
    if not abs(angle_deg) < 90.0:
        raise ValueError(
            "The shear angle_deg has to be inside the open interval "
            "(-90.0, 90.0)."
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
            "The translate_height has to be inside the open interval "
            "(-image_height, image_height). In this case that means "
            f"{-image_shape[0]} < translate_height < {image_shape[0]}."
        )
    if not width_cond:
        raise ValueError(
            "The translate_width has to be inside the open interval "
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
            "The offset_height has to be inside the half-open interval "
            "[0, image_height). In this case that means "
            f"0 <= offset_height < {image_shape[0]}."
        )
    if not width_cond:
        raise ValueError(
            "The translate_width has to be inside the half-open interval "
            "[0, image_width). In this case that means "
            f"0 <= offset_width < {image_shape[1]}."
        )


def _check_input_range(
        input_range: Tuple[float, float],
        limit_open_interval: Optional[Tuple[float, float]],
        input_name: str
) -> None:
    if limit_open_interval is None:
        if not input_range[0] < input_range[1]:
            raise ValueError(
                f"The {input_name} should be provided as "
                f"(min_value, max_value), where min_value < max_value."
            )
        return
    lower_limit, upper_limit = limit_open_interval
    if not lower_limit < input_range[0] < input_range[1] < upper_limit:
        raise ValueError(
            f"The {input_name} should be provided as "
            f"(min_value, max_value), "
            f"where {lower_limit} < min_value < max_value < {upper_limit}."
        )
