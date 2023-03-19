import typing as tp
import warnings

import numpy as np
from PIL import Image


# TODO: Handle such logger behaviour that maintains 'always'.
warnings.simplefilter('ignore')  


class FacePointsRandomCropTransform:
    def __init__(
            self,
            crop_sizes: tp.Tuple[float, float] = (1.0, 1.0),
            p: float = 0.5,
            skip: bool = False
    ):
        assert (0.0 <= crop_sizes[0] <= 1.0) and (0.0 <= crop_sizes[1] <= 1.0), \
            "Crop sizes if relative, so each of them must be between 0.0 and 1.0"
        assert 0.0 <= p <= 1.0, "Probability must be between 0.0 and 1.0"
        self._crop_sizes = crop_sizes
        self._p = p
        self._skip = skip

    def __call__(
            self,
            image: Image.Image,
            points: np.ndarray
    ) -> tp.Tuple[Image.Image, np.ndarray]:
        if (np.random.rand() < self._p) and (
                (self._crop_sizes[0] < 1.0) or (self._crop_sizes[1] < 1.0)
        ):
            new_height = round(self._crop_sizes[0] * image.height)
            new_width = round(self._crop_sizes[1] * image.width)

            left_image_border = np.random.randint(0, image.width - new_width)
            down_image_border = np.random.randint(0, image.height - new_height)
            
            croped_image = image.crop(
                (
                    left_image_border,
                    down_image_border,
                    left_image_border + new_width,
                    down_image_border + new_height
                )
            )
            
            croped_points = np.empty_like(points)
            croped_points[::2] = points[::2] - left_image_border
            croped_points[1::2] = points[1::2] - down_image_border
            
            does_points_go_too_left = croped_points[::2].min() < 0.0
            does_points_go_too_right = croped_points[::2].max() >= croped_image.width
            does_points_go_too_down = croped_points[1::2].min() < 0.0
            does_points_go_too_up = croped_points[1::2].max() >= croped_image.height
            
            good_condition = (
                not does_points_go_too_left
                and not does_points_go_too_right
                and not does_points_go_too_down
                and not does_points_go_too_up
            )
            
            if self._skip:
                warnings.warn("Cropping skiped.", category=Warning)
                if good_condition:
                    image = croped_image
                    points = croped_points
            else:
                assert good_condition, \
                    "Too rough crop sizes, points go out of image borders."                
                image = croped_image
                points = croped_points
            
        return image, points


class FacePointsRandomHorizontalFlipTransform:
    def __init__(self, p: float = 0.5):
        assert 0.0 <= p <= 1.0, "Probability must be between 0.0 and 1.0"
        self._p = p
        self._pairs_correspondence = {
            0: 3, 1: 2, 2: 1, 3: 0,   4: 9,   5: 8,   6: 7,
            7: 6, 8: 5, 9: 4, 10: 10, 11: 13, 12: 12, 13: 11
        }

    
    def _swap_points(
            self,
            arr: np.ndarray
    ) -> np.ndarray:
        blank_arr = np.empty_like(arr)
        for old_position, new_position in self._pairs_correspondence.items():
            blank_arr[new_position * 2 : new_position * 2 + 2] = \
                arr[old_position * 2 : old_position * 2 + 2]
        return blank_arr


    def __call__(
            self,
            image: Image.Image,
            points: np.ndarray
    ) -> tp.Tuple[Image.Image, np.ndarray]:
        if np.random.rand() < self._p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            points[::2] = image.width - points[::2]
            points = self._swap_points(points)
        return image, points

