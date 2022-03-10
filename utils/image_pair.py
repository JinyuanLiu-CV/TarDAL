from pathlib import Path

import cv2
import kornia
import numpy
from torch import Tensor


class ImagePair:

    def __init__(self, ir_path: Path, vi_path: Path):
        """
        Load infrared and visible image pair.
        Args:
            ir_path: infrared image path
            vi_path: visible image path
        """

        self.ir_c = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        self.vi_c = cv2.imread(str(vi_path), cv2.IMREAD_COLOR)

        self.ir_t = kornia.utils.image_to_tensor(self.ir_c / 255.).float()
        self.vi_t = kornia.utils.image_to_tensor(cv2.cvtColor(self.vi_c, cv2.COLOR_BGR2GRAY) / 255.).float()

    def save_fus(self, path: Path, fus: Tensor, color: bool = False):
        """
        Colorize fusion image with visible color channels.
        Args:
            path: save fused image to specified path, if not exist, create it.
            fus: fused image (ndarray: cv2)
            color: colorize the fused image with visible color channels.
        """

        fus = kornia.utils.tensor_to_image(fus.squeeze().cpu()) * 255.
        path.parent.mkdir(parents=True, exist_ok=True)
        if not color:
            cv2.imwrite(str(path), fus)
            return
        fus = fus.astype(numpy.uint8)
        cbcr = cv2.cvtColor(self.vi_c, cv2.COLOR_BGR2YCrCb)[:, :, -2:]
        fus_r = numpy.concatenate([fus[..., numpy.newaxis], cbcr], axis=2)
        fus_c = cv2.cvtColor(fus_r, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(str(path), fus_c)
