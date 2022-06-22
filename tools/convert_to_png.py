import argparse
import logging
from pathlib import Path

import cv2
from tqdm import tqdm


def convert_to_png(src: str | Path, color: bool):
    img_list = [x for x in Path(src).rglob('*') if x.suffix in ['.bmp', '.jpg', '.tiff']]
    process = tqdm(sorted(img_list))
    for o_path in process:
        n_path = o_path.with_suffix('.png')
        process.set_description(f'convert {o_path.name} to {n_path.name}')
        img = cv2.imread(str(o_path), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(n_path), img)
        o_path.unlink()


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    parser = argparse.ArgumentParser('convert to png')
    parser.add_argument('--src', help='folder need to be converted')
    parser.add_argument('--color', action='store_true', help='use color mode (recommend on for vis, off for ir)')
    config = parser.parse_args()
    convert_to_png(**vars(config))
