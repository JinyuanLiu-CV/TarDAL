from functools import reduce
from pathlib import Path
from typing import Literal

import cv2
import numpy


def choose_images(root: str | Path, mode: str = Literal['train', 'val', 'pred']):
    root = Path(root)
    names = [x.name for x in sorted(root.glob('ir/*')) if x.suffix in ['.png', '.jpg', '.bmp']]
    save = []
    for name in names:
        x = cv2.imread(str(root / 'ir' / name), cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(str(root / 'vi' / name), cv2.IMREAD_GRAYSCALE)
        t = numpy.hstack([x, y])
        cv2.imshow(name, t)
        if cv2.waitKey() == ord('s'):
            save.append(name)
        cv2.destroyWindow(name)
    meta = root / 'meta'
    meta.mkdir(parents=True, exist_ok=True)
    meta_f = meta / f'{mode}.txt'
    meta_f.write_text(reduce(lambda i, j: i + j, [t + '\n' for t in save]))


if __name__ == '__main__':
    choose_images('data/tno', mode='train')
