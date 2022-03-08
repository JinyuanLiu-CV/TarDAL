import cv2
import torch
from kornia.utils import image_to_tensor
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset


class FusionData(Dataset):
    """
    Loading fusion data from hard disk.
    """

    def __init__(self, folder: Path, mask: str = 'm1', mode='train', transforms=lambda x: x):
        super(FusionData, self).__init__()

        assert mode in ['eval', 'train'], 'mode should be "eval" or "train"'
        names = (folder / 'list.txt').read_text().splitlines()
        self.samples = [{
            'name': name,
            'ir': folder / 'ir' / f'{name}.bmp',
            'vi': folder / 'vi' / f'{name}.bmp',
            'mk': folder / 'mask' / mask / f'{name}.png',
            'vsm': {
                'ir': folder / 'vsm' / 's1' / f'{name}.bmp',
                'vi': folder / 'vsm' / 's2' / f'{name}.bmp'
            },
        } for name in names]
        self.transforms = transforms
        self.mode = mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        ir, vi = self.imread(sample['ir']), self.imread(sample['vi'])
        if self.mode == 'train':
            mk = self.imread(sample['mk'])
            s1, s2 = self.imread(sample['vsm']['ir']), self.imread(sample['vsm']['vi'])
            im = torch.cat([ir, vi, mk, s1, s2], dim=0)
            im = self.transforms(im)
            ir, vi, mk, s1, s2 = torch.chunk(im, 5, dim=0)
            sample = {'name': sample['name'], 'ir': ir, 'vi': vi, 'mk': mk, 'vsm': {'ir': s1, 'vi': s2}}
        elif self.mode == 'eval':
            im = torch.cat([ir, vi], dim=0)
            im = self.transforms(im)
            ir, vi = torch.chunk(im, 2, dim=0)
            sample = {'name': sample['name'], 'ir': ir, 'vi': vi}
        return sample

    @staticmethod
    def imread(path: Path) -> Tensor:
        img_n = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img_t = image_to_tensor(img_n / 255.).float()
        return img_t


if __name__ == '__main__':
    fd = FusionData(folder=Path('../data/train'), mode='train')
    s = fd[0]
    print(s)
