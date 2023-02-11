import argparse
import logging

from pipeline.saliency import Saliency


def generate_mask(url: str, src: str, dst: str):
    saliency = Saliency(url=url)
    saliency.inference(src=src, dst=dst)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    default_url = 'https://github.com/JinyuanLiu-CV/TarDAL/releases/download/v1.0.0/u2netp.pth'
    parser = argparse.ArgumentParser('mask generator')
    parser.add_argument('--url', default=default_url, help='checkpoint url')
    parser.add_argument('--src', help='folder need to be detected')
    parser.add_argument('--dst', help='mask output folder (we will create it if not exists)')
    config = parser.parse_args()
    generate_mask(**vars(config))
