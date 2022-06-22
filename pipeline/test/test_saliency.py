import logging

from pipeline.saliency import Saliency

logger = logging.getLogger(__name__)


def test_saliency():
    logging.basicConfig(level='DEBUG')
    saliency = Saliency(url='https://github.com/JinyuanLiu-CV/TarDAL/releases/download/v1.0.0/u2netp.pth')
