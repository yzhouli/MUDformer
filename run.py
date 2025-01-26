import os
import time

import numpy as np
from tqdm import tqdm

from config import Config
from my_models.models import Models
from train.train import Train


def main():
    model = Models.MS2Dformer_Base
    train = Train(model=model)
    train.train()


if __name__ == '__main__':
    main()
