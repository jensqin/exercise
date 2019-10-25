import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import sqlalchemy
import settings

engine = sqlalchemy.engine_from_config(settings.ENGINE_URL, prefix='FOOTBALL_DEV.')


x = torch.rand(3, 4)
