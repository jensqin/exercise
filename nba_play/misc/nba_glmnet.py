from glmnet_py import glmnet
from utils import load_nba_sparse

xx, yy = load_nba_sparse("test")
glmfit = glmnet(x=xx, y=yy, family="gaussian", alpha=0.2, nlambda=20)
