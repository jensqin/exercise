from numpyro.infer import SVI
from fastapi import APIRouter

router = APIRouter()


@router.get("/name")
def svi_name():
    return "svi"
