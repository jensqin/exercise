from fastapi import APIRouter

router = APIRouter()


@router.get("/user/{user_name}", tags=["user"])
def read_user(user_name: str):
    return {"user_name": user_name}
