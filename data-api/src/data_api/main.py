from fastapi import FastAPI

description = """
BLA API helps you do awesome stuff. 🚀

## Items

You can **read items**.

## Users

You will be able to:

* **Create users** (_not implemented_).
* **Read users** (_not implemented_).
"""

app = FastAPI(
    title="Data API",
    version="dev",
    summary="tes api.",
    description=description,
    contact={
        "name": "Developers",
        "url": "https://localhost/",
        "email": "info@github.com",
    },
    terms_of_service="/terms",
    license_info={"name": "All rights reserved."},
)


@app.get("/")
async def root() -> dict:
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int) -> dict:
    return {"item_id": item_id}
