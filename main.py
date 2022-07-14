from enum import Enum

from fastapi import FastAPI


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "hello world"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None, short: bool = False):
    # path는 string이나 fastapi에서 알아서 parse해주고 있다.
    # query parameter(q)를 알아서 인식해준다.
    # boolean 타입도 인식한다.(uppercase, first letter uppercase 포함)
    # bool은 true, false 뿐만 아니라 1/0, yes/no, on/off도 인식한다.
    item = {"item_id": item_id}
    if q:
        return {"item_id": item_id, "q": q}
    if not short:
        item.update({"description": "This is an amazing item that has a long description"})
    return item


@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}


@app.get("/users/me")
async def read_user_me():
    # 이 path는 호출될 수 없다. 위의 API에서 user_id: str로 이미 me를 받고 있기 때문.
    # 따라서 고정 path를 변수가 있는 path보다 항상 먼저 오게 해야 한다.
    return {"user_id": "the current user"}


@app.get("/models/{model_name}")
async def get_mode(model_name: ModelName):
    # path param을 Enum으로 지정하면 올 수 있는 parameter를 한정할 수 있다.
    # Enum 타입을 리턴하더라도 알아서 value로 파싱한다.
    if model_name == ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}


@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    # :path라고 지정만 해주면 일반적으로 불가능한, url에 path 넣기도 가능. 앞에 /가 있으면 그것도 포함할 수 있다.
    return {"file_path": file_path}


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@app.get("/items")
async def query_item(skip: int = 0, limit: int = 10):
    # path parameter와 마찬가지로 함수 파라미터에서 타입을 지정해준대로 동작.(str->int)
    return fake_items_db[skip : skip + limit]


@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(user_id: int, item_id: str, q: str | None = None, short: bool = False):
    # 매개변수 순서 상관 없음
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update({"description": "This is an amazing item that has a long description"})
    return item


@app.get("/items/{item_id}/user")
async def read_item_user(item_id: str, needy: str):
    # query parameter라도 기본값을 주지 않으면 required로 지정할 수 있다.
    item = {"item_id": item_id, "needy": needy}
    return item


@app.get("/")
async def foo():
    return
