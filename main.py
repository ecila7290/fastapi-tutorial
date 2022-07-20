from enum import Enum

from fastapi import FastAPI, Query, Path, Body, Cookie, Header, status, Form
from pydantic import BaseModel, Field, HttpUrl, EmailStr


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


class Image(BaseModel):
    url: HttpUrl
    name: str


class Item(BaseModel):
    name: str
    description: str | None = Field(default=None, title="The description of the item", max_length=300)
    price: float = Field(gt=0, description="The price must be greater than zero")
    tax: float | None = None
    tags: list[str] = []
    image: Image | None = None

    # schema의 예시를 아래와 같이 넣어줄 수 있다. 이는 apidoc에도 반영된다.
    class Config:
        schema_extra = {
            "example": {
                "name": "Foo",
                "description": "A very nice Item",
                "price": 35.4,
                "tax": 3.2,
                "tags": ["a"],
                "image": {"url": "https://www.naver.com", "name": "image_name"},
            }
        }


class Offer(BaseModel):
    name: str
    # 각 필드에도 Field함수로 example을 넣을 수 있다.
    description: str | None = Field(default=None, example="A very nice offer")
    price: float
    items: list[Item]


class UserBase(BaseModel):
    # In/Out을 나누고 추가로 DB에 들어가는 모델까지 더하게 되면 중복된 필드가 많아진다.
    # 대신 이런 base를 만들고 상속시키는 방식으로 구현하면 중복을 줄일 수 있다.
    username: str
    email: EmailStr
    full_name: str | None = None


class User(BaseModel):
    username: str
    full_name: str | None = None


class UserIn(UserBase):
    password: str


class UserOut(UserBase):
    pass


class UserInDB(UserBase):
    hashed_password: str


def fake_password_hasher(raw_password: str):
    return "supersecret" + raw_password


def fake_save_user(user_in: UserIn):
    hashed_password = fake_password_hasher(user_in.password)
    user_in_db = UserInDB(**user_in.dict(), hashed_password=hashed_password)
    print("User saved! ..not really")
    return user_in_db


items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "hello world"}


@app.get("/items/ads")
async def read_items_ads(ads_id: str | None = Cookie(default=None)):
    # 쿠키의 값을 가져와서 사용할 수 있다.
    return {"ads_id": ads_id}


@app.get("/items/header")
async def read_items_headers(
    user_agent: str | None = Header(default=None),
    strange_header: str | None = Header(default=None, convert_underscores=False),
    x_token: list[str] | None = Header(default=None),
):
    # 헤더에 있는 값을 가져와 사용할 수 있다.
    # 헤더는 보통 User-Agent처럼 -을 사용하지만, Header함수에서 이를 snake_case로 변환하여 준다.
    # 같은 헤더가 여러 값을 갖는 경우 list type으로 명시하면 알아서 이를 인식한다.
    return {"user_agent": user_agent, "strange_header": strange_header, "x-token values": x_token}


@app.get("/items/response/{item_id}", response_model=Item, response_model_exclude_unset=True)
async def read_item_response(item_id: str):
    # response_model_exclude_unset 옵션을 통해 실제로 값이 설정되지 않은(=default value가 들어갈) 필드는 제외할 수 있다.
    # default value와 같은 값이라도 명시적으로 값이 할당되었다면 제외되지 않는다.
    # 이 외에도 response_model_exclude_defaults나 response_model_exclude_none과 같은 옵션으로 제어할 수도 있다.
    return items[item_id]


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


@app.get("/items/{item_id}/name", response_model=Item, response_model_include={"name", "description"})
async def read_item_name(item_id: str):
    # response_model용 모델을 새로 작성하는 대신 response_model_include/response_model_exclude로 특정 필드만 포함/제외할수 있다.
    # 옵션의 값으로 set()이 들어갔지만, list/tuple로 넣어도 알아서 set으로 변환해준다.
    return items[item_id]


@app.get("/items/{item_id}/public", response_model=Item, response_model_exclude={"tax"})
async def read_item_public_data(item_id: str):
    return items[item_id]


@app.post("/user", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserIn):
    # response_model을 지정함으로써 필드 제한 & serialization이 가능하다.
    # 요청에 대한 성공 응답은 기본 200. 이를 status_code 옵션으로 수정할 수 있다.
    # status code는 fastapi에서 제공하는 status를 사용할 수도 있고 파이썬 내장 모듈인 http.HTTPStatus를 사용할 수도 있다.
    user_saved = fake_save_user(user)
    return user_saved


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


@app.post("/items", response_model=Item)
async def create_item(item: Item):
    # 필요하다면 body에 있는 타입을 자동으로 변환한다. 변환할 수 없을 경우 에러
    # swagger에 schema도 자동으로 생성
    # response_model을 지정함으로써 필드 제한 & serialization이 가능하다.
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict


@app.put("/items/{item_id}")
async def create_item(item_id: int, item: Item, q: str | None = None):
    # 1. path에 있는 함수 파라미터는 path param으로 인식
    # 2. 함수 파라미터의 타입이 singular type(int, float, str, bool ...)이면 query param으로 인식
    # 3. 함수 파라미터의 타입이 Pydantic model이면 request body로 인식.
    # Pydantic을 쓰지 않는다면 fastapi에서 제공하는 Body를 사용
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result


@app.get("/itemsQVal")
async def query_items(q: list[str] | None = Query(default=None, alias="item-query")):
    # Query 함수를 파라미터의 값으로 넣어서 기본값 및 query string에 대한 제약조건을 설정할 수 있다.
    # default를 없애면 required
    # query string의 타입을 list로 설정하면 타입검사와 값을 리스트로 변환하는 것도 자동으로 수행한다.
    # alias 설정 가능해서 query string의 Key가 함수 파라미터 이름과 달라도 이를 인식한다.
    results = {"items": [{"item_id": "foo"}, {"item_id": "bar"}]}
    if q:
        results.update({"q": q})
    return results


@app.get("/itemsPathParam/{item_id}")
async def query_items_path(
    q: str, item_id: int = Path(title="The ID of the item to get", gt=0, le=1000), size: float = Query(gt=0, lt=10.5)
):
    # path parameter에 대해서는 fastapi에서 제공하는 Path함수를 사용할 수 있다.
    # 함수에서 파라미터의 순서는 관계 없이 잘 인식한다.
    # path/query param이 숫자일 때는 Path/Query 함수에 숫자의 범위를 제한할 수도 있다.
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results


@app.put("/items/body/{item_id}")
async def update_item_user(item_id: int, item: Item, user: User, importance: int = Body()):
    # body parameter가 두개 이상이더라도 구분해서 처리해줄 수 있다.
    # pydantic의 baseModel을 사용하지 않더라도 Body함수를 파라미터에 지정해주면, 알아서 body로 받아야 하는 것으로 인식한다.
    results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
    return results


@app.put("/items/field/{item_id}")
async def update_item_field(item_id: int, item: Item):
    # 함수의 파라미터에 Query, Body등을 사용하는 대신 Field 함수를 baseModel안에서 사용함으로써
    # validation이나 metadata를 추가할 수 있다.
    results = {"item_id": item_id, "item": item}
    return results


@app.put("/items/nested/{item_id}")
async def update_item_nested(item_id: int, item: Item):
    # nest model의 경우에도 sub model의 type valdation & data conversion을 지원한다.
    # 기존에 파이썬에서 제공하는 type뿐만 아니라 HttpUrl과 같은, pydantic에서 제공하는 타입도 사용할 수 있다.
    results = {"item_id": item_id, "item": item}
    return results


@app.post("/offers")
async def create_offer(offer: Offer):
    # 깊이 nested된 모델이라도 문제 없이 인식한다.
    for item in offer.items:
        item.image.url
    return offer


@app.post("/index-wieghts")
async def create_index_weights(weights: dict[int, float]):
    # key, value를 모르더라도 type validation은 가능.
    # body에 담긴 json은 string만 key로 받지만, fastapi에서 변환이 가능하다면 지정한 타입대로 변환시킨다.
    return weights


@app.post("/login")
async def login(username: str = Form(), password: str = Form()):
    # form field가 강제되는 경우 반드시 Form함수를 사용해야 한다.
    return {"username": username}


@app.get("/")
async def foo():
    return
