import datetime
from enum import Enum

from fastapi import FastAPI, Query, Path, Body, Cookie, Header, Request, status, Form, HTTPException, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
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
    timestamp: datetime.datetime

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
                "timestamp": datetime.datetime.now(),
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

async def common_parameters(q: str|None=None, skip:int=0, limit:int=100):
    # dependency 함수를 설정하고 Depends를 통해 호출하면 의존성이 주입된다.
    # 동기 함수에 async dependency 함수를 주입할 수도 있고, 비동기 함수에 sync dependency 함수를 주입할 수도 있다.
    return {'q':q, 'skip':skip, 'limit':limit}

class CommonQueryParams:
    # dependency가 될 수 있는 것은 함수뿐만 아니라 모든 Callable이므로 클래스도 가능하다.
    # 위의 함수처럼 dict를 리턴하는 경우, 자동완성이 미흡할 수밖에 없어 클래스로 만드는 것이 편할 수 있다.
    def __init__(self, q:str|None=None, skip:int=0,limit:int=100) -> None:
        self.q = q
        self.skip=skip
        self.limit=limit

def query_extractor(q:str|None=None):
    return q

def query_or_cookie_extractor(q:str=Depends(query_extractor), last_query:str|None=Cookie(default=None)):
    # dependency는 nested된 형태로도 사용이 가능하다.
    # 같은 dependency를 여러번 사용하는 경우 캐싱 처리를 해서 한 번만 호출되지만
    # 캐싱된 값을 사용하지 않고 매번 호출해야 하는 경우에는 use_cache=False 옵션을 준다.
    if not q:
        return last_query
    return q

async def verify_token(x_token:str=Header()):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")

async def verify_key(x_key: str = Header()):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key

fake_db = {}

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

# tag를 달면 swagger에서 알아서 태그별로 묶어서 보여준다.
# tag를 Enum으로 만들면 하드코드된 태그 대신 코드로 만들 수 있다.
@app.get("/items/ads", tags=["items"])
async def read_items_ads(ads_id: str | None = Cookie(default=None)):
    # 쿠키의 값을 가져와서 사용할 수 있다.
    return {"ads_id": ads_id}


@app.get("/items/header", tags=["items"])
async def read_items_headers(
    user_agent: str | None = Header(default=None),
    strange_header: str | None = Header(default=None, convert_underscores=False),
    x_token: list[str] | None = Header(default=None),
):
    # 헤더에 있는 값을 가져와 사용할 수 있다.
    # 헤더는 보통 User-Agent처럼 -을 사용하지만, Header함수에서 이를 snake_case로 변환하여 준다.
    # 같은 헤더가 여러 값을 갖는 경우 list type으로 명시하면 알아서 이를 인식한다.
    return {"user_agent": user_agent, "strange_header": strange_header, "x-token values": x_token}


@app.get("/items/response/{item_id}", response_model=Item, response_model_exclude_unset=True, tags=["items"])
async def read_item_response(item_id: str):
    # response_model_exclude_unset 옵션을 통해 실제로 값이 설정되지 않은(=default value가 들어갈) 필드는 제외할 수 있다.
    # default value와 같은 값이라도 명시적으로 값이 할당되었다면 제외되지 않는다.
    # 이 외에도 response_model_exclude_defaults나 response_model_exclude_none과 같은 옵션으로 제어할 수도 있다.
    return items[item_id]


@app.get("/items/{item_id}", tags=["items"])
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
    if not item_id in item:
        # detail에는 string뿐만 아니라 list, dict 타입도 알아서 JSON으로 변환시켜준다.
        raise HTTPException(status_code=404, detail='Item not found')
    return item


@app.get("/items/{item_id}/name", response_model=Item, response_model_include={"name", "description"}, tags=["items"])
async def read_item_name(item_id: str):
    # response_model용 모델을 새로 작성하는 대신 response_model_include/response_model_exclude로 특정 필드만 포함/제외할수 있다.
    # 옵션의 값으로 set()이 들어갔지만, list/tuple로 넣어도 알아서 set으로 변환해준다.
    return items[item_id]


@app.get("/items/{item_id}/public", response_model=Item, response_model_exclude={"tax"}, tags=["items"])
async def read_item_public_data(item_id: str):
    return items[item_id]

@app.get('/itemsQuery', tags=["items"])
async def query_items(commons:dict= Depends(common_parameters)):
    return commons

@app.get('/itemsQueryClass', tags=["items"])
async def query_items_with_class_param(commons: CommonQueryParams=Depends(CommonQueryParams)):
    # dependency가 파라미터 타입인 클래스를 호출하는 경우에는 Depends()만으로 의존성을 주입할 수 있다.
    # 예) commons: CommonQueryParams=Depends()
    response = {}
    if commons.q:
        response.update({'q':commons.q})
    items=fake_items_db[commons.skip : commons.skip + commons.limit]
    response.update({'items':items})
    return response

@app.get('/itemsQueryWithDecorator', dependencies=[Depends(verify_token), Depends(verify_key)], tags=["items"])
async def query_items_with_decorator():
    # 위와 같은 dependency는 path에 놓아도 사용하지 않으므로 삭제하게 될 우려가 있다.
    # 이처럼 값을 돌려줄 필요가 없는 경우에는 decorator의 dependencies에서 처리하도록 한다.
    return [{'item':'foo'},{'item':'bar'}]

@app.post("/user", response_model=UserOut, status_code=status.HTTP_201_CREATED, tags=["users"])
async def create_user(user: UserIn):
    # response_model을 지정함으로써 필드 제한 & serialization이 가능하다.
    # 요청에 대한 성공 응답은 기본 200. 이를 status_code 옵션으로 수정할 수 있다.
    # status code는 fastapi에서 제공하는 status를 사용할 수도 있고 파이썬 내장 모듈인 http.HTTPStatus를 사용할 수도 있다.
    user_saved = fake_save_user(user)
    return user_saved

app.get('/usersQuery', tags=["users"])
async def query_users(commons:dict= Depends(common_parameters)):
    return commons


@app.get("/users/{user_id}", tags=["users"])
async def read_user(user_id: str):
    return {"user_id": user_id}


@app.get("/users/me", tags=["users"])
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


@app.get("/items", tags=["items"])
async def query_item(skip: int = 0, limit: int = 10):
    # path parameter와 마찬가지로 함수 파라미터에서 타입을 지정해준대로 동작.(str->int)
    return fake_items_db[skip : skip + limit]


@app.get("/users/{user_id}/items/{item_id}", tags=["users"])
async def read_user_item(user_id: int, item_id: str, q: str | None = None, short: bool = False):
    # 매개변수 순서 상관 없음
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update({"description": "This is an amazing item that has a long description"})
    return item


@app.get("/items/{item_id}/user", tags=["items"])
async def read_item_user(item_id: str, needy: str):
    # query parameter라도 기본값을 주지 않으면 required로 지정할 수 있다.
    item = {"item_id": item_id, "needy": needy}
    return item


@app.post("/items", response_model=Item, tags=["items"], response_description="The created item")
async def create_item(item: Item):
    # docstring 생성이 가능하며 markdown도 지원한다.
    """
    Create an item with all the information:

    - **name**: each item must have a name
    - **description**: a long description
    - **price**: required
    - **tax**: if the item doesn't have tax, you can omit this
    - **tags**: a set of unique tag strings for this item
    """
    # 필요하다면 body에 있는 타입을 자동으로 변환한다. 변환할 수 없을 경우 에러
    # swagger에 schema도 자동으로 생성
    # response_model을 지정함으로써 필드 제한 & serialization이 가능하다.
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict


@app.put("/items/{item_id}", tags=["items"])
async def update_item(item_id: str, item: Item, q: str | None = None):
    # 1. path에 있는 함수 파라미터는 path param으로 인식
    # 2. 함수 파라미터의 타입이 singular type(int, float, str, bool ...)이면 query param으로 인식
    # 3. 함수 파라미터의 타입이 Pydantic model이면 request body로 인식.
    # Pydantic을 쓰지 않는다면 fastapi에서 제공하는 Body를 사용
    result = {"item_id": item_id, **item.dict()}
    # datetime처럼 JSON 변환시 에러가 발생하는 것들도 jsonable_encoder를 통해 적절하게 변환해줄 수 있다.
    # 또한 해당 함수의 결과는 string이 아닌 dict 형태로 나오게 된다.
    json_compatible_item_data = jsonable_encoder(item)
    fake_db[item_id] = json_compatible_item_data
    if q:
        result.update({"q": q})
    return result

@app.patch('/itemsPatch/{item_id}', tags=['items'])
async def patch_item(item_id:str, item:Item):
    # exclude_unset을 통해 값이 안들어간(default value인) 필드는 제외함으로써 부분 업데이트를 구현할 수 있다.
    # 기존 데이터를 copy하고 파라미터에 update을 줌으로써 데이터 업데이트가 가능하다.
    stored_item_data = items[item_id]
    stored_item_model = Item(**stored_item_data)
    update_data = item.dict(exclude_unset=True)
    updated_item = stored_item_model.copy(update=update_data)
    items[item_id] = jsonable_encoder(updated_item)
    return updated_item



@app.get("/itemsQVal", tags=["items"])
async def query_items(q: list[str] | None = Query(default=None, alias="item-query")):
    # Query 함수를 파라미터의 값으로 넣어서 기본값 및 query string에 대한 제약조건을 설정할 수 있다.
    # default를 없애면 required
    # query string의 타입을 list로 설정하면 타입검사와 값을 리스트로 변환하는 것도 자동으로 수행한다.
    # alias 설정 가능해서 query string의 Key가 함수 파라미터 이름과 달라도 이를 인식한다.
    results = {"items": [{"item_id": "foo"}, {"item_id": "bar"}]}
    if q:
        results.update({"q": q})
    return results


@app.get("/itemsPathParam/{item_id}", tags=["items"])
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


@app.put("/items/body/{item_id}", tags=["items"])
async def update_item_user(item_id: int, item: Item, user: User, importance: int = Body()):
    # body parameter가 두개 이상이더라도 구분해서 처리해줄 수 있다.
    # pydantic의 baseModel을 사용하지 않더라도 Body함수를 파라미터에 지정해주면, 알아서 body로 받아야 하는 것으로 인식한다.
    results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
    return results


@app.put("/items/field/{item_id}", tags=["items"])
async def update_item_field(item_id: int, item: Item):
    # 함수의 파라미터에 Query, Body등을 사용하는 대신 Field 함수를 baseModel안에서 사용함으로써
    # validation이나 metadata를 추가할 수 있다.
    results = {"item_id": item_id, "item": item}
    return results


@app.put("/items/nested/{item_id}", tags=["items"])
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


@app.post("/login", tags=["users"])
async def login(username: str = Form(), password: str = Form()):
    # form field가 강제되는 경우 반드시 Form함수를 사용해야 한다.
    return {"username": username}


class UnicornException(Exception):
    def __init__(self, name:str) -> None:
        self.name=name

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    # exception_handler 데코레이터를 통해 exception이 발생했을 떄 어떻게 handling할지 custom이 가능
    return JSONResponse(status_code=418, content={'message':f'Oops! {exc.name} did something. There goes a rainbow...'})

@app.get("/unicorns/{name}")
async def read_unicorn(name:str):
    if name == 'yolo':
        raise UnicornException(name=name)
    return {'unicorn_name':name}

@app.get("/")
async def foo():
    return
