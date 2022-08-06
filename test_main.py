from fastapi.testclient import TestClient

from main import app

client = TestClient(app)
# requests 라이브러리와 같은 동작 방식이므로 세부적인 내용은 해당 문서를 참고한다.
# pydantic model을 사용하여 데이터를 넘겨야 할 때는 jsonable_encoder를 사용한다.


# def test_read_main():
#     response = client.get("/test/items")
#     assert response.status_code == 200
#     assert response.json() == {"msg": "Hello World"}


def test_read_item():
    response = client.get("/test/items/foo", headers={"X-Token": "coneofsilence"})
    assert response.status_code == 200
    assert response.json() == {"id": "foo", "title": "Foo", "description": "There goes my hero"}


def test_read_item_bad_token():
    response = client.get("/test/items/foo", headers={"X-Token": "hailhydra"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid X-Token header"}


def test_read_inexistent_item():
    response = client.get("/test/items/baz", headers={"X-Token": "coneofsilence"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Item not found"}


def test_create_item():
    response = client.post(
        "/test/items",
        headers={"X-Token": "coneofsilence"},
        json={"id": "foobar", "title": "Foo Bar", "description": "The Foo Barters"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "id": "foobar",
        "title": "Foo Bar",
        "description": "The Foo Barters",
    }


def test_create_item_bad_token():
    response = client.post(
        "test/items",
        headers={"X-Token": "hailhydra"},
        json={"id": "bazz", "title": "Bazz", "description": "Drop the bazz"},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid X-Token header"}


def test_create_existing_item():
    response = client.post(
        "test/items",
        headers={"X-Token": "coneofsilence"},
        json={
            "id": "foo",
            "title": "The Foo ID Stealers",
            "description": "There goes my stealer",
        },
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Item already exists"}
