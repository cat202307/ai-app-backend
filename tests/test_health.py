"""
健康检查接口测试。
"""


def test_root_ok(client):
    """
    根路径可正常访问（已构建前端时返回页面，否则返回 JSON）。
    """
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    """
    健康检查接口返回状态信息。
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200
    assert "message" in data


def test_docs_endpoint(client):
    """
    测试 Swagger 文档页面能正常访问。
    """
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_openapi_schema(client):
    """
    测试 OpenAPI Schema 能正常获取。
    """
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "paths" in data
