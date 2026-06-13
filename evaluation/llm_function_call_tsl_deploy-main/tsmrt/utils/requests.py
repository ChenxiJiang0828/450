import httpx
import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

_client: Optional[httpx.AsyncClient] = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        logger.info("Created shared httpx.AsyncClient")
    return _client


async def close():
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
        logger.info("Closed shared httpx.AsyncClient")


class Response:
    def __init__(self, resp: httpx.Response):
        self._resp = resp
        self.status_code: int = resp.status_code
        self.text: str = resp.text
        self.headers: httpx.Headers = resp.headers
        self.url: str = str(resp.url)
        self.encoding: Optional[str] = resp.encoding

    def json(self, **kwargs) -> Any:
        return self._resp.json(**kwargs)

    def raise_for_status(self):
        self._resp.raise_for_status()

    @property
    def ok(self) -> bool:
        return self._resp.is_success

    def __repr__(self) -> str:
        return f"<Response [{self.status_code}]>"


async def request(
    method: str,
    url: str,
    *,
    params: Optional[Dict] = None,
    data: Optional[Any] = None,
    json: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
) -> Response:
    client = await _get_client()
    if isinstance(timeout, (int, float)):
        timeout = httpx.Timeout(timeout)
    resp = await client.request(
        method, url,
        params=params,
        content=data,
        json=json,
        headers=headers,
        timeout=timeout,
        **kwargs,
    )
    return Response(resp)


async def get(url: str, **kwargs) -> Response:
    return await request("GET", url, **kwargs)


async def post(url: str, **kwargs) -> Response:
    return await request("POST", url, **kwargs)


async def put(url: str, **kwargs) -> Response:
    return await request("PUT", url, **kwargs)


async def delete(url: str, **kwargs) -> Response:
    return await request("DELETE", url, **kwargs)


async def head(url: str, **kwargs) -> Response:
    return await request("HEAD", url, **kwargs)


async def patch(url: str, **kwargs) -> Response:
    return await request("PATCH", url, **kwargs)


async def options(url: str, **kwargs) -> Response:
    return await request("OPTIONS", url, **kwargs)
