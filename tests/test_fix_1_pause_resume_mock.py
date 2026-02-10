# How to run
# cd to Project Directory
# python -m pytest tests/test_fix_1_pause_resume_mock.py -v
import asyncio
import sys
import types
from unittest.mock import patch

import pytest
from fastapi import HTTPException


# `api.main` imports `sentence_transformers` at module import time.
# For this mock-only test, stub it so the test can run without that package.
if "sentence_transformers" not in sys.modules:
    stub = types.ModuleType("sentence_transformers")

    class _DummySentenceTransformer:  # pragma: no cover - utility stub
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *args, **kwargs):
            return [0.0]

    stub.SentenceTransformer = _DummySentenceTransformer
    sys.modules["sentence_transformers"] = stub


# `api.main` imports `core.search.search_logic` at module import time.
# Stub it to avoid import-time side effects in test runtime (for example stdout wrapping).
if "core.search.search_logic" not in sys.modules:
    search_stub = types.ModuleType("core.search.search_logic")

    def _dummy_search(*args, **kwargs):  # pragma: no cover - utility stub
        return {"search_metadata": {}, "results": []}

    search_stub.search_on_demand = _dummy_search
    search_stub.search_standard = _dummy_search
    sys.modules["core.search.search_logic"] = search_stub


from api.main import SubscriptionManager, app, pause_subscription, resume_subscription


class MockUpdateResult:
    def __init__(self, matched_count: int):
        self.matched_count = matched_count


class MockMongoCollection:
    def __init__(self, matched_count: int):
        self.matched_count = matched_count
        self.calls = []

    def update_one(self, filt, update):
        self.calls.append((filt, update))
        return MockUpdateResult(self.matched_count)


@pytest.fixture
def restore_subscription_manager():
    had_attr = hasattr(app.state, "subscription_manager")
    original = getattr(app.state, "subscription_manager", None)
    yield
    if had_attr:
        app.state.subscription_manager = original
    elif hasattr(app.state, "subscription_manager"):
        delattr(app.state, "subscription_manager")


@pytest.mark.asyncio
async def test_update_subscription_status_returns_true_when_subscription_exists():
    collection = MockMongoCollection(matched_count=1)
    manager = SubscriptionManager(collection)

    with patch("api.main.asyncio.get_event_loop") as mock_get_loop:
        mock_get_loop.return_value.run_in_executor = lambda executor, fn, *args: asyncio.sleep(0, result=fn(*args))
        result = await manager.update_subscription_status("sub-1", {"active": False})

    assert result is True
    assert len(collection.calls) == 1
    filt, update = collection.calls[0]
    assert filt == {"subscription_id": "sub-1"}
    assert "$set" in update
    assert "updated_at" in update["$set"]
    assert update["$set"]["active"] is False


@pytest.mark.asyncio
async def test_update_subscription_status_returns_false_when_subscription_missing():
    collection = MockMongoCollection(matched_count=0)
    manager = SubscriptionManager(collection)

    with patch("api.main.asyncio.get_event_loop") as mock_get_loop:
        mock_get_loop.return_value.run_in_executor = lambda executor, fn, *args: asyncio.sleep(0, result=fn(*args))
        result = await manager.update_subscription_status("missing-sub", {"active": False})

    assert result is False


@pytest.mark.asyncio
async def test_pause_and_resume_return_success_for_existing_subscription(restore_subscription_manager):
    manager = SubscriptionManager(MockMongoCollection(matched_count=1))
    app.state.subscription_manager = manager

    with patch.object(manager, "update_subscription_status", return_value=True):
        pause_resp = await pause_subscription("sub-1")
        resume_resp = await resume_subscription("sub-1")

    assert pause_resp["status"] == "success"
    assert resume_resp["status"] == "success"


@pytest.mark.asyncio
async def test_pause_returns_404_for_missing_subscription(restore_subscription_manager):
    manager = SubscriptionManager(MockMongoCollection(matched_count=0))
    app.state.subscription_manager = manager

    with patch.object(manager, "update_subscription_status", return_value=False):
        with pytest.raises(HTTPException) as exc:
            await pause_subscription("missing-sub")

    assert exc.value.status_code == 404
    assert exc.value.detail == "Subscription not found"


@pytest.mark.asyncio
async def test_resume_returns_404_for_missing_subscription(restore_subscription_manager):
    manager = SubscriptionManager(MockMongoCollection(matched_count=0))
    app.state.subscription_manager = manager

    with patch.object(manager, "update_subscription_status", return_value=False):
        with pytest.raises(HTTPException) as exc:
            await resume_subscription("missing-sub")

    assert exc.value.status_code == 404
    assert exc.value.detail == "Subscription not found"
