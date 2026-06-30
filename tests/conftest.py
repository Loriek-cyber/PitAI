"""
conftest.py – Shared pytest fixtures for the merged F1 project tests.
"""

from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def slow_down_api_calls():
    """Insert a 1.5 s delay between tests to respect API rate limits."""
    yield
    time.sleep(1.5)
