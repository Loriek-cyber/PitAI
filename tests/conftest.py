"""
conftest.py – Shared pytest fixtures for the merged F1 project tests.
"""

from __future__ import annotations

import pytest


# NOTE: The 1.5s delay was removed because all API calls in tests are mocked.
# If you run tests against real APIs, add rate limiting to individual tests.
