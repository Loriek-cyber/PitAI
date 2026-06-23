import time
import pytest

@pytest.fixture(autouse=True)
def slow_down_api_calls():
    """
    Inserisce un ritardo di 1.5 secondi tra ogni test per evitare blocchi 
    (Error 429) dall'API di OpenF1 durante i cicli intensivi.
    """
    yield
    time.sleep(1.5)
