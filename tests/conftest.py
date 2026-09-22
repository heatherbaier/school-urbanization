import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_params  # noqa: E402


@pytest.fixture
def params():
    return load_params()
