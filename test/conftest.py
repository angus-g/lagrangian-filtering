import pytest

import os


@pytest.fixture
def tmp_chdir(tmp_path):
    # change to tmp_path, saving current directory
    original_dir = os.getcwd()
    os.chdir(tmp_path)

    yield True

    # restore original directory
    os.chdir(original_dir)
