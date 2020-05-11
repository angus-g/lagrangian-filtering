import pytest

import os

import filtering


@pytest.fixture
def tmp_chdir(tmp_path):
    """Change to tmp_path, saving current directory.

    This restores the current directory when the test finishes.
    """

    original_dir = os.getcwd()
    os.chdir(tmp_path)

    yield True

    os.chdir(original_dir)


@pytest.fixture(scope="session")
def nocompile_LagrangeFilter():
    """A monkey-patched version of the LagrangeFilter that won't compile.

    In tests where we don't actually perform advection, this saves us
    a lot of time.
    """

    class F(filtering.LagrangeFilter):
        def _compile(*args, **kwargs):
            pass

    return F
