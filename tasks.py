from invoke import task
import os
import os.path as op

BASE_DIR = op.dirname(op.abspath(__file__))


@task
def setup_env(c):
    """🔧 Set up Python virtual environment and install dependencies.

    Creates a virtual environment in ./env/ and installs all required packages
    ```

    Notes
    -----
    This task creates a new virtual environment from scratch. If you need to update
    dependencies in an existing environment, activate it manually and run pip install.
    """
    c.run(
        f"python -m venv {BASE_DIR}/env && "
        f"source {BASE_DIR}/env/bin/activate && "
        "pip install --upgrade pip && "
        "pip install -e . && "
        "cd env/lib/python3.10/site-packages/ && "
        "git clone git@github.com:MaxStrange/retrowrapper.git && "
        "cd retrowrapper && "
        "sed -i 's/gym-retro/stable-retro/g' setup.py && "
        "pip install -e . && "
        "cd ../../../../.."
    )