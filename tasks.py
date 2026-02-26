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

@task
def get_scenes_data(c):
    """📊 Download scene metadata and background images from Zenodo.

    Downloads and extracts:
    - scenes_mastersheet.csv: Scene boundary definitions and layouts
    - scenes_mastersheet.json: Same data in JSON format
    - mario_scenes_manual_annotation.pdf: Annotation documentation
    - level_backgrounds/: Background images for each level
    - scene_backgrounds/: Background images for individual scenes

    All files are saved to sourcedata/scenes_info/ and sourcedata/*_backgrounds/.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke get-scenes-data
    ```

    Notes
    -----
    This must be run before other analysis tasks that depend on scene definitions.
    Downloads from Zenodo record 15586709.
    """
    c.run("mkdir -p sourcedata/scenes_info")
    c.run(
        'wget "https://zenodo.org/records/15586709/files/mario_scenes_manual_annotation.pdf?download=1" -O sourcedata/scenes_info/mario_scenes_manual_annotation.pdf'
    )
    c.run(
        'wget "https://zenodo.org/records/15586709/files/scenes_mastersheet.json?download=1" -O sourcedata/scenes_info/scenes_mastersheet.json'
    )
    c.run(
        'wget "https://zenodo.org/records/15586709/files/scenes_mastersheet.csv?download=1" -O sourcedata/scenes_info/scenes_mastersheet.csv'
    )

