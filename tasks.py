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

@task
def setup_mario_dataset(c):
    """📥 Download and configure the Mario dataset using datalad.

    Installs the Courtois NeuroMod Mario dataset including:
    - Event timing .tsv files
    - Game ROM stimuli


    The Game ROM is installed into sourcedata/mario/stimuli.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke setup-mario-dataset
    ```

    Notes
    -----
    Requires datalad to be installed and SSH access to the Courtois NeuroMod
    repositories.
    """
    command = (
            "mkdir -p sourcedata && "
            "cd sourcedata && "
            "datalad install git@github.com:courtois-neuromod/mario && "
            "cd mario && "
            "git checkout events && "
            "datalad get */*/*/*.tsv && "
            "rm -rf stimuli && "
            "datalad install git@github.com:courtois-neuromod/mario.stimuli && "
            "mv mario.stimuli stimuli && "
            "cd stimuli && "
            "git checkout scenes_states && "
            "datalad get ."
    )
    c.run(command)


@task(
    help={
        "model_type": "Model type: 'ppo' or 'imitation' (default: ppo).",
        "subjects": "Space-separated subjects, e.g. 'sub-01 sub-06'.",
        "sessions": "Space-separated sessions, e.g. 'ses-001 ses-002'.",
        "levels": "Space-separated levels, e.g. 'w1l1 w1l2'.",
        "scenes": "Space-separated scenes, e.g. 'scene-1 scene-2'.",
        "n_jobs": "Number of parallel jobs (-1 = all cores, default: -1).",
        "device": "Device for inference: 'cpu' or 'cuda' (default: cpu).",
        "verbose": "Enable verbose output.",
        "save_videos": "Save playback videos.",
        "save_variables": "Save game variables as JSON.",
        "video_format": "Video format: gif, mp4, or webp (default: mp4).",
    }
)
def run_agents(
    c,
    model_type="ppo",
    subjects=None,
    sessions=None,
    levels=None,
    scenes=None,
    n_jobs=-1,
    device="cpu",
    verbose=False,
    save_videos=False,
    save_variables=False,
    video_format="mp4",
):
    """🎮 Run artificial agents on scene savestates.

    Replays scene .state files through the emulator using the specified model
    type (PPO or imitation) and saves outputs (bk2, json, optional videos).

    Examples
    --------
    ```bash
    # PPO on specific subject/session
    invoke run-agents --subjects "sub-06" --sessions "ses-001" --levels "w1l1" --scenes "scene-1"

    # Imitation models on CPU
    invoke run-agents --model-type imitation --subjects "sub-06" --sessions "ses-001" --device cpu

    # With videos
    invoke run-agents --subjects "sub-06" --save-videos --video-format webp
    ```
    """
    cmd = (
        f"source {BASE_DIR}/env/bin/activate && "
        f"python {BASE_DIR}/code/main.py "
        f"--model-type {model_type} "
        f"--device {device} "
        f"-j {n_jobs} "
        f"--video_format {video_format}"
    )

    if subjects:
        cmd += f" -sub {subjects}"
    if sessions:
        cmd += f" -ses {sessions}"
    if levels:
        cmd += f" -l {levels}"
    if scenes:
        cmd += f" -scn {scenes}"
    if verbose:
        cmd += " -v"
    if save_videos:
        cmd += " --save_videos"
    if save_variables:
        cmd += " --save_variables"

    c.run(cmd)

