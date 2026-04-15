# ROB801 Assignment 4

This project implements two small POMDP manipulation tasks in MuJoCo with a Franka Panda arm:

- Task A: choose the correct cube after noisy inspections
- Task B: place one parcel in the correct bin after a noisy cue
- Policies: baseline and tabular Q-learning
- Modes: train, run one episode, evaluate both tasks

The main command-line entry point is `pomdp_mujoco.py`.

## Project Files

- `pomdp_mujoco.py`: command-line interface
- `pomdp.py`: task definitions, belief update, training, evaluation, logging
- `mujoco_scene.py`: builds the runtime MuJoCo scene
- `mujoco_ik.py`: inverse kinematics and waypoint planning
- `mujoco_replay.py`: executes high-level actions in simulation
- `mujoco_common.py`: shared constants, scene dataclasses, object colors

## Setup

Activate the environment first, then use plain `python` commands:

```bash
git clone https://github.com/Grik301/Rob801Assignment4.git
cd Rob801Assignment4
conda create --name simtoreal python=3.11
conda activate simtoreal
pip install -r requirements.txt
python pomdp_mujoco.py --help
```

If you do not want to activate the environment, use the environment Python directly:

```bash
python pomdp_mujoco.py --help
```

## Core Commands

Train one task:

```bash
python pomdp_mujoco.py train --task A
python pomdp_mujoco.py train --task B
```

Run one episode without opening the viewer:

```bash
python pomdp_mujoco.py run --task A --policy qlearning --seed 0
python pomdp_mujoco.py run --task A --policy baseline --seed 0
python pomdp_mujoco.py run --task B --policy qlearning --seed 0
python pomdp_mujoco.py run --task B --policy baseline --seed 0
```

Run the full evaluation for both tasks:

```bash
python pomdp_mujoco.py evaluate --tasks A B --output-dir results
```

## Show The Simulation

Open the live MuJoCo viewer with the GLFW backend:

```bash
python pomdp_mujoco.py run --task A --policy baseline --seed 0 --simulate --mujoco-gl glfw
python pomdp_mujoco.py run --task A --policy qlearning --seed 0 --simulate --mujoco-gl glfw
python pomdp_mujoco.py run --task B --policy baseline --seed 0 --simulate --mujoco-gl glfw
python pomdp_mujoco.py run --task B --policy qlearning --seed 0 --simulate --mujoco-gl glfw
```

`--simulate` opens the viewer. `--mujoco-gl glfw` is the correct backend for a normal Linux desktop session.

## Save MP4 Videos

Save one MP4 per task and policy:

```bash
python pomdp_mujoco.py run --task A --policy baseline --seed 0 --mujoco-gl glfw --output-dir results --video-output results/task_A_baseline_seed0.mp4
python pomdp_mujoco.py run --task A --policy qlearning --seed 0 --mujoco-gl glfw --output-dir results --video-output results/task_A_qlearning_seed0.mp4
python pomdp_mujoco.py run --task B --policy baseline --seed 0 --mujoco-gl glfw --output-dir results --video-output results/task_B_baseline_seed0.mp4
python pomdp_mujoco.py run --task B --policy qlearning --seed 0 --mujoco-gl glfw --output-dir results --video-output results/task_B_qlearning_seed0.mp4
```

Important:

- `--video-output` does not open the live viewer in the same run.
- Video export requires OpenCV (`cv2`) in the active environment.
- The `results/` directory is created automatically when needed.

## Saved Results

Each `run` command writes a JSON episode record into `results/`, for example:

- `results/task_A_baseline_seed0.json`
- `results/task_A_qlearning_seed0.json`
- `results/task_B_baseline_seed0.json`
- `results/task_B_qlearning_seed0.json`

The `evaluate` command writes:

- `results/trial_records.csv`
- `results/summary_results.csv`
- `results/summary_results.txt`
- Q-table and training-history files for each task

## Tasks

### Task A

- Hidden state: which of `obj1`, `obj2`, `obj3` is the true target
- Inspect actions: `inspect_obj1`, `inspect_obj2`, `inspect_obj3`
- Commit actions: `pick_place_obj1`, `pick_place_obj2`, `pick_place_obj3`
- Observation model: inspecting the true target returns `likely_target` with probability `0.8`

### Task B

- Hidden state: whether the parcel belongs in the left or right bin
- Inspect action: `inspect_signal`
- Commit actions: `place_left`, `place_right`
- Observation model: the inspection cue is correct with probability `0.75`

## Notes

- If `python pomdp_mujoco.py run ... --video-output ...` fails with `OpenCV is required for --video-output`, install OpenCV in the active environment.
- If the viewer does not open, confirm that your shell has display access and that `DISPLAY` is set.
- If you want a quick smoke test without opening the viewer, start with `python pomdp_mujoco.py run --task A --policy qlearning --seed 0`.
