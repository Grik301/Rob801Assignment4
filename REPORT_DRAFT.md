# ROB801 Assignment 4 Report Draft

## 1. Setup

We implemented both required POMDP manipulation tasks in MuJoCo with a Franka Panda arm. The same tabletop simulator and low-level robot stack are reused across both tasks. The decision layer does not access hidden state directly; it only uses the observation history, the current belief, and learned Q-values.

## 2. Task A: Hidden-Target Retrieval

- Hidden state: which of three candidate objects is the true target
- Actions: inspect each object, pick-and-place each object into the goal tray, or stop
- Observation: inspecting an object returns `likely_target` or `unlikely_target`
- Observation noise:
  - true target inspected: `likely_target` with probability `0.8`
  - non-target inspected: `likely_target` with probability `0.2`
- Reward:
  - inspect: `-1`
  - correct retrieval: `+10`
  - wrong retrieval: `-8`
  - stop: `-3`

## 3. Task B: Uncertain Sorting / Placement

- Hidden state: whether the parcel belongs in the left or right bin
- Actions: inspect the signal region, place left, place right, or stop
- Observation: `looks_left` or `looks_right`
- Observation noise:
  - correct signal with probability `0.75`
  - incorrect signal with probability `0.25`
- Reward:
  - inspect: `-1`
  - correct placement: `+9`
  - wrong placement: `-8`
  - stop: `-3`

## 4. Belief Update

Belief is represented as a small discrete probability vector over the hidden task variable:

- Task A: `[P(obj1), P(obj2), P(obj3)]`
- Task B: `[P(left), P(right)]`

After an inspect action and a noisy observation, the belief is updated with Bayes' rule:

```text
b_{t+1}(x') = eta * Z(o_{t+1} | x', a_t) * sum_x T(x' | x, a_t) b_t(x)
```

In our tasks the hidden variable is static during one episode, so `T` is the identity transition.

## 5. Q-Learning Design

- Learning state: discretized belief on a `0.1` grid plus the current time step
- Policy: epsilon-greedy during training, greedy at test time
- Update:

```text
Q(b_t, a_t) <- (1 - alpha) Q(b_t, a_t) + alpha [r_t + gamma max_a' Q(b_{t+1}, a')]
```

- Default training setup:
  - `alpha = 0.25`
  - `gamma = 0.95`
  - epsilon decayed from `0.35` to `0.05`
  - `3000` training episodes per task

We also add a simple commitment guard at late horizon stages so the learned policy does not waste the last steps on extra inspection when it should commit.

## 6. Baseline

The baseline is intentionally simple and non-learning:

- Task A: inspect once, then commit to the object with the largest posterior belief
- Task B: inspect once, then place according to the larger posterior belief

This gives a lightweight comparison against the final belief-based Q-learning policy.

## 7. Low-Level Execution

Each high-level action is connected to an explicit MuJoCo routine:

- `inspect_*`: move the end-effector to a viewing pose and dwell briefly
- `pick_place_*`: approach, grasp, lift, move, place, retreat
- `place_left/right`: pick the parcel from the pickup region and place it in the chosen bin

The MuJoCo logs and saved MP4s make the executed action sequence visible.

## 8. Results

We evaluated both policies on 10 seeded test episodes per task using the same seeds for baseline and Q-learning. The aggregate numbers below are taken directly from `results/summary_results.csv`.

| Task | Policy | Avg reward | Success rate | Avg inspect | Avg length |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Baseline | 1.8 | 0.6 | 1.0 | 2.0 |
| A | Q-learning | 5.8 | 0.9 | 2.4 | 3.4 |
| B | Baseline | 6.3 | 0.9 | 1.0 | 2.0 |
| B | Q-learning | 6.3 | 0.9 | 1.0 | 2.0 |

Interpretation:

- On Task A, the learned policy benefits from spending extra inspections when the belief remains uncertain.
- On Task B, the task is small enough that one good inspect is often sufficient, so the learned policy matches the simple baseline.
- The corresponding per-episode logs are stored in `results/trial_records.csv`.

## 9. Representative Belief Trajectory

Example Task A run from `results/task_A_representative_qlearning_seed0.json`:

- prior: `[0.33, 0.33, 0.33]`
- after `inspect_obj3 -> unlikely_target`: `[0.44, 0.44, 0.11]`
- after `inspect_obj2 -> likely_target`: `[0.19, 0.76, 0.05]`
- action: `pick_place_obj2`

This trajectory shows the belief concentrating on the correct hidden target before commitment.

## 10. Failure Case

The saved failure-case JSON files in `results/` show episodes where noisy observations misled the controller. For example, `results/task_A_failure_case_baseline_seed4.json` records a Task A episode where a limited inspection budget caused the policy to commit to the wrong object. A typical failure mode is receiving a misleading positive cue early, which causes the posterior belief to concentrate on the wrong hypothesis before the final place action.

## 11. Videos

The assignment requires at least one successful visualized run per task. Those are already generated:

- `results/demo_taskA_seed0.mp4`
- `results/demo_taskB_seed0.mp4`

## 12. Reproducibility

Entry points:

```bash
conda run -n mujoco_env python ROB801_Assignment4/assignment4_pomdp_mujoco.py evaluate --tasks A B --train-episodes 3000 --test-episodes 10 --output-dir ROB801_Assignment4/results
conda run -n mujoco_env python ROB801_Assignment4/assignment4_pomdp_mujoco.py run --task A --policy qlearning --seed 0 --video-output ROB801_Assignment4/results/demo_taskA_seed0.mp4
conda run -n mujoco_env python ROB801_Assignment4/assignment4_pomdp_mujoco.py run --task B --policy qlearning --seed 0 --video-output ROB801_Assignment4/results/demo_taskB_seed0.mp4
```
