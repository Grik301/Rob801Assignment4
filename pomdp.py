from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from mujoco_common import TaskSceneSpec


ArrayLike = Sequence[float] | np.ndarray
BeliefKey = Tuple[int, Tuple[int, ...]]
QTable = Dict[BeliefKey, np.ndarray]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    hidden_states: Tuple[str, ...]
    initial_belief: np.ndarray
    horizon: int
    actions: Tuple[str, ...]
    inspect_actions: Tuple[str, ...]
    terminal_actions: Tuple[str, ...]
    success_action_for_hidden: Dict[str, str]
    observation_likelihoods: Dict[str, Dict[str, Dict[str, float]]]
    inspect_cost: float
    success_reward: float
    failure_reward: float
    stop_reward: float
    baseline_confidence: float
    baseline_max_inspects: int
    scene_spec: TaskSceneSpec


@dataclass
class StepRecord:
    step_index: int
    action: str
    observation: str
    reward: float
    belief_before: List[float]
    belief_after: List[float]


@dataclass
class EpisodeResult:
    task: str
    policy: str
    seed: int
    hidden_truth: str
    total_reward: float
    success: bool
    inspect_count: int
    episode_length: int
    actions: List[str]
    observations: List[str]
    belief_sequence: List[List[float]]
    step_records: List[StepRecord]


@dataclass(frozen=True)
class TransitionOutcome:
    reward: float
    observation: str
    done: bool
    success: bool


def _normalize(probs: ArrayLike) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full_like(arr, 1.0 / max(len(arr), 1))
    return arr / total


def _task_a_spec() -> TaskSpec:
    scene_spec = TaskSceneSpec(
        task_name="A",
        cube_names=("obj1", "obj2", "obj3"),
        region_positions={
            "src_left": (0.42, 0.22),
            "src_mid": (0.42, 0.00),
            "src_right": (0.42, -0.22),
            "goal_tray": (0.73, 0.00),
        },
        region_types={
            "src_left": "source",
            "src_mid": "source",
            "src_right": "source",
            "goal_tray": "goal",
        },
        initial_cube_regions={
            "obj1": "src_left",
            "obj2": "src_mid",
            "obj3": "src_right",
        },
        inspect_targets={
            "inspect_obj1": ("cube", "obj1"),
            "inspect_obj2": ("cube", "obj2"),
            "inspect_obj3": ("cube", "obj3"),
        },
        commit_destinations={
            "pick_place_obj1": "goal_tray",
            "pick_place_obj2": "goal_tray",
            "pick_place_obj3": "goal_tray",
        },
    )
    observation_likelihoods: Dict[str, Dict[str, Dict[str, float]]] = {}
    for obj in scene_spec.cube_names:
        action = f"inspect_{obj}"
        observation_likelihoods[action] = {}
        for hidden in scene_spec.cube_names:
            if hidden == obj:
                observation_likelihoods[action][hidden] = {"likely_target": 0.80, "unlikely_target": 0.20}
            else:
                observation_likelihoods[action][hidden] = {"likely_target": 0.20, "unlikely_target": 0.80}
    return TaskSpec(
        name="A",
        hidden_states=scene_spec.cube_names,
        initial_belief=_normalize([1.0, 1.0, 1.0]),
        horizon=5,
        actions=(
            "inspect_obj1",
            "inspect_obj2",
            "inspect_obj3",
            "pick_place_obj1",
            "pick_place_obj2",
            "pick_place_obj3",
            "stop",
        ),
        inspect_actions=("inspect_obj1", "inspect_obj2", "inspect_obj3"),
        terminal_actions=("pick_place_obj1", "pick_place_obj2", "pick_place_obj3", "stop"),
        success_action_for_hidden={
            "obj1": "pick_place_obj1",
            "obj2": "pick_place_obj2",
            "obj3": "pick_place_obj3",
        },
        observation_likelihoods=observation_likelihoods,
        inspect_cost=-1.0,
        success_reward=10.0,
        failure_reward=-8.0,
        stop_reward=-3.0,
        baseline_confidence=0.72,
        baseline_max_inspects=1,
        scene_spec=scene_spec,
    )


def _task_b_spec() -> TaskSpec:
    scene_spec = TaskSceneSpec(
        task_name="B",
        cube_names=("parcel",),
        region_positions={
            "pickup": (0.42, 0.00),
            "inspect_pad": (0.58, 0.24),
            "left_bin": (0.73, 0.18),
            "right_bin": (0.73, -0.18),
        },
        region_types={
            "pickup": "pickup",
            "inspect_pad": "inspect",
            "left_bin": "left_bin",
            "right_bin": "right_bin",
        },
        initial_cube_regions={"parcel": "pickup"},
        inspect_targets={"inspect_signal": ("region", "inspect_pad")},
        commit_destinations={
            "place_left": "left_bin",
            "place_right": "right_bin",
        },
    )
    observation_likelihoods = {
        "inspect_signal": {
            "left": {"looks_left": 0.75, "looks_right": 0.25},
            "right": {"looks_left": 0.25, "looks_right": 0.75},
        }
    }
    return TaskSpec(
        name="B",
        hidden_states=("left", "right"),
        initial_belief=_normalize([1.0, 1.0]),
        horizon=4,
        actions=("inspect_signal", "place_left", "place_right", "stop"),
        inspect_actions=("inspect_signal",),
        terminal_actions=("place_left", "place_right", "stop"),
        success_action_for_hidden={"left": "place_left", "right": "place_right"},
        observation_likelihoods=observation_likelihoods,
        inspect_cost=-1.0,
        success_reward=9.0,
        failure_reward=-8.0,
        stop_reward=-3.0,
        baseline_confidence=0.67,
        baseline_max_inspects=1,
        scene_spec=scene_spec,
    )


TASK_SPECS: Dict[str, TaskSpec] = {
    "A": _task_a_spec(),
    "B": _task_b_spec(),
}


def get_task_spec(task_name: str) -> TaskSpec:
    key = str(task_name).upper()
    if key not in TASK_SPECS:
        raise ValueError(f"Unknown task {task_name!r}. Expected one of {sorted(TASK_SPECS)}")
    return TASK_SPECS[key]


def _hidden_index(task: TaskSpec, hidden_state: str) -> int:
    return task.hidden_states.index(hidden_state)


def _state_belief_array(belief: ArrayLike) -> np.ndarray:
    return np.asarray(belief, dtype=float)


def belief_dict(task: TaskSpec, belief: ArrayLike) -> Dict[str, float]:
    return {hidden: float(prob) for hidden, prob in zip(task.hidden_states, _state_belief_array(belief))}


def action_success_probability(task: TaskSpec, belief: ArrayLike, action: str) -> float:
    probs = belief_dict(task, belief)
    success_total = 0.0
    for hidden, success_action in task.success_action_for_hidden.items():
        if success_action == action:
            success_total += probs[hidden]
    return float(success_total)


def commit_action_from_belief(task: TaskSpec, belief: ArrayLike) -> str:
    belief_arr = _state_belief_array(belief)
    hidden = task.hidden_states[int(np.argmax(belief_arr))]
    return task.success_action_for_hidden[hidden]


def transition_probability(task: TaskSpec, next_hidden: str, hidden: str, action: str) -> float:
    del task, action
    return 1.0 if next_hidden == hidden else 0.0


def predictive_hidden_belief(task: TaskSpec, prior_belief: ArrayLike, action: str) -> np.ndarray:
    prior = _normalize(prior_belief)
    predictive = np.zeros(len(task.hidden_states), dtype=float)
    for next_idx, next_hidden in enumerate(task.hidden_states):
        total = 0.0
        for hidden_idx, hidden in enumerate(task.hidden_states):
            total += transition_probability(task, next_hidden, hidden, action) * float(prior[hidden_idx])
        predictive[next_idx] = total
    return _normalize(predictive)


def observation_probability(task: TaskSpec, observation: str, next_hidden: str, action: str) -> float:
    if action not in task.inspect_actions:
        return 1.0
    return float(task.observation_likelihoods[action][next_hidden].get(observation, 0.0))


def observation_support(task: TaskSpec, action: str, hidden_state: str) -> List[str]:
    if action not in task.inspect_actions:
        return []
    return list(task.observation_likelihoods[action][hidden_state].keys())


def update_belief(task: TaskSpec, prior_belief: ArrayLike, action: str, observation: str) -> np.ndarray:
    prior = _normalize(prior_belief)
    if action not in task.inspect_actions:
        return prior.copy()
    predictive = predictive_hidden_belief(task, prior, action)
    posterior = np.zeros(len(task.hidden_states), dtype=float)
    for hidden_idx, next_hidden in enumerate(task.hidden_states):
        posterior[hidden_idx] = observation_probability(task, observation, next_hidden, action) * float(predictive[hidden_idx])
    return _normalize(posterior)


def discretize_belief(belief: ArrayLike, step: float = 0.1) -> Tuple[int, ...]:
    arr = _normalize(belief)
    bins = int(round(1.0 / step))
    scaled = arr * bins
    base = np.floor(scaled).astype(int)
    deficit = bins - int(np.sum(base))
    if deficit > 0:
        order = np.argsort(-(scaled - base))
        for idx in order[:deficit]:
            base[idx] += 1
    elif deficit < 0:
        order = np.argsort(scaled - base)
        for idx in order[: -deficit]:
            if base[idx] > 0:
                base[idx] -= 1
    return tuple(int(value) for value in base.tolist())


def learning_state_key(task: TaskSpec, belief: ArrayLike, step_index: int, belief_step: float = 0.1) -> BeliefKey:
    del task
    return int(step_index), discretize_belief(belief, step=belief_step)


def _entropy(probs: ArrayLike) -> float:
    arr = _normalize(probs)
    safe = np.clip(arr, 1e-9, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def expected_posterior_entropy(task: TaskSpec, belief: ArrayLike, action: str) -> float:
    belief_arr = _normalize(belief)
    total = 0.0
    observation_names = sorted(
        {
            obs_name
            for hidden in task.hidden_states
            for obs_name in task.observation_likelihoods[action][hidden].keys()
        }
    )
    for observation in observation_names:
        obs_prob = 0.0
        for hidden_idx, hidden in enumerate(task.hidden_states):
            obs_prob += belief_arr[hidden_idx] * observation_probability(task, observation, hidden, action)
        posterior = update_belief(task, belief_arr, action, observation)
        total += obs_prob * _entropy(posterior)
    return float(total)


def baseline_action(task: TaskSpec, belief: ArrayLike, step_index: int, action_history: Sequence[str]) -> str:
    remaining_steps = task.horizon - int(step_index)
    inspect_count = sum(1 for action in action_history if action in task.inspect_actions)
    if remaining_steps <= 1:
        return commit_action_from_belief(task, belief)
    if inspect_count >= task.baseline_max_inspects:
        return commit_action_from_belief(task, belief)
    if task.name == "A":
        return task.inspect_actions[min(inspect_count, len(task.inspect_actions) - 1)]
    return task.inspect_actions[0]


def _should_force_commit(task: TaskSpec, belief: ArrayLike, step_index: int) -> bool:
    belief_arr = _state_belief_array(belief)
    remaining_steps = task.horizon - int(step_index)
    return remaining_steps <= 2 or float(np.max(belief_arr)) >= float(task.baseline_confidence)


def _sample_epsilon_greedy_action(task: TaskSpec, q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> str:
    if float(epsilon) > 0.0 and float(rng.random()) < float(epsilon):
        return str(rng.choice(task.actions))
    best_idx = int(np.argmax(q_values))
    return task.actions[best_idx]


def q_policy_action(
    task: TaskSpec,
    belief: ArrayLike,
    step_index: int,
    q_table: QTable,
    epsilon: float,
    rng: np.random.Generator,
    belief_step: float = 0.1,
) -> str:
    belief_arr = _state_belief_array(belief)
    if _should_force_commit(task, belief_arr, step_index):
        return commit_action_from_belief(task, belief_arr)
    state_key = learning_state_key(task, belief_arr, step_index, belief_step=belief_step)
    q_values = q_table.get(state_key)
    if q_values is None:
        return commit_action_from_belief(task, belief_arr)
    return _sample_epsilon_greedy_action(task, q_values, epsilon, rng)


def sample_hidden_truth(task: TaskSpec, rng: np.random.Generator) -> str:
    return str(rng.choice(task.hidden_states, p=task.initial_belief))


def sample_observation(task: TaskSpec, hidden_truth: str, action: str, rng: np.random.Generator) -> str:
    observation_probs = task.observation_likelihoods[action][hidden_truth]
    observation_names = list(observation_probs.keys())
    probs = [observation_probs[name] for name in observation_names]
    return str(rng.choice(observation_names, p=_normalize(probs)))


def resolve_transition_outcome(task: TaskSpec, hidden_truth: str, action: str, rng: np.random.Generator) -> TransitionOutcome:
    if action in task.inspect_actions:
        observation = sample_observation(task, hidden_truth, action, rng)
        return TransitionOutcome(task.inspect_cost, observation, False, False)
    if action == "stop":
        return TransitionOutcome(task.stop_reward, "stopped", True, False)
    success = action == task.success_action_for_hidden[hidden_truth]
    reward = task.success_reward if success else task.failure_reward
    observation = "placed_correctly" if success else "placed_incorrectly"
    return TransitionOutcome(reward, observation, True, bool(success))


def environment_step(
    task: TaskSpec,
    hidden_truth: str,
    action: str,
    rng: np.random.Generator,
) -> Tuple[float, str, bool, bool]:
    outcome = resolve_transition_outcome(task, hidden_truth, action, rng)
    return outcome.reward, outcome.observation, outcome.done, outcome.success


def _run_timeout_adjustment(task: TaskSpec, reward: float, observation: str, done: bool, success: bool) -> Tuple[float, str, bool, bool]:
    if done:
        return float(reward), observation, True, bool(success)
    return float(reward) + float(task.stop_reward), f"{observation}|timeout", True, False


def run_episode(
    task: TaskSpec,
    policy_name: str,
    policy_fn: Callable[[TaskSpec, np.ndarray, int, Sequence[str], Sequence[str], np.random.Generator], str],
    seed: int,
) -> EpisodeResult:
    rng = np.random.default_rng(int(seed))
    hidden_truth = sample_hidden_truth(task, rng)
    belief = task.initial_belief.copy()
    total_reward = 0.0
    success = False
    action_history: List[str] = []
    observation_history: List[str] = []
    belief_sequence: List[List[float]] = [belief.tolist()]
    step_records: List[StepRecord] = []

    for step_index in range(task.horizon):
        action = str(policy_fn(task, belief.copy(), step_index, action_history, observation_history, rng))
        if action not in task.actions:
            raise ValueError(f"Policy returned unsupported action {action!r} for task {task.name}")
        reward, observation, done, success = environment_step(task, hidden_truth, action, rng)
        next_belief = update_belief(task, belief, action, observation)
        if not done and step_index == task.horizon - 1:
            reward, observation, done, success = _run_timeout_adjustment(task, reward, observation, done, success)
        total_reward += float(reward)
        action_history.append(action)
        observation_history.append(observation)
        step_records.append(
            StepRecord(
                step_index=step_index,
                action=action,
                observation=observation,
                reward=float(reward),
                belief_before=belief.tolist(),
                belief_after=next_belief.tolist(),
            )
        )
        belief = next_belief
        belief_sequence.append(belief.tolist())
        if done:
            break

    inspect_count = sum(1 for action in action_history if action in task.inspect_actions)
    return EpisodeResult(
        task=task.name,
        policy=policy_name,
        seed=int(seed),
        hidden_truth=hidden_truth,
        total_reward=float(total_reward),
        success=bool(success),
        inspect_count=inspect_count,
        episode_length=len(action_history),
        actions=action_history,
        observations=observation_history,
        belief_sequence=belief_sequence,
        step_records=step_records,
    )


def _epsilon_for_episode(episode_idx: int, num_episodes: int, epsilon_start: float, epsilon_end: float) -> float:
    return float(
        epsilon_end
        + (epsilon_start - epsilon_end) * max(0.0, (num_episodes - episode_idx - 1) / max(num_episodes - 1, 1))
    )


def _ensure_state(q_table: QTable, state_key: BeliefKey, num_actions: int) -> np.ndarray:
    if state_key not in q_table:
        q_table[state_key] = np.zeros(num_actions, dtype=float)
    return q_table[state_key]


def _update_q_entry(current_value: float, alpha: float, target: float) -> float:
    return (1.0 - float(alpha)) * float(current_value) + float(alpha) * float(target)


def train_q_learning(
    task: TaskSpec,
    num_episodes: int,
    seed: int = 0,
    alpha: float = 0.25,
    gamma: float = 0.95,
    epsilon_start: float = 0.35,
    epsilon_end: float = 0.05,
    belief_step: float = 0.1,
) -> Tuple[QTable, List[Dict[str, float]]]:
    rng = np.random.default_rng(int(seed))
    q_table: QTable = {}
    history: List[Dict[str, float]] = []

    for episode_idx in range(int(num_episodes)):
        hidden_truth = sample_hidden_truth(task, rng)
        belief = task.initial_belief.copy()
        total_reward = 0.0
        epsilon = _epsilon_for_episode(episode_idx, int(num_episodes), float(epsilon_start), float(epsilon_end))

        for step_index in range(task.horizon):
            state_key = learning_state_key(task, belief, step_index, belief_step=belief_step)
            state_q_values = _ensure_state(q_table, state_key, len(task.actions))
            action = q_policy_action(
                task=task,
                belief=belief,
                step_index=step_index,
                q_table=q_table,
                epsilon=epsilon,
                rng=rng,
                belief_step=belief_step,
            )
            action_idx = task.actions.index(action)
            reward, observation, done, _success = environment_step(task, hidden_truth, action, rng)
            next_belief = update_belief(task, belief, action, observation)
            if not done and step_index == task.horizon - 1:
                reward += task.stop_reward
                done = True
            target = float(reward)
            if not done:
                next_state_key = learning_state_key(task, next_belief, step_index + 1, belief_step=belief_step)
                next_q = _ensure_state(q_table, next_state_key, len(task.actions))
                target += float(gamma) * float(np.max(next_q))
            state_q_values[action_idx] = _update_q_entry(state_q_values[action_idx], alpha, target)
            total_reward += float(reward)
            belief = next_belief
            if done:
                break

        history.append(
            {
                "episode": float(episode_idx),
                "epsilon": epsilon,
                "total_reward": float(total_reward),
            }
        )
    return q_table, history


def evaluate_policy_bundle(
    task: TaskSpec,
    q_table: QTable,
    seeds: Iterable[int],
    belief_step: float = 0.1,
) -> List[EpisodeResult]:
    records: List[EpisodeResult] = []
    for seed in seeds:
        records.append(
            run_episode(
                task=task,
                policy_name="baseline",
                seed=int(seed),
                policy_fn=lambda t, b, s, ah, oh, rng: baseline_action(t, b, s, ah),
            )
        )
        records.append(
            run_episode(
                task=task,
                policy_name="qlearning",
                seed=int(seed),
                policy_fn=lambda t, b, s, ah, oh, rng: q_policy_action(
                    t,
                    b,
                    s,
                    q_table=q_table,
                    epsilon=0.0,
                    rng=rng,
                    belief_step=belief_step,
                ),
            )
        )
    return records


def summarize_records(records: Sequence[EpisodeResult]) -> List[Dict[str, float | str]]:
    grouped: Dict[Tuple[str, str], List[EpisodeResult]] = defaultdict(list)
    for record in records:
        grouped[(record.task, record.policy)].append(record)
    summary_rows: List[Dict[str, float | str]] = []
    for (task_name, policy_name), task_records in sorted(grouped.items()):
        summary_rows.append(
            {
                "task": task_name,
                "policy": policy_name,
                "episodes": len(task_records),
                "average_total_reward": float(np.mean([record.total_reward for record in task_records])),
                "success_rate": float(np.mean([1.0 if record.success else 0.0 for record in task_records])),
                "average_inspect_actions": float(np.mean([record.inspect_count for record in task_records])),
                "average_episode_length": float(np.mean([record.episode_length for record in task_records])),
            }
        )
    return summary_rows


def save_q_table(path: Path, task: TaskSpec, q_table: QTable) -> None:
    payload = {
        "task": task.name,
        "actions": list(task.actions),
        "states": {
            f"{step}|{','.join(map(str, belief_bins))}": values.tolist()
            for (step, belief_bins), values in q_table.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_q_table(path: Path, task: TaskSpec) -> QTable:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload["task"]).upper() != task.name:
        raise ValueError(f"Q-table task mismatch: expected {task.name}, found {payload['task']}")
    q_table: QTable = {}
    for key, values in payload["states"].items():
        step_text, belief_text = str(key).split("|", maxsplit=1)
        belief_bins = tuple(int(token) for token in belief_text.split(",") if token)
        q_table[(int(step_text), belief_bins)] = np.asarray(values, dtype=float)
    return q_table


def _episode_record_to_csv_row(record: EpisodeResult) -> Dict[str, str | float | int]:
    return {
        "task": record.task,
        "policy": record.policy,
        "seed": record.seed,
        "hidden_truth": record.hidden_truth,
        "total_reward": record.total_reward,
        "success": int(record.success),
        "inspect_count": record.inspect_count,
        "episode_length": record.episode_length,
        "actions": " | ".join(record.actions),
        "observations": " | ".join(record.observations),
        "belief_sequence": json.dumps(record.belief_sequence),
    }


def write_training_history_csv(path: Path, history: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "epsilon", "total_reward"])
        writer.writeheader()
        writer.writerows(history)


def write_trial_records_csv(path: Path, records: Sequence[EpisodeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_episode_record_to_csv_row(record) for record in records]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary_rows: Sequence[Dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def write_summary_text(path: Path, summary_rows: Sequence[Dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Assignment 4 POMDP results",
        "==========================",
        "",
        "task | policy | avg_reward | success_rate | avg_inspects | avg_len",
        "---- | ------ | ---------- | ------------ | ------------ | -------",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['task']} | {row['policy']} | "
            f"{float(row['average_total_reward']):.3f} | "
            f"{float(row['success_rate']):.3f} | "
            f"{float(row['average_inspect_actions']):.3f} | "
            f"{float(row['average_episode_length']):.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_episode_json(path: Path, record: EpisodeResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
