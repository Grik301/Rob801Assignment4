#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Optional, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np


def _early_configure_mujoco_backend(argv: Sequence[str]) -> None:
    chosen_backend: Optional[str] = None
    wants_video = False
    for idx, token in enumerate(argv):
        if token == "--video-output":
            wants_video = True
        if token == "--mujoco-gl" and idx + 1 < len(argv):
            chosen_backend = str(argv[idx + 1])
    if chosen_backend is None and wants_video:
        chosen_backend = "egl"
    if chosen_backend:
        os.environ["MUJOCO_GL"] = chosen_backend
        if chosen_backend == "osmesa":
            os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


_early_configure_mujoco_backend(sys.argv[1:])

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pomdp import (  # noqa: E402
    EpisodeResult,
    baseline_action,
    evaluate_policy_bundle,
    get_task_spec,
    load_q_table,
    q_policy_action,
    run_episode,
    save_q_table,
    summarize_records,
    train_q_learning,
    write_episode_json,
    write_summary_csv,
    write_summary_text,
    write_training_history_csv,
    write_trial_records_csv,
)
from mujoco_replay import execute_episode_in_viewer  # noqa: E402
from mujoco_scene import (  # noqa: E402
    configure_mujoco_backend,
    create_runtime_scene_xml,
    load_scene_handles,
    runtime_temp_paths,
)


def _apply_default_camera(camera) -> None:
    import mujoco  # type: ignore

    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 118.0
    camera.elevation = -28.0
    camera.distance = 1.7
    camera.lookat[:] = np.asarray([0.58, 0.0, 0.26], dtype=float)


class OffscreenVideoRecorder:
    def __init__(self, scene, output_path: Path, width: int, height: int, fps: float) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for --video-output.") from exc
        import mujoco  # type: ignore

        self.scene = scene
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.frame_dt = 1.0 / max(self.fps, 1e-6)
        self.next_frame_time = 0.0
        self.cv2 = cv2
        self.renderer = mujoco.Renderer(scene.model, height=self.height, width=self.width)
        self.camera = mujoco.MjvCamera()
        _apply_default_camera(self.camera)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.width, self.height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {self.output_path}")

    def _write_frame(self) -> None:
        self.renderer.update_scene(self.scene.data, camera=self.camera)
        rgb = self.renderer.render()
        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)

    def sync(self) -> None:
        current_time = float(self.scene.data.time)
        while current_time + 1e-9 >= self.next_frame_time:
            self._write_frame()
            self.next_frame_time += self.frame_dt

    def append_hold(self, duration_s: float) -> None:
        frames = max(1, int(round(max(duration_s, 0.0) * self.fps)))
        for _ in range(frames):
            self._write_frame()

    def close(self) -> None:
        self.writer.release()
        self.renderer.close()


def _policy_runner(task_name: str, policy: str, q_table, belief_step: float):
    task = get_task_spec(task_name)
    if policy == "baseline":
        return lambda seed: run_episode(
            task=task,
            policy_name="baseline",
            seed=seed,
            policy_fn=lambda t, b, s, ah, oh, rng: baseline_action(t, b, s, ah),
        )
    if policy == "qlearning":
        return lambda seed: run_episode(
            task=task,
            policy_name="qlearning",
            seed=seed,
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
    raise ValueError(f"Unsupported policy {policy!r}")


def _train_or_load_q_table(task_name: str, args) -> tuple[dict, list[dict[str, float]], Path]:
    task = get_task_spec(task_name)
    output_dir = Path(args.output_dir)
    q_table_path = Path(args.q_table) if getattr(args, "q_table", None) else output_dir / f"task_{task.name}_q_table.json"
    if q_table_path.exists():
        return load_q_table(q_table_path, task), [], q_table_path
    q_table, history = train_q_learning(
        task=task,
        num_episodes=int(args.train_episodes),
        seed=int(args.train_seed),
        alpha=float(args.alpha),
        gamma=float(args.gamma),
        epsilon_start=float(args.epsilon_start),
        epsilon_end=float(args.epsilon_end),
        belief_step=float(args.belief_step),
    )
    save_q_table(q_table_path, task, q_table)
    training_csv = output_dir / f"task_{task.name}_training.csv"
    write_training_history_csv(training_csv, history)
    return q_table, history, q_table_path


def _print_episode_summary(episode: EpisodeResult) -> None:
    print(
        json.dumps(
            {
                "task": episode.task,
                "policy": episode.policy,
                "seed": episode.seed,
                "hidden_truth": episode.hidden_truth,
                "total_reward": episode.total_reward,
                "success": episode.success,
                "inspect_count": episode.inspect_count,
                "episode_length": episode.episode_length,
                "actions": episode.actions,
                "observations": episode.observations,
                "belief_sequence": episode.belief_sequence,
            },
            indent=2,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROB801 Assignment 4 POMDP manipulation reference pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train a tabular Q-learning policy for one task")
    train_parser.add_argument("--task", required=True, choices=["A", "B"])
    train_parser.add_argument("--train-episodes", type=int, default=3000)
    train_parser.add_argument("--train-seed", type=int, default=0)
    train_parser.add_argument("--belief-step", type=float, default=0.1)
    train_parser.add_argument("--alpha", type=float, default=0.25)
    train_parser.add_argument("--gamma", type=float, default=0.95)
    train_parser.add_argument("--epsilon-start", type=float, default=0.35)
    train_parser.add_argument("--epsilon-end", type=float, default=0.05)
    train_parser.add_argument("--output-dir", type=str, default=str(SCRIPT_DIR / "results"))
    train_parser.add_argument("--q-table", type=str, default=None)
    train_parser.set_defaults(func=_train_cli)

    run_parser = subparsers.add_parser("run", help="Run one seeded episode with either the baseline or Q-learning policy")
    run_parser.add_argument("--task", required=True, choices=["A", "B"])
    run_parser.add_argument("--policy", choices=["baseline", "qlearning"], default="qlearning")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--train-episodes", type=int, default=3000)
    run_parser.add_argument("--train-seed", type=int, default=0)
    run_parser.add_argument("--belief-step", type=float, default=0.1)
    run_parser.add_argument("--alpha", type=float, default=0.25)
    run_parser.add_argument("--gamma", type=float, default=0.95)
    run_parser.add_argument("--epsilon-start", type=float, default=0.35)
    run_parser.add_argument("--epsilon-end", type=float, default=0.05)
    run_parser.add_argument("--output-dir", type=str, default=str(SCRIPT_DIR / "results"))
    run_parser.add_argument("--q-table", type=str, default=None)
    run_parser.add_argument("--simulate", action="store_true", help="Replay the chosen episode in MuJoCo.")
    run_parser.add_argument("--dry-run", action="store_true", help="Build the runtime scene without launching the viewer.")
    run_parser.add_argument("--cube-size", type=float, default=0.044)
    run_parser.add_argument("--hover-height", type=float, default=0.12)
    run_parser.add_argument("--realtime-scale", type=float, default=1.0)
    run_parser.add_argument("--mujoco-gl", type=str, default=None)
    run_parser.add_argument("--keep-runtime-xml", action="store_true")
    run_parser.add_argument("--return-home-at-end", action="store_true")
    run_parser.add_argument("--video-output", type=str, default=None)
    run_parser.add_argument("--video-width", type=int, default=1280)
    run_parser.add_argument("--video-height", type=int, default=720)
    run_parser.add_argument("--video-fps", type=float, default=30.0)
    run_parser.set_defaults(func=_run_cli)

    eval_parser = subparsers.add_parser("evaluate", help="Train Q-learning and compare it against the baseline on both tasks")
    eval_parser.add_argument("--tasks", nargs="+", default=["A", "B"], choices=["A", "B"])
    eval_parser.add_argument("--train-episodes", type=int, default=3000)
    eval_parser.add_argument("--train-seed", type=int, default=0)
    eval_parser.add_argument("--test-episodes", type=int, default=10)
    eval_parser.add_argument("--seed-offset", type=int, default=0)
    eval_parser.add_argument("--belief-step", type=float, default=0.1)
    eval_parser.add_argument("--alpha", type=float, default=0.25)
    eval_parser.add_argument("--gamma", type=float, default=0.95)
    eval_parser.add_argument("--epsilon-start", type=float, default=0.35)
    eval_parser.add_argument("--epsilon-end", type=float, default=0.05)
    eval_parser.add_argument("--output-dir", type=str, default=str(SCRIPT_DIR / "results"))
    eval_parser.set_defaults(func=_evaluate_cli)
    return parser


def _train_cli(args: argparse.Namespace) -> int:
    task = get_task_spec(args.task)
    q_table, history, q_table_path = _train_or_load_q_table(task.name, args)
    if history:
        print(f"Trained task {task.name} for {args.train_episodes} episodes.")
    else:
        print(f"Loaded existing Q-table for task {task.name}.")
    print(f"Q-table path: {q_table_path}")
    print(f"Stored {len(q_table)} discretized belief states.")
    return 0


def _run_cli(args: argparse.Namespace) -> int:
    task = get_task_spec(args.task)
    q_table = None
    if args.policy == "qlearning":
        q_table, _history, _path = _train_or_load_q_table(task.name, args)
    runner = _policy_runner(task.name, args.policy, q_table, float(args.belief_step))
    episode = runner(int(args.seed))
    _print_episode_summary(episode)

    output_dir = Path(args.output_dir)
    write_episode_json(output_dir / f"task_{task.name}_{args.policy}_seed{args.seed}.json", episode)

    if not args.simulate and not args.video_output:
        return 0

    configure_mujoco_backend(args.mujoco_gl)
    cube_half_extent = 0.5 * float(args.cube_size)
    runtime_xml_path = create_runtime_scene_xml(task.scene_spec, cube_half_extent)
    try:
        scene = load_scene_handles(task.scene_spec, runtime_xml_path)
        print(f"MuJoCo runtime scene: {runtime_xml_path}")
        if args.dry_run:
            print("Dry run requested; not launching the viewer.")
            return 0

        if args.video_output:
            recorder = OffscreenVideoRecorder(
                scene=scene,
                output_path=Path(args.video_output),
                width=int(args.video_width),
                height=int(args.video_height),
                fps=float(args.video_fps),
            )
            try:
                execute_episode_in_viewer(
                    task=task,
                    episode=episode,
                    scene=scene,
                    viewer_handle=recorder,
                    cube_half_extent=cube_half_extent,
                    hover_height=float(args.hover_height),
                    realtime_scale=float(args.realtime_scale),
                    return_home_at_end=bool(args.return_home_at_end),
                )
                recorder.append_hold(1.0 / max(float(args.realtime_scale), 1e-6))
            finally:
                recorder.close()
            print(f"Saved video to {Path(args.video_output).resolve()}")
            return 0

        import mujoco.viewer  # type: ignore

        with mujoco.viewer.launch_passive(scene.model, scene.data, show_left_ui=True, show_right_ui=True) as viewer_handle:
            with viewer_handle.lock():
                _apply_default_camera(viewer_handle.cam)
            execute_episode_in_viewer(
                task=task,
                episode=episode,
                scene=scene,
                viewer_handle=viewer_handle,
                cube_half_extent=cube_half_extent,
                hover_height=float(args.hover_height),
                realtime_scale=float(args.realtime_scale),
                return_home_at_end=bool(args.return_home_at_end),
            )
            end_t = time.perf_counter() + 1.2 / max(float(args.realtime_scale), 1e-6)
            while viewer_handle.is_running() and time.perf_counter() < end_t:
                viewer_handle.sync()
                time.sleep(0.01)
    finally:
        if not args.keep_runtime_xml:
            for temp_path in runtime_temp_paths(runtime_xml_path):
                if temp_path.exists():
                    temp_path.unlink()
            if runtime_xml_path.parent.exists():
                runtime_xml_path.parent.rmdir()
    return 0


def _evaluate_cli(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    for task_name in args.tasks:
        task = get_task_spec(task_name)
        q_table, history, q_table_path = _train_or_load_q_table(task.name, args)
        if history:
            print(f"Trained task {task.name}; saved Q-table to {q_table_path}")
        seeds = [int(args.seed_offset) + idx for idx in range(int(args.test_episodes))]
        task_records = evaluate_policy_bundle(
            task=task,
            q_table=q_table,
            seeds=seeds,
            belief_step=float(args.belief_step),
        )
        all_records.extend(task_records)
        qlearning_records = [record for record in task_records if record.policy == "qlearning"]
        if qlearning_records:
            write_episode_json(output_dir / f"task_{task.name}_representative_qlearning_seed{qlearning_records[0].seed}.json", qlearning_records[0])
        failure_records = [record for record in task_records if not record.success]
        if failure_records:
            failure = failure_records[0]
            write_episode_json(output_dir / f"task_{task.name}_failure_case_{failure.policy}_seed{failure.seed}.json", failure)
    summary_rows = summarize_records(all_records)
    write_trial_records_csv(output_dir / "trial_records.csv", all_records)
    write_summary_csv(output_dir / "summary_results.csv", summary_rows)
    write_summary_text(output_dir / "summary_results.txt", summary_rows)
    print("Saved:")
    print(f"  {output_dir / 'trial_records.csv'}")
    print(f"  {output_dir / 'summary_results.csv'}")
    print(f"  {output_dir / 'summary_results.txt'}")
    print(json.dumps(summary_rows, indent=2))
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
