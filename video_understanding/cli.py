from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .ab_eval import evaluate_ab
from .asr import transcribe_audio
from .config import load_config
from .fusion import fuse_files
from .media import (
    download_video,
    enforce_duration_limit,
    extract_audio,
    extract_frames,
    probe_duration,
    safe_stem,
    split_windows,
)
from .summarize import summarize_context
from .utils import PipelineError, ensure_dir, write_jsonl
from .vl_client import analyze_visual_segment


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config.")


def workdir_for(config: dict[str, Any], video_or_name: str, explicit: str | None) -> Path:
    if explicit:
        return ensure_dir(explicit)
    return ensure_dir(Path(config.get("workdir", "runs")) / safe_stem(video_or_name))


def max_duration_seconds_for(args: argparse.Namespace, video_config: dict[str, Any]) -> float | None:
    value = getattr(args, "max_duration_seconds", None)
    if value is None:
        value = video_config.get("max_duration_seconds")
    if value is None:
        return None
    limit = float(value)
    if limit < 0:
        raise PipelineError("max_duration_seconds must be greater than or equal to 0")
    return limit


def enforce_configured_duration_limit(
    *,
    duration: float,
    args: argparse.Namespace,
    video_config: dict[str, Any],
    video_path: str | Path,
) -> None:
    enforce_duration_limit(
        duration,
        max_duration_seconds_for(args, video_config),
        video_path=video_path,
    )


def command_download(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output_dir = Path(args.output_dir or config.get("workdir", "runs")) / "downloads"
    path = download_video(args.source, output_dir, config.get("download"))
    print(path)
    return 0


def command_vl(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    video_config = config["video"]
    vl_config = config["vl"]
    workdir = workdir_for(config, args.video, args.workdir)
    video_path = download_video(args.video, workdir / "source", config.get("download"))
    duration = probe_duration(video_path)
    enforce_configured_duration_limit(
        duration=duration,
        args=args,
        video_config=video_config,
        video_path=video_path,
    )

    output = Path(args.output or workdir / "visual.jsonl")
    visual_rows: list[dict[str, Any]] = []
    windows = split_windows(duration, float(args.segment_seconds or video_config["segment_seconds"]))
    for index, (start, end) in enumerate(windows):
        frame_dir = workdir / "frames" / f"{index:04d}_{int(start):06d}_{int(end):06d}"
        frames = extract_frames(
            video_path,
            frame_dir,
            fps=float(args.fps or video_config["fps"]),
            start=start,
            end=end,
            max_side=int(args.max_side or video_config["max_side"]),
            jpeg_quality=int(video_config.get("jpeg_quality", 3)),
        )
        row = analyze_visual_segment(
            frames,
            segment_start=start,
            segment_end=end,
            base_url=vl_config["base_url"],
            api_key=vl_config.get("api_key"),
            model=vl_config["model"],
            temperature=float(vl_config.get("temperature", 0.0)),
            max_tokens=int(vl_config.get("max_tokens", 1800)),
            timeout_seconds=int(vl_config.get("timeout_seconds", 600)),
            extra_prompt=args.prompt,
        )
        visual_rows.append(row)
        write_jsonl(output, visual_rows)
        print(f"VL {index + 1}/{len(windows)} {start:.1f}-{end:.1f}s")
    print(output)
    return 0


def command_asr(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    video_config = config["video"]
    workdir = workdir_for(config, args.video, args.workdir)
    video_path = download_video(args.video, workdir / "source", config.get("download"))
    duration = probe_duration(video_path)
    enforce_configured_duration_limit(
        duration=duration,
        args=args,
        video_config=video_config,
        video_path=video_path,
    )
    audio_path = extract_audio(video_path, args.audio or workdir / "audio.wav")
    rows = transcribe_audio(audio_path, config["asr"])
    output = Path(args.output or workdir / "asr.jsonl")
    write_jsonl(output, rows)
    print(output)
    return 0


def command_fuse(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output_jsonl = args.output_jsonl or "fused.jsonl"
    output_markdown = args.output_markdown or "context.md"
    fuse_files(
        args.visual,
        args.asr,
        output_jsonl=output_jsonl,
        output_markdown=output_markdown,
        window_seconds=float(args.window_seconds or config["fusion"]["window_seconds"]),
    )
    print(output_markdown)
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    text = summarize_context(
        args.context,
        args.output,
        question=args.question,
        config=config["summary"],
    )
    print(text)
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    workdir = workdir_for(config, args.video, args.workdir)
    video_path = download_video(args.video, workdir / "source", config.get("download"))
    duration = probe_duration(video_path)

    visual_path = workdir / "visual.jsonl"
    asr_path = workdir / "asr.jsonl"
    fused_path = workdir / "fused.jsonl"
    context_path = workdir / "context.md"
    summary_path = workdir / "summary.md"

    video_config = config["video"]
    enforce_configured_duration_limit(
        duration=duration,
        args=args,
        video_config=video_config,
        video_path=video_path,
    )
    vl_config = config["vl"]
    visual_rows: list[dict[str, Any]] = []
    windows = split_windows(duration, float(args.segment_seconds or video_config["segment_seconds"]))
    for index, (start, end) in enumerate(windows):
        frame_dir = workdir / "frames" / f"{index:04d}_{int(start):06d}_{int(end):06d}"
        frames = extract_frames(
            video_path,
            frame_dir,
            fps=float(args.fps or video_config["fps"]),
            start=start,
            end=end,
            max_side=int(args.max_side or video_config["max_side"]),
            jpeg_quality=int(video_config.get("jpeg_quality", 3)),
        )
        row = analyze_visual_segment(
            frames,
            segment_start=start,
            segment_end=end,
            base_url=vl_config["base_url"],
            api_key=vl_config.get("api_key"),
            model=vl_config["model"],
            temperature=float(vl_config.get("temperature", 0.0)),
            max_tokens=int(vl_config.get("max_tokens", 1800)),
            timeout_seconds=int(vl_config.get("timeout_seconds", 600)),
            extra_prompt=args.prompt,
        )
        visual_rows.append(row)
        write_jsonl(visual_path, visual_rows)
        print(f"VL {index + 1}/{len(windows)} {start:.1f}-{end:.1f}s")

    audio_path = extract_audio(video_path, workdir / "audio.wav")
    asr_rows = transcribe_audio(audio_path, config["asr"])
    write_jsonl(asr_path, asr_rows)

    fuse_files(
        visual_path,
        asr_path,
        output_jsonl=fused_path,
        output_markdown=context_path,
        window_seconds=float(config["fusion"]["window_seconds"]),
    )

    if not args.skip_summary:
        summarize_context(
            str(context_path),
            str(summary_path),
            question=args.question,
            config=config["summary"],
        )

    print(workdir)
    return 0


def command_ab_eval(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    text = evaluate_ab(
        args.vl_asr_context,
        args.omni_context,
        args.output,
        config=config["ab_eval"],
    )
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vu", description="Video understanding pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download a video URL through adapters.")
    add_common(download)
    download.add_argument("source", help="Video URL or local path.")
    download.add_argument("--output-dir", default=None)
    download.set_defaults(func=command_download)

    fetch = subparsers.add_parser("fetch", help="Download a video URL through configured adapters.")
    add_common(fetch)
    fetch.add_argument("source", help="Video URL, share text, or local path.")
    fetch.add_argument("--output-dir", default=None)
    fetch.set_defaults(func=command_download)

    vl = subparsers.add_parser("vl", help="Run visual/OCR understanding through vLLM.")
    add_common(vl)
    vl.add_argument("video", help="Local video path or URL.")
    vl.add_argument("--workdir", default=None)
    vl.add_argument("--output", default=None)
    vl.add_argument("--fps", type=float, default=None)
    vl.add_argument("--segment-seconds", type=float, default=None)
    vl.add_argument("--max-side", type=int, default=None)
    vl.add_argument(
        "--max-duration-seconds",
        type=float,
        default=None,
        help="Reject analysis when probed video duration exceeds this limit. Use 0 to disable.",
    )
    vl.add_argument("--prompt", default=None, help="Extra per-segment instruction.")
    vl.set_defaults(func=command_vl)

    asr = subparsers.add_parser("asr", help="Extract audio and run ASR.")
    add_common(asr)
    asr.add_argument("video", help="Local video path or URL.")
    asr.add_argument("--workdir", default=None)
    asr.add_argument("--audio", default=None)
    asr.add_argument("--output", default=None)
    asr.add_argument(
        "--max-duration-seconds",
        type=float,
        default=None,
        help="Reject analysis when probed video duration exceeds this limit. Use 0 to disable.",
    )
    asr.set_defaults(func=command_asr)

    fuse = subparsers.add_parser("fuse", help="Fuse visual/OCR and ASR JSONL by timestamp.")
    add_common(fuse)
    fuse.add_argument("--visual", required=True)
    fuse.add_argument("--asr", required=True)
    fuse.add_argument("--output-jsonl", default=None)
    fuse.add_argument("--output-markdown", default=None)
    fuse.add_argument("--window-seconds", type=float, default=None)
    fuse.set_defaults(func=command_fuse)

    summarize = subparsers.add_parser("summarize", help="Summarize or QA over fused context.")
    add_common(summarize)
    summarize.add_argument("--context", required=True)
    summarize.add_argument("--output", required=True)
    summarize.add_argument("--question", default=None)
    summarize.set_defaults(func=command_summarize)

    run = subparsers.add_parser("run", help="Run VL, ASR, fusion, and optional summary.")
    add_common(run)
    run.add_argument("video", help="Local video path or URL.")
    run.add_argument("--workdir", default=None)
    run.add_argument("--fps", type=float, default=None)
    run.add_argument("--segment-seconds", type=float, default=None)
    run.add_argument("--max-side", type=int, default=None)
    run.add_argument(
        "--max-duration-seconds",
        type=float,
        default=None,
        help="Reject analysis when probed video duration exceeds this limit. Use 0 to disable.",
    )
    run.add_argument("--prompt", default=None)
    run.add_argument("--question", default=None)
    run.add_argument("--skip-summary", action="store_true")
    run.set_defaults(func=command_run)

    ab_eval = subparsers.add_parser("ab-eval", help="Compare VL+ASR output against Omni output.")
    add_common(ab_eval)
    ab_eval.add_argument("--vl-asr-context", required=True)
    ab_eval.add_argument("--omni-context", required=True)
    ab_eval.add_argument("--output", required=True)
    ab_eval.set_defaults(func=command_ab_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
