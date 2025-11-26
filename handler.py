import os
from types import SimpleNamespace

import runpod

from generate import EXAMPLE_PROMPT, generate, _validate_args


def build_args(event_input):
    """
    Build an argparse-style namespace from the RunPod event input.

    This mirrors the arguments defined in `_parse_args` in `generate.py`
    so we can call `generate(args)` directly without going through CLI parsing.
    """
    # Basic inputs
    task = event_input.get("task", "t2v-14B")
    size = event_input.get("size", "1280*720")

    # Where model weights are stored inside the container.
    # You should mount your WAN checkpoints here, e.g. /workspace/Wan2.1-T2V-14B
    ckpt_dir = event_input.get(
        "ckpt_dir",
        os.environ.get("WAN_CKPT_DIR", "./Wan2.1-T2V-14B"),
    )

    # Common generation options
    prompt = event_input.get(
        "prompt",
        EXAMPLE_PROMPT.get(task, {}).get("prompt"),
    )

    args = SimpleNamespace(
        # core
        task=task,
        size=size,
        frame_num=event_input.get("frame_num", None),
        ckpt_dir=ckpt_dir,
        offload_model=event_input.get("offload_model", None),
        ulysses_size=int(event_input.get("ulysses_size", 1)),
        ring_size=int(event_input.get("ring_size", 1)),
        t5_fsdp=bool(event_input.get("t5_fsdp", False)),
        t5_cpu=bool(event_input.get("t5_cpu", False)),
        dit_fsdp=bool(event_input.get("dit_fsdp", False)),
        save_file=event_input.get("save_file", None),
        src_video=event_input.get("src_video", None),
        src_mask=event_input.get("src_mask", None),
        src_ref_images=event_input.get("src_ref_images", None),
        prompt=prompt,
        use_prompt_extend=bool(event_input.get("use_prompt_extend", False)),
        prompt_extend_method=event_input.get("prompt_extend_method", "local_qwen"),
        prompt_extend_model=event_input.get("prompt_extend_model", None),
        prompt_extend_target_lang=event_input.get("prompt_extend_target_lang", "zh"),
        base_seed=int(event_input.get("base_seed", -1)),
        image=event_input.get("image", None),
        first_frame=event_input.get("first_frame", None),
        last_frame=event_input.get("last_frame", None),
        sample_solver=event_input.get("sample_solver", "unipc"),
        sample_steps=event_input.get("sample_steps", None),
        sample_shift=event_input.get("sample_shift", None),
        sample_guide_scale=float(event_input.get("sample_guide_scale", 5.0)),
    )

    # Let the existing validation logic fill in any derived defaults and checks.
    _validate_args(args)

    return args


def handler(event):
    """
    RunPod serverless handler.

    Expected event format:
    {
        "input": {
            "task": "t2v-14B",
            "size": "1280*720",
            "ckpt_dir": "/workspace/Wan2.1-T2V-14B",
            "prompt": "Two anthropomorphic cats ...",
            ...
        }
    }
    """
    event_input = (event or {}).get("input", {}) or {}

    args = build_args(event_input)

    # Run generation (will write output to args.save_file)
    generate(args)

    # Return metadata about the run and where the file was saved.
    return {
        "task": args.task,
        "size": args.size,
        "prompt": args.prompt,
        "frame_num": args.frame_num,
        "save_file": args.save_file,
    }


runpod.serverless.start({"handler": handler})


