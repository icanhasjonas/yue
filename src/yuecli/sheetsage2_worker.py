"""Runs INSIDE envs/sheetsage2 (not the yue environment). Prints one JSON line at the end."""
import argparse
import json
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--melody-only", action="store_true")
    p.add_argument("--render-score", action="store_true")
    p.add_argument("--device")
    p.add_argument("--dtype", choices=("bf16", "fp32"))
    p.add_argument("--preset", choices=("default", "paper"))
    p.add_argument("--max-seconds", type=float)
    a = p.parse_args()

    import torch
    from transformers import AutoModel

    device = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    start = time.perf_counter()
    model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True).eval().to(device)
    loaded = time.perf_counter()

    def progress(value):
        if value.get("stage") == "encoding":
            print(f"window {value['window']}/{value['windows']}", flush=True)

    # Mirrors the model repo's own infer.py call.
    kwargs = {"output_dir": a.output, "dtype": a.dtype or "bf16", "progress": progress}
    if a.melody_only:
        kwargs["melody_only"] = True
    if a.render_score:
        kwargs["render_score"] = "pdf,svg,png"
    if a.preset:
        kwargs["preset"] = a.preset
    if a.max_seconds:
        kwargs["max_seconds"] = a.max_seconds
    result = model.transcribe(a.input, **kwargs)
    if not result.get("abc"):
        print(f"no ABC score was produced: {result.get('abc_error') or 'unknown reason'}", file=sys.stderr)
        return 1
    print(json.dumps({"device": device, "load_seconds": loaded - start,
                      "transcribe_seconds": time.perf_counter() - loaded,
                      "abc": result["abc"], "output": a.output}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
