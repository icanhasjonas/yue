"""`yue runpod setup | status | teardown`: a guided RunPod serverless deployment from an API key.

Every step is idempotent and recorded in ~/.config/yue/runpod.json, so a setup
interrupted halfway resumes where it stopped. Steps that cost money say what they
cost and ask first (unless --yes).

  1 key        verify the key (free)
  2 image      the worker image, ghcr.io/<owner>/yue:<sha> built by GitHub Actions
  3 registry   RunPod credential to pull a PRIVATE GHCR image (free)
  4 volume     network volume for the Hugging Face cache (paid per GB-month)
  5 template   serverless template: image + HF_HOME on the volume (free)
  6 endpoint   scale-to-zero endpoint on 24 GB GPUs (free while idle)
  7 prime      one job that downloads the weights onto the volume (paid, a few minutes)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time

from ..args import fail
from . import runpod_api as rp

DEFAULT_GPUS = ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A4500", "NVIDIA GeForce RTX 4090", "NVIDIA L4"]
WEIGHTS_GB = 9  # YuE2-3B 7.3 GB + YuE2-Vae; legacy VAE adds ~0.6


def say(reporter, message: str, level: str = "info") -> None:
    if reporter.mode == "text":
        print(("warning: " if level == "warn" else "") + message, file=sys.stderr)
    else:
        reporter.log(message, level=level, code="yue.runpod")


def confirm(s, reporter, question: str) -> bool:
    if s.get("yes"):
        return True
    if reporter.mode != "text" or not sys.stdin.isatty():
        fail(f"{question} -- needs confirmation; re-run with --yes")
    answer = input(f"{question} [y/N] ").strip().lower()
    return answer in ("y", "yes")


def setup(s, reporter) -> int:
    key = rp.api_key(s.get("api_key"))
    cfg = rp.load_config()
    reporter.run_start("runpod setup")

    # 1 key
    rp.rest("GET", "/endpoints", key)
    say(reporter, "✓ API key works")

    # 2 image
    image = s.get("image") or cfg.get("image") or default_image()
    if not image:
        fail("no worker image", ["Push the repo so GitHub Actions builds it, then pass --image ghcr.io/<owner>/yue:<sha>."])
    cfg["image"] = image
    # Regression (2026-09-17): the endpoint was created while GitHub Actions was still
    # building, RunPod failed the pull with IMAGE_NOT_FOUND and stopped the workers.
    if image.startswith("ghcr.io/"):
        wait_for_image(image, s.get("registry_user") or gh_user(), s.get("registry_token") or gh_token(),
                       reporter, wait=bool(s.get("wait_image")))
    say(reporter, f"✓ image {image}")

    # 3 registry auth (GHCR images are private unless you made the package public)
    if image.startswith("ghcr.io/") and not s.get("public_image"):
        if not cfg.get("registry_auth_id"):
            user = s.get("registry_user") or gh_user()
            token = s.get("registry_token") or gh_token()
            if not (user and token):
                fail("a private GHCR image needs a pull credential",
                     ["Pass --registry-user <github user> --registry-token <PAT with read:packages>,",
                      "or `gh auth refresh -s read:packages` and re-run, or --public-image if the package is public."])
            auth = rp.rest("POST", "/containerregistryauth", key,
                           {"name": f"yue-ghcr-{int(time.time())}", "username": user, "password": token})
            cfg["registry_auth_id"] = auth["id"]
            rp.save_config(cfg)
        say(reporter, f"✓ registry credential {cfg['registry_auth_id']}")

    # 4 network volume
    dc = s.get("data_center") or cfg.get("data_center") or pick_data_center(key, s.get("gpu") or DEFAULT_GPUS)
    cfg["data_center"] = dc
    if not s.get("no_volume") and not cfg.get("volume_id"):
        size = int(s.get("volume_gb") or 20)
        if not confirm(s, reporter, f"Create a {size} GB network volume in {dc} for the model weights? "
                                    f"It is billed per GB-month while it exists (see `yue runpod status`)."):
            fail("setup stopped before creating the network volume")
        vol = rp.rest("POST", "/networkvolumes", key, {"name": "yue-weights", "size": size, "dataCenterId": dc})
        cfg["volume_id"] = vol["id"]
        rp.save_config(cfg)
    if cfg.get("volume_id"):
        say(reporter, f"✓ network volume {cfg['volume_id']} in {dc}")

    # 5 template (a new image means a new template; the endpoint is repointed below)
    replaced_template = None
    if cfg.get("template_image") != image or not cfg.get("template_id"):
        replaced_template = cfg.get("template_id")
        body = {"name": f"yue-{image.rsplit(':', 1)[-1][:12]}", "imageName": image, "isServerless": True,
                "containerDiskInGb": 30, "env": {"HF_HOME": "/runpod-volume/hf" if cfg.get("volume_id") else "/root/hf"}}
        if cfg.get("registry_auth_id"):
            body["containerRegistryAuthId"] = cfg["registry_auth_id"]
        tpl = rp.rest("POST", "/templates", key, body)
        cfg["template_id"], cfg["template_image"] = tpl["id"], image
        rp.save_config(cfg)
    say(reporter, f"✓ template {cfg['template_id']}")

    # 6 endpoint
    gpus = s.get("gpu") or cfg.get("gpus") or DEFAULT_GPUS
    body = {"name": "yue", "templateId": cfg["template_id"], "gpuTypeIds": gpus, "gpuCount": 1,
            "workersMin": 0, "workersMax": int(s.get("max_workers") or 1), "idleTimeout": int(s.get("idle_timeout") or 5),
            "flashboot": True, "executionTimeoutMs": int(float(s.get("timeout_minutes") or 10) * 60 * 1000),
            "minCudaVersion": "12.8",
            "scalerType": "QUEUE_DELAY", "scalerValue": 4}
    if cfg.get("volume_id"):
        body["networkVolumeId"] = cfg["volume_id"]
        body["dataCenterIds"] = [dc]
    if not cfg.get("endpoint_id"):
        ep = rp.rest("POST", "/endpoints", key, body)
        cfg["endpoint_id"] = ep["id"]
    else:
        rp.rest("PATCH", f"/endpoints/{cfg['endpoint_id']}", key, body)
    cfg["gpus"] = gpus
    rp.save_config(cfg)
    if replaced_template and replaced_template != cfg["template_id"]:
        # B8: every image change made a new template and orphaned the old one
        try:
            rp.rest("DELETE", f"/templates/{replaced_template}", key)
            say(reporter, f"✓ removed the replaced template {replaced_template}")
        except rp.RunPodError as exc:
            say(reporter, f"could not remove old template {replaced_template}: {exc}", level="warn")
    say(reporter, f"✓ endpoint {cfg['endpoint_id']} ({', '.join(g.replace('NVIDIA ', '') for g in gpus)}; "
                  f"0 workers when idle)")

    # 7 prime
    if cfg.get("volume_id") and not cfg.get("primed") and not s.get("no_prime"):
        if confirm(s, reporter, "Download the model weights onto the volume now? One paid job, a few GPU-minutes."):
            prime(cfg, key, reporter)
            cfg["primed"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            rp.save_config(cfg)
    reporter.result("succeeded", 0, data={k: v for k, v in cfg.items()})
    if reporter.mode == "text":
        print(f"ready: yue generate --prompt \"...\" --lyrics @song.txt --remote runpod")
    return 0


def prime(cfg: dict, key: str, reporter) -> None:
    job = rp.run(cfg["endpoint_id"], key, {"argv": ["__prime__"]})
    say(reporter, f"priming weights (job {job}); the first worker also pulls the image, expect several minutes")
    while True:
        out = rp.stream(cfg["endpoint_id"], job, key)
        for item in out.get("stream") or []:
            part = item.get("output") or {}
            if part.get("k") == "event":
                say(reporter, "  " + part["e"].get("message", ""))
        if out.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT") and not out.get("stream"):
            if out["status"] != "COMPLETED":
                fail(f"priming ended {out['status']}: {rp.status(cfg['endpoint_id'], job, key)}")
            return
        time.sleep(3)


def status(s, reporter) -> int:
    key = rp.api_key(s.get("api_key"))
    cfg = rp.load_config()
    data = {"config": cfg}
    if cfg.get("endpoint_id"):
        data["health"] = rp.health(cfg["endpoint_id"], key)
    if cfg.get("volume_id"):
        data["volume"] = rp.rest("GET", f"/networkvolumes/{cfg['volume_id']}", key)
    if reporter.mode == "text":
        import json
        print(json.dumps(data, indent=2))
    else:
        reporter.run_start("runpod status")
        reporter.result("succeeded", 0, data=data)
    return 0


def teardown(s, reporter) -> int:
    key = rp.api_key(s.get("api_key"))
    cfg = rp.load_config()
    reporter.run_start("runpod teardown")
    for field, path, label in (("endpoint_id", "/endpoints/", "endpoint"), ("template_id", "/templates/", "template")):
        if cfg.get(field):
            rp.rest("DELETE", path + cfg[field], key)
            say(reporter, f"deleted {label} {cfg.pop(field)}")
    if cfg.get("volume_id") and s.get("volume"):
        if confirm(s, reporter, f"Delete network volume {cfg['volume_id']} and the downloaded weights on it?"):
            rp.rest("DELETE", f"/networkvolumes/{cfg['volume_id']}", key)
            say(reporter, f"deleted network volume {cfg.pop('volume_id')}")
            cfg.pop("primed", None)
    elif cfg.get("volume_id"):
        say(reporter, f"kept network volume {cfg['volume_id']} (pass --volume to delete it too)")
    if cfg.get("registry_auth_id") and s.get("volume"):
        rp.rest("DELETE", f"/containerregistryauth/{cfg['registry_auth_id']}", key)
        cfg.pop("registry_auth_id")
    cfg.pop("template_image", None)
    rp.save_config(cfg)
    reporter.result("succeeded", 0, data=cfg)
    return 0


def image_exists(image: str, user: str | None, token: str | None) -> bool:
    """Ask GHCR for the manifest with a pull token (the same check RunPod's pull makes)."""
    import base64
    import json
    import urllib.error
    import urllib.request
    name, tag = image[len("ghcr.io/"):].rsplit(":", 1)
    req = urllib.request.Request(f"https://ghcr.io/token?service=ghcr.io&scope=repository:{name}:pull")
    if user and token:
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode())
    with urllib.request.urlopen(req, timeout=20) as resp:
        bearer = json.loads(resp.read())["token"]
    head = urllib.request.Request(f"https://ghcr.io/v2/{name}/manifests/{tag}", method="HEAD", headers={
        "Authorization": f"Bearer {bearer}",
        "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json, "
                  "application/vnd.oci.image.manifest.v1+json"})
    try:
        with urllib.request.urlopen(head, timeout=20):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 401, 403):
            return False
        raise


def wait_for_image(image: str, user, token, reporter, *, wait: bool) -> None:
    if image_exists(image, user, token):
        return
    if not wait:
        fail(f"{image} is not in the registry yet (still building, or never pushed)",
             ["Check `gh run list`, then re-run setup, or add --wait-image to wait for the build."])
    say(reporter, f"waiting for {image} to be pushed (GitHub Actions build)...")
    deadline = time.monotonic() + 60 * 60
    while time.monotonic() < deadline:
        time.sleep(30)
        if image_exists(image, user, token):
            return
    fail(f"{image} did not appear within an hour")


def pick_data_center(key: str, gpus: list[str]) -> str:
    """The listed data center with the best Pod stock across the wanted GPUs (a proxy for serverless supply)."""
    rank = {"High": 3, "Medium": 2, "Low": 1}
    stock = rp.gpu_stock(key)
    scored = sorted(((sum(rank.get(av.get(g, ""), 0) for g in gpus), dc) for dc, av in stock.items()), reverse=True)
    if not scored or scored[0][0] == 0:
        fail("no data center currently lists any of the wanted GPUs", ["Pass --data-center, or widen --gpu."])
    return scored[0][1]


def default_image() -> str | None:
    if not shutil.which("git"):
        return None
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                             cwd=__file__.rsplit("/src/", 1)[0]).stdout.strip()
        owner = gh_user()
    except (subprocess.CalledProcessError, OSError):
        return None
    return f"ghcr.io/{owner}/yue:{sha}" if owner and sha else None


def gh_user() -> str | None:
    if not shutil.which("gh"):
        return None
    out = subprocess.run(["gh", "api", "user", "-q", ".login"], capture_output=True, text=True)
    return out.stdout.strip() or None


def gh_token() -> str | None:
    if not shutil.which("gh"):
        return None
    out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    return out.stdout.strip() or None
