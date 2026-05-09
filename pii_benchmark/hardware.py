"""Hardware and software metadata capture."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(args: list[str], cwd: Path | None = None, timeout: float = 5.0) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _memory_info() -> dict[str, Any]:
    try:
        import psutil  # type: ignore
    except Exception:
        return {}
    mem = psutil.virtual_memory()
    return {
        "total_bytes": int(mem.total),
        "available_bytes": int(mem.available),
    }


def _torch_info() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception:
        return {"available": False}

    info: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        ),
    }
    if torch.cuda.is_available():
        devices: list[dict[str, Any]] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "major": int(props.major),
                "minor": int(props.minor),
                "multi_processor_count": int(props.multi_processor_count),
            })
        info["cuda_devices"] = devices
    return info


def _nvidia_smi_info() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    output = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return []
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        index, name, memory_mb, driver = parts[:4]
        gpus.append({
            "index": int(index),
            "name": name,
            "memory_total_mb": int(float(memory_mb)),
            "driver_version": driver,
        })
    return gpus


def _mac_display_info() -> list[dict[str, Any]]:
    if platform.system() != "Darwin":
        return []
    output = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=10.0)
    if not output:
        return []
    try:
        raw = json.loads(output)
    except json.JSONDecodeError:
        return []
    displays = raw.get("SPDisplaysDataType", [])
    if not isinstance(displays, list):
        return []
    result: list[dict[str, Any]] = []
    for item in displays:
        if not isinstance(item, dict):
            continue
        result.append({
            "name": item.get("sppci_model") or item.get("_name"),
            "vendor": item.get("spdisplays_vendor"),
            "metal": item.get("spdisplays_metal"),
            "vram": item.get("spdisplays_vram"),
        })
    return result


def _tegrastats_sample(timeout_s: float = 3.0) -> str | None:
    if not shutil.which("tegrastats"):
        return None
    try:
        completed = subprocess.run(
            ["tegrastats", "--interval", "1000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_s,
        )
        output = completed.stdout
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout
    except Exception:
        return None

    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    if not output:
        return None
    return str(output).splitlines()[0].strip() or None


def _parse_tegrastats_power_watts(sample: str | None) -> float | None:
    if not sample:
        return None
    # Common Jetson format: "VDD_IN 4014mW/4014mW ...".
    match = re.search(r"\b(?:VDD_IN|POM_5V_IN)\s+(\d+(?:\.\d+)?)mW", sample)
    if not match:
        return None
    return float(match.group(1)) / 1000.0


def _jetson_info() -> dict[str, Any]:
    model = _read_text("/proc/device-tree/model")
    nv_tegra = _read_text("/etc/nv_tegra_release")
    tegrastats = _tegrastats_sample()
    is_jetson = bool(
        nv_tegra
        or shutil.which("tegrastats")
        or (model and re.search(r"jetson|tegra|orin", model, re.IGNORECASE))
    )
    info: dict[str, Any] = {
        "detected": is_jetson,
        "model": model,
        "nv_tegra_release": nv_tegra,
        "tegrastats_sample": tegrastats,
        "tegrastats_vdd_in_watts": _parse_tegrastats_power_watts(tegrastats),
    }
    if is_jetson:
        info["nvpmodel"] = _run(["nvpmodel", "-q"], timeout=5.0)
        if shutil.which("jetson_release"):
            info["jetson_release"] = _run(["jetson_release", "-v"], timeout=10.0)
    return info


def _float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _power_info(jetson: dict[str, Any]) -> dict[str, Any]:
    budget_watts = _float_env("PII_BENCH_POWER_WATTS")
    note = os.environ.get("PII_BENCH_POWER_NOTE")
    return {
        "budget_watts": budget_watts,
        "budget_source": "PII_BENCH_POWER_WATTS" if budget_watts else None,
        "note": note,
        "jetson_snapshot_watts": jetson.get("tegrastats_vdd_in_watts"),
    }


def git_metadata(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _run(["git", "branch", "--show-current"], cwd=root)
    dirty_output = _run(["git", "status", "--porcelain"], cwd=root)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty_output),
    }


def capture_metadata(root: str | Path) -> dict[str, Any]:
    jetson = _jetson_info()
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "cpu": {
            "logical_count": os.cpu_count(),
        },
        "memory": _memory_info(),
        "packages": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "opf": _package_version("opf"),
            "psutil": _package_version("psutil"),
        },
        "accelerators": {
            "torch": _torch_info(),
            "nvidia_smi": _nvidia_smi_info(),
            "mac_displays": _mac_display_info(),
            "jetson": jetson,
        },
        "power": _power_info(jetson),
        "git": git_metadata(root),
    }


def hardware_label(metadata: dict[str, Any]) -> str:
    accel = metadata.get("accelerators", {})
    jetson = accel.get("jetson") or {}
    if jetson.get("detected") and jetson.get("model"):
        return str(jetson.get("model"))

    nvidia = accel.get("nvidia_smi") or []
    if nvidia:
        return " + ".join(gpu.get("name", "NVIDIA GPU") for gpu in nvidia)

    torch_info = accel.get("torch") or {}
    cuda_devices = torch_info.get("cuda_devices") or []
    if cuda_devices:
        return " + ".join(gpu.get("name", "CUDA GPU") for gpu in cuda_devices)

    mac_displays = accel.get("mac_displays") or []
    if mac_displays:
        names = [item.get("name") for item in mac_displays if item.get("name")]
        if names:
            return " + ".join(str(name) for name in names)

    if torch_info.get("mps_available"):
        return "Apple Silicon MPS"
    return str(metadata.get("platform", {}).get("machine") or "CPU")
