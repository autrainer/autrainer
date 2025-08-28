import logging
import os
import platform
import threading
from typing import Any, Callable, Optional, Type

from omegaconf import OmegaConf
import psutil
import torch


class ThreadManager:
    """Thread manager singleton for tracking spawned threads."""

    _instance = None

    def __new__(cls: Type["ThreadManager"]) -> "ThreadManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self._threads = []
        self._lock = threading.Lock()

    def spawn(self, target: Callable, *args: Any) -> None:
        """Spawn and start a new thread for the target function.

        Args:
            target: Function to run in the thread.
            args: Arguments to pass to the function.
        """
        with self._lock:
            t = threading.Thread(target=target, args=args or ())
            t.start()
            self._threads.append(t)

    def join(self) -> None:
        """Join all spawned threads."""
        with self._lock:
            threads = self._threads[:]
            self._threads.clear()

        for t in threads:
            t.join()

        with self._lock:
            if self._threads:
                return


def get_gpu_info(device: torch.device) -> Optional[dict]:
    """Get GPU information of the current system.

    Args:
        device: Device to get the GPU information from.

    Returns:
        Dictionary containing GPU information. None if no GPU is available.
    """
    if not torch.cuda.is_available():
        return None

    gpu = torch.cuda.get_device_properties(device)
    return {"name": gpu.name, "memory_gb": round(gpu.total_memory / (1024**3))}


def get_system_info() -> dict:
    """Get system information of the current system.

    Returns:
        Dictionary containing system information.
    """
    s = {"os": platform.system(), "platform": platform.platform()}
    if os.environ.get("SLURMD_NODENAME"):
        s.update(
            {
                "node": os.environ.get("SLURMD_NODENAME"),
                "memory_gb": round(int(os.environ.get("SLURM_MEM_PER_NODE", 0)) / 1024),
                "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", 0)),
            }
        )
    else:
        s.update(
            {
                "memory_gb": round(psutil.virtual_memory().total / (1024**3)),
                "cpu_count": psutil.cpu_count(),
            }
        )
    return s


def get_hardware_info(device: torch.device) -> dict:
    """Get hardware information of the current system.

    Args:
        device: Device to get the hardware information from.

    Returns:
        Dictionary containing system and GPU information.
    """
    return {
        "system": get_system_info(),
        "gpu": get_gpu_info(device) if device.type == "cuda" else None,
    }


def save_hardware_info(output_directory: str, device: torch.device) -> None:
    """Save hardware information to a hardware.yaml file.

    Args:
        output_directory: Directory to save the hardware information to.
        device: Device to get the hardware information from.
    """
    os.makedirs(output_directory, exist_ok=True)
    OmegaConf.save(
        get_hardware_info(device),
        os.path.join(output_directory, "hardware.yaml"),
    )


def set_device(device_name: str) -> torch.device:
    """Set the device to use. If the specified CUDA device is unavailable,
    automatically fall back to CPU.

    Args:
        device_name: Name of the device to use.

    Returns:
        Specified device if available, otherwise CPU.
    """
    try:
        device = torch.device(device_name)
        torch.tensor(1).to(device)
    except (RuntimeError, AssertionError):
        device = torch.device("cpu")
        logging.warning(  # noqa: LOG015
            f"Device '{device_name}' is not available. Falling back to CPU.",  # noqa: G004
        )
    return device
