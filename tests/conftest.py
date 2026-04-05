import pytest
from pathlib import Path
from max.driver import CPU, Accelerator, Device, accelerator_count
from max.engine import InferenceSession
from max.graph.graph import KernelLibrary

_CUSTOM_EXTENSIONS = [Path(__file__).parent / ".." / "max_cv" / "operations_mojo"]


def get_available_devices() -> list[Device]:
    """Return list of available devices (CPU and GPU if available)."""
    devices = [CPU()]
    if accelerator_count() > 0:
        devices.append(Accelerator())
    return devices


@pytest.fixture(scope="session")
def kernel_library() -> KernelLibrary:
    """Pre-built kernel library shared across all tests."""
    lib = KernelLibrary()
    lib.load_paths(_CUSTOM_EXTENSIONS)
    return lib


@pytest.fixture(
    scope="session", params=get_available_devices(), ids=lambda d: type(d).__name__
)
def device(request) -> Device:
    """Fixture that yields each available device (CPU, and GPU if available)."""
    return request.param


@pytest.fixture(scope="session")
def session(device: Device) -> InferenceSession:
    """InferenceSession configured for the current device."""
    return InferenceSession(devices=[device])
