import pytest
from max.driver import CPU, Accelerator, Device, accelerator_count
from max.engine import InferenceSession


def get_available_devices() -> list[Device]:
    """Return list of available devices (CPU and GPU if available)."""
    devices = [CPU()]
    if accelerator_count() > 0:
        devices.append(Accelerator())
    return devices


@pytest.fixture(params=get_available_devices(), ids=lambda d: type(d).__name__)
def device(request) -> Device:
    """Fixture that yields each available device (CPU, and GPU if available)."""
    return request.param


@pytest.fixture
def session(device: Device) -> InferenceSession:
    """InferenceSession configured for the current device."""
    return InferenceSession(devices=[device])
