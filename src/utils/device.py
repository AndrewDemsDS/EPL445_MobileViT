"""Auto-detect the best available compute device."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Return a torch.device based on *preference*.

    Parameters
    ----------
    preference : str
        One of ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
        ``"auto"`` picks the best available accelerator.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)
