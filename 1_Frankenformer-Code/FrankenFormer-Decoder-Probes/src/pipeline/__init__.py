"""Pipeline modules for training and evaluation."""

__all__ = ["Trainer"]


def __getattr__(name: str):
    if name == "Trainer":
        from src.pipeline.trainer import Trainer
        return Trainer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(__all__)
