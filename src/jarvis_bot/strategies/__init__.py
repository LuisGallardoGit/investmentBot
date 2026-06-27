from .base import Signal, Strategy
from .trend_follow import TrendFollowStrategy
from .mean_reversion import MeanReversionStrategy
from .combined import CombinedStrategy
from .ml_strategy import MLStrategy

__all__ = [
    "Signal",
    "Strategy",
    "TrendFollowStrategy",
    "MeanReversionStrategy",
    "CombinedStrategy",
    "MLStrategy",
]


def get_strategy(name: str, **kwargs) -> Strategy:
    """Factory: instancia la estrategia por nombre de config."""
    strategies = {
        "trend_follow": TrendFollowStrategy,
        "mean_reversion": MeanReversionStrategy,
        "combined": CombinedStrategy,
        "ml": MLStrategy,
    }
    if name not in strategies:
        raise ValueError(
            f"Estrategia '{name}' desconocida. Opciones: {list(strategies)}"
        )
    return strategies[name](**kwargs)
