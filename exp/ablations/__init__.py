"""Ours 的运行时消融配置。"""

from .variants import ABLATION_VARIANTS, Variant, build_ablation

__all__ = ["ABLATION_VARIANTS", "Variant", "build_ablation"]
