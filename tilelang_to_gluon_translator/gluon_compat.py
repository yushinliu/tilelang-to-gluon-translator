"""Compatibility helpers for Gluon layouts missing from the installed runtime."""

from dataclasses import dataclass
from typing import Any

from triton.language.core import _unwrap_if_constexpr, constexpr_type
from triton.experimental.gluon.language._layouts import DistributedLayout as RuntimeDistributedLayout


class DistributedLayout(RuntimeDistributedLayout):
    """Minimal distributed layout base compatible with Gluon's constexpr handling."""

    @property
    def type(self):
        return constexpr_type(self)


@dataclass(frozen=True)
class DotOperandLayout(DistributedLayout):
    """Project-local fallback for Gluon's DotOperandLayout."""

    operand_index: int
    parent: Any
    k_width: int

    def __post_init__(self):
        super().__setattr__("operand_index", _unwrap_if_constexpr(self.operand_index))
        super().__setattr__("parent", _unwrap_if_constexpr(self.parent))
        super().__setattr__("k_width", _unwrap_if_constexpr(self.k_width))

    def _to_ir(self, builder):
        return builder.get_dot_operand_layout(self.operand_index, self.parent._to_ir(builder), self.k_width)

    def mangle(self) -> str:
        parent_mangle = self.parent.mangle() if hasattr(self.parent, "mangle") else str(self.parent)
        return f"DO{self.operand_index}_{parent_mangle}_{self.k_width}DO"
