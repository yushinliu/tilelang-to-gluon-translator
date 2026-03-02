"""
Example: Translate a TileLang elementwise kernel to Gluon.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.translator import TileLangToGluonTranslator


# TileLang elementwise kernel
tilelang_source = '''
@T.prim_func
def elementwise_add(
    A: T.Tensor((1024,), T.float32),
    B: T.Tensor((1024,), T.float32),
    C: T.Tensor((1024,), T.float32),
):
    with T.Kernel(T.ceildiv(1024, 128), threads=128) as (bx,):
        A_shared = T.alloc_shared([128], T.float32)
        B_shared = T.alloc_shared([128], T.float32)
        C_shared = T.alloc_shared([128], T.float32)

        T.copy(A[bx * 128:(bx + 1) * 128], A_shared)
        T.copy(B[bx * 128:(bx + 1) * 128], B_shared)

        for i in T.Parallel(128):
            C_shared[i] = A_shared[i] + B_shared[i]

        T.copy(C_shared, C[bx * 128:(bx + 1) * 128])
'''


def main():
    translator = TileLangToGluonTranslator()
    gluon_code = translator.translate(tilelang_source)

    print("=" * 80)
    print("Generated Gluon Kernel:")
    print("=" * 80)
    print(gluon_code)

    # Save to file
    output_path = Path(__file__).parent / "elementwise_add_gluon.py"
    output_path.write_text(gluon_code)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
