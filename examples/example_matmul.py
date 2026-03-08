"""
Example: Translate a TileLang matrix multiplication kernel to Gluon.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tilelang_to_gluon_translator import TileLangToGluonTranslator


# TileLang GEMM kernel
tilelang_source = '''
@T.prim_func
def matmul(
    A: T.Tensor((M, K), T.float16),
    B: T.Tensor((K, N), T.float16),
    C: T.Tensor((M, N), T.float16),
):
    with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
        A_shared = T.alloc_shared([128, 32], T.float16)
        B_shared = T.alloc_shared([128, 32], T.float16)
        C_local = T.alloc_fragment([128, 128], T.float32)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, 32), num_stages=2):
            T.copy(A[by * 128, k * 32], A_shared)
            T.copy(B[k * 32, bx * 128], B_shared)
            T.gemm(A_shared, B_shared, C_local, False, True)
        T.copy(C_local, C[by * 128, bx * 128])
'''


def main():
    translator = TileLangToGluonTranslator()
    gluon_code = translator.translate(tilelang_source)

    print("=" * 80)
    print("Generated Gluon Kernel:")
    print("=" * 80)
    print(gluon_code)

    # Save to file
    output_path = Path(__file__).parent / "matmul_gluon.py"
    output_path.write_text(gluon_code)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
