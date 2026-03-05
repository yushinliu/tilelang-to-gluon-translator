"""
Main entry point for TileLang to Gluon translation.
"""

import ast
import os
from pathlib import Path
from typing import Optional, Union, List

from .parser import TileLangParser
from .transformer import TileLangToGluonTransformer
from .codegen import GluonCodeGenerator
from .version_check import log_version_info, check_gluon_version


class TileLangToGluonTranslator:
    """
    Main translator class that orchestrates the translation pipeline.
    """

    def __init__(
        self,
        max_jobs: int = 8,
        verify: bool = True,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        check_version: bool = True
    ):
        self.parser = TileLangParser()
        self.transformer = TileLangToGluonTransformer()
        self.codegen = GluonCodeGenerator()
        self.max_jobs = max_jobs
        self.verify = verify
        self.atol = atol
        self.rtol = rtol

        # Check Gluon version
        if check_version:
            log_version_info()

    def translate(
        self,
        source: Union[str, Path],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Translate TileLang kernel to Gluon kernel.

        Args:
            source: TileLang source code or path to source file
            output_path: Optional path to write generated Gluon code

        Returns:
            Generated Gluon kernel source code
        """
        # Read source
        if isinstance(source, Path):
            source_code = source.read_text()
        else:
            source_code = source

        # Step 1: Parse TileLang AST
        tilelang_kernel = self.parser.parse(source_code)

        # Step 2: Transform to Gluon AST
        gluon_kernel = self.transformer.transform(tilelang_kernel)

        # Step 3: Generate Gluon code
        gluon_code = self.codegen.generate(gluon_kernel)

        # Step 4: Write output if requested
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(gluon_code)

        return gluon_code

    def translate_file(self, input_path: Path, output_dir: Path) -> Path:
        """
        Translate a single TileLang file to Gluon.

        Args:
            input_path: Path to TileLang source file
            output_dir: Directory to write output

        Returns:
            Path to generated Gluon file
        """
        output_path = output_dir / f"{input_path.stem}_gluon.py"
        self.translate(input_path, output_path)
        return output_path

    def translate_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.py"
    ) -> List[Path]:
        """
        Translate all TileLang files in a directory.

        Args:
            input_dir: Directory containing TileLang source files
            output_dir: Directory to write output
            pattern: File pattern to match

        Returns:
            List of paths to generated Gluon files
        """
        output_paths = []
        output_dir.mkdir(parents=True, exist_ok=True)

        for input_path in input_dir.rglob(pattern):
            try:
                output_path = self.translate_file(input_path, output_dir)
                output_paths.append(output_path)
            except Exception as e:
                print(f"Error translating {input_path}: {e}")

        return output_paths


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Translate TileLang kernels to Triton Gluon kernels"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input TileLang file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file or directory"
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=8,
        help="Maximum number of parallel compilation jobs"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification"
    )

    args = parser.parse_args()

    # Set max jobs environment variable
    os.environ['MAX_JOBS'] = str(args.max_jobs)

    translator = TileLangToGluonTranslator(
        max_jobs=args.max_jobs,
        verify=not args.no_verify
    )

    if args.input.is_file():
        output_path = args.output or args.input.with_suffix(".gluon.py")
        translator.translate(args.input, output_path)
        print(f"Translated: {args.input} -> {output_path}")
    elif args.input.is_dir():
        output_dir = args.output or args.input / "gluon_output"
        output_paths = translator.translate_directory(args.input, output_dir)
        print(f"Translated {len(output_paths)} files to {output_dir}")
    else:
        print(f"Error: {args.input} does not exist")


if __name__ == "__main__":
    main()
