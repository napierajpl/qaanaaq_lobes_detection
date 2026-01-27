import argparse
from pathlib import Path
from typing import Optional


class BaseCLIParser:
    """Base class for CLI argument parsing with common patterns."""

    def __init__(self, description: str, project_root: Path):
        """
        Initialize base CLI parser.

        Args:
            description: Description of the script
            project_root: Project root directory
        """
        self.project_root = project_root
        self.parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    def add_input_output_args(
        self,
        default_input: Path,
        default_output: Path,
        input_help: Optional[str] = None,
        output_help: Optional[str] = None,
    ) -> None:
        """
        Add common input and output file arguments.

        Args:
            default_input: Default input file path
            default_output: Default output file path
            input_help: Help text for input argument
            output_help: Help text for output argument
        """
        try:
            input_relative = default_input.relative_to(self.project_root)
        except ValueError:
            input_relative = default_input

        try:
            output_relative = default_output.relative_to(self.project_root)
        except ValueError:
            output_relative = default_output

        self.parser.add_argument(
            "-i",
            "--input",
            type=Path,
            default=default_input,
            help=input_help or f"Input file path (default: {input_relative})",
        )

        self.parser.add_argument(
            "-o",
            "--output",
            type=Path,
            default=default_output,
            help=output_help or f"Output file path (default: {output_relative})",
        )

    def add_reference_raster_arg(
        self,
        default_reference: Path,
        help_text: Optional[str] = None,
    ) -> None:
        """
        Add reference raster argument with optional flag.

        Args:
            default_reference: Default reference raster path
            help_text: Help text for reference argument
        """
        try:
            reference_relative = default_reference.relative_to(self.project_root)
        except ValueError:
            reference_relative = default_reference

        self.parser.add_argument(
            "-r",
            "--reference",
            type=Path,
            default=default_reference,
            help=help_text or f"Reference raster for extent/resolution (default: {reference_relative})",
        )

        self.parser.add_argument(
            "--no-reference",
            action="store_true",
            help="Ignore reference raster even if it exists",
        )

    def parse_args(self) -> argparse.Namespace:
        """Parse command-line arguments."""
        return self.parser.parse_args()

    def set_epilog(self, examples: str) -> None:
        """Set epilog with usage examples."""
        self.parser.epilog = examples
