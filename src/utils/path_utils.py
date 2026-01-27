from pathlib import Path


def resolve_path(path: Path, base_dir: Path) -> Path:
    """
    Resolve path to absolute, handling relative paths.
    
    Args:
        path: Path to resolve (absolute or relative)
        base_dir: Base directory for resolving relative paths
    
    Returns:
        Absolute Path
    """
    if path.is_absolute():
        return path
    return base_dir / path


def get_project_root(script_path: Path) -> Path:
    """
    Get project root directory from script path.
    
    Args:
        script_path: Path to the script file (typically __file__)
    
    Returns:
        Project root Path (parent of scripts/ directory)
    """
    current = Path(script_path).resolve()
    
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    raise ValueError("Could not find project root (pyproject.toml)")
