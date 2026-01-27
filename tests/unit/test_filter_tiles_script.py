import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.filter_tiles import main


class TestFilterTilesScript:
    """Test the filter_tiles.py script."""
    
    def test_script_requires_features_argument(self):
        """Test that --features argument is required."""
        with patch('sys.argv', ['filter_tiles.py']):
            with pytest.raises(SystemExit):
                main()
    
    def test_script_requires_targets_argument(self):
        """Test that --targets argument is required."""
        with patch('sys.argv', ['filter_tiles.py', '--features', 'test']):
            with pytest.raises(SystemExit):
                main()
    
    def test_script_requires_output_argument(self):
        """Test that --output argument is required."""
        with patch('sys.argv', ['filter_tiles.py', '--features', 'test', '--targets', 'test']):
            with pytest.raises(SystemExit):
                main()
    
    def test_script_validates_directories_exist(self, tmp_path):
        """Test that script validates directories exist."""
        features_dir = tmp_path / "nonexistent_features"
        targets_dir = tmp_path / "nonexistent_targets"
        output_file = tmp_path / "output.json"
        
        with patch('sys.argv', [
            'filter_tiles.py',
            '--features', str(features_dir),
            '--targets', str(targets_dir),
            '--output', str(output_file),
        ]):
            with pytest.raises(FileNotFoundError, match="Features directory not found"):
                main()
    
    def test_script_calls_filter_with_correct_arguments(self, tmp_path):
        """Test that script calls TileFilter with correct arguments."""
        features_dir = tmp_path / "features"
        targets_dir = tmp_path / "targets"
        output_file = tmp_path / "output.json"
        features_dir.mkdir()
        targets_dir.mkdir()
        
        with patch('scripts.filter_tiles.TileFilter') as mock_filter_class:
            mock_filter = MagicMock()
            mock_filter.filter_tile_pairs.return_value = ([], {"total_tiles": 0, "rgb_invalid": 0, "background_only": 0, "valid_tiles": 0})
            mock_filter.print_summary = MagicMock()
            mock_filter_class.return_value = mock_filter
            
            with patch('sys.argv', [
                'filter_tiles.py',
                '--features', str(features_dir),
                '--targets', str(targets_dir),
                '--output', str(output_file),
                '--min-rgb-coverage', '0.05',
                '--exclude-background',
            ]):
                main()
            
            # Verify TileFilter was initialized with correct arguments
            mock_filter_class.assert_called_once_with(
                min_rgb_coverage=0.05,
                include_background_only=False,  # --exclude-background sets this to False
                min_target_coverage=None,
            )
            
            # Verify filter_tile_pairs was called
            mock_filter.filter_tile_pairs.assert_called_once()
            call_args = mock_filter.filter_tile_pairs.call_args
            assert call_args[1]["features_dir"] == features_dir
            assert call_args[1]["targets_dir"] == targets_dir
            assert call_args[1]["output_file"] == output_file
    
    def test_script_handles_min_target_coverage(self, tmp_path):
        """Test that script handles --min-target-coverage argument."""
        features_dir = tmp_path / "features"
        targets_dir = tmp_path / "targets"
        output_file = tmp_path / "output.json"
        features_dir.mkdir()
        targets_dir.mkdir()
        
        with patch('scripts.filter_tiles.TileFilter') as mock_filter_class:
            mock_filter = MagicMock()
            mock_filter.filter_tile_pairs.return_value = ([], {"total_tiles": 0, "rgb_invalid": 0, "background_only": 0, "valid_tiles": 0})
            mock_filter.print_summary = MagicMock()
            mock_filter_class.return_value = mock_filter
            
            with patch('sys.argv', [
                'filter_tiles.py',
                '--features', str(features_dir),
                '--targets', str(targets_dir),
                '--output', str(output_file),
                '--min-target-coverage', '0.02',
            ]):
                main()
            
            # Verify TileFilter was initialized with min_target_coverage
            mock_filter_class.assert_called_once_with(
                min_rgb_coverage=0.01,  # default
                include_background_only=True,  # default (no --exclude-background)
                min_target_coverage=0.02,
            )
