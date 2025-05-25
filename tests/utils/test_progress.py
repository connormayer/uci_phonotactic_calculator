"""
tests/unit/test_progress.py — Tests for progress abstraction

Tests the various progress implementations to ensure they work correctly and follow
the expected interface contract.
"""

from unittest.mock import MagicMock, patch

import pytest

from uci_phonotactic_calculator.utils.progress_base import (
    BaseProgress,
    GradioProgress,
    RichProgress,
)


@pytest.fixture(params=[RichProgress, GradioProgress])
def progress_impl(request):
    """Parametrize over all concrete progress implementations."""
    return request.param(enabled=True)


@pytest.fixture(params=[RichProgress, GradioProgress])
def disabled_progress_impl(request):
    """Parametrize over all concrete progress implementations in disabled state."""
    return request.param(enabled=False)


def test_progress_context_manager(progress_impl):
    """Test that the progress implementation can be used as a context manager."""
    with progress_impl as p:
        assert isinstance(p, BaseProgress)


def test_add_task(progress_impl):
    """Test that tasks can be added to the progress bar."""
    with progress_impl as p:
        task_id = p.add_task("Test task", total=10)
        assert isinstance(task_id, int)


def test_update_task(progress_impl):
    """Test that tasks can be updated in the progress bar."""
    with progress_impl as p:
        task_id = p.add_task("Test task", total=10)
        p.update(task_id, advance=1)
        # No error means success


def test_disabled_progress(disabled_progress_impl):
    """Test that disabled progress bars don't break anything."""
    with disabled_progress_impl as p:
        task_id = p.add_task("Test task", total=10)
        p.update(task_id, advance=1)
        # No error means success


@patch("uci_phonotactic_calculator.utils.progress_base.RichProgress")
def test_progress_factory_enabled(mock_rich):
    """Test that the progress factory returns the correct type when enabled."""
    from uci_phonotactic_calculator.utils.progress_base import progress

    p = progress(enabled=True)
    p.__enter__()

    # Verify RichProgress was instantiated
    mock_rich.assert_called_once()


@patch("uci_phonotactic_calculator.utils.progress_base.nullcontext")
def test_progress_factory_disabled(mock_nullcontext):
    """Test that the progress factory returns nullcontext when disabled."""
    from uci_phonotactic_calculator.utils.progress_base import progress

    # Call the progress function but we don't need the result
    progress(enabled=False)

    # Verify nullcontext was used
    mock_nullcontext.assert_called_once()


def test_gradio_progress_has_description_support():
    """Test that GradioProgress handles description correctly."""
    with patch("gradio.Progress") as mock_progress:
        # Mock the Gradio Progress context manager
        mock_cm = MagicMock()
        mock_progress.return_value = mock_cm
        mock_g_prog = MagicMock()
        mock_cm.__enter__.return_value = mock_g_prog
        mock_g_prog.set_description = MagicMock()

        # Create and use GradioProgress
        with GradioProgress() as p:
            p.add_task("Test description", total=10)

            # Verify set_description was called if it exists
            if hasattr(mock_g_prog, "set_description"):
                mock_g_prog.set_description.assert_called_with("Test description")
