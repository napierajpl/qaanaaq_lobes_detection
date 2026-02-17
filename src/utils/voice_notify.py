"""
Voice notification when training finishes. Uses gTTS and pygame to speak
duration (hours/minutes) and epoch count.
"""

import logging
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_gtts = None
_pygame = None


def _ensure_deps():
    global _gtts, _pygame
    if _gtts is None:
        try:
            from gtts import gTTS
            _gtts = gTTS
        except ImportError:
            logger.warning("gtts not installed; voice notification disabled. Install with: pip install gtts")
            return False
    if _pygame is None:
        try:
            import pygame
            _pygame = pygame
        except ImportError:
            logger.warning("pygame not installed; voice notification disabled. Install with: pip install pygame")
            return False
    return True


def _format_duration(total_seconds: float) -> str:
    total_seconds = int(round(total_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    parts = []
    if hours > 0:
        parts.append(f"{hours} hour" if hours == 1 else f"{hours} hours")
    if minutes > 0 or not parts:
        parts.append(f"{minutes} minute" if minutes == 1 else f"{minutes} minutes")
    return " and ".join(parts)


def notify_training_finished(elapsed_seconds: float, epochs_run: int) -> None:
    """
    Speak a short message: training finished, duration (hours and minutes), and epochs run.
    If gtts or pygame are missing, logs a warning and returns without error.
    """
    if not _ensure_deps():
        return
    duration_str = _format_duration(elapsed_seconds)
    epoch_word = "epoch" if epochs_run == 1 else "epochs"
    text = (
        f"Training finished. It took {duration_str}. "
        f"{epochs_run} {epoch_word} were run."
    )
    tmp = Path(tempfile.gettempdir()) / "training_finished_notify.mp3"
    try:
        _gtts(text=text, lang="en").save(str(tmp))
        _pygame.mixer.init()
        _pygame.mixer.music.load(str(tmp))
        _pygame.mixer.music.play()
        while _pygame.mixer.music.get_busy():
            time.sleep(0.1)
        _pygame.mixer.quit()
    except Exception as e:
        logger.warning("Voice notification failed: %s", e)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
