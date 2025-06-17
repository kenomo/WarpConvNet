import logging
import sys
import os
from pathlib import Path
from typing import Optional, Mapping

import torch
import torch.distributed as dist


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors"""

    COLORS = {
        "DEBUG": "\033[37m",  # White
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        # Add color to the level name
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"

        return super().format(record)


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = True,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes with their rank prefixed in the log message."""
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = self.rank
            # Determine how many additional stack frames to skip so that the
            # reported filename and line number correspond to the original
            # caller (the user code) rather than this wrapper.
            # Default ``stacklevel`` is 1, which would point to the call in
            # ``logging.Logger``. Since the call stack is:
            #   user code -> RankedLogger.<level>() -> RankedLogger.log() -> logging.Logger.log()
            # we need to skip two extra frames to reach the user code.
            stacklevel = kwargs.pop("stacklevel", 1) + 2

            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, stacklevel=stacklevel, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, stacklevel=stacklevel, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, stacklevel=stacklevel, **kwargs)


def get_logger(name: str = None, rank_zero_only: bool = True) -> RankedLogger:
    """
    Create a logger with consistent formatting including filename and line number

    Args:
        name: Logger name, defaults to file name if None

    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the caller's filename if no name provided
        frame = sys._getframe(1)
        name = Path(frame.f_code.co_filename).stem

    # Create logger
    logger = RankedLogger(name=name, rank_zero_only=rank_zero_only)

    # Only add handler if logger doesn't have one
    if not logger.logger.handlers:
        # Create stderr handler
        handler = logging.StreamHandler(sys.stderr)

        # Format: [LEVEL] filename:line - message
        formatter = ColorFormatter(
            fmt="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler.setFormatter(formatter)
        logger.logger.addHandler(handler)

        # Set default level, allowing override from environment variable
        log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = log_levels.get(log_level_name, logging.INFO)  # Default to INFO if invalid
        logger.setLevel(level)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    return logger
