# utils.py
# Lightweight utilities: banners, progress bar, early stopping.

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Sequence


def print_banner(s: str, separator: str = "-", num_star: int = 60) -> None:
    """
    Print a message surrounded by a separator line (used for nice console blocks).
    """
    line = separator * num_star
    print(line, flush=True)
    print(s, flush=True)
    print(line, flush=True)


class Progress:
    """
    Simple CLI progress + speed display used in diffusion sampling.

    Example:
        progress = Progress(total_timesteps)
        ...
        progress.update({'t': i})
        ...
        progress.close()
    """

    def __init__(
        self,
        total: int,
        name: str = "Progress",
        ncol: int = 3,
        max_length: int = 20,
        indent: int = 0,
        line_width: int = 100,
        speed_update_freq: int = 100,
    ) -> None:
        self.total = total
        self.name = name
        self.ncol = ncol
        self.max_length = max_length
        self.indent = indent
        self.line_width = line_width
        self._speed_update_freq = speed_update_freq

        self._step = 0
        self._prev_line = "\033[F"
        self._clear_line = " " * self.line_width

        self._pbar_size = self.ncol * self.max_length
        self._complete_pbar = "#" * self._pbar_size
        self._incomplete_pbar = " " * self._pbar_size

        self.lines: List[str] = [""]
        self.fraction = f"0 / {self.total}"
        self._speed = "0.0 Hz"
        self._skip_lines = 1

        self.resume()

    def update(self, description: Dict[str, Any] | Sequence[Any] | None = None, n: int = 1) -> None:
        """
        Increment progress by `n` and refresh the display.
        `description` can be a dict or list of (key, val) pairs.
        """
        self._step += n
        if self._step % self._speed_update_freq == 0:
            self._time0 = time.time()
            self._step0 = self._step
        if description is None:
            description = []
        self.set_description(description)

    def resume(self) -> None:
        """Start (or restart) the progress display."""
        self._skip_lines = 1
        print("\n", end="")
        self._time0 = time.time()
        self._step0 = self._step

    def pause(self) -> None:
        """Clear the current progress lines without printing a new one."""
        self._clear()
        self._skip_lines = 1

    def set_description(self, params: Any = None) -> None:
        """
        Update the progress line with a new set of params.
        `params` can be:
            - dict {key: val}
            - list of (key, val) pairs
            - empty/None
        """
        if params is None:
            params = []

        if isinstance(params, dict):
            params = sorted((key, val) for key, val in params.items())

        # Clear old lines
        self._clear()

        # Percent and fraction
        percent_str, fraction = self._format_percent(self._step, self.total)
        self.fraction = fraction

        # Speed
        speed = self._format_speed(self._step)

        # Params formatting
        num_params = len(params)
        nrow = math.ceil(num_params / self.ncol) if num_params > 0 else 0
        params_split = self._chunk(params, self.ncol)
        params_string, lines = self._format(params_split)
        self.lines = lines

        description = f"{percent_str} | {speed}{params_string}"
        print(description)
        self._skip_lines = nrow + 1

    def append_description(self, descr: str) -> None:
        """Append an extra line below the main progress line."""
        self.lines.append(descr)

    # ---- internal helpers ---- #

    def _clear(self) -> None:
        position = self._prev_line * self._skip_lines
        empty = "\n".join([self._clear_line for _ in range(self._skip_lines)])
        print(position, end="")
        print(empty)
        print(position, end="")

    def _format_percent(self, n: int, total: int) -> tuple[str, str]:
        if total > 0:
            percent = n / float(total)
            percent = min(max(percent, 0.0), 1.0)  # clamp for safety

            complete_entries = int(percent * self._pbar_size)
            incomplete_entries = self._pbar_size - complete_entries

            pbar = (
                self._complete_pbar[:complete_entries]
                + self._incomplete_pbar[:incomplete_entries]
            )
            fraction = f"{n} / {total}"
            string = f"{fraction} [{pbar}] {int(percent * 100):3d}%"
        else:
            fraction = f"{n}"
            string = f"{n} iterations"
        return string, fraction

    def _format_speed(self, n: int) -> str:
        num_steps = n - self._step0
        t = time.time() - self._time0
        if t <= 0 or num_steps <= 0:
            # keep previous speed string if available
            return getattr(self, "_speed", "0.0 Hz")

        speed = num_steps / t
        string = f"{speed:.1f} Hz"
        self._speed = string
        return string

    @staticmethod
    def _chunk(seq: Sequence[Any], n: int) -> List[Sequence[Any]]:
        return [seq[i : i + n] for i in range(0, len(seq), n)]

    def _format(self, chunks: Sequence[Sequence[Any]]) -> tuple[str, List[str]]:
        lines = [self._format_chunk(chunk) for chunk in chunks] if chunks else []
        lines.insert(0, "")
        padding = "\n" + " " * self.indent
        string = padding.join(lines)
        return string, lines

    def _format_chunk(self, chunk: Sequence[Any]) -> str:
        return " | ".join([self._format_param(param) for param in chunk])

    def _format_param(self, param: Any) -> str:
        k, v = param
        return f"{k} : {v}"[: self.max_length]

    def stamp(self) -> None:
        """
        Print a final summary line with the last fraction, params, and speed.
        """
        if self.lines != [""]:
            params = " | ".join(self.lines)
            string = f"[ {self.name} ] {self.fraction}{params} | {self._speed}"
            self._clear()
            print(string, end="\n")
            self._skip_lines = 1
        else:
            self._clear()
            self._skip_lines = 0

    def close(self) -> None:
        """Alias for pause(), kept for API compatibility."""
        self.pause()


class Silent:
    """
    Dummy progress class that does nothing.
    Used when verbose=False in sampling.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __getattr__(self, attr: str):
        # Any method called on Silent will just be a no-op function.
        return lambda *args, **kwargs: None


class EarlyStopping:
    """
    Simple early stopping monitor.

    Usage (as in the original main.py):
        stop_check = EarlyStopping(tolerance=1, min_delta=0.0)
        ...
        early_stop = stop_check(metric, bc_loss)

    If (validation_loss - train_loss) > min_delta for `tolerance`
    consecutive calls, returns True.
    """

    def __init__(self, tolerance: int = 5, min_delta: float = 0.0) -> None:
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0

    def __call__(self, train_loss: float, validation_loss: float) -> bool:
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
        else:
            self.counter = 0
        return False
