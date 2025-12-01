"""
Lightweight experiment logger (adapted from rllab).

Provides:
    - logger.log(...)                : free-form text logging
    - logger.record_tabular(key,val) : scalar metrics per iteration
    - logger.dump_tabular()          : prints table + writes CSV
    - setup_logger(...)              : sets up log directory, files, variants
"""

from __future__ import annotations

from enum import Enum
from contextlib import contextmanager
from collections import OrderedDict
from numbers import Number
from typing import Any, Dict, List, Tuple

import csv
import datetime
import errno
import json
import os
import os.path as osp
import pickle
import sys

import dateutil.tz
import numpy as np
from tabulate import tabulate


# =========================================================================== #
# Helpers for variant / JSON logging
# =========================================================================== #

def safe_json(data: Any) -> bool:
    """Return True if `data` can be safely JSON-serialized as-is."""
    if data is None:
        return True
    if isinstance(data, (bool, int, float)):
        return True
    if isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    if isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def dict_to_safe_json(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert each value in the dictionary into a JSON'able primitive.
    Non-serializable values are converted to strings (or recursed into dicts).
    """
    new_d: Dict[str, Any] = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d


class MyEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle classes, Enums, and callables."""
    def default(self, o: Any) -> Any:
        if isinstance(o, type):
            return {"$class": o.__module__ + "." + o.__name__}
        if isinstance(o, Enum):
            return {
                "$enum": o.__module__ + "." + o.__class__.__name__ + "." + o.name
            }
        if callable(o):
            return {"$function": o.__module__ + "." + o.__name__}
        return json.JSONEncoder.default(self, o)


def mkdir_p(path: str) -> None:
    """`mkdir -p` in Python."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# =========================================================================== #
# Experiment naming / log-dir creation
# =========================================================================== #

def create_exp_name(exp_prefix: str, exp_id: int = 0, seed: int = 0) -> str:
    """
    Create a semi-unique experiment name that has a timestamp.
    Example: hopper_2025_11_30_19_12_05_0001--s-0
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    return f"{exp_prefix}_{timestamp}_{exp_id:04d}--s-{seed}"


def create_log_dir(
    exp_prefix: str,
    exp_id: int = 0,
    seed: int = 0,
    base_log_dir: str | None = None,
    include_exp_prefix_sub_dir: bool = True,
) -> str:
    """
    Creates and returns a unique log directory.
    Default: ./data/<exp_prefix>/<exp_name>
    """
    exp_name = create_exp_name(exp_prefix, exp_id=exp_id, seed=seed)

    if base_log_dir is None:
        base_log_dir = "./data"

    if include_exp_prefix_sub_dir:
        log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
    else:
        log_dir = osp.join(base_log_dir, exp_name)

    if osp.exists(log_dir):
        print(f"WARNING: Log directory already exists {log_dir}", flush=True)

    os.makedirs(log_dir, exist_ok=True)
    return log_dir


# =========================================================================== #
# Main Logger class
# =========================================================================== #

class TerminalTablePrinter:
    """Optional: prints a live-updating table when log_tabular_only=True."""
    def __init__(self) -> None:
        self.headers: List[str] | None = None
        self.tabulars: List[List[Any]] = []

    def print_tabular(self, new_tabular: List[Tuple[str, Any]]) -> None:
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self) -> None:
        try:
            rows, columns = os.popen("stty size", "r").read().split()
            rows = int(rows)
        except Exception:
            rows = 40  # fallback

        tabulars = self.tabulars[-(rows - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


class Logger:
    def __init__(self) -> None:
        self._prefixes: List[str] = []
        self._prefix_str: str = ""

        self._tabular_prefixes: List[str] = []
        self._tabular_prefix_str: str = ""

        self._tabular: List[Tuple[str, Any]] = []

        self._text_outputs: List[str] = []
        self._tabular_outputs: List[str] = []

        self._text_fds: Dict[str, Any] = {}
        self._tabular_fds: Dict[str, Any] = {}
        self._tabular_header_written: set[Any] = set()

        self._snapshot_dir: str | None = None
        self._snapshot_mode: str = "all"
        self._snapshot_gap: int = 1

        self._log_tabular_only: bool = False
        self.table_printer = TerminalTablePrinter()

    # ---------- low-level file management ---------- #

    def reset(self) -> None:
        self.__init__()

    def _add_output(self, file_name: str, arr: List[str], fds: Dict[str, Any], mode: str = "a") -> None:
        if file_name not in arr:
            mkdir_p(os.path.dirname(file_name))
            arr.append(file_name)
            fds[file_name] = open(file_name, mode)

    def _remove_output(self, file_name: str, arr: List[str], fds: Dict[str, Any]) -> None:
        if file_name in arr:
            fds[file_name].close()
            del fds[file_name]
            arr.remove(file_name)

    # ---------- prefix handling ---------- #

    def push_prefix(self, prefix: str) -> None:
        self._prefixes.append(prefix)
        self._prefix_str = "".join(self._prefixes)

    def pop_prefix(self) -> None:
        del self._prefixes[-1]
        self._prefix_str = "".join(self._prefixes)

    def push_tabular_prefix(self, key: str) -> None:
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = "".join(self._tabular_prefixes)

    def pop_tabular_prefix(self) -> None:
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = "".join(self._tabular_prefixes)

    @contextmanager
    def prefix(self, key: str):
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    @contextmanager
    def tabular_prefix(self, key: str):
        self.push_tabular_prefix(key)
        try:
            yield
        finally:
            self.pop_tabular_prefix()

    # ---------- output management ---------- #

    def add_text_output(self, file_name: str) -> None:
        self._add_output(file_name, self._text_outputs, self._text_fds, mode="a")

    def remove_text_output(self, file_name: str) -> None:
        self._remove_output(file_name, self._text_outputs, self._text_fds)

    def add_tabular_output(self, file_name: str, relative_to_snapshot_dir: bool = False) -> None:
        if relative_to_snapshot_dir and self._snapshot_dir is not None:
            file_name = osp.join(self._snapshot_dir, file_name)
        self._add_output(file_name, self._tabular_outputs, self._tabular_fds, mode="w")

    def remove_tabular_output(self, file_name: str, relative_to_snapshot_dir: bool = False) -> None:
        if relative_to_snapshot_dir and self._snapshot_dir is not None:
            file_name = osp.join(self._snapshot_dir, file_name)
        if self._tabular_fds[file_name] in self._tabular_header_written:
            self._tabular_header_written.remove(self._tabular_fds[file_name])
        self._remove_output(file_name, self._tabular_outputs, self._tabular_fds)

    # ---------- snapshot settings ---------- #

    def set_snapshot_dir(self, dir_name: str) -> None:
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self) -> str | None:
        return self._snapshot_dir

    def get_snapshot_mode(self) -> str:
        return self._snapshot_mode

    def set_snapshot_mode(self, mode: str) -> None:
        self._snapshot_mode = mode

    def get_snapshot_gap(self) -> int:
        return self._snapshot_gap

    def set_snapshot_gap(self, gap: int) -> None:
        self._snapshot_gap = gap

    def set_log_tabular_only(self, log_tabular_only: bool) -> None:
        self._log_tabular_only = log_tabular_only

    def get_log_tabular_only(self) -> bool:
        return self._log_tabular_only

    # ---------- main logging APIs ---------- #

    def log(self, s: str, with_prefix: bool = True, with_timestamp: bool = True) -> None:
        out = s
        if with_prefix:
            out = self._prefix_str + out
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime("%y-%m-%d.%H:%M")
            out = f"{timestamp}|{out}"

        if not self._log_tabular_only:
            print(out, flush=True)
            for fd in list(self._text_fds.values()):
                fd.write(out + "\n")
                fd.flush()
            sys.stdout.flush()

    def record_tabular(self, key: str, val: Any) -> None:
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))

    def record_dict(self, d: Dict[str, Any], prefix: str | None = None) -> None:
        if prefix is not None:
            self.push_tabular_prefix(prefix)
        for k, v in d.items():
            self.record_tabular(k, v)
        if prefix is not None:
            self.pop_tabular_prefix()

    def get_table_dict(self) -> Dict[str, Any]:
        return dict(self._tabular)

    def get_table_key_set(self) -> set[str]:
        return {key for key, _ in self._tabular}

    def log_variant(self, log_file: str, variant_data: Dict[str, Any]) -> None:
        mkdir_p(os.path.dirname(log_file))
        with open(log_file, "w") as f:
            json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)

    def dump_tabular(self, *args, **kwargs) -> None:
        """
        Print tabular data to stdout and write to CSV.
        Assumes keys (columns) do not change across iterations.
        """
        write_header = kwargs.pop("write_header", None)

        if len(self._tabular) == 0:
            return

        # stdout printing
        if self._log_tabular_only:
            self.table_printer.print_tabular(self._tabular)
        else:
            for line in tabulate(self._tabular).split("\n"):
                self.log(line, *args, **kwargs)

        tabular_dict = dict(self._tabular)

        # CSV writing
        for tabular_fd in list(self._tabular_fds.values()):
            writer = csv.DictWriter(tabular_fd, fieldnames=list(tabular_dict.keys()))
            if write_header or (
                write_header is None and tabular_fd not in self._tabular_header_written
            ):
                writer.writeheader()
                self._tabular_header_written.add(tabular_fd)
            writer.writerow(tabular_dict)
            tabular_fd.flush()

        # clear current buffer
        del self._tabular[:]

    # ---------- snapshot saving ---------- #

    def save_itr_params(self, itr: int, params: Any) -> None:
        """
        Save parameters according to snapshot_mode:
            - 'all'        : save every itr as itr_<k>.pkl
            - 'last'       : overwrite params.pkl
            - 'gap'        : save every snapshot_gap itrs
            - 'gap_and_last': both of the above
            - 'none'       : do nothing
        """
        if not self._snapshot_dir:
            return

        if self._snapshot_mode == "all":
            file_name = osp.join(self._snapshot_dir, f"itr_{itr}.pkl")
            pickle.dump(params, open(file_name, "wb"))

        elif self._snapshot_mode == "last":
            file_name = osp.join(self._snapshot_dir, "params.pkl")
            pickle.dump(params, open(file_name, "wb"))

        elif self._snapshot_mode == "gap":
            if itr % self._snapshot_gap == 0:
                file_name = osp.join(self._snapshot_dir, f"itr_{itr}.pkl")
                pickle.dump(params, open(file_name, "wb"))

        elif self._snapshot_mode == "gap_and_last":
            if itr % self._snapshot_gap == 0:
                file_name = osp.join(self._snapshot_dir, f"itr_{itr}.pkl")
                pickle.dump(params, open(file_name, "wb"))
            file_name = osp.join(self._snapshot_dir, "params.pkl")
            pickle.dump(params, open(file_name, "wb"))

        elif self._snapshot_mode == "none":
            pass
        else:
            raise NotImplementedError(f"Unknown snapshot_mode: {self._snapshot_mode}")


# Global singleton (matches original usage: from utils.logger import logger)
logger = Logger()


# =========================================================================== #
# setup_logger: convenience wrapper used by main.py
# =========================================================================== #

def setup_logger(
    exp_prefix: str = "default",
    variant: Dict[str, Any] | None = None,
    text_log_file: str = "debug.log",
    variant_log_file: str = "variant.json",
    tabular_log_file: str = "progress.csv",
    snapshot_mode: str = "last",
    snapshot_gap: int = 1,
    log_tabular_only: bool = False,
    log_dir: str | None = None,
    script_name: str | None = None,
    **create_log_dir_kwargs,
) -> str:
    """
    Set up logger to have some reasonable default settings.

    - Creates a log directory (unless `log_dir` is given).
    - Writes variant.json with experiment config.
    - Sets up:
        debug.log     : text logs
        progress.csv  : tabular metrics
    """
    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(exp_prefix, **create_log_dir_kwargs)

    # Save variant (config)
    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    # Set up file outputs
    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)

    if first_time:
        logger.add_tabular_output(tabular_log_path)
    else:
        # Append if reusing the same logger/log_dir
        logger._add_output(tabular_log_path, logger._tabular_outputs, logger._tabular_fds, mode="a")
        for tabular_fd in logger._tabular_fds.values():
            logger._tabular_header_written.add(tabular_fd)

    # Snapshot settings
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)

    # Prefix includes experiment name
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix(f"[{exp_name}] ")

    # Save script name if provided
    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)

    return log_dir
