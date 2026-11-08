"""Apply Web process limits, then exec the original, unchanged training command."""

import argparse
import ctypes
import os
import resource
import signal
import sys


def parent_death_signal(expected_parent=None):
    if sys.platform == "linux":
        parent = os.getppid()
        if expected_parent is not None and parent != expected_parent:
            raise SystemExit(1)
        ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0)
        if os.getppid() != parent:
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-file-bytes", type=int)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or (args.max_file_bytes is not None and args.max_file_bytes < 1):
        parser.error("a command and a positive file limit are required")
    parent_death_signal(args.parent_pid)
    if args.max_file_bytes is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (args.max_file_bytes, args.max_file_bytes))
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
