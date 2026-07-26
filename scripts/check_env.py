import shutil
import sys


REQUIRED_COMMANDS = ["curl"]
OPTIONAL_COMMANDS = ["pactl", "paplay", "aplay", "ffplay"]


def main() -> int:
    python_cmd = shutil.which("python3") or shutil.which("python")
    if python_cmd is None:
        print("Missing required command: python3/python")
        return 1

    missing = [cmd for cmd in REQUIRED_COMMANDS if shutil.which(cmd) is None]
    optional_missing = [cmd for cmd in OPTIONAL_COMMANDS if shutil.which(cmd) is None]

    if missing:
        print("Missing required commands:", ", ".join(missing))
        return 1

    if optional_missing:
        print("Optional commands not found:", ", ".join(optional_missing))

    print("Environment looks good for scaffold stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
