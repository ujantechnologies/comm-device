#!/usr/bin/env bash
# run.sh — launch comm_device with the correct display and audio environment.
#
# Works from:
#   • SSH session (exports Wayland + PipeWire socket from the active desktop session)
#   • Direct terminal on the Pi desktop
#   • systemd user service
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${REPO_ROOT}/.venv"

# -- Wayland display -----------------------------------------------------------
# The Wayland compositor socket is usually wayland-0 or wayland-1 under
# /run/user/<uid>/. Detect it automatically when running over SSH.
if [ -z "${WAYLAND_DISPLAY:-}" ]; then
    XDG_RT="/run/user/$(id -u)"
    for sock in "${XDG_RT}"/wayland-{1,0,2}; do
        if [ -S "${sock}" ]; then
            export WAYLAND_DISPLAY="$(basename "${sock}")"
            export XDG_RUNTIME_DIR="${XDG_RT}"
            break
        fi
    done
fi

# -- PipeWire / audio ----------------------------------------------------------
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export PATH="${VENV}/bin:${PATH}"

# -- Launch --------------------------------------------------------------------
echo "WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<not set>}  XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}"
exec "${VENV}/bin/python" -m comm_device "$@"
