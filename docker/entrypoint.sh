#!/usr/bin/env sh
set -e

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="${PYTHONPATH:-/app}"

exec "$@"
