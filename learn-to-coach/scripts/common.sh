#!/usr/bin/env bash

l2c_die() {
    echo "ERROR: $*" >&2
    exit 2
}

l2c_is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        0|false|no|off) return 1 ;;
        *) l2c_die "expected a boolean, got '$1'" ;;
    esac
}

l2c_run_command() {
    local dry_run="$1"
    shift
    if l2c_is_true "${dry_run}"; then
        printf 'DRY RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

l2c_require_file() {
    local path="$1"
    local label="$2"
    [[ -f "${path}" ]] || l2c_die "missing ${label}: ${path}"
}
