#!/usr/bin/env bash

# Install idempotent shell aliases for speedy-utils command-line tools.
set -euo pipefail

readonly MARKER_BEGIN="# >>> speedy-utils uvx aliases >>>"
readonly MARKER_END="# <<< speedy-utils uvx aliases <<<"
readonly PCAT_ALIAS="alias pcat='uvx --from speedy-utils pcat'"
readonly SP_CHAT_ALIAS="alias sp_chat='uvx --from speedy-utils sp_chat'"

usage() {
    cat <<'EOF'
Usage: ./install-tools.sh [--shell bash|zsh|both]

Adds aliases for pcat and sp_chat to the selected shell startup file(s). The
operation is idempotent: existing matching aliases are not duplicated.
EOF
}

install_aliases() {
    local rc_file="$1"
    local -a missing_aliases=()

    if [[ -e "$rc_file" && ! -f "$rc_file" ]]; then
        echo "Skipping $rc_file: it exists but is not a regular file." >&2
        return 1
    fi

    touch "$rc_file"

    if ! grep -Fqx "$PCAT_ALIAS" "$rc_file"; then
        missing_aliases+=("$PCAT_ALIAS")
    fi
    if ! grep -Fqx "$SP_CHAT_ALIAS" "$rc_file"; then
        missing_aliases+=("$SP_CHAT_ALIAS")
    fi

    if [[ "${#missing_aliases[@]}" -eq 0 ]]; then
        echo "Already configured: $rc_file"
        return 0
    fi

    {
        printf '\n%s\n' "$MARKER_BEGIN"
        printf '%s\n' "${missing_aliases[@]}"
        printf '%s\n' "$MARKER_END"
    } >>"$rc_file"

    echo "Added speedy-utils aliases to $rc_file"
}

shell_target="both"
if [[ "${1:-}" == "--shell" ]]; then
    shell_target="${2:-}"
    shift 2
fi

if [[ "$#" -ne 0 ]]; then
    usage >&2
    exit 2
fi

case "$shell_target" in
    bash)
        targets=("$HOME/.bashrc")
        ;;
    zsh)
        targets=("$HOME/.zshrc")
        ;;
    both)
        targets=("$HOME/.bashrc" "$HOME/.zshrc")
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if ! command -v uvx >/dev/null 2>&1; then
    echo "Warning: uvx is not currently on PATH. Install uv, then open a new shell." >&2
fi

for rc_file in "${targets[@]}"; do
    install_aliases "$rc_file"
done

echo "Reload the relevant config, e.g. source ~/.bashrc or source ~/.zshrc."
