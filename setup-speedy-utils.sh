#!/bin/bash

set -euo pipefail

REPO_URL="${SPEEDY_UTILS_REPO_URL:-https://github.com/anhvth/speedy_utils}"
REPO_REF="${SPEEDY_UTILS_REF:-master}"
SKILLS_ROOT_REL=".github/skills"
REMOTE_CACHE_ROOT="${XDG_CONFIG_HOME:-$HOME/.config}/ai-skills/sources/speedy_utils"
ACTIVE_TARGETS=()

echo "Setting up speedy-utils AI agent skills..."

ensure_target_dir() {
    local dir_path="$1"
    local parent_dir

    parent_dir="$(dirname "$dir_path")"

    if [[ -e "$parent_dir" && ! -d "$parent_dir" ]]; then
        echo "Skipping $dir_path because parent path $parent_dir is not a directory" >&2
        return 1
    fi

    if [[ -e "$dir_path" && ! -d "$dir_path" ]]; then
        echo "Skipping $dir_path because it exists and is not a directory" >&2
        return 1
    fi

    mkdir -p "$dir_path"
    ACTIVE_TARGETS+=("$dir_path")
}

link_skill() {
    local target_dir="$1"
    local skill_name="$2"
    local skill_target="$3"

    if [[ ! -d "$target_dir" ]]; then
        return 1
    fi

    ln -sfn "$skill_target" "$target_dir/$skill_name"
}

cleanup_repo_managed_links() {
    local target_dir="$1"
    local managed_root="$2"
    local cleanup_mode="${3:-target-prefix}"

    if [[ ! -d "$target_dir" ]]; then
        return 0
    fi

    shopt -s nullglob
    for existing_path in "$target_dir"/*; do
        local existing_name
        local existing_target

        existing_name="$(basename "$existing_path")"
        if [[ -e "$managed_root/$existing_name" ]]; then
            continue
        fi
        if [[ ! -L "$existing_path" ]]; then
            continue
        fi

        if [[ "$cleanup_mode" == "missing-name" ]]; then
            rm -f "$existing_path"
            echo "Removed stale $existing_name from $target_dir"
            continue
        fi

        existing_target="$(readlink "$existing_path")"
        case "$existing_target" in
            "$managed_root"/*)
                rm -f "$existing_path"
                echo "Removed stale $existing_name from $target_dir"
                ;;
        esac
    done
    shopt -u nullglob
}

resolve_script_dir() {
    local source_path="${BASH_SOURCE[0]:-}"
    if [[ -n "$source_path" && -f "$source_path" ]]; then
        cd -- "$(dirname -- "$source_path")" && pwd
        return 0
    fi
    return 1
}

resolve_local_repo_root() {
    local candidate=""

    if candidate="$(resolve_script_dir)"; then
        if [[ -d "$candidate/$SKILLS_ROOT_REL" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi

    if [[ -d "$PWD/$SKILLS_ROOT_REL" ]]; then
        printf '%s\n' "$PWD"
        return 0
    fi

    return 1
}

bootstrap_remote_repo() {
    local source_root="$REMOTE_CACHE_ROOT"
    local tmp_root

    if ! command -v git >/dev/null 2>&1; then
        echo "Remote bootstrap requires git to be installed." >&2
        exit 1
    fi

    mkdir -p "$(dirname "$source_root")"
    tmp_root="$(mktemp -d "${source_root}.tmp.XXXXXX")"
    trap 'rm -rf "$tmp_root"' EXIT

    echo "Cloning $REPO_URL@$REPO_REF into $source_root" >&2
    git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$tmp_root/repo" >/dev/null 2>&1

    rm -rf "$source_root"
    mkdir -p "$(dirname "$source_root")"
    mv "$tmp_root/repo" "$source_root"

    trap - EXIT
    rm -rf "$tmp_root"
    printf '%s\n' "$source_root"
}

mode="remote"
source_root=""

if source_root="$(resolve_local_repo_root)"; then
    mode="local"
else
    source_root="$(bootstrap_remote_repo)"
fi

skills_root="$source_root/$SKILLS_ROOT_REL"
if [[ ! -d "$skills_root" ]]; then
    echo "Missing skills directory: $skills_root" >&2
    exit 1
fi

if [[ "$mode" == "local" ]]; then
    CLAUDE_SKILLS_DIR="$source_root/.claude/skills"
    CODEX_SKILLS_DIR="$source_root/.codex/skills"
else
    CLAUDE_SKILLS_DIR="$HOME/.claude/skills"
    CODEX_SKILLS_DIR="$HOME/.codex/skills"
fi
GLOBAL_SKILLS_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/ai-skills"

ensure_target_dir "$CLAUDE_SKILLS_DIR" || true
ensure_target_dir "$CODEX_SKILLS_DIR" || true
ensure_target_dir "$GLOBAL_SKILLS_DIR" || true

if [[ "$mode" == "local" ]]; then
    cleanup_repo_managed_links "$CLAUDE_SKILLS_DIR" "$skills_root" "missing-name"
    cleanup_repo_managed_links "$CODEX_SKILLS_DIR" "$skills_root" "missing-name"
else
    cleanup_repo_managed_links "$CLAUDE_SKILLS_DIR" "$skills_root"
    cleanup_repo_managed_links "$CODEX_SKILLS_DIR" "$skills_root"
fi
cleanup_repo_managed_links "$GLOBAL_SKILLS_DIR" "$skills_root"

skill_count=0
for skill_path in "$skills_root"/*; do
    if [[ ! -d "$skill_path" || ! -f "$skill_path/SKILL.md" ]]; then
        continue
    fi

    skill_name="$(basename "$skill_path")"

    link_skill "$CLAUDE_SKILLS_DIR" "$skill_name" "$skill_path" || true
    link_skill "$CODEX_SKILLS_DIR" "$skill_name" "$skill_path" || true
    link_skill "$GLOBAL_SKILLS_DIR" "$skill_name" "$skill_path" || true

    echo "Linked $skill_name"
    ((skill_count += 1))
done

if [[ "$skill_count" -eq 0 ]]; then
    echo "No skills found under $skills_root" >&2
    exit 1
fi

if [[ "${#ACTIVE_TARGETS[@]}" -eq 0 ]]; then
    echo "No install targets were available" >&2
    exit 1
fi

echo "Done. Linked $skill_count skill(s) into:"
for target_dir in "${ACTIVE_TARGETS[@]}"; do
    echo "  - $target_dir"
done
