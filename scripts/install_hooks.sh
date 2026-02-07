#!/bin/bash
# Install git hooks from .githooks directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$PROJECT_ROOT/.githooks"
GIT_HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

echo "Installing git hooks from $HOOKS_DIR..."

# Create .git/hooks if it doesn't exist
mkdir -p "$GIT_HOOKS_DIR"

# Copy each hook from .githooks to .git/hooks and make executable
for hook in "$HOOKS_DIR"/*; do
    if [ -f "$hook" ]; then
        hook_name=$(basename "$hook")
        echo "  Installing $hook_name..."
        cp "$hook" "$GIT_HOOKS_DIR/$hook_name"
        chmod +x "$GIT_HOOKS_DIR/$hook_name"
    fi
done

echo "Git hooks installed successfully!"
