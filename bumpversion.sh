#!/bin/zsh
# Bump version using uv, commit, and push
set -e

if ! command -v uv >/dev/null 2>&1; then
	echo "uv is required but not installed."
	echo "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
	exit 1
fi

echo "Current version: $(uv version --short)"

# Determine bump type (patch, minor, major)
BUMP_TYPE=${1:-patch}
if [[ "$BUMP_TYPE" != "patch" && "$BUMP_TYPE" != "minor" && "$BUMP_TYPE" != "major" ]]; then
	echo "Usage: $0 [patch|minor|major]"
	exit 1
fi

echo "Bumping $BUMP_TYPE version..."
uv version --bump "$BUMP_TYPE" --frozen

NEW_VERSION=$(uv version --short)
echo "New version: $NEW_VERSION"

git add pyproject.toml
git commit -m "bumpversion"
git push
echo 'Version bumped, committed, and pushed.'
