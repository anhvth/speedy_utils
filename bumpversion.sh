#!/bin/zsh
# Bump patch version using poetry, commit, and push
set -e

echo 'Current version:'
poetry version

# Determine bump type (patch, minor, major)
BUMP_TYPE=${1:-patch}
if [[ "$BUMP_TYPE" != "patch" && "$BUMP_TYPE" != "minor" && "$BUMP_TYPE" != "major" ]]; then
	echo "Usage: $0 [patch|minor|major]"
	exit 1
fi

echo "Bumping $BUMP_TYPE version..."
poetry version $BUMP_TYPE

NEW_VERSION=$(poetry version -s)
echo "New version: $NEW_VERSION"

git add pyproject.toml
git commit -m "bumpversion"
git push
echo 'Version bumped, committed, and pushed.'
