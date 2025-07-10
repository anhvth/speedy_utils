#!/bin/zsh
# Bump patch version using poetry, commit, and push
set -e

echo 'Current version:'
poetry version

echo 'Bumping patch version...'
poetry version patch

NEW_VERSION=$(poetry version -s)
echo "New version: $NEW_VERSION"

git add pyproject.toml
git commit -m "bumpversion"
git push
echo 'Version bumped, committed, and pushed.'
