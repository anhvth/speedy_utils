#!/bin/bash

# Deploy script for publishing to PyPI
# Based on the GitHub Actions workflow

set -e  # Exit on any error






# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required tools
echo_info "Checking required tools..."

if ! command_exists python3; then
    echo_error "Python 3 is required but not installed"
    exit 1
fi

if ! command_exists git; then
    echo_error "Git is required but not installed"
    exit 1
fi

# Install/upgrade required packages
echo_info "Installing/upgrading required packages..."
python3 -m pip install --upgrade pip
pip install poetry twine

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo_error "Not in a git repository"
    exit 1
fi

# Check if we're on master branch
current_branch=$(git branch --show-current)
if [ "$current_branch" != "master" ]; then
    echo_warn "Current branch is '$current_branch', not 'master'"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo_info "Deployment cancelled"
        exit 0
    fi
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo_error "There are uncommitted changes. Please commit or stash them first"
    git status --short
    exit 1
fi

# Check if PYPI_API_TOKEN is set
if [ -z "$PYPI_API_TOKEN" ]; then
    echo_error "PYPI_API_TOKEN environment variable is not set"
    echo_info "Please set it with: export PYPI_API_TOKEN=your_token_here"
    exit 1
fi

# Get current version from pyproject.toml
current_version=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo_info "Current version: $current_version"

# Ask for new version
echo_info "Enter new version (or press Enter to skip version bump):"
read -r new_version

if [ -n "$new_version" ]; then
    echo_info "Bumping version to $new_version..."
    
    # Update version in pyproject.toml
    sed -i "s/^version = \".*\"/version = \"$new_version\"/" pyproject.toml
    
    # Commit version bump
    git add pyproject.toml
    git commit -m "Bump version to $new_version"
    
    echo_info "Version bumped and committed"
else
    echo_info "Skipping version bump"
fi

# Check if last commit message starts with "Bump version"
last_commit_msg=$(git log -1 --pretty=%B)
if [[ ! $last_commit_msg =~ ^"Bump version" ]]; then
    echo_warn "Last commit message doesn't start with 'Bump version'"
    echo_warn "Last commit: $last_commit_msg"
    read -p "Continue deployment anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo_info "Deployment cancelled"
        exit 0
    fi
fi

# Clean previous builds
echo_info "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build package
echo_info "Building package with Poetry..."
poetry build

# Check if dist directory was created and contains files
if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
    echo_error "Build failed - no files in dist directory"
    exit 1
fi

echo_info "Build completed. Files in dist/:"
ls -la dist/

# Upload to PyPI
echo_info "Publishing to PyPI..."
twine upload dist/* \
    --username "__token__" \
    --password "$PYPI_API_TOKEN" \
    --non-interactive

echo_info "🎉 Package published successfully to PyPI!"

# Push to remote if version was bumped
if [ -n "$new_version" ]; then
    echo_info "Pushing version bump to remote..."
    git push origin "$current_branch"
    echo_info "Version bump pushed to remote"
fi

echo_info "Deployment completed successfully!"