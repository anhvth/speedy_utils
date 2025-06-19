if ! bump2version --allow-dirty patch; then
    echo "Error: bump2version failed"
    exit 1
fi
version=$(bump2version --allow-dirty --dry-run --list patch | grep new_version | cut -d '=' -f2 | xargs)
message="Bumped version to $version"
git add -A && git commit -m "$message"
git push
