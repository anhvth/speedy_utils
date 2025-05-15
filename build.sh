if ! bump2version --allow-dirty patch; then
    echo "Error: bump2version failed"
    exit 1
fi

git add -A && git commit -m "Bumped version"
git push
