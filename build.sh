# rm -rf dist/ build/ *.egg-info
# pip install -r requirements.txt
# poetry build
# twine upload dist/* 
bump2version patch
git add -A && git commit -m "Bumped version"
git push 