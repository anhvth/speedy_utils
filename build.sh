rm -rf dist/ build/ *.egg-info
pip install -r requirements.txt
poetry build
twine upload dist/* 