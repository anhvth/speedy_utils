#!/bin/zsh
# Bump version using uv, run tests, commit, and push
set -e
set -u
set -o pipefail

if ! command -v uv >/dev/null 2>&1; then
	echo "uv is required but not installed."
	echo "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
	exit 1
fi

if ! command -v git >/dev/null 2>&1; then
	echo "git is required but not installed."
	exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
	echo "Not inside a git repository; aborting."
	exit 1
fi

MIN_PYTEST_PASS_RATE="${BUMP_PYTEST_MIN_PASS_RATE:-0.90}"
DRY_RUN="${BUMPVERSION_DRY_RUN:-0}"

ensure_git_clean() {
	if [[ -n "$(git status --porcelain)" ]]; then
		echo "Working tree is not clean; please commit or stash changes before bumping:"
		git status --short
		return 1
	fi
	return 0
}

ensure_git_clean

run_pytest_with_min_pass_rate() {
	echo "Running pytest (min pass rate: ${MIN_PYTEST_PASS_RATE})..."

	local junit_xml
	junit_xml="$(mktemp)"
	trap "rm -f '$junit_xml'" EXIT

	set +e
	uv run pytest --junitxml "$junit_xml"
	local pytest_status="$?"
	set -e

	if [[ ! -s "$junit_xml" ]]; then
		echo "pytest did not produce a JUnit report; aborting."
		return 1
	fi

	set +e
	uv run python - "$junit_xml" "$MIN_PYTEST_PASS_RATE" <<'PY'
import sys
from xml.etree import ElementTree as ET


def _as_int(value: str | None) -> int:
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return int(float(value))


path = sys.argv[1]
min_rate = float(sys.argv[2])

tree = ET.parse(path)
root = tree.getroot()

if root.tag == "testsuite":
    suites = [root]
else:
    suites = list(root.findall(".//testsuite"))

tests = failures = errors = skipped = 0
for suite in suites:
    tests += _as_int(suite.attrib.get("tests"))
    failures += _as_int(suite.attrib.get("failures"))
    errors += _as_int(suite.attrib.get("errors"))
    skipped += _as_int(suite.attrib.get("skipped"))

executed = tests - skipped
passed = executed - failures - errors

if tests <= 0:
    print("pytest reported 0 tests; aborting.", file=sys.stderr)
    raise SystemExit(2)

if errors > 0:
    print(f"pytest reported errors={errors}; aborting.", file=sys.stderr)
    raise SystemExit(1)

pass_rate = passed / tests
print(
    "pytest summary: "
    f"total={tests}, executed={executed}, passed={passed}, failures={failures}, skipped={skipped}, "
    f"pass_rate={pass_rate:.1%}"
)

if pass_rate < min_rate:
    print(f"pass_rate {pass_rate:.1%} < required {min_rate:.1%}", file=sys.stderr)
    raise SystemExit(1)
PY
	local passrate_status="$?"
	set -e

	if [[ "$passrate_status" -ne 0 ]]; then
		return "$passrate_status"
	fi

	if [[ "$pytest_status" -ne 0 ]]; then
		echo "WARNING: pytest exited with status $pytest_status but pass rate met threshold; continuing."
	fi

	return 0
}

run_pytest_with_min_pass_rate
ensure_git_clean

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

if [[ "$DRY_RUN" == "1" ]]; then
	echo "Dry run enabled (BUMPVERSION_DRY_RUN=1): skipping git commit/push."
	exit 0
fi

git add pyproject.toml
git commit -m "bumpversion"
ensure_git_clean
git push
echo 'Version bumped, committed, and pushed.'
