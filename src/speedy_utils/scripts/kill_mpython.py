#!/usr/bin/env python3
"""Script to kill all tmux sessions matching 'mpython*' pattern."""

import subprocess
import sys


def main():
    """Kill all tmux sessions with names starting with 'mpython'."""
    try:
        # Get list of tmux sessions matching the pattern
        result = subprocess.run(
            ["tmux", "ls"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            if "no server running" in result.stderr.lower():
                print("No tmux server running.")
                return
            print(f"Error listing tmux sessions: {result.stderr}")
            sys.exit(result.returncode)

        sessions = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                session_name = line.split(':')[0]
                if session_name.startswith('mpython'):
                    sessions.append(session_name)

        if not sessions:
            print("No tmux sessions found matching 'mpython*'")
            return

        print(f"Found {len(sessions)} tmux session(s) to kill: {', '.join(sessions)}")

        # Kill each session
        for session in sessions:
            kill_result = subprocess.run(
                ["tmux", "kill-session", "-t", session],
                capture_output=True,
                text=True
            )
            if kill_result.returncode == 0:
                print(f"Successfully killed tmux session '{session}'")
            else:
                print(f"Error killing tmux session '{session}': {kill_result.stderr}")

    except FileNotFoundError:
        print("Error: tmux command not found. Please ensure tmux is installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
