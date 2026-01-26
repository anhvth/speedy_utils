#!/usr/bin/env python3
import argparse
import itertools
import multiprocessing  # Import multiprocessing module
import os
import re
import shlex  # To properly escape command line arguments
import shutil
import subprocess
import sys

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text
    from rich.syntax import Syntax
except ImportError:
    Console = None
    Group = None
    Panel = None
    Text = None
    Syntax = None


taskset_path = shutil.which('taskset')


def get_existing_tmux_sessions():
    """Get list of existing tmux session names."""
    try:
        result = subprocess.run(
            ['tmux', 'list-sessions', '-F', '#{session_name}'],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []
    except FileNotFoundError:
        # tmux not installed
        return []


def get_next_session_name(base_name='mpython'):
    """Get next available session name.

    If 'mpython' doesn't exist, return 'mpython'.
    If 'mpython' exists, return 'mpython-1', 'mpython-2', etc.
    """
    existing_sessions = get_existing_tmux_sessions()

    if base_name not in existing_sessions:
        return base_name

    # Find all existing mpython-N sessions
    pattern = re.compile(rf'^{re.escape(base_name)}-(\d+)$')
    existing_numbers = []

    for session in existing_sessions:
        match = pattern.match(session)
        if match:
            existing_numbers.append(int(match.group(1)))

    # Find the next available number
    next_num = 1
    if existing_numbers:
        next_num = max(existing_numbers) + 1

    return f'{base_name}-{next_num}'


def assert_script(python_path):
    with open(python_path) as f:
        code_str = f.read()
    if 'MP_ID' not in code_str or 'MP_TOTAL' not in code_str:
        helper_code = (
            'import os\n'
            'MP_ID = int(os.getenv("MP_ID", "0"))\n'
            'MP_TOTAL = int(os.getenv("MP_TOTAL", "1"))\n'
            'inputs = list(inputs[MP_ID::MP_TOTAL])'
        )
        if Console and Panel and Text and Syntax and Group:
            console = Console(stderr=True, force_terminal=True)
            syntax = Syntax(helper_code, "python", theme="monokai", line_numbers=False)
            console.print()
            console.print(
                Panel(
                    f'Your script {python_path} is missing MP_ID and/or MP_TOTAL variables.\n\n'
                    f'Add the following code to enable multi-process sharding:',
                    title='[bold yellow]Warning: Missing Multi-Process Variables[/bold yellow]',
                    border_style='yellow',
                    expand=False,
                )
            )
            console.print()
            console.print("```python")
            console.print(syntax)
            console.print("```")
            console.print("-"*80)
        else:
            # Fallback to plain text
            print(f'Warning: MP_ID and MP_TOTAL not found in {python_path}, please add them.', file=sys.stderr)
            print(f'Example:\n{helper_code}', file=sys.stderr)


def run_in_tmux(commands_to_run, tmux_name, num_windows):
    with open('/tmp/start_multirun_tmux.sh', 'w') as script_file:
        script_file.write('#!/bin/bash\n\n')
        script_file.write(f'tmux new-session -d -s {tmux_name}\n')
        for i, cmd in enumerate(itertools.cycle(commands_to_run)):
            if i >= num_windows:
                break
            window_name = f'{tmux_name}:{i}'
            if i == 0:
                script_file.write(f"tmux send-keys -t {window_name} '{cmd}' C-m\n")
            else:
                script_file.write(f'tmux new-window -t {tmux_name}\n')
                script_file.write(f"tmux send-keys -t {window_name} '{cmd}' C-m\n")

        # Make the script executable
        script_file.write('chmod +x /tmp/start_multirun_tmux.sh\n')
        print('Run /tmp/start_multirun_tmux.sh')


def main():
    # Assert that MP_ID and MP_TOTAL are not already set

    helper_code = (
        'import os\n'
        'MP_ID = int(os.getenv("MP_ID", "0"))\n'
        'MP_TOTAL = int(os.getenv("MP_TOTAL", "1"))\n'
        'inputs = list(inputs[MP_ID::MP_TOTAL])'
    )

    parser = argparse.ArgumentParser(
        description='Process fold arguments',
        epilog=f'Helper code for multi-process sharding:\n{helper_code}',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--total_fold', '-t', default=16, type=int, help='total number of folds'
    )
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--ignore_gpus', '-ig', type=str, default='')
    parser.add_argument(
        '--total_cpu',
        type=int,
        default=multiprocessing.cpu_count(),
        help='total number of cpu cores available',
    )
    parser.add_argument(
        'cmd', nargs=argparse.REMAINDER
    )  # This will gather the remaining unparsed arguments

    args = parser.parse_args()
    if not args.cmd or (args.cmd[0] == '--' and len(args.cmd) == 1):
        parser.error('Invalid command provided')
    assert_script(args.cmd[0])

    cmd_str = None
    if args.cmd[0] == '--':
        cmd_str = shlex.join(args.cmd[1:])
    else:
        cmd_str = shlex.join(args.cmd)

    gpus = args.gpus.split(',')
    gpus = [gpu for gpu in gpus if gpu not in args.ignore_gpus.split(',')]
    num_gpus = len(gpus)

    cpu_per_process = max(args.total_cpu // args.total_fold, 1)
    cmds = []
    path_python = shutil.which('python')
    for i in range(args.total_fold):
        gpu = gpus[i % num_gpus]
        cpu_start = (i * cpu_per_process) % args.total_cpu
        cpu_end = ((i + 1) * cpu_per_process - 1) % args.total_cpu
        ENV = f'CUDA_VISIBLE_DEVICES={gpu} MP_ID={i} MP_TOTAL={args.total_fold}'
        if taskset_path:
            fold_cmd = f'{ENV} {taskset_path} -c {cpu_start}-{cpu_end}  {path_python} {cmd_str}'
        else:
            fold_cmd = f'{ENV} {path_python} {cmd_str}'

        cmds.append(fold_cmd)

    session_name = get_next_session_name('mpython')
    run_in_tmux(cmds, session_name, args.total_fold)
    os.chmod('/tmp/start_multirun_tmux.sh', 0o755)  # Make the script executable
    os.system('/tmp/start_multirun_tmux.sh')
    print(f'Started tmux session: {session_name}')


if __name__ == '__main__':
    main()
