import builtins

import pytest

from speedy_utils.scripts import mpython


def test_resolve_session_name_no_collision():
    assert mpython.resolve_session_name(base_name="unique-session") == "unique-session"


def test_resolve_session_name_overwrite_kills_only_base(monkeypatch):
    killed = []

    monkeypatch.setattr(
        mpython,
        "get_existing_tmux_sessions",
        lambda: ["mpython", "mpython-1", "mpython-2"],
    )
    monkeypatch.setattr(mpython.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        mpython,
        "_prompt_collision_action",
        lambda base_name="mpython": "overwrite",
    )
    monkeypatch.setattr(
        mpython, "_kill_tmux_session", lambda session: killed.append(session)
    )
    monkeypatch.setattr(
        mpython, "get_next_session_name", lambda base_name="mpython": "BAD"
    )

    session = mpython.resolve_session_name("mpython")
    assert session == "mpython"
    assert killed == ["mpython"]


def test_resolve_session_name_increment(monkeypatch):
    killed = []
    monkeypatch.setattr(mpython, "get_existing_tmux_sessions", lambda: ["mpython"])
    monkeypatch.setattr(mpython.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        mpython,
        "_prompt_collision_action",
        lambda base_name="mpython": "increment",
    )
    monkeypatch.setattr(
        mpython, "_kill_tmux_session", lambda session: killed.append(session)
    )
    monkeypatch.setattr(
        mpython, "get_next_session_name", lambda base_name="mpython": "mpython-7"
    )

    session = mpython.resolve_session_name("mpython")
    assert session == "mpython-7"
    assert killed == []


def test_resolve_session_name_non_interactive_collision_exits(monkeypatch, capsys):
    monkeypatch.setattr(mpython, "get_existing_tmux_sessions", lambda: ["mpython"])
    monkeypatch.setattr(mpython.sys.stdin, "isatty", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        mpython.resolve_session_name("mpython")

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert "interactive" in captured.err


def test_prompt_collision_action_reprompts_on_invalid(monkeypatch, capsys):
    monkeypatch.setattr(mpython, "Console", None)
    answers = iter(["oops", "2"])
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))

    action = mpython._prompt_collision_action("mpython")

    assert action == "increment"
    captured = capsys.readouterr()
    assert "Invalid choice" in captured.err


def test_main_uses_resolved_session_name(monkeypatch):
    captured = {}

    monkeypatch.setattr(mpython.sys, "argv", ["mpython", "-t", "2", "demo.py"])
    monkeypatch.setattr(mpython, "assert_script", lambda _: None)
    monkeypatch.setattr(mpython.shutil, "which", lambda name: "/usr/bin/python")
    monkeypatch.setattr(
        mpython, "resolve_session_name", lambda base_name="mpython": "mpython"
    )
    monkeypatch.setattr(
        mpython,
        "run_in_tmux",
        lambda commands_to_run, tmux_name, num_windows: captured.update(
            {"name": tmux_name, "num_windows": num_windows, "commands": commands_to_run}
        ),
    )
    monkeypatch.setattr(mpython.os, "chmod", lambda *args, **kwargs: None)
    monkeypatch.setattr(mpython.os, "system", lambda *args, **kwargs: 0)

    mpython.main()

    assert captured["name"] == "mpython"
    assert captured["num_windows"] == 2
    assert len(captured["commands"]) == 2


def test_main_distributes_workers_across_nodes(monkeypatch):
    captured = {}
    monkeypatch.setattr(mpython.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(mpython.os, "access", lambda _path, _mode: True)
    monkeypatch.setattr(
        mpython.sys,
        "argv",
        [
            "mpython",
            "--nodes",
            "h2-14,h2-15",
            "--gpus",
            "0-1",
            "--python",
            "/opt/project/bin/python",
            "--",
            "demo.py",
        ],
    )
    monkeypatch.setattr(mpython, "assert_script", lambda _: None)
    monkeypatch.setattr(
        mpython, "resolve_session_name", lambda base_name="mpython": base_name
    )
    monkeypatch.setattr(
        mpython,
        "run_in_tmux",
        lambda commands_to_run, tmux_name, num_windows: captured.update(
            {"num_windows": num_windows, "commands": commands_to_run}
        ),
    )
    monkeypatch.setattr(mpython.os, "chmod", lambda *args, **kwargs: None)
    monkeypatch.setattr(mpython.os, "system", lambda *args, **kwargs: 0)

    mpython.main()

    assert captured["num_windows"] == 4
    commands = captured["commands"]
    assert "ssh h2-14" in commands[0]
    assert (
        "MP_ID=0 MP_TOTAL=4 MP_NODE=h2-14 MP_LOCAL_ID=0 MP_LOCAL_TOTAL=2" in commands[0]
    )
    assert "CUDA_VISIBLE_DEVICES=0" in commands[0]
    assert "/opt/project/bin/python demo.py" in commands[0]
    assert "ssh h2-15" in commands[2]
    assert (
        "MP_ID=2 MP_TOTAL=4 MP_NODE=h2-15 MP_LOCAL_ID=0 MP_LOCAL_TOTAL=2" in commands[2]
    )


def test_main_rejects_missing_python_before_launch(monkeypatch):
    monkeypatch.setattr(
        mpython.sys,
        "argv",
        ["mpython", "--python", "/missing/python", "demo.py"],
    )
    monkeypatch.setattr(mpython, "assert_script", lambda _: None)

    with pytest.raises(SystemExit) as error:
        mpython.main()

    assert error.value.code == 2
