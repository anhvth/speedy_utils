"""Regression tests for bare SSH endpoints accepted by ``LLM``."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from llm_utils.lm import ssh_tunnel
from llm_utils.lm.utils import get_base_client


class TestSshTunnel(TestCase):
    def setUp(self) -> None:
        ssh_tunnel._tunnels.clear()

    def tearDown(self) -> None:
        ssh_tunnel._close_tunnels()

    @patch("llm_utils.lm.ssh_tunnel._wait_for_listener")
    @patch("llm_utils.lm.ssh_tunnel._reserve_local_port", return_value=43123)
    @patch("llm_utils.lm.ssh_tunnel.subprocess.Popen")
    def test_reuses_one_tunnel_for_repeated_target(self, popen, _reserve, _wait):
        process = SimpleNamespace(poll=lambda: None, terminate=lambda: None)
        popen.return_value = process

        first = ssh_tunnel.resolve_ssh_endpoint("h1-27:8000")
        second = ssh_tunnel.resolve_ssh_endpoint("h1-27:8000")

        self.assertEqual(first, "http://127.0.0.1:43123/v1")
        self.assertEqual(second, first)
        popen.assert_called_once()

    def test_http_url_bypasses_ssh(self):
        self.assertEqual(
            ssh_tunnel.resolve_ssh_endpoint("http://worker:8000/v1"),
            "http://worker:8000/v1",
        )

    def test_ssh_command_is_loopback_only(self):
        command = ssh_tunnel._ssh_command("h1-27", 8000, 43123)
        self.assertIn("127.0.0.1:43123:127.0.0.1:8000", command)
        self.assertIn("BatchMode=yes", command)

    @patch("llm_utils.lm.ssh_tunnel.os.getpid")
    def test_forked_child_does_not_close_parent_tunnels(self, getpid):
        process = SimpleNamespace(poll=lambda: None, terminate=lambda: None)
        ssh_tunnel._tunnels[("h1-27", 8000)] = (43123, process)
        getpid.return_value = ssh_tunnel._tunnel_owner_pid + 1

        with patch.object(process, "terminate") as terminate:
            ssh_tunnel._close_tunnels()

        terminate.assert_not_called()
        self.assertIn(("h1-27", 8000), ssh_tunnel._tunnels)

    @patch("llm_utils.lm.ssh_tunnel._wait_for_listener")
    @patch(
        "llm_utils.lm.ssh_tunnel._reserve_local_port",
        side_effect=[43121, 43122, 43123, 43124],
    )
    @patch("llm_utils.lm.ssh_tunnel.subprocess.Popen")
    def test_mixed_pool_tunnels_only_bare_ssh_endpoints(
        self, popen, _reserve, _wait
    ):
        process = SimpleNamespace(poll=lambda: None, terminate=lambda: None)
        popen.return_value = process

        clients = get_base_client(
            client=[
                "h1-27:8000",
                "h1-31:8000",
                "h2-15:8100",
                "h2-17:8100",
                8100,
            ]
        )

        self.assertEqual(
            [str(client.base_url) for client in clients],
            [
                "http://127.0.0.1:43121/v1/",
                "http://127.0.0.1:43122/v1/",
                "http://127.0.0.1:43123/v1/",
                "http://127.0.0.1:43124/v1/",
                "http://localhost:8100/v1/",
            ],
        )
        self.assertEqual(popen.call_count, 4)
