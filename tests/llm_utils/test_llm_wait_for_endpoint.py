"""Tests for LLM(..., wait_for_endpoint=...) bootstrap retry behaviour."""

from itertools import count
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from llm_utils.lm.llm import LLM, _client_bootstrap_cache


def _make_alive_client(model_id: str = "test-model", base_url: str = "http://ok:8000/v1"):
    client = SimpleNamespace(
        base_url=base_url,
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id=model_id)])
        ),
    )
    return client


def _make_dead_client(base_url: str = "http://dead:8000/v1"):
    def _raise():
        raise ConnectionError(f"refused: {base_url}")

    return SimpleNamespace(
        base_url=base_url,
        models=SimpleNamespace(list=_raise),
    )


class TestLLMWaitForEndpoint(TestCase):
    def setUp(self) -> None:
        # Ensure a clean module-level bootstrap cache between tests.
        _client_bootstrap_cache.clear()

    @patch("llm_utils.lm.llm.get_base_client")
    def test_waits_for_endpoint_with_progress_logging(self, mock_get_client):
        """Bootstrap fails twice then succeeds; we observe wait progress logs."""
        attempts = count()
        client = _make_dead_client()

        def fake_get_client(*_args, **_kwargs):
            return client

        mock_get_client.side_effect = fake_get_client

        def fake_sleep(seconds):
            n = next(attempts)
            # n=0 is the first sleep inside the wait loop
            if n >= 2:
                # Make the endpoint "come up" after 2 failed attempts
                client.models = SimpleNamespace(
                    list=lambda: SimpleNamespace(
                        data=[SimpleNamespace(id="late-model")]
                    )
                )

        with (
            patch("llm_utils.lm.llm.time.sleep", side_effect=fake_sleep),
            patch("llm_utils.lm.llm.print") as mock_print,
        ):
            llm = LLM(wait_for_endpoint=10.0)

        # Model discovered after the wait
        self.assertEqual(llm.model, "late-model")

        # Progress messages were emitted with the expected format
        progress_calls = [
            call
            for call in mock_print.call_args_list
            if call.args and "waiting for" in str(call.args[0])
        ]
        self.assertTrue(
            progress_calls, msg=f"no progress log emitted: {mock_print.call_args_list}"
        )
        # Spot-check the format on the first emitted line
        first = progress_calls[0]
        # print("\rwaiting for {:.0f}/{:.0f}...", end="", flush=True)
        self.assertEqual(first.args[0], "\rwaiting for 1/10...")
        self.assertEqual(first.kwargs.get("end"), "")

        # And the "ready" message was logged at the end
        ready_calls = [
            call
            for call in mock_print.call_args_list
            if call.args and "became ready" in str(call.args[0])
        ]
        self.assertTrue(ready_calls)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_fast_fails_when_wait_for_endpoint_is_zero(self, mock_get_client):
        """Setting wait_for_endpoint=0 preserves the old fast-fail behaviour."""
        client = _make_dead_client()
        mock_get_client.return_value = client

        sleep_calls = []

        def fake_sleep(seconds):
            sleep_calls.append(seconds)

        with (
            patch("llm_utils.lm.llm.time.sleep", side_effect=fake_sleep),
            self.assertRaises(ConnectionError),
        ):
            LLM(wait_for_endpoint=0)

        # No waiting occurred: bootstrap was attempted exactly once and failed.
        self.assertEqual(sleep_calls, [])

    @patch("llm_utils.lm.llm.get_base_client")
    def test_raises_when_budget_exhausted(self, mock_get_client):
        """If the endpoint never comes up, the bootstrap error surfaces."""
        client = _make_dead_client()
        mock_get_client.return_value = client

        # Keep the endpoint dead for all attempts
        with (
            patch("llm_utils.lm.llm.time.sleep", lambda _s: None),
            self.assertRaises(ConnectionError),
        ):
            LLM(wait_for_endpoint=3.0)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_succeeds_immediately_when_endpoint_alive(self, mock_get_client):
        """No wait/sleep happens if the endpoint is alive on the first try."""
        client = _make_alive_client(model_id="ready-model")
        mock_get_client.return_value = client

        sleep_calls = []

        def fake_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("llm_utils.lm.llm.time.sleep", side_effect=fake_sleep):
            llm = LLM(wait_for_endpoint=600.0)

        self.assertEqual(llm.model, "ready-model")
        self.assertEqual(sleep_calls, [])

    @patch("llm_utils.lm.llm.get_base_client")
    def test_wait_loop_does_not_spam_bootstrap_error_logs(self, mock_get_client):
        """While waiting, the per-attempt bootstrap ERROR must stay quiet.

        The user is intentionally waiting for the endpoint to come up, so
        we should not flood the terminal with one ``Failed to connect to
        OpenAI client`` line per second. The only output during the wait
        is the ``print()`` progress line.
        """
        attempts = count()
        client = _make_dead_client()
        mock_get_client.return_value = client

        def fake_sleep(seconds):
            n = next(attempts)
            if n >= 2:
                # Endpoint comes up after two failed attempts.
                client.models = SimpleNamespace(
                    list=lambda: SimpleNamespace(
                        data=[SimpleNamespace(id="late-model")]
                    )
                )

        with (
            patch("llm_utils.lm.llm.time.sleep", side_effect=fake_sleep),
            patch("llm_utils.lm.llm.logger") as mock_logger,
            patch("llm_utils.lm.llm.print"),
        ):
            LLM(wait_for_endpoint=10.0)

        # No "Failed to connect to OpenAI client" ERROR was emitted while
        # we were waiting. (The final budget-exhausted ERROR path is a
        # different logger.error call and is also absent here because the
        # endpoint came up in time.)
        bootstrap_errors = [
            call
            for call in mock_logger.error.call_args_list
            if call.args and "Failed to connect to OpenAI client" in str(call.args[0])
        ]
        self.assertEqual(
            bootstrap_errors,
            [],
            msg=f"unexpected per-attempt ERROR spam: {mock_logger.error.call_args_list}",
        )
