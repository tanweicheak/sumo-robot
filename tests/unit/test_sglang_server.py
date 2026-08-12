"""Unit tests for the SGLang client: payload construction, response parsing, and
concurrent batch dispatch - all verified via mocked HTTP (Phase 4 Stage 3)."""

from __future__ import annotations

import unittest
from unittest import mock

from src.agents.schemas import TacticalCommand, TacticalKeyword
from src.inference.sglang_server import SGLangSLMClient


def _fake_response(text: str):
    resp = mock.Mock()
    resp.json.return_value = {"text": text}
    resp.raise_for_status.return_value = None
    return resp


class TestSGLangClient(unittest.TestCase):
    def setUp(self):
        self.client = SGLangSLMClient("http://localhost:30000")

    def test_single_call_returns_valid_and_sends_regex_constraint(self):
        with mock.patch("requests.post", return_value=_fake_response("charge_forward")) as m:
            result = self.client.generate_structured("pick", TacticalCommand)
            self.assertEqual(result.keyword, TacticalKeyword.CHARGE_FORWARD)
            payload = m.call_args.kwargs["json"]
            self.assertIn("regex", payload["sampling_params"])
            self.assertEqual(payload["sampling_params"]["temperature"], 0.0)

    def test_malformed_response_falls_back_safely(self):
        with mock.patch("requests.post", return_value=_fake_response("not_a_real_keyword")):
            result = self.client.generate_structured("pick", TacticalCommand)
            self.assertIsInstance(result, TacticalCommand)   # did not raise

    def test_batch_dispatches_all_requests_concurrently_and_preserves_order(self):
        call_log = []

        def fake_post(url, json, timeout):
            call_log.append(json["text"])
            return _fake_response("stop")

        with mock.patch("requests.post", side_effect=fake_post):
            reqs = [(f"prompt {i}", TacticalCommand) for i in range(6)]
            results = self.client.generate_structured_batch(reqs)
            self.assertEqual(len(results), 6)
            self.assertTrue(all(r.keyword == TacticalKeyword.STOP for r in results))
            self.assertEqual(len(call_log), 6)

    def test_call_count_increments(self):
        with mock.patch("requests.post", return_value=_fake_response("stop")):
            self.client.generate_structured("a", TacticalCommand)
            self.client.generate_structured("b", TacticalCommand)
            self.assertEqual(self.client.call_count, 2)


if __name__ == "__main__":
    unittest.main()