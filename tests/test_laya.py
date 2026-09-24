"""The local Laya engine: translation, normalisation, clipping, and the kind switch.

Every test here is offline. The model itself is 843 MB and takes a minute to load, so the
agent object is injected instead: what is worth testing without it is the translation, the
answer normalisation and the error paths, not the model's judgement, which the live
evaluation covers.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cascade.config import ConfigError, JevConfig, config_from_dict  # noqa: E402
from jev_cascade.jev import (  # noqa: E402
    JevError,
    StubJev,
    build_jev,
    choice_question,
    noul_question,
    score_question,
)
from jev_cascade.laya import LayaJev, _patched_checkpoint, checkpoint_for  # noqa: E402


class FakeLaya:
    """Stands in for a loaded laya_mlx Agent: records the call, returns a canned answer."""

    def __init__(self, answers=None, usage=None, error=None) -> None:
        self.answers = answers if answers is not None else {}
        self.usage = usage if usage is not None else {"input_tokens": 10, "output_tokens": 0}
        self.error = error
        self.seen: list[tuple[str, dict]] = []

    def predict(self, state, questions):
        self.seen.append((state, questions))
        if self.error is not None:
            raise self.error
        return {"model": "laya-rl-agent", "answers": self.answers, "usage": self.usage}


def make(agent: FakeLaya, **overrides) -> LayaJev:
    settings = {"kind": "laya", "model": "aac6fef/laya-mlx"}
    settings.update(overrides)
    return LayaJev(JevConfig(**settings), agent=agent)


class TestTranslation(unittest.TestCase):
    def test_a_jev_choice_becomes_a_laya_choice(self) -> None:
        agent = FakeLaya({"q": {"type": "choice", "choice": "billing", "confidence": 0.9,
                                "probabilities": {"billing": 0.9, "sales": 0.1}}})
        client = make(agent)

        result = client.ask(
            "a state",
            {"q": choice_question({"billing": "money", "sales": "new"}, "Which team?")},
        )

        _, questions = agent.seen[0]
        self.assertEqual(questions["q"]["type"], "choice")
        self.assertEqual(questions["q"]["criteria"], {"billing": "money", "sales": "new"})
        self.assertEqual(questions["q"]["instructions"], "Which team?")
        self.assertEqual(result.choice("q"), ("billing", 0.9))
        self.assertEqual(result.probabilities("q"), {"billing": 0.9, "sales": 0.1})

    def test_noul_and_score_survive_the_round_trip(self) -> None:
        agent = FakeLaya({
            "meets": {"type": "noul", "noul": 0.83, "confidence": 0.83},
            "grade": {"type": "score", "score": 2.0, "confidence": 0.7, "legend": {"0": "a", "2": "b"}},
        })
        client = make(agent)

        result = client.ask(
            "a state",
            {"meets": noul_question("it is done", "it is not done"),
             "grade": score_question(["low", "mid", "high"])},
        )

        self.assertAlmostEqual(result.noul("meets"), 0.83)
        self.assertEqual(result.score("grade"), (2.0, 0.7))
        self.assertEqual(result.answers["grade"]["legend"], {"0": "a", "2": "b"})

    def test_an_instruction_is_synthesised_when_the_caller_omits_one(self) -> None:
        agent = FakeLaya({"executor": {"type": "choice", "choice": "cheap", "confidence": 0.5}})
        client = make(agent)

        client.ask("a state", {"executor": choice_question({"cheap": None, "expensive": None})})

        _, questions = agent.seen[0]
        self.assertIn("executor", questions["executor"]["instructions"])

    def test_a_noul_without_instructions_asks_the_true_case(self) -> None:
        agent = FakeLaya({"meets": {"type": "noul", "noul": 0.5}})
        client = make(agent)

        client.ask("a state", {"meets": noul_question("every requirement is met", "it is not")})

        _, questions = agent.seen[0]
        self.assertIn("every requirement is met", questions["meets"]["instructions"])


class TestAccounting(unittest.TestCase):
    def test_the_local_engine_costs_nothing_and_counts_its_calls(self) -> None:
        agent = FakeLaya({"q": {"type": "choice", "choice": "a", "confidence": 1.0}},
                         usage={"input_tokens": 321, "output_tokens": 0})
        client = make(agent)

        result = client.ask("a state", {"q": choice_question({"a": None})})

        self.assertEqual(result.cost_usd, 0.0)
        self.assertEqual(result.tokens_in, 321)
        self.assertEqual(result.tokens_out, 0)
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.total_cost_usd, 0.0)
        self.assertEqual(client.total_tokens_in, 321)
        self.assertFalse(result.stub)

    def test_an_empty_question_set_is_an_error(self) -> None:
        client = make(FakeLaya())
        with self.assertRaises(JevError):
            client.ask("a state", {})


class TestClipping(unittest.TestCase):
    def test_a_long_state_is_clipped_visibly_rather_than_silently(self) -> None:
        agent = FakeLaya({"q": {"type": "choice", "choice": "a", "confidence": 1.0}})
        client = make(agent)

        client.ask("x" * 20_000, {"q": choice_question({"a": None})})

        sent, _ = agent.seen[0]
        self.assertLess(len(sent), 20_000)
        self.assertEqual(client.truncated_states, 1)

    def test_a_state_that_fits_is_left_alone(self) -> None:
        agent = FakeLaya({"q": {"type": "choice", "choice": "a", "confidence": 1.0}})
        client = make(agent)

        client.ask("short state", {"q": choice_question({"a": None})})

        sent, _ = agent.seen[0]
        self.assertEqual(sent, "short state")
        self.assertEqual(client.truncated_states, 0)

    def test_a_raised_window_holds_more_state(self) -> None:
        agent = FakeLaya({"q": {"type": "choice", "choice": "a", "confidence": 1.0}})
        narrow = make(agent, max_len=512, head_max_len=192)
        wide = make(agent, max_len=2048, head_max_len=1024)

        narrow.ask("x" * 20_000, {"q": choice_question({"a": None})})
        wide.ask("x" * 20_000, {"q": choice_question({"a": None})})

        narrow_sent = agent.seen[0][0]
        wide_sent = agent.seen[1][0]
        self.assertGreater(len(wide_sent), len(narrow_sent))


class TestErrors(unittest.TestCase):
    def test_too_many_options_becomes_a_typed_error(self) -> None:
        agent = FakeLaya(error=ValueError("Question 'item' has too many options for the token budget"))
        client = make(agent)

        with self.assertRaises(JevError) as ctx:
            client.ask("a state", {"item": choice_question({str(i): None for i in range(300)})})

        self.assertIn("too many options", str(ctx.exception))

    def test_a_missing_package_says_how_to_get_it(self) -> None:
        with mock.patch.dict(sys.modules, {"laya_mlx": None}):
            with self.assertRaises(JevError) as ctx:
                LayaJev(JevConfig(kind="laya"))

        message = str(ctx.exception)
        self.assertIn("laya-mlx", message)
        self.assertIn("kind", message)

    def test_half_a_context_raise_is_refused(self) -> None:
        with self.assertRaises(JevError) as ctx:
            LayaJev(JevConfig(kind="laya", max_len=2048))
        self.assertIn("together", str(ctx.exception))


class TestCheckpointResolution(unittest.TestCase):
    """A directly built config must not send the hosted model name to Hugging Face."""

    def test_the_hosted_default_becomes_the_laya_checkpoint(self) -> None:
        # the bug this covers: JevConfig() carries model="jev-latest"
        self.assertEqual(checkpoint_for(JevConfig(kind="laya")), "aac6fef/laya-mlx")
        self.assertEqual(
            checkpoint_for(JevConfig(kind="laya", model="jev-latest")), "aac6fef/laya-mlx"
        )

    def test_an_empty_model_falls_back(self) -> None:
        self.assertEqual(checkpoint_for(JevConfig(kind="laya", model="")), "aac6fef/laya-mlx")

    def test_a_chosen_checkpoint_is_kept(self) -> None:
        for name in ("aac6fef/laya-multilingual-mlx", "convaiinnovations/laya-typed-decisions"):
            self.assertEqual(checkpoint_for(JevConfig(kind="laya", model=name)), name)


class TestKindSwitch(unittest.TestCase):
    def test_the_configured_kind_picks_the_engine(self) -> None:
        with mock.patch.object(LayaJev, "_load", lambda self: FakeLaya()):
            local = build_jev(JevConfig(kind="laya"))

        self.assertIsInstance(local, LayaJev)
        hosted = build_jev(JevConfig(kind="http"))
        self.assertNotIsInstance(hosted, LayaJev)
        self.assertEqual(hosted.config.model, "jev-latest")

    def test_a_dry_run_stubs_either_kind(self) -> None:
        self.assertIsInstance(build_jev(JevConfig(kind="laya"), dry_run=True), StubJev)
        self.assertIsInstance(build_jev(JevConfig(kind="http"), dry_run=True), StubJev)


class TestPatchedCheckpoint(unittest.TestCase):
    def _fake_checkpoint(self, root: Path) -> None:
        (root / "encoder").mkdir(parents=True)
        (root / "tokenizer").mkdir(parents=True)
        (root / "encoder" / "config.json").write_text("{}")
        (root / "tokenizer" / "tokenizer.json").write_text("{}")
        (root / "mlx_config.json").write_text("{}")
        (root / "model.safetensors").write_text("weights")
        (root / "rl_agent_config.json").write_text(json.dumps({"max_len": 512, "head_max_len": 192}))

    def test_it_raises_the_limits_and_links_the_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            self._fake_checkpoint(root)
            fake_module = mock.Mock()
            fake_module.agent.resolve_model.return_value = str(root)

            patched = Path(_patched_checkpoint(fake_module, "some/model", 2048, 1024))

            settings = json.loads((patched / "rl_agent_config.json").read_text())
            self.assertEqual(settings["max_len"], 2048)
            self.assertEqual(settings["head_max_len"], 1024)
            self.assertTrue((patched / "model.safetensors").is_symlink())
            self.assertTrue((patched / "encoder" / "config.json").is_file())

    def test_a_checkpoint_without_a_config_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            root.mkdir()
            fake_module = mock.Mock()
            fake_module.agent.resolve_model.return_value = str(root)

            with self.assertRaises(JevError) as ctx:
                _patched_checkpoint(fake_module, "some/model", 2048, 1024)
            self.assertIn("rl_agent_config.json", str(ctx.exception))


class TestConfigForTheLocalEngine(unittest.TestCase):
    def test_laya_defaults_to_a_checkpoint_and_a_zero_price(self) -> None:
        config = config_from_dict({"tiers": [{"name": "cheap"}], "jev": {"kind": "laya"}})
        self.assertEqual(config.jev.kind, "laya")
        self.assertEqual(config.jev.model, "aac6fef/laya-mlx")
        self.assertEqual(config.jev.price.input, 0.0)
        self.assertEqual(config.jev.max_len, 0)

    def test_the_hosted_defaults_are_untouched(self) -> None:
        config = config_from_dict({"tiers": [{"name": "cheap"}]})
        self.assertEqual(config.jev.kind, "http")
        self.assertEqual(config.jev.model, "jev-latest")
        self.assertEqual(config.jev.price.input, 0.042)

    def test_an_unknown_kind_is_refused(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict({"tiers": [{"name": "cheap"}], "jev": {"kind": "telepathy"}})

    def test_half_a_context_raise_is_refused(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict(
                {"tiers": [{"name": "cheap"}], "jev": {"kind": "laya", "max_len": 2048}}
            )

    def test_the_context_raise_must_be_ordered(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict(
                {
                    "tiers": [{"name": "cheap"}],
                    "jev": {"kind": "laya", "max_len": 512, "head_max_len": 1024},
                }
            )

    def test_the_example_config_still_loads(self) -> None:
        from jev_cascade.config import load_config

        root = Path(__file__).resolve().parent.parent
        self.assertEqual(load_config(root / "config.example.toml").jev.kind, "http")


@unittest.skipUnless(
    os.environ.get("JEV_CASCADE_LAYA_LIVE") == "1",
    "set JEV_CASCADE_LAYA_LIVE=1 to load the real model (downloads 843 MB)",
)
class TestLiveModel(unittest.TestCase):
    """Opt-in: the real checkpoint, the real answer. Run with JEV_CASCADE_LAYA_LIVE=1."""

    def test_routing_style_choice_answers_with_a_distribution(self) -> None:
        client = LayaJev(JevConfig(kind="laya"))
        result = client.ask(
            "I was charged twice this month and need a refund.",
            {"department": choice_question(
                {"billing": "invoices, payments, refunds", "technical": "bugs and outages"},
                "Which team should handle this request?",
            )},
        )
        self.assertEqual(result.choice("department")[0], "billing")
        self.assertEqual(result.cost_usd, 0.0)


if __name__ == "__main__":
    unittest.main()
