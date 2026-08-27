"""One-shot fix: adds opponent_type logging so a Baseline1-vs-Baseline2 win-rate
breakdown becomes computable after the real run, instead of guessed at. Currently
opp_type is a purely local loop variable in MatchLevelSBSOTrainer.run() - never
stored anywhere _on_episode_end (a different file) could read it. Two-part fix,
both exact-match, run together:

1. src/sbso/match_trainer.py: store opp_type as self.last_opponent_type right after
   it's sampled, so it survives past the loop iteration that set it.
2. scripts/run_phase4_pilot.py: read trainer.last_opponent_type into each
   training_pairs.jsonl entry, tagged per-episode (all decisions within one episode
   share the same opponent, so this is correct to write once per pair, not once
   per episode - matches how "episode"/"lssd_text"/"strategy" are already written
   per-pair in the same loop).

Run once on the pod, from the repo root:
    python fix_opponent_type_logging.py
"""

from pathlib import Path

# --- Fix 1: match_trainer.py ------------------------------------------------------

target1 = Path("src/sbso/match_trainer.py")
content1 = target1.read_text()

old1 = (
    '            opp_type = self.opponent_pool.sample(ep, self.checkpoint_mgr.has_checkpoint())\n'
    '            self.env.opponent_policy = self.opponent_factory(opp_type)'
)
new1 = (
    '            opp_type = self.opponent_pool.sample(ep, self.checkpoint_mgr.has_checkpoint())\n'
    '            self.last_opponent_type = opp_type   # exposed for _on_episode_end callbacks -\n'
    '                                                   # previously local-only, no way to log\n'
    '                                                   # which opponent a given episode played\n'
    '            self.env.opponent_policy = self.opponent_factory(opp_type)'
)

if old1 not in content1:
    print("ERROR: Fix 1 anchor not found in match_trainer.py - file may differ from expected.")
    raise SystemExit(1)
content1 = content1.replace(old1, new1)
target1.write_text(content1)
print("Fix 1 applied: match_trainer.py now stores self.last_opponent_type")

# --- Fix 2: run_phase4_pilot.py ----------------------------------------------------

target2 = Path("scripts/run_phase4_pilot.py")
content2 = target2.read_text()

old2 = (
    '            pairs_file.write(json.dumps({"episode": tagged_ep, "lssd_text": text, "strategy": strat}) + "\\n")'
)
new2 = (
    '            pairs_file.write(json.dumps({\n'
    '                "episode": tagged_ep, "lssd_text": text, "strategy": strat,\n'
    '                "opponent_type": getattr(trainer, "last_opponent_type", None),\n'
    '            }) + "\\n")'
)

if old2 not in content2:
    print("ERROR: Fix 2 anchor not found in run_phase4_pilot.py - file may differ from expected.")
    raise SystemExit(1)
content2 = content2.replace(old2, new2)
target2.write_text(content2)
print("Fix 2 applied: run_phase4_pilot.py now writes opponent_type into training_pairs.jsonl")

import ast
ast.parse(target1.read_text())
ast.parse(target2.read_text())
print("Both files parse correctly - syntax OK.")
