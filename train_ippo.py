#!/usr/bin/env python3
"""
scripts/train_ippo.py

Smoke test:
  python -m scripts.train_ippo --max_steps 2000 --evo_steps 1000 --checkpoint_steps 1000 --eval_steps 1000 --eval_loop 1
"""

import os
import csv
import json
import argparse
from typing import Dict, Optional

import numpy as np
import torch

from envs.ecogrid_env_v2 import make_env

from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_multi_agent_on_policy import train_multi_agent_on_policy


# ---------------------------------------------------------------------
# AgileRL population builder (version-safe)
# ---------------------------------------------------------------------
def _net_config_variants(net_config: dict) -> list[dict]:
    """
    AgileRL network kwargs have changed across versions.
    Some versions crash if you pass ANY unknown key to StochasticActor/critic.

    Strategy:
      0) try {} first (let AgileRL defaults handle it)  ✅ most robust
      1) as-is
      2) remove "arch"
      3) rename h_size -> hidden_size
      4) rename h_size -> hidden_sizes
      5) rename h_size -> hidden_dims
    """

    def freeze(x):
        """Make nested structures hashable (lists->tuples, dicts->sorted tuples)."""
        if isinstance(x, dict):
            return tuple((k, freeze(v)) for k, v in sorted(x.items(), key=lambda kv: kv[0]))
        if isinstance(x, (list, tuple)):
            return tuple(freeze(v) for v in x)
        if isinstance(x, set):
            return tuple(sorted(freeze(v) for v in x))
        return x

    base = dict(net_config or {})

    variants = []
    variants.append({})  # ✅ try empty first
    variants.append(base)

    # remove arch
    v_no_arch = {k: v for k, v in base.items() if k != "arch"}
    variants.append(v_no_arch)

    def rename_key(d: dict, src: str, dst: str) -> dict:
        if src in d and dst not in d:
            d = dict(d)
            d[dst] = d.pop(src)
        return d

    variants.append(rename_key(v_no_arch, "h_size", "hidden_size"))
    variants.append(rename_key(v_no_arch, "h_size", "hidden_sizes"))
    variants.append(rename_key(v_no_arch, "h_size", "hidden_dims"))

    # de-duplicate while preserving order
    out = []
    seen = set()
    for v in variants:
        key = freeze(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def build_initial_population(
    *,
    algo: str,
    state_dim,          # list of gymnasium spaces (one per agent)
    action_dim,         # list of gymnasium spaces (one per agent)
    one_hot: bool,
    net_config: dict,
    init_hp: dict,
    population_size: int,
    device: torch.device
):
    """
    Create an AgileRL population in a version-safe way.

    Tries:
      1) legacy initialPopulation(...)
      2) create_population(...) with signature introspection

    Retries multiple net_config variants to survive AgileRL API differences.
    """
    net_variants = _net_config_variants(net_config)

    # 1) Legacy helper (some versions/examples use this)
    for nc in net_variants:
        try:
            from agilerl.utils.utils import initialPopulation  # type: ignore
            return initialPopulation(
                algo=algo,
                state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                net_config=nc,
                INIT_HP=init_hp,
                population_size=population_size,
                device=device,
            )
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument" in msg or "got an unexpected keyword argument" in msg:
                continue
            raise
        except Exception:
            # initialPopulation may not exist; fall back to create_population
            break

    # 2) Newer helper: create_population
    from agilerl.utils.utils import create_population  # type: ignore
    import inspect

    sig = inspect.signature(create_population)
    params = set(sig.parameters.keys())

    def _call_create_population(nc: dict):
        kwargs = {
            "algo": algo,
            "net_config": nc,
            "population_size": population_size,
            "device": device,
        }

        # init_hp / INIT_HP naming differences
        if "init_hp" in params:
            kwargs["init_hp"] = init_hp
        elif "INIT_HP" in params:
            kwargs["INIT_HP"] = init_hp

        # observation/action naming differences
        if "observation_spaces" in params:
            kwargs["observation_spaces"] = state_dim
        elif "observation_space" in params:
            kwargs["observation_space"] = state_dim
        else:
            raise TypeError(f"create_population() missing observation space param. Params={sorted(params)}")

        if "action_spaces" in params:
            kwargs["action_spaces"] = action_dim
        elif "action_space" in params:
            kwargs["action_space"] = action_dim
        else:
            raise TypeError(f"create_population() missing action space param. Params={sorted(params)}")

        if "one_hot" in params:
            kwargs["one_hot"] = one_hot

        return create_population(**kwargs)

    last_err = None
    for nc in net_variants:
        try:
            return _call_create_population(nc)
        except TypeError as e:
            last_err = e
            msg = str(e)
            if "unexpected keyword argument" in msg or "got an unexpected keyword argument" in msg:
                continue
            raise

    raise last_err if last_err is not None else RuntimeError("Failed to create population for unknown reasons.")


# ---------------------------------------------------------------------
# Logging wrapper for PettingZoo ParallelEnv
# ---------------------------------------------------------------------
class LoggingParallelEnv:
    def __init__(self, env, run_dir: str, flush_every_steps: int = 2000):
        self.env = env
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.flush_every_steps = int(flush_every_steps)

        self.global_step = 0
        self.episode_id = 0
        self.step_in_ep = 0

        self.step_csv = os.path.join(self.run_dir, "steps.csv")
        self.episode_csv = os.path.join(self.run_dir, "episodes.csv")

        self._step_rows = []
        self._episode_rows = []
        self._step_header_written = False
        self._ep_header_written = False

        self._ep_return = {}
        self._ep_unmet = {}
        self._ep_shed = {}
        self._ep_import = {}
        self._ep_export = {}

        self._ep_cost_energy = {}
        self._ep_cost_reliability = {}
        self._ep_cost_shed = {}
        self._ep_cost_smooth = {}
        self._ep_cost_soc = {}

    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def agents(self):
        return self.env.agents

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, infos = self.env.reset(seed=seed, options=options)

        self.episode_id += 1
        self.step_in_ep = 0

        self._ep_return = {a: 0.0 for a in self.possible_agents}
        self._ep_unmet = {a: 0.0 for a in self.possible_agents}
        self._ep_shed = {a: 0.0 for a in self.possible_agents}
        self._ep_import = {a: 0.0 for a in self.possible_agents}
        self._ep_export = {a: 0.0 for a in self.possible_agents}

        self._ep_cost_energy = {a: 0.0 for a in self.possible_agents}
        self._ep_cost_reliability = {a: 0.0 for a in self.possible_agents}
        self._ep_cost_shed = {a: 0.0 for a in self.possible_agents}
        self._ep_cost_smooth = {a: 0.0 for a in self.possible_agents}
        self._ep_cost_soc = {a: 0.0 for a in self.possible_agents}

        return obs, infos

    def step(self, actions: Dict[str, np.ndarray]):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        self.global_step += 1
        self.step_in_ep += 1

        for a in self.possible_agents:
            info = infos.get(a, {}) or {}
            r = float(rewards.get(a, 0.0))

            self._ep_return[a] += r
            self._ep_unmet[a] += float(info.get("unmet_kw", 0.0))
            self._ep_shed[a] += float(info.get("shed_kw", 0.0))
            self._ep_import[a] += float(info.get("import_kw", 0.0))
            self._ep_export[a] += float(info.get("export_kw", 0.0))

            self._ep_cost_energy[a] += float(info.get("cost_energy", 0.0))
            self._ep_cost_reliability[a] += float(info.get("cost_reliability", 0.0))
            self._ep_cost_shed[a] += float(info.get("cost_shed", 0.0))
            self._ep_cost_smooth[a] += float(info.get("cost_smooth", 0.0))
            self._ep_cost_soc[a] += float(info.get("cost_soc", 0.0))

            act = actions.get(a, None)
            if act is None:
                act_batt, act_shed = np.nan, np.nan
            else:
                act = np.asarray(act, dtype=np.float32).reshape(-1)
                act_batt = float(act[0]) if act.size > 0 else np.nan
                act_shed = float(act[1]) if act.size > 1 else np.nan

            row = {
                "global_step": self.global_step,
                "episode_id": self.episode_id,
                "step_in_ep": self.step_in_ep,
                "agent": a,
                "reward": r,
                "action_batt": act_batt,
                "action_shed": act_shed,
                "soc": float(info.get("soc", np.nan)),
                "import_kw": float(info.get("import_kw", 0.0)),
                "export_kw": float(info.get("export_kw", 0.0)),
                "shed_kw": float(info.get("shed_kw", 0.0)),
                "unmet_kw": float(info.get("unmet_kw", 0.0)),
                "cost_energy": float(info.get("cost_energy", 0.0)),
                "cost_reliability": float(info.get("cost_reliability", 0.0)),
                "cost_shed": float(info.get("cost_shed", 0.0)),
                "cost_smooth": float(info.get("cost_smooth", 0.0)),
                "cost_soc": float(info.get("cost_soc", 0.0)),
                "price": float(info.get("price", np.nan)),
                "t": int(info.get("t", -1)),
            }
            self._step_rows.append(row)

        all_done = all(
            bool(terminations.get(a, False) or truncations.get(a, False))
            for a in self.possible_agents
        )
        if all_done:
            for a in self.possible_agents:
                self._episode_rows.append({
                    "episode_id": self.episode_id,
                    "agent": a,
                    "ep_return": self._ep_return[a],
                    "ep_unmet_kw_sum": self._ep_unmet[a],
                    "ep_shed_kw_sum": self._ep_shed[a],
                    "ep_import_kw_sum": self._ep_import[a],
                    "ep_export_kw_sum": self._ep_export[a],
                    "ep_cost_energy": self._ep_cost_energy[a],
                    "ep_cost_reliability": self._ep_cost_reliability[a],
                    "ep_cost_shed": self._ep_cost_shed[a],
                    "ep_cost_smooth": self._ep_cost_smooth[a],
                    "ep_cost_soc": self._ep_cost_soc[a],
                })

        if (self.global_step % self.flush_every_steps) == 0 or all_done:
            self.flush()

        return obs, rewards, terminations, truncations, infos

    def flush(self):
        if self._step_rows:
            self._write_csv(self.step_csv, self._step_rows, header_attr="_step_header_written")
            self._step_rows.clear()
        if self._episode_rows:
            self._write_csv(self.episode_csv, self._episode_rows, header_attr="_ep_header_written")
            self._episode_rows.clear()

    def _write_csv(self, path: str, rows: list, header_attr: str):
        header_written = getattr(self, header_attr)
        fieldnames = list(rows[0].keys())
        need_header = (not header_written) or (not os.path.exists(path))

        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if need_header:
                w.writeheader()
                setattr(self, header_attr, True)
            w.writerows(rows)

    def close(self):
        self.flush()
        return self.env.close()


def _ensure_agilerl_hp_defaults(hp: dict) -> dict:
    defaults = {
        "TARGET_KL": 0.02,
        "CLIP_COEF": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "GAE_LAMBDA": 0.95,
        "GAMMA": 0.99,
        "LR": 3e-4,
        "UPDATE_EPOCHS": 4,
        "BATCH_SIZE": 256,
        "LEARN_STEP": 512,
        "ACTION_STD_INIT": 0.6,
    }
    for k, v in defaults.items():
        hp.setdefault(k, v)
    return hp


def main():
    parser = argparse.ArgumentParser(description="Train IPPO on EcoGrid-MARL v2 using AgileRL")
    parser.add_argument("--run_dir", type=str, default="runs/ecogrid_ippo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200_000)
    parser.add_argument("--evo_steps", type=int, default=20_000)
    parser.add_argument("--checkpoint_steps", type=int, default=10_000)
    parser.add_argument("--eval_steps", type=int, default=20_000)
    parser.add_argument("--eval_loop", type=int, default=2)
    parser.add_argument("--flush_every_steps", type=int, default=2000)
    parser.add_argument("--n_homes", type=int, default=3)
    parser.add_argument("--n_hospitals", type=int, default=1)
    parser.add_argument("--n_factories", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[IPPO] device={device}")

    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "run_meta.json"), "w") as f:
        json.dump(vars(args) | {"device": str(device)}, f, indent=2)

    base_env = make_env(
        n_homes=args.n_homes,
        n_hospitals=args.n_hospitals,
        n_factories=args.n_factories,
    )
    env = LoggingParallelEnv(base_env, run_dir=args.run_dir, flush_every_steps=args.flush_every_steps)

    env.reset(seed=args.seed)

    agent_ids = list(env.possible_agents)
    observation_spaces = [env.observation_space(a) for a in agent_ids]
    action_spaces = [env.action_space(a) for a in agent_ids]

    # We keep a candidate config, but builder will try {} first (most compatible)
    NET_CONFIG = {"h_size": [128, 128]}

    INIT_HP = {
        "AGENT_IDS": agent_ids,
        "POPULATION_SIZE": 6,
        "ALGO": "IPPO",
        "BATCH_SIZE": 256,
        "LR": 3e-4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "ACTION_STD_INIT": 0.6,
        "CLIP_COEF": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "UPDATE_EPOCHS": 4,
        "LEARN_STEP": 512,
        "TARGET_KL": 0.02,
    }
    INIT_HP = _ensure_agilerl_hp_defaults(INIT_HP)

    pop = build_initial_population(
        algo=INIT_HP["ALGO"],
        state_dim=observation_spaces,
        action_dim=action_spaces,
        one_hot=False,
        net_config=NET_CONFIG,
        init_hp=INIT_HP,
        population_size=INIT_HP["POPULATION_SIZE"],
        device=device,
    )

    tournament = TournamentSelection(
        tournament_size=2,
        elitism=True,
        population_size=INIT_HP["POPULATION_SIZE"],
        eval_loop=1,
    )

    # ✅ AgileRL 2.2.1: Mutations() does NOT accept algo=
    mutations = Mutations(
        no_mutation=0.35,
        architecture=0.15,
        new_layer_prob=0.2,
        parameters=0.2,
        activation=0.0,
        rl_hp=0.3,
        mutation_sd=0.1,
        rand_seed=args.seed,
        device=device,
    )

    print(f"[IPPO] Training start: max_steps={args.max_steps}, evo_steps={args.evo_steps}")
    trained_pop, pop_fitnesses = train_multi_agent_on_policy(
        env=env,
        env_name="ecogrid_marl_v2",
        algo="IPPO",
        pop=pop,
        max_steps=args.max_steps,
        evo_steps=args.evo_steps,
        eval_steps=args.eval_steps,
        eval_loop=args.eval_loop,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        checkpoint=args.checkpoint_steps,
        checkpoint_path=os.path.join(args.run_dir, "checkpoint_ippo.pt"),
    )

    env.close()

    with open(os.path.join(args.run_dir, "pop_fitnesses.json"), "w") as f:
        json.dump(pop_fitnesses, f, indent=2)

    print("[IPPO] Training complete.")
    print("[IPPO] Elite fitnesses:", pop_fitnesses)


if __name__ == "__main__":
    main()

