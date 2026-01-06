#!/usr/bin/env python3
"""
scripts/train_matd3.py

EcoGrid-MARL v2 + AgileRL (off-policy) + MATD3.

Smoke test:
  python -m scripts.train_matd3 --max_steps 2000 --evo_steps 1000 --checkpoint_steps 1000 --eval_steps 1000 --eval_loop 1

Outputs (under --run_dir, default runs/ecogrid_matd3):
  - run_meta.json
  - steps.csv, episodes.csv         (from LoggingParallelEnv)
  - checkpoint_matd3.pt
  - pop_fitnesses.json
"""

import os
import json
import argparse
import random
from typing import Dict

import numpy as np
import torch

from envs.ecogrid_env_v2 import make_env

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy

# Reuse your proven version-safe population builder + logging wrapper
from scripts.train_ippo import build_initial_population, LoggingParallelEnv


def set_global_seeds(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # More reproducible, sometimes slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _ensure_matd3_hp_defaults(hp: Dict) -> Dict:
    """
    Keep this minimal + safe. The population builder / AgileRL algo class will
    pick what it needs from init_hp. Extra keys won't hurt.
    """
    defaults = {
        "ALGO": "MATD3",
        "POPULATION_SIZE": 6,

        # RL core
        "BATCH_SIZE": 256,
        "LR": 1e-3,
        "GAMMA": 0.95,
        "TAU": 0.01,
        "LEARN_STEP": 100,

        # TD3/MATD3-specific knobs (common conventions)
        # (names are kept generic; AgileRL versions differ slightly,
        # but extra keys are harmless if unused)
        "POLICY_FREQ": 2,              # actor update every N critic updates
        "POLICY_NOISE": 0.2,           # target policy smoothing noise
        "NOISE_CLIP": 0.5,             # clip target noise
        "EXPLORATION_NOISE": 0.1,      # action exploration noise
    }
    for k, v in defaults.items():
        hp.setdefault(k, v)
    return hp


def main():
    parser = argparse.ArgumentParser(
        description="Train MATD3 on EcoGrid-MARL v2 using AgileRL (off-policy)"
    )
    parser.add_argument("--run_dir", type=str, default="runs/ecogrid_matd3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="More reproducible (may be slower).")

    parser.add_argument("--max_steps", type=int, default=200_000)
    parser.add_argument("--evo_steps", type=int, default=20_000)
    parser.add_argument("--checkpoint_steps", type=int, default=10_000)
    parser.add_argument("--eval_steps", type=int, default=20_000)
    parser.add_argument("--eval_loop", type=int, default=2)

    parser.add_argument("--flush_every_steps", type=int, default=2000)
    parser.add_argument("--learning_delay", type=int, default=1000)

    parser.add_argument("--n_homes", type=int, default=3)
    parser.add_argument("--n_hospitals", type=int, default=1)
    parser.add_argument("--n_factories", type=int, default=1)

    # Optional overrides for quick experimentation
    parser.add_argument("--population_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--policy_freq", type=int, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seeds(args.seed, deterministic=args.deterministic)

    print(f"[MATD3] device={device}")
    print(f"[MATD3] run_dir={args.run_dir}")
    print(f"[MATD3] seed={args.seed} deterministic={args.deterministic}")

    os.makedirs(args.run_dir, exist_ok=True)

    # --- Env ---
    base_env = make_env(
        n_homes=args.n_homes,
        n_hospitals=args.n_hospitals,
        n_factories=args.n_factories,
    )
    env = LoggingParallelEnv(
        base_env,
        run_dir=args.run_dir,
        flush_every_steps=args.flush_every_steps,
    )
    env.reset(seed=args.seed)

    agent_ids = list(env.possible_agents)
    observation_spaces = [env.observation_space(a) for a in agent_ids]
    action_spaces = [env.action_space(a) for a in agent_ids]

    # --- HP / NET ---
    NET_CONFIG = {"h_size": [128, 128]}

    INIT_HP = _ensure_matd3_hp_defaults({"AGENT_IDS": agent_ids})

    # Apply CLI overrides (if provided)
    if args.population_size is not None:
        INIT_HP["POPULATION_SIZE"] = int(args.population_size)
    if args.batch_size is not None:
        INIT_HP["BATCH_SIZE"] = int(args.batch_size)
    if args.lr is not None:
        INIT_HP["LR"] = float(args.lr)
    if args.gamma is not None:
        INIT_HP["GAMMA"] = float(args.gamma)
    if args.tau is not None:
        INIT_HP["TAU"] = float(args.tau)
    if args.policy_freq is not None:
        INIT_HP["POLICY_FREQ"] = int(args.policy_freq)

    # --- Save run meta (great for reproducibility) ---
    run_meta = {
        **vars(args),
        "device": str(device),
        "agent_ids": agent_ids,
        "init_hp": INIT_HP,
        "net_config": NET_CONFIG,
    }
    with open(os.path.join(args.run_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    # --- Population ---
    pop = build_initial_population(
        algo=INIT_HP["ALGO"],                  # "MATD3"
        state_dim=observation_spaces,
        action_dim=action_spaces,
        one_hot=False,
        net_config=NET_CONFIG,
        init_hp=INIT_HP,
        population_size=INIT_HP["POPULATION_SIZE"],
        device=device,
    )

    # --- Replay Buffer ---
    memory = MultiAgentReplayBuffer(
        memory_size=1_000_000,
        field_names=["state", "action", "reward", "next_state", "done"],
        agent_ids=agent_ids,
        device=device,
    )

    # --- Evolution / HPO ---
    tournament = TournamentSelection(
        tournament_size=2,
        elitism=True,
        population_size=INIT_HP["POPULATION_SIZE"],
        eval_loop=1,
    )

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

    print(
        "[MATD3] Training start: "
        f"max_steps={args.max_steps}, evo_steps={args.evo_steps}, "
        f"checkpoint_steps={args.checkpoint_steps}, eval_steps={args.eval_steps}, eval_loop={args.eval_loop}"
    )

    checkpoint_path = os.path.join(args.run_dir, "checkpoint_matd3.pt")
    try:
        trained_pop, pop_fitnesses = train_multi_agent_off_policy(
            env=env,
            env_name="ecogrid_marl_v2",
            algo="MATD3",  # keep consistent with INIT_HP["ALGO"]
            pop=pop,
            memory=memory,
            max_steps=args.max_steps,
            evo_steps=args.evo_steps,
            eval_steps=args.eval_steps,
            eval_loop=args.eval_loop,
            learning_delay=args.learning_delay,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            checkpoint=args.checkpoint_steps,
            checkpoint_path=checkpoint_path,
        )
    finally:
        env.close()

    with open(os.path.join(args.run_dir, "pop_fitnesses.json"), "w") as f:
        json.dump(pop_fitnesses, f, indent=2)

    print("[MATD3] Training complete.")
    print(f"[MATD3] Saved checkpoint: {checkpoint_path}")
    print("[MATD3] Elite fitnesses:", pop_fitnesses)


if __name__ == "__main__":
    main()
