#!/usr/bin/env python3
"""
scripts/train_maddpg.py

Smoke test:
  python -m scripts.train_maddpg --max_steps 2000 --evo_steps 1000 --checkpoint_steps 1000 --eval_steps 1000 --eval_loop 1

Outputs (under --run_dir, default runs/ecogrid_maddpg):
  - run_meta.json
  - steps.csv, episodes.csv         (from LoggingParallelEnv)
  - checkpoint_maddpg.pt
  - pop_fitnesses.json
"""

import os
import json
import argparse

import torch

from envs.ecogrid_env_v2 import make_env

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy

# Reuse your proven version-safe population builder + logging wrapper
from scripts.train_ippo import build_initial_population, LoggingParallelEnv


def _ensure_maddpg_hp_defaults(hp: dict) -> dict:
    defaults = {
        "ALGO": "MADDPG",
        "POPULATION_SIZE": 6,
        "BATCH_SIZE": 256,
        "LR": 1e-3,
        "GAMMA": 0.95,
        "TAU": 0.01,
        "LEARN_STEP": 100,
    }
    for k, v in defaults.items():
        hp.setdefault(k, v)
    return hp


def main():
    parser = argparse.ArgumentParser(description="Train MADDPG on EcoGrid-MARL v2 using AgileRL 2.2.1 (off-policy)")
    parser.add_argument("--run_dir", type=str, default="runs/ecogrid_maddpg")
    parser.add_argument("--seed", type=int, default=0)
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MADDPG] device={device}")
    print(f"[MADDPG] run_dir={args.run_dir}")

    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "run_meta.json"), "w") as f:
        json.dump(vars(args) | {"device": str(device)}, f, indent=2)

    # --- Env ---
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

    NET_CONFIG = {"h_size": [128, 128]}  # builder tries {} first if needed

    INIT_HP = _ensure_maddpg_hp_defaults({
        "AGENT_IDS": agent_ids,
    })

    # --- Population ---
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

    # --- Replay Buffer (doc canonical field_names for 2.2.1) ---
    memory = MultiAgentReplayBuffer(
        memory_size=1_000_000,
        field_names=["state", "action", "reward", "next_state", "done"],
        agent_ids=agent_ids,
        device=device,
    )

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

    print(f"[MADDPG] Training start: max_steps={args.max_steps}, evo_steps={args.evo_steps}")
    try:
        trained_pop, pop_fitnesses = train_multi_agent_off_policy(
            env=env,
            env_name="ecogrid_marl_v2",
            algo="MADDPG",
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
            checkpoint_path=os.path.join(args.run_dir, "checkpoint_maddpg.pt"),
        )
    finally:
        env.close()

    with open(os.path.join(args.run_dir, "pop_fitnesses.json"), "w") as f:
        json.dump(pop_fitnesses, f, indent=2)

    print("[MADDPG] Training complete.")
    print("[MADDPG] Elite fitnesses:", pop_fitnesses)


if __name__ == "__main__":
    main()
