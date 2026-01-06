# envs/ecogrid_env_v2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import csv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv


# -----------------------------
# Config
# -----------------------------
@dataclass
class AgentSpec:
    kind: str  # "home" | "hospital" | "factory"
    base_demand_kw: float
    demand_amp_kw: float
    renewable_peak_kw: float
    battery_capacity_kwh: float
    battery_max_power_kw: float
    shed_max_frac: float
    reliability_weight: float     # penalty on unmet kWh
    comfort_weight: float         # penalty on shed kWh
    priority: float               # larger => protected in curtailment


DEFAULT_SPECS = {
    "home": AgentSpec(
        kind="home",
        base_demand_kw=2.0,
        demand_amp_kw=1.5,
        renewable_peak_kw=2.0,
        battery_capacity_kwh=6.0,
        battery_max_power_kw=2.0,
        shed_max_frac=0.40,
        reliability_weight=5.0,
        comfort_weight=0.5,
        priority=1.0,
    ),
    "hospital": AgentSpec(
        kind="hospital",
        base_demand_kw=8.0,
        demand_amp_kw=2.0,
        renewable_peak_kw=3.0,
        battery_capacity_kwh=20.0,
        battery_max_power_kw=6.0,
        shed_max_frac=0.00,       # critical: no shedding
        reliability_weight=50.0,  # huge penalty if unmet
        comfort_weight=2.0,
        priority=5.0,             # protected under curtailment
    ),
    "factory": AgentSpec(
        kind="factory",
        base_demand_kw=12.0,
        demand_amp_kw=6.0,
        renewable_peak_kw=4.0,
        battery_capacity_kwh=30.0,
        battery_max_power_kw=8.0,
        shed_max_frac=0.50,
        reliability_weight=10.0,
        comfort_weight=0.3,
        priority=1.5,
    ),
}


class EcoGridParallelEnvV2(ParallelEnv):
    """
    EcoGrid-MARL PettingZoo ParallelEnv with:
    - export-aware cost
    - overload curtailment -> unmet demand (hospital reliability matters)
    - proportional overload penalty (credit assignment)
    - per-agent behaviour logging in infos + episode_log
    """

    metadata = {"name": "ecogrid_marl_v2", "render_modes": ["human"], "is_parallelizable": True}

    def __init__(
        self,
        n_homes: int = 3,
        n_hospitals: int = 1,
        n_factories: int = 1,
        max_steps: int = 24,
        dt_hours: float = 1.0,
        grid_capacity_kw: float = 40.0,
        base_price: float = 6.0,
        price_slope: float = 5.0,               # price increases after threshold
        price_knee: float = 0.70,               # starts rising after 70% capacity
        overload_penalty_coeff: float = 5.0,    # penalty scale for overload ratio
        blackout_ratio: float = 1.50,           # terminate if >150% capacity
        export_credit_factor: float = 0.30,     # export credit weaker than import cost
        battery_cycle_cost_coeff: float = 0.01, # degradation proxy
        action_mode: str = "continuous",        # "continuous" or "discrete"
        include_action_mask: bool = False,
        debug: bool = False,
        seed: Optional[int] = None,
    ):
        assert action_mode in ("continuous", "discrete")
        self.n_homes = int(n_homes)
        self.n_hospitals = int(n_hospitals)
        self.n_factories = int(n_factories)

        self.max_steps = int(max_steps)
        self.dt = float(dt_hours)

        self.grid_capacity_kw = float(grid_capacity_kw)
        self.base_price = float(base_price)
        self.price_slope = float(price_slope)
        self.price_knee = float(price_knee)

        self.overload_penalty_coeff = float(overload_penalty_coeff)
        self.blackout_ratio = float(blackout_ratio)

        self.export_credit_factor = float(export_credit_factor)
        self.battery_cycle_cost_coeff = float(battery_cycle_cost_coeff)

        self.action_mode = action_mode
        self.include_action_mask = bool(include_action_mask)
        self.debug = bool(debug)

        self._np_random = np.random.default_rng(seed)

        # Agents
        agents: List[str] = []
        agents += [f"home_{i}" for i in range(self.n_homes)]
        agents += [f"hospital_{i}" for i in range(self.n_hospitals)]
        agents += [f"factory_{i}" for i in range(self.n_factories)]
        self.possible_agents = agents[:]
        self.agents = agents[:]

        # Specs
        self._specs: Dict[str, AgentSpec] = {}
        for a in self.possible_agents:
            prefix = a.split("_")[0]
            if prefix not in DEFAULT_SPECS:
                raise ValueError(f"Unknown agent type: {a}")
            self._specs[a] = DEFAULT_SPECS[prefix]

        # State
        self.t = 0
        self._soc: Dict[str, float] = {}
        self._last_price = self.base_price
        self._last_total_import_kw = 0.0

        self._demand_profile: Optional[np.ndarray] = None
        self._solar_profile: Optional[np.ndarray] = None

        # Behaviour logs
        self.episode_log: Dict[str, List[Dict[str, Any]]] = {}

        # Spaces
        self._build_spaces()

    # -----------------------------
    # Spaces
    # -----------------------------
    def _build_spaces(self) -> None:
        # Observation: [demand_kw, renewable_kw, soc, price, sin_t, cos_t]
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        # Reasonable bounds for NN stability (tuned)
        obs_high = np.array([80.0, 30.0, 1.0, 80.0, 1.0, 1.0], dtype=np.float32)
        base_obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._observation_spaces: Dict[str, gym.Space] = {}
        for a in self.possible_agents:
            if self.include_action_mask and self.action_mode == "discrete":
                self._observation_spaces[a] = spaces.Dict(
                    {
                        "obs": base_obs_space,
                        "action_mask": spaces.Box(low=0, high=1, shape=(self._discrete_n_actions(),), dtype=np.int8),
                    }
                )
            else:
                self._observation_spaces[a] = base_obs_space

        self._action_spaces: Dict[str, gym.Space] = {}
        for a in self.possible_agents:
            spec = self._specs[a]
            if self.action_mode == "continuous":
                # [battery_norm in [-1,1], shed_norm in [-1,1]]
                # shed_norm mapped to [0, shed_max_frac]
                low = np.array([-1.0, -1.0], dtype=np.float32)
                high = np.array([1.0, 1.0], dtype=np.float32)
                self._action_spaces[a] = spaces.Box(low=low, high=high, dtype=np.float32)
            else:
                self._action_spaces[a] = spaces.Discrete(self._discrete_n_actions())

    def _discrete_n_actions(self) -> int:
        return 15  # 3 battery x 5 shed

    def observation_space(self, agent: str) -> gym.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self._action_spaces[agent]

    # -----------------------------
    # Reset / Step
    # -----------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self.t = 0
        self._last_price = self.base_price
        self._last_total_import_kw = 0.0

        # Random SOC init
        self._soc = {a: float(self._np_random.uniform(0.3, 0.8)) for a in self.possible_agents}

        # Profiles for the episode length
        T = max(self.max_steps, 1)
        self._demand_profile = self._make_daily_demand_profile(T)
        self._solar_profile = self._make_daily_solar_profile(T)

        # Initialize log
        self.episode_log = {a: [] for a in self.possible_agents}

        obs = {a: self._observe(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        # Fill missing actions with safe no-op
        for a in self.agents:
            if a not in actions:
                actions[a] = self._default_action()

        t = self.t
        time_sin, time_cos = self._time_features(t, self.max_steps)

        # 1) Local signals for this step
        demand_kw: Dict[str, float] = {}
        renewable_kw: Dict[str, float] = {}
        for a in self.agents:
            spec = self._specs[a]
            demand_kw[a] = self._agent_demand_kw(spec, t)
            renewable_kw[a] = self._agent_renewable_kw(spec, t)

        # 2) Decode actions -> desired battery power (kW) and shed fraction
        # Convention: battery_kw > 0 means discharge to serve load/export
        #            battery_kw < 0 means charge (from grid/excess renewable)
        desired_batt_kw: Dict[str, float] = {}
        shed_frac: Dict[str, float] = {}
        raw_action_batt: Dict[str, float] = {}
        raw_action_shed: Dict[str, float] = {}

        for a in self.agents:
            spec = self._specs[a]
            batt_kw, sf, ab, ashed = self._decode_action(actions[a], spec)
            desired_batt_kw[a] = batt_kw
            shed_frac[a] = sf
            raw_action_batt[a] = ab
            raw_action_shed[a] = ashed

        # 3) Apply local constraints + compute preliminary grid import/export
        battery_eff = 0.95

        pre_import_kw: Dict[str, float] = {}   # >=0 (grid import)
        pre_export_kw: Dict[str, float] = {}   # >=0 (grid export)
        batt_kw_actual: Dict[str, float] = {}  # signed (+ discharge, - charge)
        shed_kw: Dict[str, float] = {}
        unmet_kw: Dict[str, float] = {}        # computed after curtailment
        batt_cycle_proxy: Dict[str, float] = {}

        # For unmet computation, we track each agent's "needed grid import" for serving load
        # separate from battery charging demand.
        grid_needed_for_load_kw: Dict[str, float] = {}
        grid_for_charge_kw: Dict[str, float] = {}

        for a in self.agents:
            spec = self._specs[a]

            # Shedding (kW)
            sf = float(np.clip(shed_frac[a], 0.0, spec.shed_max_frac))
            shed_frac[a] = sf
            shed_kw[a] = demand_kw[a] * sf

            demand_after_shed = demand_kw[a] * (1.0 - sf)

            # Use renewable first
            ren_used = min(renewable_kw[a], demand_after_shed)
            remaining_load = demand_after_shed - ren_used
            excess_ren = max(renewable_kw[a] - ren_used, 0.0)

            # Battery constraints by SOC and power limits
            soc = self._soc[a]
            cap = spec.battery_capacity_kwh
            max_p = spec.battery_max_power_kw
            energy_kwh = soc * cap
            room_kwh = (1.0 - soc) * cap

            batt_cmd = float(np.clip(desired_batt_kw[a], -max_p, max_p))

            # Split into discharge/charge
            discharge_kw = max(batt_cmd, 0.0)
            charge_kw = max(-batt_cmd, 0.0)

            # Limit discharge by available energy
            discharge_kw = min(discharge_kw, energy_kwh / self.dt, max_p)
            # Limit charge by room
            charge_kw = min(charge_kw, room_kwh / self.dt, max_p)

            # Charge from excess renewable first
            charge_from_ren = min(excess_ren, charge_kw)
            remaining_charge = charge_kw - charge_from_ren

            # Discharge to cover remaining load first
            discharge_to_load = min(discharge_kw, remaining_load)
            remaining_load -= discharge_to_load

            # Extra discharge becomes export
            discharge_to_export = discharge_kw - discharge_to_load

            # Preliminary grid needs:
            # - grid import for remaining load
            # - grid import for remaining battery charge (if any)
            grid_load_import = remaining_load
            grid_charge_import = remaining_charge

            grid_needed_for_load_kw[a] = float(grid_load_import)
            grid_for_charge_kw[a] = float(grid_charge_import)

            # Preliminary import/export totals (before curtailment)
            pre_import_kw[a] = float(grid_load_import + grid_charge_import)
            pre_export_kw[a] = float(discharge_to_export)  # export to grid

            # Actual battery power used
            batt_kw_actual[a] = float(discharge_kw - charge_kw)  # signed

            # Update SOC (kWh), include efficiency
            delta_kwh = (charge_from_ren * battery_eff + remaining_charge * battery_eff - discharge_kw / battery_eff) * self.dt
            new_energy = np.clip(energy_kwh + delta_kwh, 0.0, cap)
            self._soc[a] = float(new_energy / cap)

            batt_cycle_proxy[a] = float(discharge_kw + charge_kw)

            # initialize unmet (we'll compute after curtailment)
            unmet_kw[a] = 0.0

        # 4) Curtailment under overload -> unmet demand (proposal-consistent)
        # We curtail only the portion needed to serve LOAD (not charging), by priority.
        total_grid_load_need = float(sum(grid_needed_for_load_kw[a] for a in self.agents))
        total_grid_charge_need = float(sum(grid_for_charge_kw[a] for a in self.agents))
        total_pre_import = float(sum(pre_import_kw[a] for a in self.agents))

        load_ratio = total_pre_import / max(self.grid_capacity_kw, 1e-6)

        # If total_pre_import > capacity, first stop all grid charging if needed,
        # then curtail load-serving imports using priority weighting.
        remaining_capacity = self.grid_capacity_kw

        # Allow export always; capacity applies to import side
        # Step A: allocate capacity to load-serving imports first (priority-based)
        if total_pre_import <= self.grid_capacity_kw:
            # no curtailment
            served_load_import = {a: grid_needed_for_load_kw[a] for a in self.agents}
            served_charge_import = {a: grid_for_charge_kw[a] for a in self.agents}
            unmet_load = {a: 0.0 for a in self.agents}
        else:
            # Capacity insufficient: allocate to loads by priority
            # priority-weighted proportional allocation
            weights = np.array([self._specs[a].priority for a in self.agents], dtype=np.float64)
            weights = weights / max(weights.sum(), 1e-9)

            # First, serve load imports with capacity
            served_load_import = {}
            unmet_load = {}
            if total_grid_load_need <= remaining_capacity:
                # serve all load needs
                for a in self.agents:
                    served_load_import[a] = grid_needed_for_load_kw[a]
                    unmet_load[a] = 0.0
                remaining_capacity -= total_grid_load_need
            else:
                # allocate remaining capacity by priority share
                for i, a in enumerate(self.agents):
                    alloc = float(remaining_capacity * weights[i])
                    served = min(grid_needed_for_load_kw[a], alloc)
                    served_load_import[a] = served
                    unmet_load[a] = max(grid_needed_for_load_kw[a] - served, 0.0)
                remaining_capacity = 0.0

            # Step B: any remaining capacity can go to charging imports (usually 0 under overload)
            served_charge_import = {}
            if remaining_capacity <= 1e-9:
                for a in self.agents:
                    served_charge_import[a] = 0.0
            else:
                # serve charge needs proportionally (rare)
                total_charge = total_grid_charge_need
                if total_charge <= remaining_capacity:
                    for a in self.agents:
                        served_charge_import[a] = grid_for_charge_kw[a]
                    remaining_capacity -= total_charge
                else:
                    for i, a in enumerate(self.agents):
                        alloc = float(remaining_capacity * weights[i])
                        served_charge_import[a] = min(grid_for_charge_kw[a], alloc)
                    remaining_capacity = 0.0

        # Unmet demand is unmet load import (kW). Convert to kWh in reward via dt.
        for a in self.agents:
            unmet_kw[a] = float(unmet_load[a])

        # Actual import/export after curtailment:
        # - import = served_load_import + served_charge_import
        # - export unchanged
        imp_kw = {a: float(served_load_import[a] + served_charge_import[a]) for a in self.agents}
        exp_kw = {a: float(pre_export_kw[a]) for a in self.agents}

        total_actual_import = float(sum(imp_kw[a] for a in self.agents))
        self._last_total_import_kw = total_actual_import

        # 5) Price + overload penalty (based on actual import)
        actual_ratio = total_actual_import / max(self.grid_capacity_kw, 1e-6)

        # price rises after knee
        price = float(self.base_price * (1.0 + self.price_slope * max(actual_ratio - self.price_knee, 0.0)))
        self._last_price = price

        overload_amount = max(total_actual_import - self.grid_capacity_kw, 0.0)
        overload_frac = overload_amount / max(self.grid_capacity_kw, 1e-6)
        overload_penalty_total = float(self.overload_penalty_coeff * overload_frac)

        # Allocate overload penalty proportional to each agent's import (credit assignment)
        denom = total_actual_import + 1e-9

        # 6) Rewards + infos + log
        rewards: Dict[str, float] = {}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos: Dict[str, Dict[str, Any]] = {}

        for a in self.agents:
            spec = self._specs[a]

            import_kwh = imp_kw[a] * self.dt
            export_kwh = exp_kw[a] * self.dt

            energy_cost = price * import_kwh
            export_credit = self.export_credit_factor * price * export_kwh

            reliability_cost = spec.reliability_weight * (unmet_kw[a] * self.dt)
            comfort_cost = spec.comfort_weight * (shed_kw[a] * self.dt)
            battery_cost = self.battery_cycle_cost_coeff * (batt_cycle_proxy[a] * self.dt)

            overload_cost = overload_penalty_total * (imp_kw[a] / denom)

            # Reward is negative total cost, but export credit reduces cost
            total_cost = (energy_cost - export_credit) + reliability_cost + comfort_cost + battery_cost + overload_cost
            rewards[a] = float(-total_cost)

            infos[a] = {
                # global
                "t": int(self.t),
                "price": float(price),
                "grid_capacity_kw": float(self.grid_capacity_kw),
                "total_import_kw": float(total_actual_import),
                "load_ratio": float(actual_ratio),
                "overload_frac": float(overload_frac),
                # per-agent behaviour
                "agent_kind": spec.kind,
                "soc": float(self._soc[a]),
                "demand_kw": float(demand_kw[a]),
                "renewable_kw": float(renewable_kw[a]),
                "shed_frac": float(shed_frac[a]),
                "shed_kw": float(shed_kw[a]),
                "battery_kw": float(batt_kw_actual[a]),
                "import_kw": float(imp_kw[a]),
                "export_kw": float(exp_kw[a]),
                "unmet_kw": float(unmet_kw[a]),
                # raw actions (useful for debugging)
                "action_batt_raw": float(raw_action_batt[a]),
                "action_shed_raw": float(raw_action_shed[a]),
                # cost decomposition (for plots)
                "cost_energy": float(energy_cost),
                "credit_export": float(export_credit),
                "cost_reliability": float(reliability_cost),
                "cost_comfort": float(comfort_cost),
                "cost_battery": float(battery_cost),
                "cost_overload": float(overload_cost),
                "total_cost": float(total_cost),
            }

            # append to per-episode log
            self.episode_log[a].append(infos[a].copy())

        # 7) Termination / truncation
        blackout = (actual_ratio > self.blackout_ratio)
        if blackout:
            for a in self.agents:
                terminations[a] = True

        self.t += 1
        if self.t >= self.max_steps:
            for a in self.agents:
                truncations[a] = True

        # 8) Observations (clamped, no out-of-bounds)
        obs = {a: self._observe(a) for a in self.agents}

        if self.debug:
            print(f"[t={self.t}] price={price:.2f} import={total_actual_import:.2f} ratio={actual_ratio:.2f} blackout={blackout}")
            for a in self.agents:
                print(
                    f"  {a:12s} soc={self._soc[a]:.2f} imp={imp_kw[a]:.2f} exp={exp_kw[a]:.2f} "
                    f"shed={shed_kw[a]:.2f} unmet={unmet_kw[a]:.2f} batt={batt_kw_actual[a]:+.2f}"
                )

        # Clear agents when episode ends
        if blackout or (self.t >= self.max_steps):
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # -----------------------------
    # Observe
    # -----------------------------
    def _observe(self, agent: str):
        # Clamp t to avoid out-of-bounds
        t = min(self.t, self.max_steps - 1) if self.max_steps > 0 else 0
        spec = self._specs[agent]

        demand = self._agent_demand_kw(spec, t)
        solar = self._agent_renewable_kw(spec, t)
        soc = self._soc.get(agent, 0.5)
        sin_t, cos_t = self._time_features(t, self.max_steps)
        price = self._last_price

        vec = np.array([demand, solar, soc, price, sin_t, cos_t], dtype=np.float32)

        if self.include_action_mask and self.action_mode == "discrete":
            mask = self._action_mask(spec)
            return {"obs": vec, "action_mask": mask}
        return vec

    def _action_mask(self, spec: AgentSpec) -> np.ndarray:
        # For discrete mode: hospital cannot shed (only shed_idx=0)
        n = self._discrete_n_actions()
        mask = np.ones((n,), dtype=np.int8)
        if spec.shed_max_frac <= 1e-9:
            for idx in range(n):
                shed_idx = idx % 5
                if shed_idx != 0:
                    mask[idx] = 0
        return mask

    # -----------------------------
    # Action decode
    # -----------------------------
    def _default_action(self):
        if self.action_mode == "continuous":
            return np.array([0.0, 0.0], dtype=np.float32)
        return 0

    def _decode_action(self, act: Any, spec: AgentSpec) -> Tuple[float, float, float, float]:
        """
        Returns: (battery_kw, shed_frac, raw_batt, raw_shed)
        Continuous mode:
          act[0] in [-1,1] -> battery_kw in [-max_power, +max_power]
          act[1] in [-1,1] -> shed_frac in [0, shed_max_frac]
        """
        if self.action_mode == "continuous":
            a = np.asarray(act, dtype=np.float32).reshape(-1)
            if a.shape[0] != 2:
                raise ValueError(f"Continuous action must be shape (2,), got {a.shape}")

            raw_batt = float(np.clip(a[0], -1.0, 1.0))
            raw_shed = float(np.clip(a[1], -1.0, 1.0))

            batt_kw = raw_batt * spec.battery_max_power_kw
            shed_frac = ((raw_shed + 1.0) / 2.0) * spec.shed_max_frac
            shed_frac = float(np.clip(shed_frac, 0.0, spec.shed_max_frac))
            return float(batt_kw), float(shed_frac), raw_batt, raw_shed

        # Discrete:
        idx = int(act)
        idx = int(np.clip(idx, 0, self._discrete_n_actions() - 1))
        batt_idx = idx // 5
        shed_idx = idx % 5

        batt_levels = [-1.0, 0.0, 1.0]
        shed_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

        raw_batt = batt_levels[batt_idx]
        raw_shed = shed_levels[shed_idx] * 2.0 - 1.0  # roughly map back to [-1,1] for logging

        batt_kw = raw_batt * spec.battery_max_power_kw
        shed_frac = shed_levels[shed_idx] * spec.shed_max_frac
        return float(batt_kw), float(shed_frac), float(raw_batt), float(raw_shed)

    # -----------------------------
    # Profiles
    # -----------------------------
    def _make_daily_demand_profile(self, T: int) -> np.ndarray:
        x = np.linspace(0, 2 * np.pi, T, endpoint=False)
        base = 0.6 + 0.25 * np.sin(x - 0.5) + 0.15 * np.sin(2 * x + 1.0)
        noise = self._np_random.normal(0.0, 0.03, size=T)
        return np.clip(base + noise, 0.1, 1.3).astype(np.float32)

    def _make_daily_solar_profile(self, T: int) -> np.ndarray:
        x = np.linspace(0, 2 * np.pi, T, endpoint=False)
        raw = np.sin(x - np.pi / 2)
        prof = np.clip(raw, 0.0, None) ** 1.5
        noise = self._np_random.normal(0.0, 0.02, size=T)
        return np.clip(prof + noise, 0.0, 1.2).astype(np.float32)

    def _agent_demand_kw(self, spec: AgentSpec, t: int) -> float:
        assert self._demand_profile is not None
        mult = float(self._demand_profile[t])
        jitter = float(self._np_random.normal(0.0, 0.05))
        val = spec.base_demand_kw + spec.demand_amp_kw * mult * (1.0 + jitter)
        return float(max(val, 0.0))

    def _agent_renewable_kw(self, spec: AgentSpec, t: int) -> float:
        assert self._solar_profile is not None
        mult = float(self._solar_profile[t])
        jitter = float(self._np_random.normal(0.0, 0.05))
        val = spec.renewable_peak_kw * mult * (1.0 + jitter)
        return float(max(val, 0.0))

    @staticmethod
    def _time_features(t: int, T: int) -> Tuple[float, float]:
        if T <= 0:
            return 0.0, 1.0
        phase = 2.0 * np.pi * (t % T) / float(T)
        return float(np.sin(phase)), float(np.cos(phase))

    # -----------------------------
    # Logging helpers
    # -----------------------------
    def dump_episode_csv(self, path: str) -> None:
        """
        Dump self.episode_log to a CSV file for plotting/debugging.
        One row per (t, agent).
        """
        rows: List[Dict[str, Any]] = []
        for agent, entries in self.episode_log.items():
            for e in entries:
                row = {"agent": agent, **e}
                rows.append(row)

        if not rows:
            return

        # stable column order
        fieldnames = sorted(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    # -----------------------------
    # Render
    # -----------------------------
    def render(self):
        print(f"[t={self.t}] price={self._last_price:.2f} total_import_kw={self._last_total_import_kw:.2f}")
        for a in self.possible_agents:
            if a in self._soc:
                print(f"  - {a}: soc={self._soc[a]:.2f}")


def make_env(**kwargs) -> EcoGridParallelEnvV2:
    return EcoGridParallelEnvV2(**kwargs)
