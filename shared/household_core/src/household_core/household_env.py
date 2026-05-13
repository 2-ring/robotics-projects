"""Household grid-world environment with locked doors, keys, charging pads, and a finite battery."""

import os
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional during non-rendered runs
    plt = None

XY = tuple[int, int]
ObsType = dict[str, np.ndarray]
ActType = int

COLOR_ORDER = ("red", "blue", "green", "yellow", "orange", "purple", "pink", "gray")
COLOR_TO_INDEX = {color: index for index, color in enumerate(COLOR_ORDER)}
COLOR_TO_RGB = {
    "red": (214, 48, 49),
    "blue": (39, 114, 240),
    "green": (46, 204, 113),
    "yellow": (241, 196, 15),
    "orange": (230, 126, 34),
    "purple": (142, 68, 173),
    "pink": (232, 67, 147),
    "gray": (99, 110, 114),
}

FLOOR_RGB = (236, 240, 241)
WALL_RGB = (52, 73, 94)
AGENT_RGB = (243, 156, 18)
GOAL_RGB = (39, 174, 96)
CHARGER_RGB = (0, 188, 212)
LOCKED_DOOR_DARKEN = 0.55
OPEN_DOOR_LIGHTEN = 0.65

DOOR_OBS_DIM = 7
KEY_OBS_DIM = 8
CHARGER_OBS_DIM = 4


class PrimitiveAction(IntEnum):
    """Primitive actions in the low-level environment."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    INTERACT = 4
    WAIT = 5


class CellCode(IntEnum):
    """Integer cell codes used in the raw grid observation."""

    EMPTY = 0
    WALL = 1
    LOCKED_DOOR = 2
    OPEN_DOOR = 3
    CHARGER = 4
    GOAL = 5
    KEY = 6


@dataclass(frozen=True)
class HouseholdConfig:
    """Static configuration for generated household instances."""

    room_width: int = 5
    room_height: int = 6
    min_locked_doors: int = 1
    max_locked_doors: int = 4
    max_distractor_keys: int = 3
    battery_capacity_bonus: int = 0
    battery_safety_margin: int = 2
    max_episode_steps_scale: int = 40

    @property
    def max_rooms(self) -> int:
        """Return the maximum number of rooms in a generated household."""
        return self.max_locked_doors + 1

    @property
    def max_doors(self) -> int:
        """Return the maximum number of doors in a generated household."""
        return self.max_locked_doors

    @property
    def max_keys(self) -> int:
        """Return the maximum number of matching and distractor keys."""
        return self.max_locked_doors + self.max_distractor_keys

    @property
    def max_chargers(self) -> int:
        """Return the maximum number of chargers."""
        return self.max_rooms

    @property
    def grid_width(self) -> int:
        """Return the padded raw-observation grid width."""
        return (self.max_rooms * self.room_width) + 1

    @property
    def grid_height(self) -> int:
        """Return the padded raw-observation grid height."""
        return self.room_height + 1

    @property
    def max_battery_capacity(self) -> int:
        """Return the largest battery capacity this config can generate."""
        return (self.room_width * 2) + self.room_height + 6 + self.battery_capacity_bonus


@dataclass(frozen=True)
class StochasticityConfig:
    """Seeded stochasticity used by option execution in Q2."""

    navigation_failure_prob: float = 0.0
    interaction_failure_prob: float = 0.0


@dataclass(frozen=True)
class Room:
    """A rectangular room in the generated household."""

    room_id: str
    index: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    anchor: XY

    def contains(self, position: XY) -> bool:
        """Return whether a position lies inside this room."""
        x, y = position
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass(frozen=True)
class Charger:
    """A charger with fixed position and room membership."""

    charger_id: str
    index: int
    position: XY
    room_id: str


@dataclass
class Door:
    """A door whose open/locked state changes during an episode."""

    door_id: str
    index: int
    position: XY
    color: str
    room_a: str
    room_b: str
    matching_key_id: str
    is_open: bool = False


@dataclass
class Key:
    """A key that can move, be carried, and be consumed by a matching door."""

    key_id: str
    index: int
    position: XY | None
    color: str
    initial_room_id: str
    matching_door_id: str | None
    carried: bool = False
    consumed: bool = False


@dataclass
class EpisodeSpec:
    """Layout and mutable object state for one generated episode."""

    seed: int
    grid_width: int
    grid_height: int
    rooms: dict[str, Room]
    room_order: tuple[str, ...]
    doors: dict[str, Door]
    door_order: tuple[str, ...]
    keys: dict[str, Key]
    chargers: dict[str, Charger]
    goal_position: XY
    goal_room_id: str
    agent_start: XY
    start_room_id: str
    battery_capacity: int
    initial_battery: int
    max_steps: int


@dataclass
class ControllerObservation:
    """Controller-facing snapshot with raw tensors and symbolic helpers."""

    raw_observation: ObsType
    agent_pos: XY
    current_room_id: str
    battery_level: int
    battery_capacity: int
    carried_key_id: str | None
    goal_position: XY
    goal_room_id: str
    goal_reachable: bool
    terminated: bool
    success: bool
    door_order: tuple[str, ...]
    room_order: tuple[str, ...]
    doors: dict[str, Door]
    keys: dict[str, Key]
    chargers: dict[str, Charger]
    target_distances: dict[str, int | None]
    recovery_distances: dict[str, int | None]
    reachable_room_ids: tuple[str, ...]
    safety_margin: int = 2

    def at(self, target_id: str) -> bool:
        """Return whether the agent is already at the named target."""
        return self.target_distances.get(target_id) == 0

    def primitive_distance_to(self, target_id: str) -> int | None:
        """Return shortest primitive-step distance to a target, or `None`."""
        return self.target_distances.get(target_id)

    def is_reachable(self, target_id: str) -> bool:
        """Return whether the named target is currently reachable."""
        return self.primitive_distance_to(target_id) is not None

    def door_is_open(self, door_id: str) -> bool:
        """Return whether a specific door has already been unlocked."""
        return self.doors[door_id].is_open

    @property
    def locked_doors(self) -> tuple[str, ...]:
        """Return all currently locked doors in canonical order."""
        return tuple(door_id for door_id in self.door_order if not self.door_is_open(door_id))

    @property
    def reachable_locked_doors(self) -> tuple[str, ...]:
        """Return locked doors whose interaction cell is currently reachable."""
        return tuple(door_id for door_id in self.locked_doors if self.is_reachable(door_id))

    @property
    def reachable_keys(self) -> tuple[str, ...]:
        """Return currently reachable keys that are on the ground."""
        return tuple(
            key_id
            for key_id, key in self.keys.items()
            if not key.consumed and not key.carried and self.is_reachable(key_id)
        )

    def reachable_chargers(self) -> tuple[str, ...]:
        """Return currently reachable chargers."""
        return tuple(charger_id for charger_id in self.chargers if self.is_reachable(charger_id))

    def reachable_charger_distances(self) -> dict[str, int]:
        """Return distances to each currently reachable charger."""
        distances: dict[str, int] = {}
        for charger_id in self.chargers:
            distance = self.primitive_distance_to(charger_id)
            if distance is not None:
                distances[charger_id] = distance
        return distances

    @property
    def current_door_target(self) -> str | None:
        """Return the next unopened critical door, if any."""
        return next(
            (door_id for door_id in self.door_order if not self.door_is_open(door_id)),
            None,
        )

    @property
    def nearest_charger_id(self) -> str | None:
        """Return the nearest currently reachable charger."""
        distances = self.reachable_charger_distances()
        if not distances:
            return None
        return min(distances, key=lambda charger_id: (distances[charger_id], charger_id))

    @property
    def nearest_charger_distance(self) -> int | None:
        """Return the shortest distance to a reachable charger."""
        charger_id = self.nearest_charger_id
        return None if charger_id is None else self.target_distances[charger_id]

    @property
    def battery_low(self) -> bool:
        """Return whether the current battery is low relative to recovery distance."""
        charger_distance = self.nearest_charger_distance
        if charger_distance is None:
            return False
        required_buffer = (charger_distance * 3) + self.safety_margin + 1
        return self.battery_level <= required_buffer

    def can_unlock(self, door_id: str) -> bool:
        """Return whether the held key matches a given door."""
        return self.carried_key_id == self.doors[door_id].matching_key_id

    def can_safely_reach(self, target_id: str, *, extra_buffer: int = 0) -> bool:
        """Return whether the battery budget safely supports committing to a target."""
        distance = self.primitive_distance_to(target_id)
        if distance is None:
            return False
        if target_id == "goal" or target_id in self.chargers:
            required = distance + extra_buffer
        else:
            recovery = self.recovery_distances.get(target_id) or 0
            required = distance + recovery + self.safety_margin + extra_buffer
        return self.battery_level > required


class HouseholdEnv(gym.Env[ObsType, ActType]):
    """Fully observable household environment with option-friendly APIs."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(
        self,
        config: HouseholdConfig | None = None,
        *,
        stochasticity: StochasticityConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize a new seeded environment generator."""
        super().__init__()
        self.config = config or HouseholdConfig()
        self.stochasticity = stochasticity or StochasticityConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(PrimitiveAction))
        self.observation_space = self._build_observation_space()
        self.np_random = np.random.default_rng()
        self.spec_data: EpisodeSpec | None = None
        self.grid_width = self.config.grid_width
        self.grid_height = self.config.grid_height
        self.agent_pos = (1, 1)
        self.current_room_id = "room_0"
        self.battery_level = 0
        self.battery_capacity = self.config.max_battery_capacity
        self.carried_key_id: str | None = None
        self.terminated = False
        self.truncated = False
        self.success = False
        self.failure_reason: str | None = None
        self.step_count = 0
        self._last_pickup_previous_key_id: str | None = None
        self.figure = None
        self.axes = None
        self.legend_ax = None

    def _build_observation_space(self) -> spaces.Dict:
        """Construct the padded Gymnasium observation space."""
        max_width = self.config.grid_width
        max_height = self.config.grid_height
        max_scalar = max(max_width, max_height, len(COLOR_ORDER))
        return spaces.Dict(
            {
                "grid": spaces.Box(
                    low=-1,
                    high=int(CellCode.KEY),
                    shape=(max_height, max_width),
                    dtype=np.int16,
                ),
                "agent_pos": spaces.Box(
                    low=0,
                    high=max(max_width, max_height),
                    shape=(2,),
                    dtype=np.int16,
                ),
                "battery": spaces.Box(
                    low=0,
                    high=self.config.max_battery_capacity,
                    shape=(1,),
                    dtype=np.int16,
                ),
                "carried_key_index": spaces.Box(
                    low=-1,
                    high=self.config.max_keys - 1,
                    shape=(1,),
                    dtype=np.int16,
                ),
                "doors": spaces.Box(
                    low=-1,
                    high=max_scalar,
                    shape=(self.config.max_doors, DOOR_OBS_DIM),
                    dtype=np.int16,
                ),
                "keys": spaces.Box(
                    low=-1,
                    high=max_scalar,
                    shape=(self.config.max_keys, KEY_OBS_DIM),
                    dtype=np.int16,
                ),
                "chargers": spaces.Box(
                    low=-1,
                    high=max(max_width, max_height),
                    shape=(self.config.max_chargers, CHARGER_OBS_DIM),
                    dtype=np.int16,
                ),
                "goal_pos": spaces.Box(
                    low=0,
                    high=max(max_width, max_height),
                    shape=(2,),
                    dtype=np.int16,
                ),
            },
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and generate a new seeded household instance."""
        _ = options
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        self.spec_data = self._generate_episode(int(seed or 0))
        self.grid_width = self.spec_data.grid_width
        self.grid_height = self.spec_data.grid_height
        self.agent_pos = self.spec_data.agent_start
        self.current_room_id = self.spec_data.start_room_id
        self.battery_capacity = self.spec_data.battery_capacity
        self.battery_level = self.spec_data.initial_battery
        self.carried_key_id = None
        self.terminated = False
        self.truncated = False
        self.success = False
        self.failure_reason = None
        self.step_count = 0
        self._last_pickup_previous_key_id = None
        observation = self._raw_observation()
        info = {"controller_observation": self.controller_observation(), "seed": seed}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _generate_episode(self, seed: int) -> EpisodeSpec:
        """Generate one seeded room-chain household instance."""
        rng = np.random.default_rng(seed)
        num_doors = int(
            rng.integers(self.config.min_locked_doors, self.config.max_locked_doors + 1),
        )
        num_rooms = num_doors + 1
        grid_width = (num_rooms * self.config.room_width) + 1
        grid_height = self.config.room_height + 1

        rooms: dict[str, Room] = {}
        room_order: list[str] = []
        y_min = 1
        y_max = grid_height - 2
        y_anchor = (y_min + y_max) // 2
        for room_index in range(num_rooms):
            room_id = f"room_{room_index}"
            room_order.append(room_id)
            left_wall = room_index * self.config.room_width
            right_wall = left_wall + self.config.room_width
            x_min = left_wall + 1
            x_max = right_wall - 1
            x_anchor = (x_min + x_max) // 2
            rooms[room_id] = Room(
                room_id=room_id,
                index=room_index,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                anchor=(x_anchor, y_anchor),
            )

        occupied = {rooms["room_0"].anchor}
        doors: dict[str, Door] = {}
        keys: dict[str, Key] = {}
        door_order: list[str] = []

        for door_index in range(num_doors):
            color = COLOR_ORDER[door_index]
            door_id = f"door_{door_index}"
            key_id = f"key_{door_index}"
            room_a = room_order[door_index]
            room_b = room_order[door_index + 1]
            position = (
                (door_index + 1) * self.config.room_width,
                int(rng.integers(y_min + 1, y_max)),
            )
            door_order.append(door_id)
            doors[door_id] = Door(
                door_id=door_id,
                index=door_index,
                position=position,
                color=color,
                room_a=room_a,
                room_b=room_b,
                matching_key_id=key_id,
            )

            key_room_id = room_order[int(rng.integers(0, door_index + 1))]
            key_position = self._sample_room_cell(rng, rooms[key_room_id], occupied)
            occupied.add(key_position)
            keys[key_id] = Key(
                key_id=key_id,
                index=door_index,
                position=key_position,
                color=color,
                initial_room_id=key_room_id,
                matching_door_id=door_id,
            )

        distractor_count = int(rng.integers(0, self.config.max_distractor_keys + 1))
        for distractor_index in range(distractor_count):
            key_index = num_doors + distractor_index
            color = COLOR_ORDER[key_index]
            room_id = room_order[int(rng.integers(0, num_rooms))]
            key_position = self._sample_room_cell(rng, rooms[room_id], occupied)
            occupied.add(key_position)
            key_id = f"distractor_key_{distractor_index}"
            keys[key_id] = Key(
                key_id=key_id,
                index=key_index,
                position=key_position,
                color=color,
                initial_room_id=room_id,
                matching_door_id=None,
            )

        chargers: dict[str, Charger] = {}
        for charger_index, room_id in enumerate(room_order):
            position = self._sample_room_cell(rng, rooms[room_id], occupied)
            occupied.add(position)
            charger_id = f"charger_{charger_index}"
            chargers[charger_id] = Charger(
                charger_id=charger_id,
                index=charger_index,
                position=position,
                room_id=room_id,
            )

        goal_room_id = room_order[-1]
        goal_position = self._sample_room_cell(rng, rooms[goal_room_id], occupied)
        occupied.add(goal_position)
        agent_start = self._sample_room_cell(rng, rooms["room_0"], occupied)
        battery_capacity = self.config.max_battery_capacity
        initial_battery = max(self.config.room_width + 4, battery_capacity - 6)
        max_steps = num_rooms * self.config.room_width * self.config.max_episode_steps_scale

        return EpisodeSpec(
            seed=seed,
            grid_width=grid_width,
            grid_height=grid_height,
            rooms=rooms,
            room_order=tuple(room_order),
            doors=doors,
            door_order=tuple(door_order),
            keys=keys,
            chargers=chargers,
            goal_position=goal_position,
            goal_room_id=goal_room_id,
            agent_start=agent_start,
            start_room_id="room_0",
            battery_capacity=battery_capacity,
            initial_battery=initial_battery,
            max_steps=max_steps,
        )

    def _sample_room_cell(
        self,
        rng: np.random.Generator,
        room: Room,
        occupied: set[XY],
    ) -> XY:
        """Sample an unoccupied traversable cell from a room."""
        while True:
            position = (
                int(rng.integers(room.x_min, room.x_max + 1)),
                int(rng.integers(room.y_min, room.y_max + 1)),
            )
            if position not in occupied:
                return position

    @property
    def episode(self) -> EpisodeSpec:
        """Return the current episode state."""
        if self.spec_data is None:
            raise RuntimeError("reset() must be called before accessing the episode.")
        return self.spec_data

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one primitive action."""
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot step a terminated environment. Call reset() first.")

        primitive_action = PrimitiveAction(int(action))
        interaction_succeeded = False
        charged_this_step = False
        previous_position = self.agent_pos

        if primitive_action in {
            PrimitiveAction.UP,
            PrimitiveAction.DOWN,
            PrimitiveAction.LEFT,
            PrimitiveAction.RIGHT,
        }:
            interaction_succeeded = self._attempt_move(primitive_action, previous_position)
        elif primitive_action is PrimitiveAction.INTERACT:
            interaction_succeeded, charged_this_step = self._attempt_interaction()

        self.step_count += 1
        if charged_this_step:
            self.battery_level = self.battery_capacity
        else:
            self.battery_level = max(0, self.battery_level - 1)

        if self.agent_pos == self.episode.goal_position:
            self.terminated = True
            self.success = True
            self.failure_reason = None
        elif self.battery_level <= 0:
            self.terminated = True
            self.success = False
            self.failure_reason = "battery_depleted"

        if self.step_count >= self.episode.max_steps and not self.terminated:
            self.truncated = True
            self.success = False
            self.failure_reason = "max_steps"

        reward = 1.0 if self.success else -1.0 if self.terminated or self.truncated else -0.01
        observation = self._raw_observation()
        info = {
            "controller_observation": self.controller_observation(),
            "interaction_succeeded": interaction_succeeded,
            "failure_reason": self.failure_reason,
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, self.terminated, self.truncated, info

    def _attempt_move(self, action: PrimitiveAction, previous_position: XY) -> bool:
        """Attempt one deterministic grid movement."""
        direction = {
            PrimitiveAction.UP: (0, -1),
            PrimitiveAction.DOWN: (0, 1),
            PrimitiveAction.LEFT: (-1, 0),
            PrimitiveAction.RIGHT: (1, 0),
        }[action]
        candidate = (previous_position[0] + direction[0], previous_position[1] + direction[1])
        if not self._is_passable(candidate):
            return False
        self.agent_pos = candidate
        self.current_room_id = self._updated_room_id(previous_position, candidate)
        return True

    def _attempt_interaction(self) -> tuple[bool, bool]:
        """Attempt pickup, unlock, or charging at the current cell."""
        if self._maybe_unlock_door():
            return True, False
        if self._maybe_pickup_key():
            if self._interaction_failed():
                self._undo_last_pickup()
                return False, False
            return True, False
        if self._standing_on_charger():
            succeeded = not self._interaction_failed()
            return succeeded, succeeded
        return False, False

    def _maybe_pickup_key(self) -> bool:
        """Pick up a key at the current position if one is available."""
        key_id = self._key_at_position(self.agent_pos)
        if key_id is None:
            return False
        key = self.episode.keys[key_id]
        if key.consumed:
            return False
        previous_carried_key = self.carried_key_id
        if previous_carried_key is not None:
            previous_key = self.episode.keys[previous_carried_key]
            previous_key.carried = False
            previous_key.position = self.agent_pos
        key.carried = True
        key.position = None
        self.carried_key_id = key_id
        self._last_pickup_previous_key_id = previous_carried_key
        return True

    def _undo_last_pickup(self) -> None:
        """Roll back the most recent pickup after a stochastic interaction failure."""
        key_id = self.carried_key_id
        if key_id is None:
            return
        key = self.episode.keys[key_id]
        key.carried = False
        key.position = self.agent_pos
        previous_key_id = self._last_pickup_previous_key_id
        self.carried_key_id = previous_key_id
        if previous_key_id is not None:
            previous_key = self.episode.keys[previous_key_id]
            previous_key.carried = True
            previous_key.position = None
        self._last_pickup_previous_key_id = None

    def _maybe_unlock_door(self) -> bool:
        """Attempt to unlock one adjacent locked door with the carried key."""
        if self.carried_key_id is None:
            return False
        carried_key = self.episode.keys[self.carried_key_id]
        for door_id in self._adjacent_locked_doors(self.agent_pos):
            door = self.episode.doors[door_id]
            if door.matching_key_id != self.carried_key_id:
                continue
            if self._interaction_failed():
                return False
            door.is_open = True
            carried_key.consumed = True
            carried_key.carried = False
            carried_key.position = None
            self.carried_key_id = None
            return True
        return False

    def _interaction_failed(self) -> bool:
        """Sample whether the current interaction fails stochastically."""
        return bool(self.np_random.random() < self.stochasticity.interaction_failure_prob)

    def _standing_on_charger(self) -> bool:
        """Return whether the agent is currently standing on a charging pad."""
        return any(charger.position == self.agent_pos for charger in self.episode.chargers.values())

    def _key_at_position(self, position: XY) -> str | None:
        """Return the key id occupying a specific cell, if any."""
        for key_id, key in self.episode.keys.items():
            if key.position == position and not key.consumed:
                return key_id
        return None

    def _adjacent_locked_doors(self, position: XY) -> tuple[str, ...]:
        """Return all still-locked doors adjacent to a position."""
        adjacent = []
        for door_id, door in self.episode.doors.items():
            if door.is_open:
                continue
            distance = abs(position[0] - door.position[0]) + abs(position[1] - door.position[1])
            if distance == 1:
                adjacent.append(door_id)
        return tuple(adjacent)

    def _is_passable(self, position: XY) -> bool:
        """Return whether the agent may occupy a grid cell."""
        x, y = position
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return False
        if x == 0 or x == self.grid_width - 1 or y == 0 or y == self.grid_height - 1:
            return False
        for door in self.episode.doors.values():
            if door.position == position:
                return door.is_open
        return position[0] % self.config.room_width != 0

    def _updated_room_id(self, previous_position: XY, candidate: XY) -> str:
        """Infer the room containing the agent after a successful move."""
        for room in self.episode.rooms.values():
            if room.contains(candidate):
                return room.room_id
        for door in self.episode.doors.values():
            if door.position != candidate:
                continue
            if self.current_room_id == door.room_a or previous_position[0] < candidate[0]:
                return door.room_b
            return door.room_a
        return self.current_room_id

    def _grid_codes(self) -> np.ndarray:
        """Encode the current environment state as integer cell codes."""
        grid = np.full((self.grid_height, self.grid_width), CellCode.EMPTY, dtype=np.int16)
        grid[0, :] = CellCode.WALL
        grid[-1, :] = CellCode.WALL
        grid[:, 0] = CellCode.WALL
        grid[:, -1] = CellCode.WALL
        for wall_x in range(self.config.room_width, self.grid_width - 1, self.config.room_width):
            grid[:, wall_x] = CellCode.WALL
        for door in self.episode.doors.values():
            grid[door.position[1], door.position[0]] = (
                CellCode.OPEN_DOOR if door.is_open else CellCode.LOCKED_DOOR
            )
        for charger in self.episode.chargers.values():
            grid[charger.position[1], charger.position[0]] = CellCode.CHARGER
        for key in self.episode.keys.values():
            if key.position is not None and not key.consumed:
                grid[key.position[1], key.position[0]] = CellCode.KEY
        grid[self.episode.goal_position[1], self.episode.goal_position[0]] = CellCode.GOAL
        return grid

    def _raw_observation(self) -> ObsType:
        """Build the padded raw observation dictionary exposed through Gymnasium."""
        grid = np.full(
            (self.config.grid_height, self.config.grid_width),
            fill_value=-1,
            dtype=np.int16,
        )
        episode_grid = self._grid_codes()
        grid[: self.grid_height, : self.grid_width] = episode_grid

        doors = np.full((self.config.max_doors, DOOR_OBS_DIM), fill_value=-1, dtype=np.int16)
        for door in self.episode.doors.values():
            doors[door.index] = np.array(
                [
                    1,
                    door.position[0],
                    door.position[1],
                    COLOR_TO_INDEX[door.color],
                    int(door.is_open),
                    self.episode.rooms[door.room_a].index,
                    self.episode.rooms[door.room_b].index,
                ],
                dtype=np.int16,
            )

        keys = np.full((self.config.max_keys, KEY_OBS_DIM), fill_value=-1, dtype=np.int16)
        for key in self.episode.keys.values():
            x, y = (-1, -1) if key.position is None else key.position
            matching_door_index = (
                -1
                if key.matching_door_id is None
                else self.episode.doors[key.matching_door_id].index
            )
            keys[key.index] = np.array(
                [
                    1,
                    x,
                    y,
                    COLOR_TO_INDEX[key.color],
                    int(key.position is not None),
                    int(key.carried),
                    int(key.consumed),
                    matching_door_index,
                ],
                dtype=np.int16,
            )

        chargers = np.full(
            (self.config.max_chargers, CHARGER_OBS_DIM),
            fill_value=-1,
            dtype=np.int16,
        )
        for charger in self.episode.chargers.values():
            chargers[charger.index] = np.array(
                [
                    1,
                    charger.position[0],
                    charger.position[1],
                    self.episode.rooms[charger.room_id].index,
                ],
                dtype=np.int16,
            )

        carried_index = (
            -1 if self.carried_key_id is None else self.episode.keys[self.carried_key_id].index
        )
        return {
            "grid": grid,
            "agent_pos": np.array(self.agent_pos, dtype=np.int16),
            "battery": np.array([self.battery_level], dtype=np.int16),
            "carried_key_index": np.array([carried_index], dtype=np.int16),
            "doors": doors,
            "keys": keys,
            "chargers": chargers,
            "goal_pos": np.array(self.episode.goal_position, dtype=np.int16),
        }

    def controller_observation(self) -> ControllerObservation:
        """Build a fresh controller-facing snapshot."""
        raw = self._raw_observation()
        distances: dict[str, int | None] = {}
        recoveries: dict[str, int | None] = {}

        for room_id, room in self.episode.rooms.items():
            distances[room_id] = self._distance_to_room_anchor(room_id)
            recoveries[room_id] = self._distance_from_position_to_nearest_charger(room.anchor)

        for door_id in self.episode.door_order:
            distances[door_id] = self._distance_to_door(door_id)
            recoveries[door_id] = self._distance_from_door_to_nearest_charger(door_id)

        for key_id, key in self.episode.keys.items():
            distances[key_id] = self._distance_to_key(key_id)
            recoveries[key_id] = (
                None
                if key.position is None
                else self._distance_from_position_to_nearest_charger(key.position)
            )

        for charger_id, charger in self.episode.chargers.items():
            distances[charger_id] = self._shortest_path_distance(
                self.agent_pos,
                lambda position, target=charger.position: position == target,
            )
            recoveries[charger_id] = 0

        distances["goal"] = self._shortest_path_distance(
            self.agent_pos,
            lambda position: position == self.episode.goal_position,
        )
        recoveries["goal"] = self._distance_from_position_to_nearest_charger(
            self.episode.goal_position,
        )
        reachable_rooms = tuple(
            room_id for room_id in self.episode.room_order if self._room_is_reachable(room_id)
        )

        return ControllerObservation(
            raw_observation={name: value.copy() for name, value in raw.items()},
            agent_pos=self.agent_pos,
            current_room_id=self.current_room_id,
            battery_level=self.battery_level,
            battery_capacity=self.battery_capacity,
            carried_key_id=self.carried_key_id,
            goal_position=self.episode.goal_position,
            goal_room_id=self.episode.goal_room_id,
            goal_reachable=distances["goal"] is not None,
            terminated=self.terminated or self.truncated,
            success=self.success,
            door_order=self.episode.door_order,
            room_order=self.episode.room_order,
            doors=deepcopy(self.episode.doors),
            keys=deepcopy(self.episode.keys),
            chargers=deepcopy(self.episode.chargers),
            target_distances=distances,
            recovery_distances=recoveries,
            reachable_room_ids=reachable_rooms,
            safety_margin=self.config.battery_safety_margin,
        )

    def _room_is_reachable(self, room_id: str) -> bool:
        """Return whether the named room anchor is currently reachable."""
        return self._distance_to_room_anchor(room_id) is not None

    def _distance_to_key(self, key_id: str) -> int | None:
        """Return the current primitive distance to a key target."""
        key = self.episode.keys[key_id]
        if key.consumed:
            return None
        if key.carried:
            return 0
        if key.position is None:
            return None
        return self._shortest_path_distance(
            self.agent_pos,
            lambda position: position == key.position,
        )

    def _distance_to_door(self, door_id: str) -> int | None:
        """Return distance to an open door cell or locked-door interaction frontier."""
        door = self.episode.doors[door_id]
        if door.is_open:
            return self._shortest_path_distance(
                self.agent_pos,
                lambda position, target=door.position: position == target,
            )
        return self._shortest_path_distance(
            self.agent_pos,
            lambda position, target=door.position: (
                abs(position[0] - target[0]) + abs(position[1] - target[1]) == 1
            ),
        )

    def _distance_to_room_anchor(self, room_id: str) -> int | None:
        """Return the primitive distance to a room's anchor cell."""
        anchor = self.episode.rooms[room_id].anchor
        return self._shortest_path_distance(self.agent_pos, lambda position: position == anchor)

    def _distance_from_door_to_nearest_charger(self, door_id: str) -> int | None:
        """Estimate recovery distance from a door interaction site to a charger."""
        door = self.episode.doors[door_id]
        candidate_positions = []
        if door.is_open:
            candidate_positions.append(door.position)
        else:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                candidate = (door.position[0] + dx, door.position[1] + dy)
                if self._is_passable(candidate) or candidate == self.agent_pos:
                    candidate_positions.append(candidate)
        distances = [
            distance
            for candidate in candidate_positions
            if (distance := self._distance_from_position_to_nearest_charger(candidate)) is not None
        ]
        return min(distances) if distances else None

    def _distance_from_position_to_nearest_charger(self, start: XY) -> int | None:
        """Return the distance from a position to the nearest reachable charger."""
        distances = [
            self._shortest_path_distance(
                start,
                lambda position, target=charger.position: position == target,
            )
            for charger in self.episode.chargers.values()
        ]
        reachable_distances = [distance for distance in distances if distance is not None]
        return min(reachable_distances) if reachable_distances else None

    def _shortest_path_distance(
        self,
        start: XY,
        goal_test: Callable[[XY], bool],
    ) -> int | None:
        """Compute an unweighted shortest-path distance on the current grid."""
        queue: deque[tuple[XY, int]] = deque([(start, 0)])
        visited = {start}
        while queue:
            position, distance = queue.popleft()
            if goal_test(position):
                return distance
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                candidate = (position[0] + dx, position[1] + dy)
                if candidate in visited or not self._is_passable(candidate):
                    continue
                visited.add(candidate)
                queue.append((candidate, distance + 1))
        return None

    def next_action_toward(self, target_id: str) -> PrimitiveAction | None:
        """Choose the first primitive step on a shortest path to a named target."""
        if target_id in self.episode.rooms:
            goal_test = lambda position: position == self.episode.rooms[target_id].anchor
        elif target_id == "goal":
            goal_test = lambda position: position == self.episode.goal_position
        elif target_id in self.episode.chargers:
            goal = self.episode.chargers[target_id].position
            goal_test = lambda position, current=goal: position == current
        elif target_id in self.episode.keys:
            key = self.episode.keys[target_id]
            if key.position is None:
                return None
            goal_test = lambda position, current=key.position: position == current
        elif target_id in self.episode.doors:
            door = self.episode.doors[target_id]
            if door.is_open:
                goal_test = lambda position, current=door.position: position == current
            else:
                goal_test = lambda position, current=door.position: (
                    abs(position[0] - current[0]) + abs(position[1] - current[1]) == 1
                )
        else:
            raise KeyError(f"Unknown target id: {target_id}")

        frontier: deque[tuple[XY, list[PrimitiveAction]]] = deque([(self.agent_pos, [])])
        visited = {self.agent_pos}
        while frontier:
            position, path = frontier.popleft()
            if goal_test(position):
                return PrimitiveAction.WAIT if not path else path[0]
            for action, (dx, dy) in (
                (PrimitiveAction.RIGHT, (1, 0)),
                (PrimitiveAction.LEFT, (-1, 0)),
                (PrimitiveAction.DOWN, (0, 1)),
                (PrimitiveAction.UP, (0, -1)),
            ):
                candidate = (position[0] + dx, position[1] + dy)
                if candidate in visited or not self._is_passable(candidate):
                    continue
                visited.add(candidate)
                frontier.append((candidate, [*path, action]))
        return None

    def execute_option(self, option: Any) -> Any:
        """Run an option until it returns a non-running status."""
        status = option.step(self)
        while status == "RUNNING" and not (self.terminated or self.truncated):
            status = option.step(self)
        return status

    def render(self) -> np.ndarray | None:
        """Render the environment in the configured Gymnasium render mode."""
        frame = self._render_rgb_array()
        if self.render_mode == "rgb_array":
            return frame
        if self.render_mode != "human":
            return None
        if plt is None:
            raise RuntimeError("matplotlib is required for human rendering.")
        if self.figure is None or self.axes is None:
            self.figure, axes = plt.subplots(
                2,
                1,
                figsize=(12, 4.5),
                gridspec_kw={"height_ratios": [5, 1]},
            )
            self.axes, self.legend_ax = axes
        self.axes.clear()
        self.legend_ax.clear()
        self.axes.imshow(frame)
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self._draw_cell_labels()
        self._draw_legend()
        title = (
            f"battery={self.battery_level}/{self.battery_capacity}  "
            f"carrying={self.carried_key_id or 'none'}  "
            f"steps={self.step_count}"
        )
        self.axes.set_title(title)
        self.figure.tight_layout()
        plt.pause(1 / self.metadata["render_fps"])
        return None

    def _draw_cell_labels(self) -> None:
        """Overlay single-letter labels on entities so cells are identifiable."""
        scale = 24
        half = scale / 2
        agent_x, agent_y = self.agent_pos

        def annotate(pos: XY, label: str, color: str, *, size: int = 10) -> None:
            x, y = pos
            if (x, y) == (agent_x, agent_y):
                return
            self.axes.text(
                x * scale + half,
                y * scale + half,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=size,
                fontweight="bold",
            )

        for charger in self.episode.chargers.values():
            annotate(charger.position, "C", "black")
        for door in self.episode.doors.values():
            if not door.is_open:
                annotate(door.position, "L", "white")
        for key in self.episode.keys.values():
            if key.position is not None and not key.consumed:
                annotate(key.position, "K", "black")
        annotate(self.episode.goal_position, "G", "white")
        self.axes.text(
            agent_x * scale + half,
            agent_y * scale + half,
            "A",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )

    def _draw_legend(self) -> None:
        """Draw a color/letter legend below the grid."""
        from matplotlib.patches import Rectangle

        self.legend_ax.set_xlim(0, 14)
        self.legend_ax.set_ylim(0, 1)
        self.legend_ax.axis("off")

        def normalize(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
            return tuple(channel / 255.0 for channel in rgb)

        sample_door_color = COLOR_TO_RGB["red"]
        entries = [
            (normalize(AGENT_RGB), "A", "Agent"),
            (normalize(GOAL_RGB), "G", "Goal"),
            (normalize(CHARGER_RGB), "C", "Charger"),
            (normalize(COLOR_TO_RGB["red"]), "K", "Key"),
            (self._shade_normalized(sample_door_color, LOCKED_DOOR_DARKEN), "L", "Locked door"),
            (self._lighten_normalized(sample_door_color, OPEN_DOOR_LIGHTEN), " ", "Open door"),
        ]
        cursor = 0.2
        slot_width = 2.2
        for color, letter, label in entries:
            self.legend_ax.add_patch(
                Rectangle((cursor, 0.35), 0.5, 0.4, facecolor=color, edgecolor="black"),
            )
            text_color = "white" if letter in {"A", "G", "L"} else "black"
            self.legend_ax.text(
                cursor + 0.25,
                0.55,
                letter,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )
            self.legend_ax.text(cursor + 0.65, 0.55, label, ha="left", va="center", fontsize=9)
            cursor += slot_width

    @staticmethod
    def _shade_normalized(
        rgb: tuple[int, int, int],
        factor: float,
    ) -> tuple[float, float, float]:
        """Darken an RGB triple and normalize to matplotlib's 0-1 range."""
        return tuple((channel * factor) / 255.0 for channel in rgb)

    @staticmethod
    def _lighten_normalized(
        rgb: tuple[int, int, int],
        factor: float,
    ) -> tuple[float, float, float]:
        """Lighten an RGB triple and normalize to matplotlib's 0-1 range."""
        return tuple((channel + ((255 - channel) * factor)) / 255.0 for channel in rgb)

    def _render_rgb_array(self) -> np.ndarray:
        """Render the current grid state into an upscaled RGB array."""
        scale = 24
        frame = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        codes = self._grid_codes()
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell_rgb = FLOOR_RGB
                if codes[y, x] == CellCode.WALL:
                    cell_rgb = WALL_RGB
                elif codes[y, x] == CellCode.LOCKED_DOOR:
                    door = self._door_by_position((x, y))
                    cell_rgb = self._shade(COLOR_TO_RGB[door.color], LOCKED_DOOR_DARKEN)
                elif codes[y, x] == CellCode.OPEN_DOOR:
                    door = self._door_by_position((x, y))
                    cell_rgb = self._lighten(COLOR_TO_RGB[door.color], OPEN_DOOR_LIGHTEN)
                elif codes[y, x] == CellCode.CHARGER:
                    cell_rgb = CHARGER_RGB
                elif codes[y, x] == CellCode.GOAL:
                    cell_rgb = GOAL_RGB
                elif codes[y, x] == CellCode.KEY:
                    key_id = self._key_at_position((x, y))
                    if key_id is not None:
                        cell_rgb = COLOR_TO_RGB[self.episode.keys[key_id].color]
                frame[y, x, :] = np.array(cell_rgb, dtype=np.uint8)

        frame[self.agent_pos[1], self.agent_pos[0], :] = np.array(AGENT_RGB, dtype=np.uint8)
        return np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)

    def _door_by_position(self, position: XY) -> Door:
        """Return the door occupying a particular grid cell."""
        for door in self.episode.doors.values():
            if door.position == position:
                return door
        raise KeyError(f"No door at position {position}.")

    def close(self) -> None:
        """Release the render window if one exists."""
        if self.figure is not None and plt is not None:
            plt.close(self.figure)
        self.figure = None
        self.axes = None
        self.legend_ax = None

    @staticmethod
    def _shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        """Darken an RGB color by a multiplicative factor."""
        return tuple(int(channel * factor) for channel in rgb)

    @staticmethod
    def _lighten(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        """Lighten an RGB color by interpolating it toward white."""
        return tuple(int(channel + ((255 - channel) * factor)) for channel in rgb)
