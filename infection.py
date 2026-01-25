from __future__ import annotations

from pydantic import BaseModel, PositiveInt, model_validator, Field, PositiveFloat, ConfigDict, NonNegativeFloat, ValidationError
from enum import Enum, Flag, auto
from pyvis.network import Network
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import networkx as nx
from typing import Any, Final, Self
from collections import defaultdict
from itertools import combinations, product
import heapq
from collections import Counter
import logging
from matplotlib.lines import Line2D
from params import *



import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from dataclasses import dataclass
import random

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AgentState(Flag):
    S = auto()
    E = auto()
    I0 = auto()
    I1 = auto()
    I2 = auto()
    R = auto()
    H = auto()
    D = auto()

    INFECTIOUS = I0 | I1 | I2 | H
    IMMUNE = R | D | E
    #States with special rules during simulation
    RESTRICTED = I2 | H | D

class AgentType(Enum):
    CHILD = auto()
    ADULT = auto()
    RETIRED = auto()

class TimeType(Enum):
    FREE = auto()
    HOME = auto()
    WORK = auto()

class Place(Flag):
    MISSING = auto()
    HOUSEHOLD = auto()
    HOSPITAL = auto()
    SCHOOL = auto()
    PRIMARY_CARE = auto()
    OTHER_WORKING_PLACE = auto()
    PUBLIC_PLACE = auto()

    ADULT_WORKSPACE = HOSPITAL | SCHOOL | PRIMARY_CARE | OTHER_WORKING_PLACE | PUBLIC_PLACE
    FREE_TIME = HOUSEHOLD | PRIMARY_CARE | PUBLIC_PLACE

class AgentCurrentState(BaseModel):
    location_id: int
    arrived_at: float
    stay_until: float

    @model_validator(mode="after")
    def check_valid_visit_time(self):
        if self.arrived_at > self.stay_until:
            raise ValueError("Arrived is consequtive for exiting time.")
        return self
    
MISSING_STATE = AgentCurrentState(location_id=-1, arrived_at=-1.0, stay_until=-1.0)

class Agent:
    avg_time_table = {
        TimeType.FREE: 3.0,
        TimeType.HOME: 3.0,
        TimeType.WORK: 6.0,
    }
    transition_table = TRASITION_TABLE
    
    agents: list["Agent"] = []

    @staticmethod
    def reset() -> None:
        Agent.agents = []

    def __init__(self, state: AgentState, type: AgentType, is_multispreader: bool = False) -> None:
        self._id = len(Agent.agents)
        self._state = state
        self.current = MISSING_STATE
        self.house_id: int | None = None
        self.work_id: int | None = None
        self.hospital_id: int | None = None
        self._travel_destinations: list[int] = []
        self._type = type
        self._is_multispreader = is_multispreader
        self.death_mark: PositiveInt | None = None
        Agent.agents.append(self)
    
    @property
    def location(self) -> "Location":
        if self.current == MISSING_STATE:
            msg = f"Location is missing for agent {self._id}"
            raise RuntimeError(msg)
        return Location.location[self.current.location_id]
    
    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def type(self) -> AgentType:
        return self._type

    @property
    def house(self) -> "Location":
        if self.house_id is None:
            msg = f"House is missing for agent {self._id}"
            raise RuntimeError(msg)
        return Location.location[self.house_id]

    @property
    def is_multispreader(self) -> bool:
        return self._is_multispreader
    
    @is_multispreader.setter
    def is_multispreader(self, is_multispreader: bool) -> None:
        self._is_multispreader = is_multispreader

    def add_travel_destination(self, dest_id: int | list[int]| NDArray[np.int64]) -> None:
        if self.house_id is None or (self.work_id is None and self._type != AgentType.RETIRED):
            msg = f"Agent {self._id} is not completely initialized. Adding travel destinations is disabled."
            raise RuntimeError(msg)
        if isinstance(dest_id, int):
            self._travel_destinations.append(dest_id)
        elif isinstance(dest_id, np.ndarray):
            self._travel_destinations.extend(dest_id.tolist())
        elif isinstance(dest_id, list):
            self._travel_destinations.extend(dest_id)
        else:
            msg = f"Invalid dest_id type: {type(dest_id)}."
            raise TypeError(msg)
    
    def change_current_place_deterministically(self, location_id: int, duration: PositiveFloat) -> QueueEvent:
        arrived_time = self.current.stay_until
        left_time = arrived_time + duration
        self._change_current_state(
            AgentCurrentState(
                location_id=location_id,
                arrived_at=arrived_time,
                stay_until=left_time,
            )
        )
        return QueueEvent(
            time=left_time,
                agent=self,
                type=EventType.CHANGE_PLACE,
        )

    def change_current_place(self, rng: np.random.Generator) -> QueueEvent | None:
        if self._is_restricted_state():
            return self._apply_state_rules_to_movement(rng)
        time_type = rng.choice(TimeType)
        if self.type == AgentType.RETIRED and time_type == TimeType.WORK:
            time_type = TimeType.HOME
        match time_type:
            case TimeType.WORK:
                new_location_id = self.work_id
            case TimeType.HOME:
                new_location_id = self.house_id
            case TimeType.FREE:
                matching_locations = [id for id in self._travel_destinations if Location.locations[id].type in Place.FREE_TIME]
                new_location_id = rng.choice(matching_locations)
            case _:
                msg = f"TimeType {time_type} is unallowed in change place context."
                raise RuntimeError(msg)
        duration = np.abs(rng.logistic(loc=Agent.avg_time_table[time_type]))            
        return self.change_current_place_deterministically(new_location_id, duration)
    
    def _is_restricted_state(self) -> bool:
        return self.state in AgentState.RESTRICTED
    
    def _apply_state_rules_to_movement(self, rng: np.random.Generator) -> QueueEvent | None:
        match self.state:
            case AgentState.I2:
                event = self._stay_at_home(rng)
            case AgentState.H:
                event = self._stay_at_hospital(rng)
            case AgentState.D:
                event = None
                logger.warning(f"Evaluating move at {self.current.stay_until} for agent {self._id}")
            case _:
                msg = f"State {self.state} is not restricted."
                raise RuntimeError(msg)
        return event
    
    def _stay_at_home(self, rng: np.random.Generator) -> QueueEvent:
        duration = np.abs(rng.logistic(loc=Agent.avg_time_table[TimeType.HOME]))
        return self.change_current_place_deterministically(self.house_id, duration)
        
    def _stay_at_hospital(self, rng: np.random.Generator) -> QueueEvent:
        duration = np.abs(rng.logistic(loc=Agent.avg_time_table[TimeType.HOME]))
        return self.change_current_place_deterministically(self.hospital_id, duration)

    def convert_to_dead(self, time: PositiveFloat) -> None:
        if self.state is not AgentState.D:
            msg = f"Cannot convert to dead agent {self._id} with state {self.state}"
            raise RuntimeError(msg)
        if self.death_mark is not None:
            msg = f"Agent {self._id} is dead since {self.death_mark}. Cannot reset time to {time}."
            raise RuntimeError(msg)
        Location.locations[self.current.location_id].remove_agent(self)
        self.current = MISSING_STATE
        self.death_mark = time

    def _change_current_state(self, new_state: AgentCurrentState) -> None:
        """Actual current state is used to find locations which should be exited.
        Then the current state is overwritten.
        """
        Location.move_agent(self, new_state.location_id)
        self.current = new_state
    
    def resolve_contact_wih_spreader(self, time: PositiveFloat, rng: np.random.Generator) -> QueueEvent | None:
        if rng.random() <= Agent.transition_table.SE:
            return self.resolve_state_conversion(time, rng)

    def resolve_state_conversion(self, current_time: PositiveFloat, rng: np.random.Generator) -> QueueEvent | None:
        """The QueueEvent triggers a new event which is responsible for setting
        a new state and eventually, creating next event for next state conversion.
        """
        match self.state:
            case AgentState.S:
                self._state = AgentState.E
                activation_time = np.abs(rng.logistic(loc=STATE_AVG_TIMETABLE.E))
            case AgentState.E:
                self._state = AgentState.I0
                activation_time = np.abs(rng.logistic(loc=STATE_AVG_TIMETABLE.I0))
            case AgentState.I0:
                if rng.random() < Agent.transition_table.I0I1:
                    self._state = AgentState.I1
                    activation_time = np.abs(rng.logistic(loc=STATE_AVG_TIMETABLE.I1))
                else:
                    self._state = AgentState.I2
                    activation_time = np.abs(rng.logistic(loc=STATE_AVG_TIMETABLE.I2))
            case AgentState.I1:
                self._state = AgentState.R
                activation_time = None
            case AgentState.I2:
                if rng.random() < Agent.transition_table.I2R:
                    self._state = AgentState.R
                    activation_time = None
                else:
                    self._state = AgentState.H
                    activation_time = np.abs(rng.logistic(loc=STATE_AVG_TIMETABLE.H))
            case AgentState.H:
                if rng.random() < Agent.transition_table.HR:
                    self._state = AgentState.R
                    activation_time = None
                else:
                    self._state = AgentState.D
                    activation_time = None
            case _:
                msg = f"State {self._state} of agent {self._id} do not has conversion rule."
                raise RuntimeError(msg)
        if activation_time is not None:
            return QueueEvent(
                time=current_time + activation_time,
                agent=self,
                type=EventType.CHANGE_STATE,
            )


class EventType(Enum):
    CHANGE_PLACE = auto()
    CHANGE_STATE = auto()
    WORLD_EVENT = auto()
    LOCKDOWN = auto()

    GET_STATS = auto()

class QueueEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str | None = Field(None, description="Name of public event, not applicable for other events")
    time: PositiveFloat
    agent: Agent | None
    type: EventType

    def __lt__(self, other: "QueueEvent") -> bool:
        return self.time < other.time
    
    @model_validator(mode='after')
    def validate_after(self: Self) -> Self:
        if self.agent is None and self.type is not EventType.GET_STATS:
            msg = "Non-agent event only allowed for GET_STATS."
            raise ValueError(msg)
        return self

class AgentsActionsQueue:
    def __init__(self) -> None:
        self._heap = []
        heapq.heapify(self._heap)
    
    def put(self, e: QueueEvent) -> None:
        heapq.heappush(self._heap, e)
    
    def pop(self) -> QueueEvent:
        return heapq.heappop(self._heap)

    def peek(self) -> QueueEvent:
        return heapq.nsmallest(1, self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0
    

class GuestEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    agent: Agent
    from_: NonNegativeFloat
    to_: NonNegativeFloat
    @model_validator(mode="after")
    def check_time(self):
        if self.from_ >= self.to_ and self.from_ != 0.0:
            msg = f"Guest Entry has broken arrival/exit time. {self.from_} -> {self.to_}"
            raise ValueError(msg)
        return self

class Location:
    locations: list["Location"] = []

    @staticmethod
    def reset() -> None:
        Location.locations = []

    @staticmethod
    def get_ids(type: Place) -> list[int]:
        return [i._id for i in Location.locations if i._type == type]
    
    @staticmethod
    def move_agent(agent: Agent, new_location_id: int) -> None:
        Location.locations[agent.current.location_id].remove_agent(agent)
        Location.locations[new_location_id].add_agent(agent)

    def __init__(self, type: Place, size: int | None = None) -> None:
        self._id = len(Location.locations)
        self.max_size = size
        self._agents: list[Agent] = []
        self._type = type
        self._guests_list: list[GuestEntry] = []
        Location.locations.append(self)
    
    @property
    def id(self) -> int:
        return self._id
    
    @property
    def type(self) -> Place:
        return self._type
    
    def add_agent(self, agent: Agent):
        if len(self._agents) == self.max_size:
            msg = f"Reached slots limit in location {self._id}."
            raise RuntimeError(msg)
        self._agents.append(agent)
        entry = GuestEntry(
            agent=agent,
            from_=agent.current.arrived_at,
            to_=agent.current.stay_until,
        )
        self._guests_list.append(entry)
    
    def remove_agent(self, agent: Agent):
        try:
            self._agents.remove(agent)
        except:
            msg = f"Cannot remove non-existing agent from location {self._id}"
            raise RuntimeError(msg)
    
    @property
    def agents(self) -> list[Agent]:
        return self._agents

    def inspect_guests_list(self, from_: PositiveFloat, to_: PositiveFloat) -> list[Agent]:
        return list(set([guest_entry.agent for guest_entry in self._guests_list if from_ <= guest_entry.from_ <= to_]))

class NodeColor(Enum):
    HOUSEHOLD = "gray"
    SCHOOL = "brown"
    PRIMARY_CARE = "orange"
    HOSPITAL = "red"
    OTHER_WORKING_PLACE = "cyan"
    PUBLIC_PLACE = "green"


class WorldParams(BaseModel):
    N: PositiveInt
    schools: PositiveInt
    primary_care: PositiveInt
    hospitals: PositiveInt
    other_working_places: PositiveInt
    public_places: PositiveInt

class World:
    def __init__(self, params: WorldParams, seed: int | None = None) -> None:
        self.params = params
        self.seed = seed
        self.households_capacities: NDArray[np.int64] | None = None
        self.reset()

        self.households_ids: list[int] = []
        self.agents_households_assignments: dict[int, int] = {}
    
    def init(self) -> None:
        self.reset()
        self._add_places()
        network: dict[str, Any] = self._create_friendship_network()
        self._add_befriend_households_links(network)
        self._enforce_avg_household_friend_degree(target_avg_degree=7)

    
    def reset(self) -> None:
        Location.reset()
        self.rng = np.random.default_rng(self.seed)
        self._g = nx.Graph()
        self._get_household_sizes()
        
    def _add_places(self) -> nx.Graph:
        households = [Location(Place.HOUSEHOLD) for _ in range(len(self.households_capacities))]
        public = []
        for (type_, N) in [
            (Place.HOSPITAL, self.params.hospitals),
            (Place.SCHOOL, self.params.schools),
            (Place.PRIMARY_CARE, self.params.primary_care),
            (Place.OTHER_WORKING_PLACE, self.params.other_working_places),
            (Place.PUBLIC_PLACE, self.params.public_places),
        ]:
            locations = [Location(type_) for _ in range(N)]
            public.extend(locations)
        self.households_ids = [h.id for h in households]
        public_ids = [p.id for p in public]
        self._g.add_nodes_from(self.households_ids + public_ids) #TODO: use objects instead integers
        self._g.add_edges_from(combinations(public_ids, 2))
        self._g.add_edges_from(product(public_ids, self.households_ids))
    
    def __call__(self) -> nx.Graph:
        return self._g
    
    def _create_friendship_network(self) -> None:
        p_rewire = 0.55
        G_people = nx.relaxed_caveman_graph(
            l=len(self.households_capacities),
            k=self.households_capacities.max(),
            p=p_rewire,
            seed=self.seed,
        )
        total_generated_nodes = len(G_people.nodes.keys())
        to_remove = np.max(total_generated_nodes - self.households_capacities.sum(), 0)
        nodes_to_remove = self.rng.choice(np.arange(total_generated_nodes), size=to_remove, replace=False)
        G_people.remove_nodes_from(nodes_to_remove)
        G_people = nx.relabel_nodes(G_people, mapping={prev: new for prev, new in zip(G_people.nodes.keys(), range(self.params.N))})
        
        person_to_household = {}
        current = 0
        for household_id, size in zip(self.households_ids, self.households_capacities):
            members = list(G_people.nodes)[current : current + size]
            for m in members:
                person_to_household[m] = household_id
            current += size
        self.agents_households_assignments = person_to_household
        return {"relations": G_people, "house_assign": person_to_household}

    def _get_household_sizes(self) -> None:
        p = [0.1, 0.3, 0.5, 0.05, 0.03, 0.02]
        size_range = np.arange(1, 6 + 1)
        sizes = []
        total = 0
        while total < self.params.N:
            s = self.rng.choice(size_range, p=p)
            sizes.append(s)
            total += s
        sizes = np.array(sizes)
        excess = total - self.params.N
        if excess > 0:
            sizes[-1] -= excess
            if sizes[-1] == 0:
                sizes = sizes[:-1]
        self.households_capacities = sizes
        
    def _add_befriend_households_links(self, network_details) -> None:
        G_people = network_details["relations"]
        p2h = network_details["house_assign"]
        household_edges = set()
        for u, v in G_people.edges:
            hu, hv = p2h[u], p2h[v]
            if hu == hv:
                continue
            key = tuple(sorted((hu, hv)))
            household_edges.add(key)
        for hu, hv in household_edges:
            self._g.add_edge(hu, hv)
        
    def _enforce_avg_household_friend_degree(
        self,
        target_avg_degree: int = 7,
    ) -> None:
        households = self.households_ids
        H = len(households)
        current_edges = sum(
            1
            for u, v in self._g.edges
            if u in households and v in households
        )
        target_edges = int(H * target_avg_degree / 2)
        missing_edges = max(0, target_edges - current_edges)
        if missing_edges == 0:
            return

        possible_edges = [
            (u, v)
            for i, u in enumerate(households)
            for v in households[i + 1 :]
            if not self._g.has_edge(u, v)
        ]

        if missing_edges > len(possible_edges):
            raise RuntimeError(
                "Cannot reach target average household degree"
            )

        chosen = self.rng.choice(
            len(possible_edges),
            size=missing_edges,
            replace=False,
        )

        for idx in chosen:
            u, v = possible_edges[idx]
            self._g.add_edge(u, v)
    
    def _household_friend_degree(self, h: int) -> int:
        return sum(
            1
            for v in self._g.neighbors(h)
            if v in self.households_ids
        )

    def draw(self) -> None:
        pos = nx.spring_layout(self._g, seed=self.seed)
        nx.draw_networkx_edges(self._g, pos)
        options = {"edgecolors": "tab:gray", "node_size": 100, "alpha": 0.9}
        for place in Place:
            if place == Place.MISSING:
                continue
            nx.draw_networkx_nodes(self._g, pos, nodelist=Location.get_ids(place), node_color=f"tab:{NodeColor[place.name].value}", **options)
        
class PublicEvent(BaseModel):
    name: str
    frequency: PositiveFloat = Field(gt=0.0, le=7.0*DAY_HOURS, description="Frequency expressed as every X hours")
    size: PositiveFloat = Field(lt=1.0, description="Fraction of a population")
    duration: PositiveFloat = Field(ge=1.0, le=10.0)
    composition: NonNegativeFloat = Field(le=1.0, description="Fraction of reappearing agents")

STANDARD_EVENT = PublicEvent(
    name="STANDARD_EVENT",
    frequency=7.0*DAY_HOURS,
    size=0.05,
    duration=4.0,
    composition=0.0,
)

class PublicEventDetails(PublicEvent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    reappearing_agents: list[Agent]
    location: Location

@dataclass
class Snapshot:
    time: float
    agent_locations: dict[int, int]  # agent_id -> location_id
    agent_states: dict[int, AgentState]

class AgentTypeRatio(BaseModel):
    retired: float = Field(..., ge=0.0, le=1.0)
    adult: float = Field(..., ge=0.0, le=1.0)
    child: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_sum_is_one(self):
        if not abs(self.retired + self.adult + self.child - 1.0) < 1e-9:
            raise ValueError("Sum of retired, adult, and child must be 1.0")
        return self

class Params(BaseModel):
    T: PositiveInt
    N: PositiveInt
    alpha: PositiveInt = Field(..., description="Number of contacts per hour by multispreaders.")
    exposed: float | PositiveInt
    ratios: AgentTypeRatio
    schools: PositiveInt
    primary_care: PositiveInt
    hospitals: PositiveInt
    other_working_places: PositiveInt
    public_places: PositiveInt

class Simulation:
    def __init__(self, params: Params, public_events: list[PublicEvent] | None = None, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.params = params
        self.world_params = WorldParams(
            N=self.params.N,
            schools=self.params.schools,
            primary_care=self.params.primary_care,
            hospitals=self.params.hospitals,
            other_working_places=self.params.other_working_places,
            public_places=self.params.public_places,
        )
        
        self.world = World(self.world_params, seed)
        self.event_queue = AgentsActionsQueue()
        self.event_evaluator = EventEvaluator(self.world, self.rng, self.params.alpha)
        self._public_events_map: dict[str, PublicEvent] = {e.name: e for e in public_events}
        self._public_events_details: dict[str, PublicEventDetails] = dict()
        self._safely_select_houselods_fails_counter: int = 0

        self._collect_snapshot = False # Feature for vizualization
        self._snapshots: list[Snapshot] = []

    def init(self):
        self.reset()
        self.world.init()
        self._init_agents()
    
    def reset(self):
        Agent.reset()
        self.world.reset()
        self.event_queue = AgentsActionsQueue()
        self.rng = np.random.default_rng(self.seed)
        self.event_evaluator = EventEvaluator(self.world, self.rng, self.params.alpha)
        self._safely_select_houselods_fails_counter = 0

    def _init_agents(self) -> None:
        agents: list[Agent] = []
        for type_, ratio in [
            (AgentType.ADULT, self.params.ratios.adult),
            (AgentType.CHILD, self.params.ratios.child),
            (AgentType.RETIRED, self.params.ratios.retired),
        ]:
            total = int(self.params.N * ratio)
            agents.extend([Agent(state=AgentState.S, type=type_) for _ in range(total)])
        if (missing:=self.params.N - len(agents)) > 0:
            agents.extend([Agent(state=AgentState.S, type=AgentType.ADULT) for _ in range(missing)])
        self._init_exposed_agents()
        self._set_multipreaders()
        self._assign_agents_to_households()
        self._assign_agents_to_hospitals()
        self._assign_agents_to_workspace()
        self._assign_travel_destinations_for_agents()
        self._init_agents_positions()
    
    def _init_exposed_agents(self):
        total_exposed = (
            self.params.exposed if isinstance(self.params.exposed, int) else int(self.params.N * self.params.exposed)
        )
        exposed_ids = self.rng.choice(range(self.params.N), size=total_exposed, replace=False)
        for idx in exposed_ids:
            Agent.agents[idx]._state = AgentState.E
        
    def _set_multipreaders(self) -> None:
        ms_ratio = 0.19
        ms_no = int(ms_ratio * self.params.N)
        ms_ids = self.rng.choice(range(self.params.N), size=ms_no, replace=False)
        for idx in ms_ids:
            Agent.agents[idx].is_multispreader = True
    
    def _assign_agents_to_households(self) -> None:
        for a_id, h_id in self.world.agents_households_assignments.items():
            Agent.agents[a_id].house_id = h_id
    
    def _assign_agents_to_hospitals(self) -> None:
        hospitals_ids = [l._id for l in Location.locations if l.type == Place.HOSPITAL]
        agent_hospital_assignment = self.rng.choice(hospitals_ids, size=len(Agent.agents))
        for a, h_id in zip(Agent.agents, agent_hospital_assignment):
            a.hospital_id = h_id
    
    def _assign_agents_to_workspace(self) -> None:
        schools_ids = [l._id for l in Location.locations if l.type == Place.SCHOOL]
        kids = [a for a in Agent.agents if a.type == AgentType.CHILD]
        kid_school_assignment = self.rng.choice(schools_ids, size=len(kids))
        for kid, school_id in zip(kids, kid_school_assignment):
            kid.work_id = school_id

        workspaces_ids = [l._id for l in Location.locations if l.type in Place.ADULT_WORKSPACE]
        adults = [a for a in Agent.agents if a.type == AgentType.ADULT]
        adult_work_assignment = self.rng.choice(workspaces_ids, size=len(adults))
        for adult, work_id in zip(adults, adult_work_assignment):
            adult.work_id = work_id
    
    def _assign_travel_destinations_for_agents(self) -> None:
        city = self.world()
        public_places_no = 5
        primary_care_no = 3
        households_no = 4
        public_places_id: list[int] = [l._id for l in Location.locations if l.type == Place.PUBLIC_PLACE]
        primary_care_id: list[int] = [l._id for l in Location.locations if l.type == Place.PRIMARY_CARE]
        for agent in Agent.agents:
            agent.add_travel_destination(int(agent.house_id))
            if agent.type != AgentType.RETIRED:
                agent.add_travel_destination(int(agent.work_id))
            selected_primary_care_id = self.rng.choice(primary_care_id, size=primary_care_no, replace=False)
            agent.add_travel_destination(selected_primary_care_id)
            selected_public_places_id = self.rng.choice(public_places_id, size=public_places_no, replace=False)
            agent.add_travel_destination(selected_public_places_id)
            selected_households_id = self._safely_select_houselods(agent.house_id, households_no)
            agent.add_travel_destination(selected_households_id)
    
    def _safely_select_houselods(self, house_id: int, to_select: PositiveInt) -> list[int]:
        try:
            city = self.world()
            households_id = [i for i in city.neighbors(house_id) if Location.locations[i].type == Place.HOUSEHOLD]
            selected_households_id = self.rng.choice(households_id, size=to_select, replace=False)
        except ValueError:
            logger.warning(f"Cannot take {to_select} neighbours for household {house_id}.")
            logger.warning(f"Total households neighbours: {len(households_id)}.")
            logger.warning(f"Passing all those neighbours to destination list.")
            selected_households_id = households_id
            self._safely_select_houselods_fails_counter += 1
        finally:
            return selected_households_id

    def _init_agents_positions(self) -> None:
        for agent in Agent.agents:
            agent.current = AgentCurrentState(location_id=agent.house_id, arrived_at=0.0, stay_until=0.0)
            Location.locations[agent.house_id].add_agent(agent)
    
    def run(self, T: float, dt: PositiveFloat) -> None:
        self._create_initial_events()
        self._init_public_events()
        self._set_stats_collecting_events(dt, T)
        t_current = 0.0
        while t_current < T and not self.event_queue.is_empty():
            t_current = self._evaluate_step()
            if self._collect_snapshot:
                self._record_snapshot(t_current)
    
    def _create_initial_events(self) -> None:
        for agent in Agent.agents:
            agent.change_current_place(self.rng)
            event = QueueEvent(
                time=agent.current.stay_until,
                agent=agent,
                type=EventType.CHANGE_PLACE,
            )
            self.event_queue.put(event)
            if agent.state == AgentState.E:
                event = QueueEvent(
                    time=np.abs(self.rng.logistic(loc=STATE_AVG_TIMETABLE.E)),
                    agent=agent,
                    type=EventType.CHANGE_STATE
                )
                self.event_queue.put(event)

    def _init_public_events(self) -> None:
        first_event_date = self.rng.uniform(low=0.0, high=7.0*DAY_HOURS, size=len(self._public_events_map))
        for idx, (name, p_event) in enumerate(self._public_events_map.items()):
            agents_no = int(p_event.size * self.params.N)
            agents = self.rng.choice(Agent.agents, size=agents_no)
            reappearing_fraction = int(p_event.composition * self.params.N)
            reappearing_agents = self.rng.choice(agents, size=reappearing_fraction)
            public_locations = [l for l in Location.locations if l.type == Place.PUBLIC_PLACE]
            location = self.rng.choice(public_locations)
            self._public_events_details[name] = PublicEventDetails(
                **p_event.model_dump(),
                reappearing_agents=reappearing_agents,
                location=location,
            )
            events = [
                QueueEvent(
                    name=name,
                    time=first_event_date[idx],
                    agent=agent,
                    type=EventType.WORLD_EVENT,
                ) for agent in agents 
            ]
            for e in events:
                self.event_queue.put(e)
        self.event_evaluator.public_events_details = self._public_events_details
    
    def _set_stats_collecting_events(self, frequency: PositiveFloat, stop: PositiveFloat) -> None:
        steps_no = int(stop / frequency)
        steps = [i*frequency for i in range(1, steps_no)] + [stop]
        events = [
            QueueEvent(
                time=t,
                agent=None,
                type=EventType.GET_STATS
            ) for t in steps
        ]
        for e in events:
            self.event_queue.put(e)

    def _evaluate_step(self) -> PositiveFloat: 
        event = self.event_queue.pop()
        new_events = self.event_evaluator.eval(event)
        while len(new_events) > 0:
            self.event_queue.put(new_events.pop())
        return event.time

    def get_stats(self) -> Any:
        return self.event_evaluator._statistics

    def snapshot_collection_on(self) -> None:
        self._collect_snapshot = True

    def snapshot_collection_off(self) -> None:
        self._collect_snapshot = False

    def get_snapshots(self) -> list[Snapshot]:
        return self._snapshots
    
    def _record_snapshot(self, time: float) -> None:
        self._snapshots.append(
            Snapshot(
                time=time,
                agent_locations={
                    a._id: a.current.location_id
                    for a in Agent.agents
                    if a.current != MISSING_STATE
                },
                agent_states={a._id: a.state for a in Agent.agents},
            )
        )

class EventEvaluator:
    def __init__(self, world: World, rng: np.random.Generator, alpha: PositiveInt) -> None:
        self._world = world
        self.rng = rng
        self._alpha = alpha
        
        self._public_events_details : dict[int, PublicEventDetails] = dict()
        self._is_fully_initialized = False

        self._statistics: dict[str, Any] = defaultdict(list)
    
    def _raise_on_not_fully_initialized(self) -> None:
        if not self._is_fully_initialized:
            msg = "Event evaluator should have initialized public_events_details, before using."
            raise RuntimeError(msg)
    
    @property
    def public_events_details(self) -> dict[int, PublicEventDetails]:
        return self._public_events_details
    
    @public_events_details.setter
    def public_events_details(self, value: dict[int, PublicEventDetails]) -> None:
        self._public_events_details = value
        self._is_fully_initialized = True

    
    def eval(self, event: QueueEvent) -> list[QueueEvent]:
        self._raise_on_not_fully_initialized()
        match event.type:
            case EventType.CHANGE_PLACE:
                new_events = self._eval_change_place(event)
            case EventType.CHANGE_STATE:
                new_events = self._eval_change_state(event)
            case EventType.LOCKDOWN: ... #TODO: lockdown
            case EventType.GET_STATS:
                new_events = self._eval_get_stats(event)
            case EventType.WORLD_EVENT: 
                new_events = self._eval_world_event(event)
        return new_events
        
    
    def _eval_change_place(self, event: QueueEvent) -> list[QueueEvent]:
        events: list[QueueEvent] = []
        if event.agent._state in AgentState.INFECTIOUS:
            events.extend(self._resolve_contacts(event.agent, event.time))
        elif event.agent._state == AgentState.S:
            events.extend(self._resolve_contacts_as_susceptible(event.agent, event.time))
        move_event = event.agent.change_current_place(self.rng)
        if move_event is not None:
            events.append(move_event)
        return events
    
    def _resolve_contacts(self, agent: Agent, time: PositiveFloat) -> list[QueueEvent]:
        from_ = agent.current.arrived_at
        to_ = agent.current.stay_until
        location_id = agent.current.location_id
        agents_in_location = Location.locations[location_id].inspect_guests_list(from_, to_)
        interactions_no = int(to_ - from_)
        interactions_no = interactions_no if not agent.is_multispreader else interactions_no * self._alpha
        contanctable_agents_ids = [a._id for a in agents_in_location if (a._id != agent._id and a.state == AgentState.S)]
        events: list[QueueEvent] = []
        if len(contanctable_agents_ids) > 0:
            contacted_agents_ids = self.rng.choice(contanctable_agents_ids, size=interactions_no)
            for idx in contacted_agents_ids:
                # This check is important, because multipsreade may scuccesfully contanct with the same
                # Sucpetible agent and evaluate state conversion many times which is invalid behaviour.
                if Agent.agents[idx].state != AgentState.S:
                    continue
                e = Agent.agents[idx].resolve_contact_wih_spreader(time, self.rng)
                if e is not None:
                    events.append(e)
        return events
    
    def _resolve_contacts_as_susceptible(self, agent: Agent, time: PositiveFloat) -> list[QueueEvent]:
        from_ = agent.current.arrived_at
        to_ = agent.current.stay_until
        location_id = agent.current.location_id
        agents_in_location = Location.locations[location_id].inspect_guests_list(from_, to_)
        interactions_no = int(to_ - from_)
        contanctable_infestors_ids = [a._id for a in agents_in_location if (a._id != agent._id and a.state in AgentState.INFECTIOUS)]
        events: list[QueueEvent] = []
        if len(contanctable_infestors_ids) > 0:
            contacted_infestors_ids = self.rng.choice(contanctable_infestors_ids, size=interactions_no)
            for _ in contacted_infestors_ids:
                e = agent.resolve_contact_wih_spreader(time, self.rng)
                if e is not None:
                    events.append(e)
                    break
        return events
    
    

    def _eval_change_state(self, event: QueueEvent) -> list[QueueEvent]:
        e = event.agent.resolve_state_conversion(event.time, self.rng)
        if event.agent.state is AgentState.D:
            event.agent.convert_to_dead(event.time)
        events = [e] if e is not None else []
        return events

    def _eval_world_event(self, event: QueueEvent) -> list[QueueEvent]:
        """World event may colide with different type of event. In that case we
        stop currently running event, resolve contacts and go to the event location.
        Moreover, we check if the agent is reappearing agent to reschedule this event
        for the agent or to take another not reappearing agent on the event.
        """
        events: list[QueueEvent] = []
        public_event = self.public_events_details[event.name]
        agent = event.agent
        agent.current.stay_until = event.time
        events.extend(self._resolve_contacts(agent))
        move_event = agent.change_current_place_deterministically(
            location_id=public_event.location._id,
            duration=public_event.duration,
        )
        events.append(move_event)
        next_schedule = event.time + public_event.frequency
        if agent in public_event.reappearing_agents:
            agent_in_next_event = agent
        else:
            available_agents = [a for a in Agent.agents if a not in public_event.reappearing_agents]
            agent_in_next_event = self.rng.choice(available_agents)
        next_public_event = QueueEvent(
            name=public_event.name,
            agent=agent_in_next_event,
            time=next_schedule,
            type=EventType.WORLD_EVENT,
        )
        events.append(next_public_event)
        return events
        

    def _eval_get_stats(self, event: QueueEvent) -> list[QueueEvent]:
        type_counter = Counter([a.state for a in Agent.agents])
        self._statistics["Agent_fraction"].append({
            "time": event.time,
            "frac": type_counter
        })
        return []
