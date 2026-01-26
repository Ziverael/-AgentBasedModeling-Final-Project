from __future__ import annotations
import logging
import heapq
import random
from collections import defaultdict, Counter, deque
from enum import Enum, Flag, auto
from itertools import combinations, product
from typing import Any, Final, Optional, Union, List, Dict, Tuple

import numpy as np
import networkx as nx
from numpy.typing import NDArray
from pydantic import (
    BaseModel, 
    PositiveInt, 
    Field, 
    PositiveFloat, 
    ConfigDict,
    NonNegativeInt
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DAY_HOURS: Final[int] = 24

# --- BATCH RNG (Fastest Python Randomness) ---
class BatchRNG:
    __slots__ = ('_rng', '_logistic_buffer', '_logistic_idx', '_float_buffer', '_float_idx', '_buf_size')
    
    def __init__(self, seed: Optional[int] = None, buf_size: int = 100000):
        self._rng = np.random.default_rng(seed)
        self._buf_size = buf_size
        self._logistic_buffer = np.empty(0)
        self._logistic_idx = 0
        self._float_buffer = np.empty(0)
        self._float_idx = 0
        self._refill_logistic()
        self._refill_float()

    def _refill_logistic(self):
        self._logistic_buffer = self._rng.logistic(size=self._buf_size)
        self._logistic_idx = 0

    def _refill_float(self):
        self._float_buffer = self._rng.random(size=self._buf_size)
        self._float_idx = 0

    def get_logistic(self, loc: float) -> float:
        if self._logistic_idx >= self._buf_size:
            self._refill_logistic()
        val = self._logistic_buffer[self._logistic_idx]
        self._logistic_idx += 1
        return abs(val + loc)

    def random(self) -> float:
        if self._float_idx >= self._buf_size:
            self._refill_float()
        val = self._float_buffer[self._float_idx]
        self._float_idx += 1
        return val
    
    def choice(self, seq: List[Any]) -> Any:
        return random.choice(seq)
        
    def sample(self, population: List[Any], k: int) -> List[Any]:
        return random.sample(population, k)

# --- Enums ---
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

class EventType(Enum):
    CHANGE_PLACE = auto()
    CHANGE_STATE = auto()
    WORLD_EVENT = auto()
    LOCKDOWN = auto()
    GET_STATS = auto()

# --- Config ---
class TransitionTable(BaseModel):
    model_config = ConfigDict(frozen=True)
    SE: float = Field(default=0.01)
    I0I1: float = Field(default=0.4)
    I0I2: float = Field(default=0.6)
    I2R: float = Field(default=0.83)
    I2H: float = Field(default=0.17)
    HD: float = Field(default=0.049)
    HR: float = Field(default=0.951)

class StateAvgTimeTable(BaseModel):
    model_config = ConfigDict(frozen=True)
    E: PositiveFloat = 4.0 * DAY_HOURS
    I0: PositiveFloat = 2.0 * DAY_HOURS
    I1: PositiveFloat = 7.0 * DAY_HOURS
    I2: PositiveFloat = 7.0 * DAY_HOURS
    H: PositiveFloat = 10.0 * DAY_HOURS

class AgentTypeRatio(BaseModel):
    retired: float
    adult: float
    child: float

class WorldParams(BaseModel):
    N: PositiveInt
    schools: PositiveInt
    primary_care: PositiveInt
    hospitals: PositiveInt
    other_working_places: PositiveInt
    public_places: PositiveInt

class Params(BaseModel):
    T: PositiveInt
    N: PositiveInt
    alpha: PositiveInt
    exposed: Union[float, NonNegativeInt] = Field(default=0, description="Initial Exposed count or ratio")
    
    # NEW PARAMETERS
    initial_infected_normal: NonNegativeInt = Field(default=0, description="Count of NORMAL agents starting as I0")
    initial_infected_multispreader: NonNegativeInt = Field(default=0, description="Count of MULTISPREADER agents starting as I0")
    
    multispreader_share: float = 0.19
    ratios: AgentTypeRatio
    schools: PositiveInt = 1
    primary_care: PositiveInt = 1
    hospitals: PositiveInt = 1
    other_working_places: PositiveInt = 1
    public_places: PositiveInt = 1
    transitions: TransitionTable = Field(default_factory=TransitionTable)
    state_times: StateAvgTimeTable = Field(default_factory=StateAvgTimeTable)

# --- Core Logic ---

class AgentCurrentState:
    __slots__ = ('location_id', 'arrived_at', 'stay_until')
    def __init__(self, location_id: int, arrived_at: float, stay_until: float):
        self.location_id = location_id
        self.arrived_at = arrived_at
        self.stay_until = stay_until

MISSING_STATE = AgentCurrentState(-1, -1.0, -1.0)

class QueueEvent:
    __slots__ = ('time', 'type', 'agent', 'name')
    def __init__(self, time: float, type: EventType, agent: Optional["Agent"] = None, name: Optional[str] = None):
        self.time = time
        self.type = type
        self.agent = agent
        self.name = name
    def __lt__(self, other: "QueueEvent") -> bool:
        return self.time < other.time

class Location:
    locations: List["Location"] = []

    @staticmethod
    def reset() -> None:
        Location.locations = []

    @staticmethod
    def get_ids(type: Place) -> List[int]:
        return [i._id for i in Location.locations if i._type == type]
    
    @staticmethod
    def move_agent(agent: "Agent", new_location_id: int) -> None:
        Location.locations[agent.current.location_id].remove_agent(agent)
        Location.locations[new_location_id].add_agent(agent)

    def __init__(self, type: Place, size: Optional[int] = None) -> None:
        self._id = len(Location.locations)
        self.max_size = size
        self._type = type
        self._guests_list: deque[Tuple[Agent, float, float]] = deque()
        self._agent_count = 0
        Location.locations.append(self)
    
    @property
    def id(self) -> int: return self._id
    @property
    def type(self) -> Place: return self._type
    
    def add_agent(self, agent: "Agent"):
        if self.max_size is not None and self._agent_count >= self.max_size:
            raise RuntimeError(f"Reached slots limit in location {self._id}.")
        self._agent_count += 1
        self._guests_list.append((agent, agent.current.arrived_at, agent.current.stay_until))
    
    def remove_agent(self, agent: "Agent"):
        self._agent_count -= 1
    
    def get_contacts(self, from_: float, to_: float, target_state: Optional[AgentState] = None) -> List[Agent]:
        while self._guests_list and self._guests_list[0][2] < from_:
            self._guests_list.popleft()
            
        result = []
        for g_agent, g_from, g_to in self._guests_list:
            if (from_ <= g_from <= to_ or
                from_ <= g_to <= to_ or
                (g_from <= from_ and g_to >= to_)):
                
                if target_state is not None:
                    if g_agent.state == target_state:
                        result.append(g_agent)
                else:
                    result.append(g_agent)
        return result

class Agent:
    avg_time_table: Dict[TimeType, float] = {
        TimeType.FREE: 3.0,
        TimeType.HOME: 3.0,
        TimeType.WORK: 6.0,
    }
    transition_table: Optional[TransitionTable] = None
    state_time_table: Optional[StateAvgTimeTable] = None
    agents: List["Agent"] = []

    @staticmethod
    def reset() -> None:
        Agent.agents = []

    __slots__ = (
        '_id', 'state', 'type', 'is_multispreader',
        'current', 'house_id', 'work_id', 'hospital_id',
        '_travel_destinations', '_free_destinations', 'death_mark'
    )

    def __init__(self, state: AgentState, type: AgentType, is_multispreader: bool = False) -> None:
        self._id = len(Agent.agents)
        self.state = state
        self.current = MISSING_STATE
        self.house_id: Optional[int] = None
        self.work_id: Optional[int] = None
        self.hospital_id: Optional[int] = None
        self._travel_destinations: List[int] = []
        self._free_destinations: List[int] = [] 
        self.type = type
        self.is_multispreader = is_multispreader
        self.death_mark: Optional[PositiveInt] = None
        Agent.agents.append(self)

    def add_travel_destination(self, dest_id: Union[int, List[int], NDArray[np.int64]]) -> None:
        if isinstance(dest_id, int):
            self._travel_destinations.append(dest_id)
            if Location.locations[dest_id].type in Place.FREE_TIME:
                self._free_destinations.append(dest_id)
        elif isinstance(dest_id, (np.ndarray, list)):
             ids = list(dest_id)
             self._travel_destinations.extend(ids)
             for i in ids:
                 if Location.locations[i].type in Place.FREE_TIME:
                     self._free_destinations.append(i)

    def change_current_place_deterministically(self, location_id: int, duration: float) -> QueueEvent:
        arrived_time = self.current.stay_until
        left_time = arrived_time + duration
        self._change_current_state(AgentCurrentState(location_id, arrived_time, left_time))
        return QueueEvent(time=left_time, agent=self, type=EventType.CHANGE_PLACE)

    def change_current_place(self, rng: BatchRNG) -> Optional[QueueEvent]:
        if self.state in AgentState.RESTRICTED:
            return self._apply_state_rules_to_movement(rng)
        
        time_type = rng.choice([TimeType.FREE, TimeType.HOME, TimeType.WORK])
        
        if self.type == AgentType.RETIRED and time_type == TimeType.WORK:
            time_type = TimeType.HOME
            
        new_location_id = self.house_id
        if time_type == TimeType.WORK:
            new_location_id = self.work_id if self.work_id is not None else self.house_id
        elif time_type == TimeType.FREE:
            if self._free_destinations:
                new_location_id = rng.choice(self._free_destinations)
        
        duration = rng.get_logistic(loc=Agent.avg_time_table[time_type])            
        return self.change_current_place_deterministically(new_location_id, duration)

    def _apply_state_rules_to_movement(self, rng: BatchRNG) -> Optional[QueueEvent]:
        duration_home = rng.get_logistic(loc=Agent.avg_time_table[TimeType.HOME])
        if self.state == AgentState.I2:
            return self.change_current_place_deterministically(self.house_id, duration_home)
        elif self.state == AgentState.H:
            return self.change_current_place_deterministically(self.hospital_id, duration_home)
        return None

    def convert_to_dead(self, time: float) -> None:
        if self.state is not AgentState.D: return
        Location.locations[self.current.location_id].remove_agent(self)
        self.current = MISSING_STATE
        self.death_mark = time

    def _change_current_state(self, new_state: AgentCurrentState) -> None:
        Location.move_agent(self, new_state.location_id)
        self.current = new_state
    
    def resolve_contact_wih_spreader(self, time: float, rng: BatchRNG) -> Optional[QueueEvent]:
        if rng.random() <= Agent.transition_table.SE:
            return self.resolve_state_conversion(time, rng)
        return None

    def resolve_state_conversion(self, current_time: float, rng: BatchRNG) -> Optional[QueueEvent]:
        tt = Agent.transition_table
        st = Agent.state_time_table
        activation_time = None
        
        if self.state == AgentState.S:
            self.state = AgentState.E
            activation_time = rng.get_logistic(loc=st.E)
        elif self.state == AgentState.E:
            self.state = AgentState.I0
            activation_time = rng.get_logistic(loc=st.I0)
        elif self.state == AgentState.I0:
            if rng.random() < tt.I0I1:
                self.state = AgentState.I1
                activation_time = rng.get_logistic(loc=st.I1)
            else:
                self.state = AgentState.I2
                activation_time = rng.get_logistic(loc=st.I2)
        elif self.state == AgentState.I1:
            self.state = AgentState.R
        elif self.state == AgentState.I2:
            if rng.random() < tt.I2R:
                self.state = AgentState.R
            else:
                self.state = AgentState.H
                activation_time = rng.get_logistic(loc=st.H)
        elif self.state == AgentState.H:
            if rng.random() < tt.HR:
                self.state = AgentState.R
            else:
                self.state = AgentState.D
        
        if activation_time is not None:
            return QueueEvent(time=current_time + activation_time, agent=self, type=EventType.CHANGE_STATE)
        return None

# --- World and Simulation ---

class World:
    def __init__(self, params: WorldParams, seed: Optional[int] = None) -> None:
        self.params = params
        self.seed = seed
        self.households_capacities: Optional[NDArray[np.int64]] = None
        self.reset()
        self.households_ids: List[int] = []
        self.agents_households_assignments: Dict[int, int] = {}
        self._g = nx.Graph()

    def init(self) -> None:
        self.reset()
        self._add_places()
        network = self._create_friendship_network()
        self._add_befriend_households_links(network)
        self._enforce_avg_household_friend_degree(target_avg_degree=7)

    def reset(self) -> None:
        Location.reset()
        self.rng = np.random.default_rng(self.seed)
        self._g = nx.Graph()
        self._get_household_sizes()

    def _add_places(self):
        households = [Location(Place.HOUSEHOLD) for _ in range(len(self.households_capacities))]
        public = []
        for (type_, N) in [
            (Place.HOSPITAL, self.params.hospitals),
            (Place.SCHOOL, self.params.schools),
            (Place.PRIMARY_CARE, self.params.primary_care),
            (Place.OTHER_WORKING_PLACE, self.params.other_working_places),
            (Place.PUBLIC_PLACE, self.params.public_places),
        ]:
            public.extend([Location(type_) for _ in range(N)])
        
        self.households_ids = [h.id for h in households]
        public_ids = [p.id for p in public]
        self._g.add_nodes_from(self.households_ids + public_ids)
        self._g.add_edges_from(combinations(public_ids, 2))
        self._g.add_edges_from(product(public_ids, self.households_ids))

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
            if sizes[-1] <= 0: sizes = sizes[:-1]
        self.households_capacities = sizes

    def _create_friendship_network(self) -> dict:
        G_people = nx.relaxed_caveman_graph(
            l=len(self.households_capacities),
            k=int(self.households_capacities.max()),
            p=0.55,
            seed=self.seed,
        )
        total_generated = G_people.number_of_nodes()
        to_remove = max(total_generated - self.households_capacities.sum(), 0)
        if to_remove > 0:
            nodes_to_remove = self.rng.choice(list(G_people.nodes), size=int(to_remove), replace=False)
            G_people.remove_nodes_from(nodes_to_remove)
        
        mapping = {old: new for new, old in enumerate(G_people.nodes)}
        G_people = nx.relabel_nodes(G_people, mapping)
        
        person_to_household = {}
        current = 0
        sorted_nodes = sorted(list(G_people.nodes))
        for h_id, size in zip(self.households_ids, self.households_capacities):
            members = sorted_nodes[current : current + size]
            for m in members:
                person_to_household[m] = h_id
            current += size
        
        self.agents_households_assignments = person_to_household
        return {"relations": G_people, "house_assign": person_to_household}

    def _add_befriend_households_links(self, network) -> None:
        G_p = network["relations"]
        p2h = network["house_assign"]
        edges = set()
        for u, v in G_p.edges:
            hu, hv = p2h.get(u), p2h.get(v)
            if hu is not None and hv is not None and hu != hv:
                edges.add(tuple(sorted((hu, hv))))
        for u, v in edges:
            self._g.add_edge(u, v)

    def _enforce_avg_household_friend_degree(self, target_avg_degree: int = 7) -> None:
        households = self.households_ids
        H = len(households)
        current_edges = self._g.subgraph(households).number_of_edges()
        target_edges = int(H * target_avg_degree / 2)
        missing = max(0, target_edges - current_edges)
        
        if missing > 0:
            existing = set(self._g.edges)
            attempts = 0
            added = 0
            while added < missing and attempts < missing * 10:
                u, v = self.rng.choice(households, size=2, replace=False)
                pair = tuple(sorted((u, v)))
                if pair not in existing:
                    self._g.add_edge(u, v)
                    existing.add(pair)
                    added += 1
                attempts += 1

    def __call__(self) -> nx.Graph:
        return self._g

class AgentsActionsQueue:
    def __init__(self) -> None:
        self._heap = []
    def put(self, e: QueueEvent) -> None:
        heapq.heappush(self._heap, e)
    def pop(self) -> QueueEvent:
        return heapq.heappop(self._heap)
    def is_empty(self) -> bool:
        return not self._heap

class PublicEvent(BaseModel):
    name: str
    frequency: PositiveFloat = 7.0 * DAY_HOURS
    size: float = 0.05
    duration: float = 4.0
    composition: float = 0.0

class PublicEventDetails(PublicEvent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    reappearing_agents: List[Agent]
    location: Location

class EventEvaluator:
    def __init__(self, world: World, rng: BatchRNG, alpha: int) -> None:
        self._world = world
        self.rng = rng
        self._alpha = alpha
        self.public_events_details: Dict[str, PublicEventDetails] = {}
        self._statistics: Dict[str, List] = defaultdict(list)
        self.state_counts: Counter[AgentState] = Counter()

    def init_stats(self):
        self.state_counts = Counter(a.state for a in Agent.agents)
    
    def eval(self, event: QueueEvent) -> List[QueueEvent]:
        match event.type:
            case EventType.CHANGE_PLACE:
                return self._eval_change_place(event)
            case EventType.CHANGE_STATE:
                return self._eval_change_state(event)
            case EventType.GET_STATS:
                return self._eval_get_stats(event)
            case EventType.WORLD_EVENT: 
                return self._eval_world_event(event)
            case _:
                return []

    def _eval_change_place(self, event: QueueEvent) -> List[QueueEvent]:
        events = []
        if event.agent.state in AgentState.INFECTIOUS:
            events.extend(self._resolve_contacts(event.agent, event.time))
        elif event.agent.state == AgentState.S:
            events.extend(self._resolve_contacts_as_susceptible(event.agent, event.time))
        
        move_event = event.agent.change_current_place(self.rng)
        if move_event:
            events.append(move_event)
        return events

    def _resolve_contacts(self, agent: Agent, time: float) -> List[QueueEvent]:
        if agent.current.location_id == -1: return []
        
        loc = Location.locations[agent.current.location_id]
        
        susceptible = loc.get_contacts(agent.current.arrived_at, agent.current.stay_until, target_state=AgentState.S)
        if agent in susceptible: susceptible.remove(agent)
            
        interactions_no = int(agent.current.stay_until - agent.current.arrived_at)
        if agent.is_multispreader:
            interactions_no *= self._alpha
        
        events = []
        if susceptible:
            count = min(len(susceptible), interactions_no)
            if count > 0:
                contacted = self.rng.sample(susceptible, k=count)
                for target in contacted:
                    if target.state == AgentState.S:
                        old_state = target.state 
                        e = target.resolve_contact_wih_spreader(time, self.rng)
                        if e: 
                            if target.state != old_state:
                                self.state_counts[old_state] -= 1
                                self.state_counts[target.state] += 1
                            events.append(e)
        return events

    def _resolve_contacts_as_susceptible(self, agent: Agent, time: float) -> List[QueueEvent]:
        if agent.current.location_id == -1: return []
        loc = Location.locations[agent.current.location_id]
        
        guests = loc.get_contacts(agent.current.arrived_at, agent.current.stay_until)
        infectious = [a for a in guests if a != agent and a.state in AgentState.INFECTIOUS]
        
        events = []
        if infectious:
            interactions_no = int(agent.current.stay_until - agent.current.arrived_at)
            count = min(len(infectious), interactions_no)
            if count > 0:
                contacted = self.rng.sample(infectious, k=count)
                for _ in contacted:
                    old_state = agent.state
                    e = agent.resolve_contact_wih_spreader(time, self.rng)
                    if e:
                        if agent.state != old_state:
                            self.state_counts[old_state] -= 1
                            self.state_counts[agent.state] += 1
                        events.append(e)
                        break
        return events

    def _eval_change_state(self, event: QueueEvent) -> List[QueueEvent]:
        old_state = event.agent.state
        e = event.agent.resolve_state_conversion(event.time, self.rng)
        
        if event.agent.state != old_state:
            self.state_counts[old_state] -= 1
            self.state_counts[event.agent.state] += 1

        if event.agent.state is AgentState.D:
            event.agent.convert_to_dead(event.time)
        return [e] if e else []

    def _eval_world_event(self, event: QueueEvent) -> List[QueueEvent]:
        events = []
        if event.name not in self.public_events_details: return []
        
        pub_event = self.public_events_details[event.name]
        agent = event.agent
        
        agent.current.stay_until = event.time
        events.extend(self._resolve_contacts(agent, event.time))
        
        move_event = agent.change_current_place_deterministically(
            location_id=pub_event.location.id,
            duration=pub_event.duration,
        )
        events.append(move_event)
        
        next_time = event.time + pub_event.frequency
        next_agent = agent if agent in pub_event.reappearing_agents else self.rng.choice(Agent.agents)
        
        events.append(QueueEvent(
            name=pub_event.name, 
            time=next_time, 
            agent=next_agent, 
            type=EventType.WORLD_EVENT
        ))
        return events

    def _eval_get_stats(self, event: QueueEvent) -> List[QueueEvent]:
        self._statistics["Agent_fraction"].append({
            "time": event.time,
            "frac": self.state_counts.copy()
        })
        return []

class Simulation:
    def __init__(self, params: Params, public_events: Optional[List[PublicEvent]] = None, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.params = params
        
        Agent.transition_table = params.transitions
        Agent.state_time_table = params.state_times
        
        self.rng = BatchRNG(seed=seed)
        
        self.world_params = WorldParams(
            N=params.N,
            schools=params.schools,
            primary_care=params.primary_care,
            hospitals=params.hospitals,
            other_working_places=params.other_working_places,
            public_places=params.public_places,
        )
        
        self.world = World(self.world_params, seed)
        self.event_queue = AgentsActionsQueue()
        self.event_evaluator = EventEvaluator(self.world, self.rng, self.params.alpha)
        self._public_events_map = {e.name: e for e in public_events or []}
        self._public_events_details = {}

    def init(self):
        self.reset()
        self.world.init()
        self._init_agents()
        self.event_evaluator.init_stats()
        
    def reset(self):
        Agent.reset()
        self.world.reset()
        self.event_queue = AgentsActionsQueue()
        self.rng = BatchRNG(seed=self.seed)
        self.event_evaluator = EventEvaluator(self.world, self.rng, self.params.alpha)

    def _init_agents(self) -> None:
        agents = []
        for type_, ratio in [
            (AgentType.ADULT, self.params.ratios.adult),
            (AgentType.CHILD, self.params.ratios.child),
            (AgentType.RETIRED, self.params.ratios.retired),
        ]:
            count = int(self.params.N * ratio)
            agents.extend([Agent(state=AgentState.S, type=type_) for _ in range(count)])
        
        missing = self.params.N - len(agents)
        if missing > 0:
            agents.extend([Agent(state=AgentState.S, type=AgentType.ADULT) for _ in range(missing)])
            
        self._set_multipreaders()
        self._apply_initial_infections()
        self._assign_agents_to_households()
        self._assign_locations()
        self._init_agents_positions()

    def _set_multipreaders(self) -> None:
        ms_no = int(self.params.multispreader_share * self.params.N)
        if ms_no > 0:
            ms_ids = self.rng.sample(list(range(len(Agent.agents))), k=ms_no)
            for idx in ms_ids:
                Agent.agents[idx].is_multispreader = True

    def _apply_initial_infections(self):
        # 1. Identify pools
        all_indices = list(range(len(Agent.agents)))
        multi_indices = [i for i in all_indices if Agent.agents[i].is_multispreader]
        normal_indices = [i for i in all_indices if not Agent.agents[i].is_multispreader]
        
        # 2. Infect Multispreaders (Start at I0)
        k_multi = min(self.params.initial_infected_multispreader, len(multi_indices))
        chosen_multi = self.rng.sample(multi_indices, k=k_multi)
        for idx in chosen_multi:
            Agent.agents[idx].state = AgentState.I0
            
        # 3. Infect Normals (Start at I0)
        k_normal = min(self.params.initial_infected_normal, len(normal_indices))
        chosen_normal = self.rng.sample(normal_indices, k=k_normal)
        for idx in chosen_normal:
            Agent.agents[idx].state = AgentState.I0
            
        # 4. Exposed (Remaining S)
        infected_indices = set(chosen_multi) | set(chosen_normal)
        remaining_indices = [i for i in all_indices if i not in infected_indices]
        
        total_exposed = self.params.exposed
        if isinstance(total_exposed, float):
            total_exposed = int(len(Agent.agents) * total_exposed)
        
        total_exposed = min(total_exposed, len(remaining_indices))
        chosen_exposed = self.rng.sample(remaining_indices, k=total_exposed)
        for idx in chosen_exposed:
            Agent.agents[idx].state = AgentState.E

    def _assign_agents_to_households(self):
        for a_id, h_id in self.world.agents_households_assignments.items():
            Agent.agents[a_id].house_id = h_id

    def _assign_locations(self):
        hosp_ids = Location.get_ids(Place.HOSPITAL)
        if hosp_ids:
            assignments = [self.rng.choice(hosp_ids) for _ in Agent.agents]
            for a, h in zip(Agent.agents, assignments): a.hospital_id = h
        
        school_ids = Location.get_ids(Place.SCHOOL)
        if school_ids:
            kids = [a for a in Agent.agents if a.type == AgentType.CHILD]
            if kids:
                assigns = [self.rng.choice(school_ids) for _ in kids]
                for a, s in zip(kids, assigns): a.work_id = s
        
        work_ids = [l._id for l in Location.locations if l.type in Place.ADULT_WORKSPACE]
        if work_ids:
            adults = [a for a in Agent.agents if a.type == AgentType.ADULT]
            if adults:
                assigns = [self.rng.choice(work_ids) for _ in adults]
                for a, w in zip(adults, assigns): a.work_id = w
        
        pub_ids = Location.get_ids(Place.PUBLIC_PLACE)
        care_ids = Location.get_ids(Place.PRIMARY_CARE)
        
        city_graph = self.world()
        
        for agent in Agent.agents:
            agent.add_travel_destination(int(agent.house_id))
            if agent.work_id is not None:
                agent.add_travel_destination(int(agent.work_id))
            
            if care_ids:
                agent.add_travel_destination(self.rng.sample(care_ids, k=min(3, len(care_ids))))
            if pub_ids:
                agent.add_travel_destination(self.rng.sample(pub_ids, k=min(5, len(pub_ids))))
            
            try:
                neighbors = [n for n in city_graph.neighbors(agent.house_id) if Location.locations[n].type == Place.HOUSEHOLD]
                if neighbors:
                    agent.add_travel_destination(self.rng.sample(neighbors, k=min(4, len(neighbors))))
            except Exception:
                pass

    def _init_agents_positions(self) -> None:
        for agent in Agent.agents:
            agent.current = AgentCurrentState(agent.house_id, 0.0, 0.0)
            Location.locations[agent.house_id].add_agent(agent)

    def run(self, dt: float = 1.0) -> None:
        self._create_initial_events()
        self._init_public_events()
        self._set_stats_collecting_events(dt)
        
        t_current = 0.0
        while t_current < self.params.T and not self.event_queue.is_empty():
            t_current = self._evaluate_step()

    def _create_initial_events(self) -> None:
        for agent in Agent.agents:
            agent.change_current_place(self.rng)
            self.event_queue.put(QueueEvent(time=agent.current.stay_until, agent=agent, type=EventType.CHANGE_PLACE))
            
            # Initial state logic
            if agent.state == AgentState.E:
                duration = self.rng.get_logistic(loc=Agent.state_time_table.E)
                self.event_queue.put(QueueEvent(time=duration, agent=agent, type=EventType.CHANGE_STATE))
            elif agent.state == AgentState.I0:
                duration = self.rng.get_logistic(loc=Agent.state_time_table.I0)
                self.event_queue.put(QueueEvent(time=duration, agent=agent, type=EventType.CHANGE_STATE))

    def _init_public_events(self):
        if not self._public_events_map: return
        
        pub_locs = Location.get_ids(Place.PUBLIC_PLACE)
        if not pub_locs: return
        
        for name, p_event in self._public_events_map.items():
            loc = Location.locations[self.rng.choice(pub_locs)]
            k = int(p_event.size * self.params.N)
            agents = self.rng.sample(Agent.agents, k=k)
            k_re = int(p_event.composition * self.params.N)
            reappearing = self.rng.sample(agents, k=k_re) if k_re > 0 else []
            
            self.event_evaluator.public_events_details[name] = PublicEventDetails(
                **p_event.model_dump(),
                reappearing_agents=list(reappearing),
                location=loc
            )
            
            starts = self.rng._rng.uniform(0, 7*DAY_HOURS, size=len(agents))
            for a, t in zip(agents, starts):
                self.event_queue.put(QueueEvent(name=name, time=t, agent=a, type=EventType.WORLD_EVENT))

    def _set_stats_collecting_events(self, frequency: float) -> None:
        steps = np.arange(frequency, self.params.T + frequency, frequency)
        for t in steps:
            self.event_queue.put(QueueEvent(time=t, agent=None, type=EventType.GET_STATS))

    def _evaluate_step(self) -> float:
        event = self.event_queue.pop()
        new_events = self.event_evaluator.eval(event)
        for e in new_events:
            self.event_queue.put(e)
        return event.time

    def get_stats_df(self):
        data = self.event_evaluator._statistics["Agent_fraction"]
        result = defaultdict(list)
        for entry in data:
            result["time"].append(entry["time"])
            counts = entry["frac"]
            for state in AgentState:
                result[state.name].append(counts.get(state, 0))
        return result