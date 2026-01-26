from __future__ import annotations
import logging
import heapq
import random
from collections import defaultdict, Counter, deque
from enum import IntEnum
from typing import Any, Final, Optional, Union, List, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel, 
    PositiveInt, 
    Field, 
    PositiveFloat, 
    ConfigDict
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DAY_HOURS: Final[int] = 24

# --- OPTIMIZATION: Batch Random Generator ---
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

# --- Integer Enums ---
class AgentState(IntEnum):
    S = 0
    E = 1
    I0 = 2
    I1 = 3
    I2 = 4
    R = 5
    H = 6
    D = 7

class AgentType(IntEnum):
    CHILD = 0
    ADULT = 1
    RETIRED = 2

class TimeType(IntEnum):
    FREE = 0
    HOME = 1
    WORK = 2

class Place(IntEnum):
    MISSING = 0
    HOUSEHOLD = 1
    HOSPITAL = 2
    SCHOOL = 3
    PRIMARY_CARE = 4
    OTHER_WORKING_PLACE = 5
    PUBLIC_PLACE = 6

class EventType(IntEnum):
    CHANGE_PLACE = 0
    CHANGE_STATE = 1
    WORLD_EVENT = 2
    GET_STATS = 3

# --- Helper Sets for Logic (Defined globally) ---
INFECTIOUS_STATES = {AgentState.I0, AgentState.I1, AgentState.I2, AgentState.H}
RESTRICTED_STATES = {AgentState.I2, AgentState.H, AgentState.D}
# These correspond to the IntEnum values
FREE_TIME_PLACES = {Place.HOUSEHOLD, Place.PRIMARY_CARE, Place.PUBLIC_PLACE}
ADULT_WORK_PLACES = {Place.HOSPITAL, Place.SCHOOL, Place.PRIMARY_CARE, Place.OTHER_WORKING_PLACE, Place.PUBLIC_PLACE}

# --- Config Models ---
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
    exposed: Union[float, PositiveInt]
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

class Location:
    locations: List["Location"] = []
    
    @staticmethod
    def reset() -> None:
        Location.locations = []
    
    @staticmethod
    def get_ids(type: int) -> List[int]:
        return [loc.id for loc in Location.locations if loc.type == type]

    __slots__ = ('id', 'type', 'max_size', '_guests_list', '_agent_count')

    def __init__(self, type: int, size: Optional[int] = None):
        self.id = len(Location.locations)
        self.type = type
        self.max_size = size
        # Tuple: (agent_idx, from, to)
        self._guests_list: deque[Tuple[int, float, float]] = deque()
        self._agent_count = 0
        Location.locations.append(self)
    
    def add_agent(self, agent_idx: int, current_state: Tuple[int, float, float]):
        if self.max_size is not None and self._agent_count >= self.max_size:
            raise RuntimeError(f"Reached slots limit in location {self.id}.")
        self._agent_count += 1
        self._guests_list.append((agent_idx, current_state[1], current_state[2]))
    
    def remove_agent(self):
        self._agent_count -= 1
    
    def inspect_guests_list(self, from_: float, to_: float) -> List[int]:
        while self._guests_list and self._guests_list[0][2] < from_:
            self._guests_list.popleft()
            
        result = []
        for g_idx, g_from, g_to in self._guests_list:
            if (from_ <= g_from <= to_ or
                from_ <= g_to <= to_ or
                (g_from <= from_ and g_to >= to_)):
                result.append(g_idx)
        return result

class Agent:
    avg_time_table: Dict[int, float] = {
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
        'id', 'state', 'type', 'is_multispreader',
        'house_id', 'work_id', 'hospital_id',
        'current_loc', 'current_arr', 'current_dep',
        '_travel_destinations', '_free_destinations',
        'death_mark'
    )

    def __init__(self, state: int, type: int, is_multispreader: bool = False):
        self.id = len(Agent.agents)
        self.state = state
        self.type = type
        self.is_multispreader = is_multispreader
        
        self.house_id = -1
        self.work_id = -1
        self.hospital_id = -1
        
        self.current_loc = -1
        self.current_arr = -1.0
        self.current_dep = -1.0
        
        self._travel_destinations: List[int] = []
        self._free_destinations: List[int] = []
        self.death_mark = -1.0
        
        Agent.agents.append(self)

    def add_travel_destination(self, dest_id: Union[int, List[int], NDArray[np.int64]]):
        if isinstance(dest_id, int):
            self._travel_destinations.append(dest_id)
            # Correctly use global set lookup
            if Location.locations[dest_id].type in FREE_TIME_PLACES:
                self._free_destinations.append(dest_id)
        elif isinstance(dest_id, (np.ndarray, list)):
             ids = list(dest_id)
             self._travel_destinations.extend(ids)
             for i in ids:
                 if Location.locations[i].type in FREE_TIME_PLACES:
                     self._free_destinations.append(i)

    def move_to(self, location_id: int, duration: float) -> Tuple[float, int, int, int]:
        arrived = self.current_dep
        left = arrived + duration
        
        if self.current_loc != -1:
            Location.locations[self.current_loc].remove_agent()
        
        self.current_loc = location_id
        self.current_arr = arrived
        self.current_dep = left
        
        Location.locations[location_id].add_agent(self.id, (location_id, arrived, left))
        
        return (left, EventType.CHANGE_PLACE, self.id, 0)

    def change_current_place(self, rng: BatchRNG) -> Optional[Tuple]:
        if self.state in RESTRICTED_STATES:
            duration_home = rng.get_logistic(loc=Agent.avg_time_table[TimeType.HOME])
            if self.state == AgentState.I2:
                return self.move_to(self.house_id, duration_home)
            elif self.state == AgentState.H:
                target = self.hospital_id if self.hospital_id != -1 else self.house_id
                return self.move_to(target, duration_home)
            return None 

        time_type = rng.choice([TimeType.FREE, TimeType.HOME, TimeType.WORK])
        
        if self.type == AgentType.RETIRED and time_type == TimeType.WORK:
            time_type = TimeType.HOME
            
        new_loc = self.house_id
        if time_type == TimeType.WORK:
            new_loc = self.work_id if self.work_id != -1 else self.house_id
        elif time_type == TimeType.FREE:
            if self._free_destinations:
                new_loc = rng.choice(self._free_destinations)
        
        duration = rng.get_logistic(loc=Agent.avg_time_table[time_type])            
        return self.move_to(new_loc, duration)

    def resolve_contact(self, time: float, rng: BatchRNG) -> Optional[Tuple]:
        if rng.random() <= Agent.transition_table.SE:
            return self.resolve_state_conversion(time, rng)
        return None

    def resolve_state_conversion(self, current_time: float, rng: BatchRNG) -> Optional[Tuple]:
        tt = Agent.transition_table
        st = Agent.state_time_table
        delay = None
        
        if self.state == AgentState.S:
            self.state = AgentState.E
            delay = rng.get_logistic(loc=st.E)
        elif self.state == AgentState.E:
            self.state = AgentState.I0
            delay = rng.get_logistic(loc=st.I0)
        elif self.state == AgentState.I0:
            if rng.random() < tt.I0I1:
                self.state = AgentState.I1
                delay = rng.get_logistic(loc=st.I1)
            else:
                self.state = AgentState.I2
                delay = rng.get_logistic(loc=st.I2)
        elif self.state == AgentState.I1:
            self.state = AgentState.R
        elif self.state == AgentState.I2:
            if rng.random() < tt.I2R:
                self.state = AgentState.R
            else:
                self.state = AgentState.H
                delay = rng.get_logistic(loc=st.H)
        elif self.state == AgentState.H:
            if rng.random() < tt.HR:
                self.state = AgentState.R
            else:
                self.state = AgentState.D
        
        if delay is not None:
            return (current_time + delay, EventType.CHANGE_STATE, self.id, 0)
        return None

# --- Simulation Core ---

class PublicEvent(BaseModel):
    name: str
    frequency: PositiveFloat = 7.0 * DAY_HOURS
    size: float = 0.05
    duration: float = 4.0
    composition: float = 0.0

class PublicEventDetails(PublicEvent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    reappearing_agents: List[int]
    location_id: int

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
        
        self.event_queue: List[Tuple[float, int, int, int]] = []
        self._public_events_map = {e.name: e for e in public_events or []}
        self._state_counts = Counter()
        self._stats = defaultdict(list)

    def init(self):
        self.reset()
        self._init_world()
        self._init_agents()
        self._state_counts = Counter(a.state for a in Agent.agents)

    def reset(self):
        Agent.reset()
        Location.reset()
        self.event_queue = []
        self.rng = BatchRNG(seed=self.seed)
        self._stats = defaultdict(list)

    def _init_world(self):
        # 1. Household Sizes
        p = [0.1, 0.3, 0.5, 0.05, 0.03, 0.02]
        sizes = []
        total = 0
        while total < self.params.N:
            s = self.rng.choice([1,2,3,4,5,6]) 
            sizes.append(s)
            total += s
        if total > self.params.N:
            sizes[-1] -= (total - self.params.N)
            if sizes[-1] <= 0: sizes.pop()
        
        self.households_ids = [Location(Place.HOUSEHOLD).id for _ in sizes]
        self.household_sizes = sizes
        
        # 2. Places
        for t, n in [
            (Place.HOSPITAL, self.params.hospitals),
            (Place.SCHOOL, self.params.schools),
            (Place.PRIMARY_CARE, self.params.primary_care),
            (Place.OTHER_WORKING_PLACE, self.params.other_working_places),
            (Place.PUBLIC_PLACE, self.params.public_places),
        ]:
            for _ in range(n): Location(t)
        
        self.agent_house_map = {}
        current_agent = 0
        for hid, size in zip(self.households_ids, self.household_sizes):
            for _ in range(size):
                self.agent_house_map[current_agent] = hid
                current_agent += 1

    def _init_agents(self):
        # Create Agents
        for type_, ratio in [
            (AgentType.ADULT, self.params.ratios.adult),
            (AgentType.CHILD, self.params.ratios.child),
            (AgentType.RETIRED, self.params.ratios.retired),
        ]:
            c = int(self.params.N * ratio)
            for _ in range(c): Agent(AgentState.S, type_)
        
        while len(Agent.agents) < self.params.N:
            Agent(AgentState.S, AgentType.ADULT)
            
        # Exposed
        exposed_n = self.params.exposed if isinstance(self.params.exposed, int) else int(self.params.N * self.params.exposed)
        # Use simple choice for startup (random.sample is fine)
        exposed_ids = random.sample(range(len(Agent.agents)), k=exposed_n)
        for idx in exposed_ids:
            Agent.agents[idx].state = AgentState.E
            
        # Assignments
        # FIXED: Use global sets for lookup, not Enum attributes
        hosp_ids = Location.get_ids(Place.HOSPITAL)
        school_ids = Location.get_ids(Place.SCHOOL)
        
        work_ids = []
        for p in ADULT_WORK_PLACES:
            work_ids.extend(Location.get_ids(p))
            
        free_ids = []
        for p in FREE_TIME_PLACES:
            free_ids.extend(Location.get_ids(p))
        
        for a in Agent.agents:
            a.house_id = self.agent_house_map.get(a.id, self.households_ids[0])
            if hosp_ids: a.hospital_id = self.rng.choice(hosp_ids)
            
            if a.type == AgentType.CHILD and school_ids: 
                a.work_id = self.rng.choice(school_ids)
            elif a.type == AgentType.ADULT and work_ids: 
                a.work_id = self.rng.choice(work_ids)
            
            # Destinations
            a.add_travel_destination(a.house_id)
            if a.work_id != -1: a.add_travel_destination(a.work_id)
            if free_ids: a.add_travel_destination(self.rng.sample(free_ids, k=min(5, len(free_ids))))

            # Init Position
            a.current_loc = a.house_id
            Location.locations[a.house_id].add_agent(a.id, (a.house_id, 0.0, 0.0))
            
            # Init Events
            heapq.heappush(self.event_queue, a.change_current_place(self.rng))
            if a.state == AgentState.E:
                t = self.rng.get_logistic(Agent.state_time_table.E)
                heapq.heappush(self.event_queue, (t, EventType.CHANGE_STATE, a.id, 0))

    def run(self, dt: float = 1.0):
        # Stats Events
        for t in np.arange(dt, self.params.T + dt, dt):
            heapq.heappush(self.event_queue, (t, EventType.GET_STATS, -1, 0))

        while self.event_queue:
            time, type_, a_id, extra = heapq.heappop(self.event_queue)
            if time > self.params.T: break
            
            if type_ == EventType.CHANGE_PLACE:
                self._handle_move(time, a_id)
            elif type_ == EventType.CHANGE_STATE:
                self._handle_state(time, a_id)
            elif type_ == EventType.GET_STATS:
                self._handle_stats(time)
    
    def _handle_move(self, time: float, a_id: int):
        agent = Agent.agents[a_id]
        
        if agent.current_loc != -1:
            loc = Location.locations[agent.current_loc]
            guests = loc.inspect_guests_list(agent.current_arr, agent.current_dep)
            
            interactions = int(agent.current_dep - agent.current_arr)
            if agent.is_multispreader: interactions *= self.params.alpha
            
            if agent.state in INFECTIOUS_STATES:
                susceptible = [i for i in guests if i != a_id and Agent.agents[i].state == AgentState.S]
                if susceptible:
                    targets = self.rng.sample(susceptible, k=min(len(susceptible), interactions))
                    for t_id in targets:
                        t_agent = Agent.agents[t_id]
                        if t_agent.state == AgentState.S:
                            evt = t_agent.resolve_contact(time, self.rng)
                            if evt:
                                self._state_counts[AgentState.S] -= 1
                                self._state_counts[AgentState.E] += 1
                                heapq.heappush(self.event_queue, evt)
                                
            elif agent.state == AgentState.S:
                infectious = [i for i in guests if i != a_id and Agent.agents[i].state in INFECTIOUS_STATES]
                if infectious:
                    targets = self.rng.sample(infectious, k=min(len(infectious), interactions))
                    for _ in targets:
                        evt = agent.resolve_contact(time, self.rng)
                        if evt:
                            self._state_counts[AgentState.S] -= 1
                            self._state_counts[AgentState.E] += 1
                            heapq.heappush(self.event_queue, evt)
                            break

        evt = agent.change_current_place(self.rng)
        if evt: heapq.heappush(self.event_queue, evt)

    def _handle_state(self, time: float, a_id: int):
        agent = Agent.agents[a_id]
        old_state = agent.state
        evt = agent.resolve_state_conversion(time, self.rng)
        
        if agent.state != old_state:
            self._state_counts[old_state] -= 1
            self._state_counts[agent.state] += 1
            
        if evt: heapq.heappush(self.event_queue, evt)
        
        if agent.state == AgentState.D:
            Location.locations[agent.current_loc].remove_agent()
            agent.current_loc = -1

    def _handle_stats(self, time: float):
        self._stats["time"].append(time)
        counts = {s.name: self._state_counts[s] for s in AgentState}
        self._stats["frac"].append(counts)

    def get_stats_df(self):
        res = defaultdict(list)
        for t, c in zip(self._stats["time"], self._stats["frac"]):
            res["time"].append(t)
            for s in AgentState:
                res[s.name].append(c.get(s.name, 0))
        return res