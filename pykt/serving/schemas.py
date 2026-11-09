"""Strict request models shared by Python callers and FastAPI (Pydantic 2)."""
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

ID = Annotated[StrictStr, Field(min_length=1, max_length=256)]
Revision = Annotated[StrictInt, Field(ge=0)]
Response = Annotated[StrictInt, Field(ge=0, le=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Candidate(StrictModel):
    candidate_id: ID
    kc_vector: list[Any] = Field(min_length=1, max_length=256)


class Event(StrictModel):
    event_id: ID
    kc_vector: list[Any] = Field(min_length=1, max_length=256)
    response: Response


class Step(StrictModel):
    kc_vector: list[Any] = Field(min_length=1, max_length=256)
    response: Response


class RolloutStep(StrictModel):
    kc_vector: list[Any] = Field(min_length=1, max_length=256)
    response: Response | None = None


class Features(StrictModel):
    kc_feature_names: list[ID] = Field(min_length=1, max_length=256)


class PredictRequest(Features):
    candidates: list[Candidate] = Field(min_length=1, max_length=4096)


class ObserveRequest(Features):
    event: Event
    expected_revision: Revision


class BatchRequest(Features):
    attempt_id: ID
    events: list[Event] = Field(min_length=1, max_length=4096)
    expected_revision: Revision


class ForkRequest(StrictModel):
    student_id: ID


class StepRequest(Features, Step):
    pass


class RolloutRequest(Features):
    student_id: ID
    sequence: list[RolloutStep] = Field(min_length=1, max_length=4096)
    mode: Literal['forced', 'sample']
    num_trajectories: Annotated[StrictInt, Field(ge=1, le=1024)] = 1
    seed: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)] | None = None
    forecast_candidates: list[Candidate] = Field(default_factory=list, max_length=4096)
    retain_branch: StrictBool = False
    include_steps: StrictBool = True


class StateInfo(StrictModel):
    student_id: str
    history_revision: int
    next_position: int
    cache_length: int
    model_version: str
    feature_version: str


class Prediction(StrictModel):
    candidate_id: str
    probability: float
    attrs: list[int]
    pid: int
    unknown_fields: list[str]
    unknown_question: bool


class PredictionResult(StrictModel):
    state: StateInfo
    predictions: list[Prediction]


class Observation(StrictModel):
    event_id: str
    probability: float
    duplicate: bool


class ObservationResult(StrictModel):
    state: StateInfo
    observations: list[Observation]
    atomic: bool = True


class BranchInfo(StrictModel):
    branch_id: str
    student_id: str
    base_revision: int
    simulated_steps: int
    next_position: int
    model_version: str
    feature_version: str
    created_at: float
    last_access_at: float
    expires_at: float


class BranchPredictionResult(StrictModel):
    branch: BranchInfo
    predictions: list[Prediction]


class SimulationResult(StrictModel):
    branch: BranchInfo
    probability: float
    hypothetical_response: int


class Statistics(StrictModel):
    mean: float
    std: float
    q05: float
    q50: float
    q95: float


class TrajectoryStep(StrictModel):
    probability: float
    hypothetical_response: int


class Trajectory(StrictModel):
    trajectory_index: int
    correct_count: int
    correct_rate: float
    final_position: int
    steps: list[TrajectoryStep] | None
    forecast_predictions: list[Prediction]


class RolloutResult(StrictModel):
    state: StateInfo
    base_revision: int
    seed: int
    mode: Literal['forced', 'sample']
    trajectories: list[Trajectory]
    correct_count: Statistics
    correct_rate: Statistics
    forecast_statistics: dict[str, Statistics]
    branch: BranchInfo | None
