from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional, List, Dict
from datetime import date as date_type, datetime

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class OfficerTwin(StrictModel):
    officer_id: str = Field(pattern=r'^OFF-\d{6}$')
    twin_date: date_type
    d1_approval_rate: float = Field(ge=0, le=1)
    d1_approval_rate_z: float
    d2_garage_concentration: float = Field(ge=0, le=1)
    d2_garage_concentration_z: float
    d3_threshold_proximity: float = Field(ge=0, le=1)
    d3_threshold_proximity_z: float
    d4_surveyor_diversity: float = Field(ge=0)
    d4_surveyor_diversity_z: float
    d5_turnaround_variance: float = Field(ge=0)
    d5_turnaround_variance_z: float
    d6_after_hours_pct: float = Field(ge=0, le=1)
    d6_after_hours_pct_z: float
    twin_confidence: float = Field(ge=0.0, le=1.0)
    days_of_history: int = Field(ge=0, le=60)
    normalization_baseline: Literal["population", "blended", "personal"]

class AnomalyFlag(StrictModel):
    officer_id: str = Field(pattern=r'^OFF-\d{6}$')
    flag_date: date_type
    per_dimension_zscore: Dict[str, float]
    autoencoder_reconstruction_error: float = Field(ge=0.0, le=1.0)
    sustained_days: int = Field(ge=0, le=7)
    network_ring_id: Optional[str] = Field(default=None, pattern=r'^RING-\d{6}$')
    hard_flag: bool

class ShapContribution(StrictModel):
    dim_code: str = Field(pattern=r"^d[1-6]$")
    dim_label: str
    raw_value: float
    shap_contribution_pts: float

class RiskScore(StrictModel):
    officer_id: str = Field(pattern=r'^OFF-\d{6}$')
    score: int = Field(ge=0, le=100)
    score_band: Literal["clean","watch","suspected","high_risk","confirmed"]
    top_3_shap: List[ShapContribution] = Field(min_length=3, max_length=3)
    score_date: date_type
    decay_applied: bool
    financial_exposure_inr: Optional[float] = None
    twin_confidence_at_scoring: float = Field(ge=0.0, le=1.0)

class AlertPayload(StrictModel):
    officer_id: Optional[str] = None
    officer_ids: Optional[List[str]] = None
    tier: int = Field(ge=2, le=5)
    recipients: List[str]
    channel: List[Literal["dashboard","email","sms"]]
    generated_at: datetime
    evidence_pdf_url: Optional[str] = None

class AlertReviewUpdate(StrictModel):
    review_status: Literal["reviewed_no_action","confirmed"]
    reviewed_by: str
    reviewed_at: Optional[datetime] = None
    pattern_signature: Optional[str] = None
