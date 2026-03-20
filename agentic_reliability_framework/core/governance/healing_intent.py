"""
Healing Intent - OSS creates, Enterprise executes
Enhanced with probabilistic confidence, risk scoring, cost projection,
and full audit trail integration.

This is the core contract between OSS advisory and Enterprise execution.
All intents are immutable and self-validating, ensuring consistency
across the OSS/Enterprise boundary.

The design follows ARF governing principles:
- OSS = advisory intelligence only
- Enterprise = governed execution
- Immutable contracts between layers
- Full provenance and explainability
- Probabilistic uncertainty quantification
- Deep immutability and cryptographic integrity

Copyright 2025 Juan Petter

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from dataclasses import dataclass, field, fields
from typing import Dict, Any, Optional, List, ClassVar, Tuple, Union, Mapping
from datetime import datetime
import hashlib
import json
import time
import uuid
from enum import Enum
import numpy as np
from types import MappingProxyType
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives import serialization

# Import from local infrastructure modules
from .intents import InfrastructureIntent
from agentic_reliability_framework.core.config.constants import (
    OSS_EDITION,
    OSS_LICENSE,
    ENTERPRISE_UPGRADE_URL,
    EXECUTION_ALLOWED,
    MAX_SIMILARITY_CACHE,
    SIMILARITY_THRESHOLD,
    MAX_POLICY_VIOLATIONS,
    MAX_RISK_FACTORS,
    MAX_COST_PROJECTIONS,
    MAX_DECISION_TREE_DEPTH,
    MAX_ALTERNATIVE_ACTIONS,
    OSSBoundaryError,
)


# ============================================================================
# Exceptions
# ============================================================================
class HealingIntentError(Exception):
    """Base exception for HealingIntent errors"""
    pass


class SerializationError(HealingIntentError):
    """Error during serialization/deserialization"""
    pass


class ValidationError(HealingIntentError):
    """Error during intent validation"""
    pass


class IntegrityError(HealingIntentError):
    """Error during signature verification"""
    pass


# ============================================================================
# Enums
# ============================================================================
class IntentSource(str, Enum):
    """Source of the healing intent - matches old ARF patterns"""
    OSS_ANALYSIS = "oss_analysis"
    HUMAN_OVERRIDE = "human_override"
    AUTOMATED_LEARNING = "automated_learning"  # Enterprise only
    SCHEDULED_ACTION = "scheduled_action"  # Enterprise only
    RAG_SIMILARITY = "rag_similarity"  # From RAG graph similarity
    INFRASTRUCTURE_ANALYSIS = "infrastructure_analysis"  # From infra module
    POLICY_VIOLATION = "policy_violation"  # From policy engine
    COST_OPTIMIZATION = "cost_optimization"  # From cost analysis


class IntentStatus(str, Enum):
    """Status of the healing intent - enhanced with partial states"""
    CREATED = "created"
    PENDING_EXECUTION = "pending_execution"
    EXECUTING = "executing"
    EXECUTING_PARTIAL = "executing_partial"
    COMPLETED = "completed"
    COMPLETED_PARTIAL = "completed_partial"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"
    OSS_ADVISORY_ONLY = "oss_advisory_only"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    APPROVED_WITH_OVERRIDES = "approved_with_overrides"


class RecommendedAction(str, Enum):
    """
    Advisory recommendation from the OSS engine.
    Matches the infrastructure module's RecommendedAction.
    """
    APPROVE = "approve"
    DENY = "deny"
    ESCALATE = "escalate"
    DEFER = "defer"  # Wait for more information


# ============================================================================
# Confidence Distribution (Deterministic)
# ============================================================================
class ConfidenceDistribution:
    """
    Probabilistic confidence representation with deterministic seeding.
    """

    def __init__(self, mean: float, std: float = 0.05, samples: Optional[List[float]] = None):
        self.mean = max(0.0, min(mean, 1.0))
        self.std = max(0.0, min(std, 0.5))
        if samples is None:
            seed = int(hashlib.sha256(f"{self.mean}-{self.std}".encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            self._samples = list(rng.normal(self.mean, self.std, 1000).clip(0, 1))
        else:
            self._samples = samples

    @property
    def p5(self) -> float:
        return float(np.percentile(self._samples, 5))

    @property
    def p50(self) -> float:
        return float(np.percentile(self._samples, 50))

    @property
    def p95(self) -> float:
        return float(np.percentile(self._samples, 95))

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        return (self.p5, self.p95)

    def to_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std, "p5": self.p5, "p50": self.p50, "p95": self.p95}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ConfidenceDistribution":
        return cls(mean=data["mean"], std=data.get("std", 0.05), samples=None)

    def __repr__(self) -> str:
        return f"ConfidenceDistribution(mean={self.mean:.3f}, 95% CI=[{self.p5:.3f}, {self.p95:.3f}])"


# ============================================================================
# Deep Freeze / Unfreeze Utilities
# ============================================================================
def _deep_freeze(obj: Any) -> Any:
    """Recursively freeze dictionaries, lists, and sets into immutable structures."""
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(v) for v in obj)
    elif isinstance(obj, set):
        return frozenset(_deep_freeze(v) for v in obj)
    else:
        return obj


def _unfreeze(obj: Any) -> Any:
    """Recursively convert frozen structures back to mutable Python types."""
    if isinstance(obj, MappingProxyType):
        return {k: _unfreeze(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return [_unfreeze(v) for v in obj]
    elif isinstance(obj, frozenset):
        return {_unfreeze(v) for v in obj}
    elif isinstance(obj, dict):
        return {k: _unfreeze(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_unfreeze(v) for v in obj]
    else:
        return obj


# ============================================================================
# Healing Intent (Main Class)
# ============================================================================
@dataclass(frozen=True, slots=True)
class HealingIntent:
    """
    Immutable healing recommendation contract.
    All mutable fields are deeply frozen after construction.
    """

    # === CORE ACTION FIELDS ===
    action: str
    component: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    justification: str = ""

    # === CONFIDENCE & METADATA ===
    confidence: float = 0.85
    confidence_distribution: Optional[Dict[str, float]] = None
    incident_id: str = ""
    detected_at: float = field(default_factory=time.time)

    # === RISK AND COST INTEGRATION ===
    risk_score: Optional[float] = None
    risk_factors: Optional[Dict[str, float]] = None
    cost_projection: Optional[float] = None
    cost_confidence_interval: Optional[Tuple[float, float]] = None
    recommended_action: Optional[RecommendedAction] = None

    # === DECISION TRACKING ===
    decision_tree: Optional[List[Dict[str, Any]]] = None
    alternative_actions: Optional[List[Dict[str, Any]]] = None
    risk_profile: Optional[str] = None

    # === OSS ANALYSIS CONTEXT ===
    reasoning_chain: Optional[List[Dict[str, Any]]] = None
    similar_incidents: Optional[List[Dict[str, Any]]] = None
    rag_similarity_score: Optional[float] = None
    source: IntentSource = IntentSource.OSS_ANALYSIS

    # === IMMUTABLE IDENTIFIERS ===
    intent_id: str = field(default_factory=lambda: f"intent_{uuid.uuid4().hex[:16]}")
    created_at: float = field(default_factory=time.time)

    # === EXECUTION METADATA ===
    status: IntentStatus = IntentStatus.CREATED
    execution_id: Optional[str] = None
    executed_at: Optional[float] = None
    execution_result: Optional[Dict[str, Any]] = None
    enterprise_metadata: Dict[str, Any] = field(default_factory=dict)

    # === HUMAN INTERACTION TRACKING ===
    human_overrides: List[Dict[str, Any]] = field(default_factory=list)
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    comments: List[Dict[str, Any]] = field(default_factory=list)

    # === OSS EDITION METADATA ===
    oss_edition: str = OSS_EDITION
    oss_license: str = OSS_LICENSE
    requires_enterprise: bool = True
    execution_allowed: bool = EXECUTION_ALLOWED

    # === INFRASTRUCTURE INTEGRATION ===
    infrastructure_intent_id: Optional[str] = None
    policy_violations: List[str] = field(default_factory=list)
    infrastructure_intent: Optional[Dict[str, Any]] = None

    # === EXTENDED METADATA (deep frozen) ===
    metadata: Dict[str, Any] = field(default_factory=dict)

    # === CAUSAL LINKAGE ===
    # Stores IDs of ancestor intents (excluding self). The root is the first element.
    ancestor_chain: Tuple[str, ...] = field(default_factory=tuple)
    parent_intent_id: Optional[str] = None
    root_intent_id: Optional[str] = None

    # === EXECUTION CONSTRAINTS (deep frozen) ===
    execution_constraints: Dict[str, Any] = field(default_factory=dict)

    # === CRYPTOGRAPHIC INTEGRITY ===
    signature: Optional[str] = None
    public_key_fingerprint: Optional[str] = None

    # Class constants
    MIN_CONFIDENCE: ClassVar[float] = 0.0
    MAX_CONFIDENCE: ClassVar[float] = 1.0
    MAX_JUSTIFICATION_LENGTH: ClassVar[int] = 5000
    MAX_PARAMETERS_SIZE: ClassVar[int] = 100
    MAX_SIMILAR_INCIDENTS: ClassVar[int] = MAX_SIMILARITY_CACHE
    VERSION: ClassVar[str] = "2.1.0"
    MAX_INTENT_AGE_SECONDS: ClassVar[int] = 3600

    def __post_init__(self) -> None:
        """Validate and deeply freeze all mutable fields."""
        self._validate_oss_boundaries()
        self._validate_risk_integration()
        self._validate_causal_chain()
        self._validate_execution_constraints()

        # Deep freeze every field that is a dict, list, or set.
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (dict, list, set, tuple)):
                # Only freeze if not already frozen (skip tuple because it's immutable)
                if not isinstance(val, (tuple, MappingProxyType)):
                    frozen = _deep_freeze(val)
                    object.__setattr__(self, f.name, frozen)

    # ------------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------------
    def _validate_oss_boundaries(self) -> None:
        errors = []
        if not (self.MIN_CONFIDENCE <= self.confidence <= self.MAX_CONFIDENCE):
            errors.append(f"Confidence must be between {self.MIN_CONFIDENCE} and {self.MAX_CONFIDENCE}, got {self.confidence}")
        if len(self.justification) > self.MAX_JUSTIFICATION_LENGTH:
            errors.append(f"Justification exceeds max length {self.MAX_JUSTIFICATION_LENGTH}")
        if not self.action or not self.action.strip():
            errors.append("Action cannot be empty")
        if not self.component or not self.component.strip():
            errors.append("Component cannot be empty")
        if len(self.parameters) > self.MAX_PARAMETERS_SIZE:
            errors.append(f"Too many parameters: {len(self.parameters)} > {self.MAX_PARAMETERS_SIZE}")
        try:
            json.dumps(self.parameters)
        except (TypeError, ValueError) as e:
            errors.append(f"Parameters must be JSON serializable: {e}")
        try:
            json.dumps(dict(self.metadata))
        except (TypeError, ValueError) as e:
            errors.append(f"Metadata must be JSON serializable: {e}")
        try:
            json.dumps(dict(self.execution_constraints))
        except (TypeError, ValueError) as e:
            errors.append(f"Execution constraints must be JSON serializable: {e}")
        if self.similar_incidents and len(self.similar_incidents) > self.MAX_SIMILAR_INCIDENTS:
            errors.append(f"Too many similar incidents: {len(self.similar_incidents)} > {self.MAX_SIMILAR_INCIDENTS}")
        if self.oss_edition == OSS_EDITION:
            if self.execution_allowed:
                errors.append("Execution not allowed in OSS edition")
            if self.status == IntentStatus.EXECUTING:
                errors.append("EXECUTING status not allowed in OSS edition")
            if self.executed_at is not None:
                errors.append("executed_at should not be set in OSS edition")
            if self.execution_id is not None:
                errors.append("execution_id should not be set in OSS edition")
        if errors:
            raise ValidationError(f"HealingIntent validation failed:\n" + "\n".join(f"  • {error}" for error in errors))

    def _validate_risk_integration(self) -> None:
        if self.risk_score is not None and not (0.0 <= self.risk_score <= 1.0):
            raise ValidationError(f"Risk score must be between 0 and 1, got {self.risk_score}")
        if self.cost_projection is not None and self.cost_projection < 0:
            raise ValidationError(f"Cost projection cannot be negative, got {self.cost_projection}")
        if self.cost_confidence_interval is not None:
            low, high = self.cost_confidence_interval
            if low > high:
                raise ValidationError(f"Invalid confidence interval: [{low}, {high}]")

    def _validate_causal_chain(self) -> None:
        # root_intent_id must match first element of ancestor_chain (if chain non‑empty)
        if self.ancestor_chain:
            if self.root_intent_id != self.ancestor_chain[0]:
                raise ValidationError("root_intent_id must match first element of ancestor_chain")
        if self.parent_intent_id:
            if self.parent_intent_id not in self.ancestor_chain:
                raise ValidationError("parent_intent_id must be present in ancestor_chain")

    def _validate_execution_constraints(self) -> None:
        if "max_retries" in self.execution_constraints:
            if not isinstance(self.execution_constraints["max_retries"], int) or self.execution_constraints["max_retries"] < 0:
                raise ValidationError("execution_constraints.max_retries must be a non‑negative integer")
        if "timeout_seconds" in self.execution_constraints:
            if not isinstance(self.execution_constraints["timeout_seconds"], (int, float)) or self.execution_constraints["timeout_seconds"] <= 0:
                raise ValidationError("execution_constraints.timeout_seconds must be a positive number")

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------
    @property
    def deterministic_id(self) -> str:
        data = {
            "action": self.action,
            "component": self.component,
            "parameters": self._normalize_parameters(self.parameters),
            "incident_id": self.incident_id,
            "oss_edition": self.oss_edition,
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        hash_digest = hashlib.sha256(json_str.encode()).hexdigest()
        return f"intent_{hash_digest[:16]}"

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_stale(self) -> bool:
        return self.age_seconds > self.MAX_INTENT_AGE_SECONDS

    @property
    def instance_hash(self) -> str:
        """
        Cryptographic hash of the intent's core data (including instance‑specific fields).
        This can be used as a unique fingerprint for this specific intent.
        """
        data = self._get_canonical_data()
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _get_canonical_data(self) -> Dict[str, Any]:
        """Return a plain dict of all fields (except computed ones) for hashing/signing."""
        # Convert all fields to plain Python types (unfreeze)
        data = {
            "action": self.action,
            "component": self.component,
            "parameters": _unfreeze(self.parameters),
            "justification": self.justification,
            "confidence": self.confidence,
            "confidence_distribution": _unfreeze(self.confidence_distribution),
            "incident_id": self.incident_id,
            "detected_at": self.detected_at,
            "risk_score": self.risk_score,
            "risk_factors": _unfreeze(self.risk_factors),
            "cost_projection": self.cost_projection,
            "cost_confidence_interval": self.cost_confidence_interval,
            "recommended_action": self.recommended_action.value if self.recommended_action else None,
            "decision_tree": _unfreeze(self.decision_tree),
            "alternative_actions": _unfreeze(self.alternative_actions),
            "risk_profile": self.risk_profile,
            "reasoning_chain": _unfreeze(self.reasoning_chain),
            "similar_incidents": _unfreeze(self.similar_incidents),
            "rag_similarity_score": self.rag_similarity_score,
            "source": self.source.value,
            "intent_id": self.intent_id,
            "created_at": self.created_at,
            "oss_edition": self.oss_edition,
            "oss_license": self.oss_license,
            "requires_enterprise": self.requires_enterprise,
            "execution_allowed": self.execution_allowed,
            "infrastructure_intent_id": self.infrastructure_intent_id,
            "policy_violations": list(self.policy_violations),
            "infrastructure_intent": _unfreeze(self.infrastructure_intent),
            "metadata": _unfreeze(self.metadata),
            "parent_intent_id": self.parent_intent_id,
            "root_intent_id": self.root_intent_id,
            "ancestor_chain": list(self.ancestor_chain),
            "execution_constraints": _unfreeze(self.execution_constraints),
            # Exclude execution_result and enterprise_metadata because they can be set later
        }
        return data

    @property
    def expected_value(self) -> float:
        max_cost = 10000.0
        normalized_cost = min(1.0, (self.cost_projection or 0.0) / max_cost)
        risk = self.risk_score or 0.0
        return self.confidence * (1 - risk) - normalized_cost

    @property
    def is_executable(self) -> bool:
        if self.oss_edition == OSS_EDITION:
            return False
        if self.is_stale:
            return False
        return self.status in [IntentStatus.CREATED, IntentStatus.PENDING_EXECUTION, IntentStatus.APPROVED]

    @property
    def is_oss_advisory(self) -> bool:
        return self.oss_edition == OSS_EDITION and not self.execution_allowed

    @property
    def requires_enterprise_upgrade(self) -> bool:
        return self.requires_enterprise and self.oss_edition == OSS_EDITION

    @property
    def confidence_interval(self) -> Optional[Tuple[float, float]]:
        if self.confidence_distribution:
            return (self.confidence_distribution.get("p5", self.confidence),
                    self.confidence_distribution.get("p95", self.confidence))
        return None

    def is_immutable(self) -> bool:
        # Frozen dataclass with deep‑frozen nested structures is truly immutable.
        return True

    # ------------------------------------------------------------------------
    # Cloning and building (immutable updates)
    # ------------------------------------------------------------------------
    def _clone(self, **updates) -> "HealingIntent":
        """Create a new intent by applying updates to the current one."""
        plain = self._to_plain_dict()
        plain.update(updates)
        return HealingIntent(**plain)

    def _to_plain_dict(self) -> Dict[str, Any]:
        """Convert the intent to a plain dict suitable for constructor (unfreezes)."""
        data = {
            "action": self.action,
            "component": self.component,
            "parameters": _unfreeze(self.parameters),
            "justification": self.justification,
            "confidence": self.confidence,
            "confidence_distribution": _unfreeze(self.confidence_distribution),
            "incident_id": self.incident_id,
            "detected_at": self.detected_at,
            "risk_score": self.risk_score,
            "risk_factors": _unfreeze(self.risk_factors),
            "cost_projection": self.cost_projection,
            "cost_confidence_interval": self.cost_confidence_interval,
            "recommended_action": self.recommended_action,
            "decision_tree": _unfreeze(self.decision_tree),
            "alternative_actions": _unfreeze(self.alternative_actions),
            "risk_profile": self.risk_profile,
            "reasoning_chain": _unfreeze(self.reasoning_chain),
            "similar_incidents": _unfreeze(self.similar_incidents),
            "rag_similarity_score": self.rag_similarity_score,
            "source": self.source,
            "intent_id": self.intent_id,
            "created_at": self.created_at,
            "status": self.status,
            "execution_id": self.execution_id,
            "executed_at": self.executed_at,
            "execution_result": _unfreeze(self.execution_result),
            "enterprise_metadata": _unfreeze(self.enterprise_metadata),
            "human_overrides": _unfreeze(self.human_overrides),
            "approvals": _unfreeze(self.approvals),
            "comments": _unfreeze(self.comments),
            "oss_edition": self.oss_edition,
            "oss_license": self.oss_license,
            "requires_enterprise": self.requires_enterprise,
            "execution_allowed": self.execution_allowed,
            "infrastructure_intent_id": self.infrastructure_intent_id,
            "policy_violations": list(self.policy_violations),
            "infrastructure_intent": _unfreeze(self.infrastructure_intent),
            "metadata": _unfreeze(self.metadata),
            "parent_intent_id": self.parent_intent_id,
            "root_intent_id": self.root_intent_id,
            "ancestor_chain": list(self.ancestor_chain),
            "execution_constraints": _unfreeze(self.execution_constraints),
            "signature": self.signature,
            "public_key_fingerprint": self.public_key_fingerprint,
        }
        return data

    def with_execution_result(
        self,
        execution_id: str,
        executed_at: float,
        result: Dict[str, Any],
        status: IntentStatus = IntentStatus.COMPLETED,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "HealingIntent":
        updates = {
            "status": status,
            "execution_id": execution_id,
            "executed_at": executed_at,
            "execution_result": result,
            "enterprise_metadata": {**(self.enterprise_metadata or {}), **(metadata or {})},
        }
        return self._clone(**updates)

    def with_human_approval(
        self,
        approver: str,
        approval_time: float,
        comments: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> "HealingIntent":
        approval_record = {"approver": approver, "timestamp": approval_time, "comments": comments, "overrides": overrides}
        new_approvals = list(self.approvals) + [approval_record]
        new_overrides = list(self.human_overrides)
        if overrides:
            new_overrides.append({"overrider": approver, "timestamp": approval_time, "overrides": overrides, "reason": comments})
        new_comments = list(self.comments)
        if comments:
            new_comments.append({"author": approver, "timestamp": approval_time, "comment": comments})
        updates = {
            "status": IntentStatus.APPROVED_WITH_OVERRIDES if overrides else IntentStatus.APPROVED,
            "human_overrides": new_overrides,
            "approvals": new_approvals,
            "comments": new_comments,
        }
        return self._clone(**updates)

    def mark_as_sent_to_enterprise(self) -> "HealingIntent":
        return self._clone(status=IntentStatus.PENDING_EXECUTION)

    def mark_as_oss_advisory(self) -> "HealingIntent":
        return self._clone(status=IntentStatus.OSS_ADVISORY_ONLY, execution_allowed=False)

    # ------------------------------------------------------------------------
    # Cryptographic signing
    # ------------------------------------------------------------------------
    def sign(self, private_key: Any) -> "HealingIntent":
        """Return a new signed intent."""
        data_to_sign = self._get_signable_data()
        signature = private_key.sign(
            data_to_sign.encode(),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        b64_sig = base64.b64encode(signature).decode()
        fp = self._get_public_key_fingerprint(private_key.public_key())
        return self._clone(signature=b64_sig, public_key_fingerprint=fp)

    def verify(self, public_key: Any) -> bool:
        """Verify the signature."""
        if not self.signature:
            return False
        data_to_verify = self._get_signable_data()
        try:
            signature_bytes = base64.b64decode(self.signature)
            public_key.verify(
                signature_bytes,
                data_to_verify.encode(),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

    def _get_signable_data(self) -> str:
        """Return canonical string of core fields for signing."""
        data = self._get_canonical_data()
        # Remove signature and fingerprint
        data.pop("signature", None)
        data.pop("public_key_fingerprint", None)
        # Remove any ephemeral execution fields that could change after signing
        data.pop("execution_id", None)
        data.pop("executed_at", None)
        data.pop("execution_result", None)
        data.pop("enterprise_metadata", None)
        data.pop("status", None)
        return json.dumps(data, sort_keys=True, default=str)

    def _get_public_key_fingerprint(self, public_key: Any) -> str:
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(pem).hexdigest()[:16]

    # ------------------------------------------------------------------------
    # Serialization and external interfaces
    # ------------------------------------------------------------------------
    def to_enterprise_request(self) -> Dict[str, Any]:
        """Payload for Enterprise API."""
        return {
            "intent_id": self.deterministic_id,
            "action": self.action,
            "component": self.component,
            "parameters": _unfreeze(self.parameters),
            "justification": self.justification,
            "confidence": self.confidence,
            "confidence_interval": self.confidence_interval,
            "risk_score": self.risk_score,
            "cost_projection": self.cost_projection,
            "incident_id": self.incident_id,
            "detected_at": self.detected_at,
            "created_at": self.created_at,
            "source": self.source.value,
            "recommended_action": self.recommended_action.value if self.recommended_action else None,
            "oss_edition": self.oss_edition,
            "oss_license": self.oss_license,
            "requires_enterprise": self.requires_enterprise,
            "execution_allowed": self.execution_allowed,
            "version": self.VERSION,
            "expected_value": self.expected_value,
            "is_stale": self.is_stale,
            "parent_intent_id": self.parent_intent_id,
            "root_intent_id": self.root_intent_id,
            "ancestor_chain": list(self.ancestor_chain),
            "execution_constraints": _unfreeze(self.execution_constraints),
            "signature": self.signature,
            "public_key_fingerprint": self.public_key_fingerprint,
            "oss_metadata": {
                "similar_incidents_count": len(self.similar_incidents) if self.similar_incidents else 0,
                "rag_similarity_score": self.rag_similarity_score,
                "has_reasoning_chain": self.reasoning_chain is not None,
                "source": self.source.value,
                "is_oss_advisory": self.is_oss_advisory,
                "risk_factors": _unfreeze(self.risk_factors),
                "policy_violations_count": len(self.policy_violations),
                "confidence_basis": self._get_confidence_basis(),
                "learning_applied": False,
                "learning_reason": "OSS advisory mode does not persist or learn from outcomes",
            },
            "upgrade_url": ENTERPRISE_UPGRADE_URL,
            "enterprise_features": [
                "autonomous_execution", "approval_workflows", "persistent_storage",
                "learning_engine", "audit_trails", "compliance_reports", "multi_tenant_support",
                "sso_integration", "24_7_support", "probabilistic_confidence", "risk_analytics",
                "cost_optimization"
            ]
        }

    def _get_confidence_basis(self) -> str:
        if self.recommended_action == RecommendedAction.DENY and self.policy_violations:
            return "policy_violation"
        if self.rag_similarity_score and self.rag_similarity_score > SIMILARITY_THRESHOLD:
            return "historical_similarity"
        if self.risk_score is not None:
            return "risk_based"
        return "policy_only"

    def to_dict(self, include_oss_context: bool = False) -> Dict[str, Any]:
        """Full dictionary representation (for serialization)."""
        data = self._to_plain_dict()
        # Convert enums to strings
        if "source" in data and isinstance(data["source"], IntentSource):
            data["source"] = data["source"].value
        if "status" in data and isinstance(data["status"], IntentStatus):
            data["status"] = data["status"].value
        if "recommended_action" in data and isinstance(data["recommended_action"], RecommendedAction):
            data["recommended_action"] = data["recommended_action"].value if data["recommended_action"] else None

        if not include_oss_context:
            data.pop("reasoning_chain", None)
            data.pop("similar_incidents", None)
            data.pop("rag_similarity_score", None)
            data.pop("decision_tree", None)
            data.pop("alternative_actions", None)
            data.pop("infrastructure_intent", None)

        # Add computed properties
        data["deterministic_id"] = self.deterministic_id
        data["age_seconds"] = self.age_seconds
        data["is_executable"] = self.is_executable
        data["is_oss_advisory"] = self.is_oss_advisory
        data["requires_enterprise_upgrade"] = self.requires_enterprise_upgrade
        data["version"] = self.VERSION
        data["confidence_interval"] = self.confidence_interval
        data["is_stale"] = self.is_stale
        data["expected_value"] = self.expected_value
        data["instance_hash"] = self.instance_hash
        return data

    def get_oss_context(self) -> Dict[str, Any]:
        return {
            "reasoning_chain": _unfreeze(self.reasoning_chain),
            "similar_incidents": _unfreeze(self.similar_incidents),
            "rag_similarity_score": self.rag_similarity_score,
            "decision_tree": _unfreeze(self.decision_tree),
            "alternative_actions": _unfreeze(self.alternative_actions),
            "analysis_timestamp": datetime.fromtimestamp(self.detected_at).isoformat(),
            "source": self.source.value,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "oss_edition": self.oss_edition,
            "is_oss_advisory": self.is_oss_advisory,
            "infrastructure_intent": _unfreeze(self.infrastructure_intent),
            "metadata": _unfreeze(self.metadata),
            "parent_intent_id": self.parent_intent_id,
            "root_intent_id": self.root_intent_id,
            "ancestor_chain": list(self.ancestor_chain),
        }

    def get_execution_summary(self) -> Dict[str, Any]:
        summary = {
            "intent_id": self.deterministic_id,
            "action": self.action,
            "component": self.component,
            "confidence": self.confidence,
            "confidence_interval": self.confidence_interval,
            "risk_score": self.risk_score,
            "cost_projection": self.cost_projection,
            "status": self.status.value,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "age_seconds": self.age_seconds,
            "oss_edition": self.oss_edition,
            "requires_enterprise": self.requires_enterprise,
            "is_oss_advisory": self.is_oss_advisory,
            "source": self.source.value,
            "policy_violations_count": len(self.policy_violations),
            "confidence_basis": self._get_confidence_basis(),
            "expected_value": self.expected_value,
            "is_stale": self.is_stale,
            "parent_intent_id": self.parent_intent_id,
            "root_intent_id": self.root_intent_id,
            "ancestor_chain_length": len(self.ancestor_chain),
            "has_signature": self.signature is not None,
        }
        if self.executed_at:
            summary["executed_at"] = datetime.fromtimestamp(self.executed_at).isoformat()
            summary["execution_duration_seconds"] = self.executed_at - self.created_at
        if self.execution_result:
            summary["execution_success"] = self.execution_result.get("success", False)
            summary["execution_message"] = self.execution_result.get("message", "")
        if self.rag_similarity_score:
            summary["rag_similarity_score"] = self.rag_similarity_score
        if self.similar_incidents:
            summary["similar_incidents_count"] = len(self.similar_incidents)
        if self.approvals:
            summary["approvals_count"] = len(self.approvals)
            summary["approved_by"] = [a.get("approver") for a in self.approvals if a.get("approver")]
        if self.human_overrides:
            summary["overrides_count"] = len(self.human_overrides)
        return summary

    # ------------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------------
    @classmethod
    def from_infrastructure_intent(
        cls,
        infrastructure_intent: Any,
        action: str,
        component: str,
        parameters: Dict[str, Any],
        justification: str,
        confidence: float = 0.85,
        risk_score: Optional[float] = None,
        risk_factors: Optional[Dict[str, float]] = None,
        cost_projection: Optional[float] = None,
        policy_violations: Optional[List[str]] = None,
        recommended_action: Optional[RecommendedAction] = None,
        source: IntentSource = IntentSource.INFRASTRUCTURE_ANALYSIS,
        metadata: Optional[Dict[str, Any]] = None,
        parent_intent_id: Optional[str] = None,
        ancestor_chain: Optional[List[str]] = None,
        execution_constraints: Optional[Dict[str, Any]] = None,
    ) -> "HealingIntent":
        """Create from infrastructure module analysis."""
        infrastructure_intent_id = getattr(infrastructure_intent, 'intent_id', None)
        if hasattr(infrastructure_intent, 'model_dump'):
            intent_dict = infrastructure_intent.model_dump()
        elif hasattr(infrastructure_intent, 'to_dict'):
            intent_dict = infrastructure_intent.to_dict()
        else:
            intent_dict = {"type": str(type(infrastructure_intent))}

        # Build ancestor chain and root ID
        if ancestor_chain is None:
            if parent_intent_id:
                ancestor_chain = [parent_intent_id]
            else:
                ancestor_chain = []
        root_intent_id = ancestor_chain[0] if ancestor_chain else None

        return cls(
            action=action,
            component=component,
            parameters=parameters,
            justification=justification,
            confidence=confidence,
            risk_score=risk_score,
            risk_factors=risk_factors,
            cost_projection=cost_projection,
            policy_violations=policy_violations or [],
            recommended_action=recommended_action,
            source=source,
            infrastructure_intent_id=infrastructure_intent_id,
            infrastructure_intent=intent_dict,
            metadata=metadata or {},
            parent_intent_id=parent_intent_id,
            root_intent_id=root_intent_id,
            ancestor_chain=tuple(ancestor_chain),
            execution_constraints=execution_constraints or {},
        )

    @classmethod
    def from_analysis(
        cls,
        action: str,
        component: str,
        parameters: Dict[str, Any],
        justification: str,
        confidence: float,
        confidence_std: float = 0.05,
        similar_incidents: Optional[List[Dict[str, Any]]] = None,
        reasoning_chain: Optional[List[Dict[str, Any]]] = None,
        incident_id: str = "",
        source: IntentSource = IntentSource.OSS_ANALYSIS,
        rag_similarity_score: Optional[float] = None,
        risk_score: Optional[float] = None,
        cost_projection: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        policy_violations: Optional[List[str]] = None,
        parent_intent_id: Optional[str] = None,
        ancestor_chain: Optional[List[str]] = None,
        execution_constraints: Optional[Dict[str, Any]] = None,
    ) -> "HealingIntent":
        """Primary OSS factory."""
        if similar_incidents and len(similar_incidents) > cls.MAX_SIMILAR_INCIDENTS:
            similar_incidents = similar_incidents[:cls.MAX_SIMILAR_INCIDENTS]

        conf_dist = ConfidenceDistribution(confidence, confidence_std)
        enhanced_confidence = confidence
        if similar_incidents:
            similarity_scores = [inc.get("similarity", 0.0) for inc in similar_incidents if "similarity" in inc]
            if similarity_scores:
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                confidence_boost = min(0.2, avg_similarity * 0.3)
                enhanced_confidence = min(confidence * (1.0 + confidence_boost), cls.MAX_CONFIDENCE)

        final_rag_score = rag_similarity_score
        if final_rag_score is None and similar_incidents:
            top_similarities = [inc.get("similarity", 0.0) for inc in similar_incidents[:3] if "similarity" in inc]
            if top_similarities:
                final_rag_score = sum(top_similarities) / len(top_similarities)

        # Build ancestor chain and root ID
        if ancestor_chain is None:
            if parent_intent_id:
                ancestor_chain = [parent_intent_id]
            else:
                ancestor_chain = []
        root_intent_id = ancestor_chain[0] if ancestor_chain else None

        return cls(
            action=action,
            component=component,
            parameters=parameters,
            justification=justification,
            confidence=enhanced_confidence,
            confidence_distribution=conf_dist.to_dict(),
            incident_id=incident_id,
            similar_incidents=similar_incidents,
            reasoning_chain=reasoning_chain,
            rag_similarity_score=final_rag_score,
            source=source,
            risk_score=risk_score,
            cost_projection=cost_projection,
            metadata=metadata or {},
            policy_violations=policy_violations or [],
            parent_intent_id=parent_intent_id,
            root_intent_id=root_intent_id,
            ancestor_chain=tuple(ancestor_chain),
            execution_constraints=execution_constraints or {},
        )

    @classmethod
    def from_rag_recommendation(
        cls,
        action: str,
        component: str,
        parameters: Dict[str, Any],
        rag_similarity_score: float,
        similar_incidents: List[Dict[str, Any]],
        justification_template: str = "Based on {count} similar historical incidents with {success_rate:.0%} success rate",
        success_rate: Optional[float] = None,
        risk_score: Optional[float] = None,
        cost_projection: Optional[float] = None,
    ) -> "HealingIntent":
        if not similar_incidents:
            raise ValidationError("RAG recommendation requires similar incidents")
        if success_rate is None:
            successful = sum(1 for inc in similar_incidents if inc.get("success", False))
            success_rate = successful / len(similar_incidents)
        justification = justification_template.format(
            count=len(similar_incidents),
            success_rate=success_rate or 0.0,
            action=action,
            component=component,
        )
        base_confidence = rag_similarity_score * 0.8
        if success_rate:
            base_confidence = base_confidence * (0.7 + success_rate * 0.3)
        return cls.from_analysis(
            action=action,
            component=component,
            parameters=parameters,
            justification=justification,
            confidence=min(base_confidence, 0.95),
            similar_incidents=similar_incidents,
            incident_id=similar_incidents[0].get("incident_id", "") if similar_incidents else "",
            source=IntentSource.RAG_SIMILARITY,
            rag_similarity_score=rag_similarity_score,
            risk_score=risk_score,
            cost_projection=cost_projection,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealingIntent":
        clean = data.copy()
        # Convert string enums
        if "source" in clean and isinstance(clean["source"], str):
            clean["source"] = IntentSource(clean["source"])
        if "status" in clean and isinstance(clean["status"], str):
            clean["status"] = IntentStatus(clean["status"])
        if "recommended_action" in clean and isinstance(clean["recommended_action"], str):
            try:
                clean["recommended_action"] = RecommendedAction(clean["recommended_action"])
            except ValueError:
                clean["recommended_action"] = None
        # Remove computed fields
        for f in ["deterministic_id", "age_seconds", "is_executable", "is_oss_advisory",
                  "requires_enterprise_upgrade", "version", "confidence_interval",
                  "is_stale", "expected_value", "instance_hash"]:
            clean.pop(f, None)
        return cls(**clean)

    # ------------------------------------------------------------------------
    # Utility: parameter normalization (order‑preserving)
    # ------------------------------------------------------------------------
    def _normalize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for key, value in sorted(params.items()):
            normalized[key] = self._normalize_value(value)
        return normalized

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, set):
            return tuple(sorted(self._normalize_value(v) for v in value))
        elif isinstance(value, (list, tuple)):
            return tuple(self._normalize_value(v) for v in value)
        elif isinstance(value, dict):
            return self._normalize_parameters(value)
        elif hasattr(value, '__dict__'):
            return self._normalize_parameters(value.__dict__)
        else:
            try:
                return str(value)
            except Exception:
                return f"<unserializable:{type(value).__name__}>"

    def __repr__(self) -> str:
        risk_str = f"{self.risk_score:.2f}" if self.risk_score is not None else "N/A"
        return (
            f"HealingIntent("
            f"id={self.deterministic_id[:8]}..., "
            f"action={self.action}, "
            f"component={self.component}, "
            f"confidence={self.confidence:.2f}, "
            f"risk={risk_str}, "
            f"status={self.status.value}, "
            f"age={self.age_seconds:.0f}s"
            f")"
        )


# ============================================================================
# Serializer
# ============================================================================
class HealingIntentSerializer:
    SCHEMA_VERSION: ClassVar[str] = "2.1.0"

    @classmethod
    def serialize(cls, intent: HealingIntent, version: str = "2.1.0") -> Dict[str, Any]:
        try:
            if version == "2.1.0":
                return {
                    "version": version,
                    "schema_version": cls.SCHEMA_VERSION,
                    "data": intent.to_dict(include_oss_context=True),
                    "metadata": {
                        "serialized_at": time.time(),
                        "deterministic_id": intent.deterministic_id,
                        "is_executable": intent.is_executable,
                        "is_oss_advisory": intent.is_oss_advisory,
                        "requires_enterprise_upgrade": intent.requires_enterprise_upgrade,
                        "oss_edition": intent.oss_edition,
                        "has_probabilistic_confidence": intent.confidence_distribution is not None,
                        "has_risk_assessment": intent.risk_score is not None,
                        "has_cost_projection": intent.cost_projection is not None,
                        "metadata_keys": list(intent.metadata.keys()) if intent.metadata else [],
                        "has_signature": intent.signature is not None,
                        "ancestor_chain_length": len(intent.ancestor_chain),
                        "expected_value": intent.expected_value,
                        "is_stale": intent.is_stale,
                    }
                }
            elif version in ("2.0.0", "1.1.0", "1.0.0"):
                # Legacy support
                data = intent.to_dict(include_oss_context=True)
                for field in ["metadata", "parent_intent_id", "root_intent_id", "ancestor_chain",
                              "execution_constraints", "signature", "public_key_fingerprint"]:
                    data.pop(field, None)
                if version.startswith("1."):
                    for f in ["confidence_distribution", "risk_score", "risk_factors", "cost_projection",
                              "cost_confidence_interval", "recommended_action", "decision_tree",
                              "alternative_actions", "risk_profile", "human_overrides", "approvals",
                              "comments", "infrastructure_intent_id", "policy_violations",
                              "infrastructure_intent", "confidence_interval"]:
                        data.pop(f, None)
                return {
                    "version": version,
                    "schema_version": "1.1.0" if version == "1.1.0" else "1.0.0",
                    "data": data,
                    "metadata": {
                        "serialized_at": time.time(),
                        "deterministic_id": intent.deterministic_id,
                        "is_executable": intent.is_executable,
                        "is_oss_advisory": intent.is_oss_advisory,
                    }
                }
            else:
                raise SerializationError(f"Unsupported version: {version}")
        except Exception as e:
            raise SerializationError(f"Failed to serialize HealingIntent: {e}") from e

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> HealingIntent:
        try:
            version = data.get("version", "1.0.0")
            intent_data = data.get("data", data)
            if version in ["2.1.0", "2.0.0", "1.1.0", "1.0.0"]:
                if version.startswith("1."):
                    # Add defaults for missing fields
                    intent_data.setdefault("confidence_distribution", None)
                    intent_data.setdefault("risk_score", None)
                    intent_data.setdefault("risk_factors", None)
                    intent_data.setdefault("cost_projection", None)
                    intent_data.setdefault("cost_confidence_interval", None)
                    intent_data.setdefault("recommended_action", None)
                    intent_data.setdefault("decision_tree", None)
                    intent_data.setdefault("alternative_actions", None)
                    intent_data.setdefault("risk_profile", None)
                    intent_data.setdefault("human_overrides", [])
                    intent_data.setdefault("approvals", [])
                    intent_data.setdefault("comments", [])
                    intent_data.setdefault("infrastructure_intent_id", None)
                    intent_data.setdefault("policy_violations", [])
                    intent_data.setdefault("infrastructure_intent", None)
                    intent_data.setdefault("metadata", {})
                    intent_data.setdefault("parent_intent_id", None)
                    intent_data.setdefault("root_intent_id", None)
                    intent_data.setdefault("ancestor_chain", [])
                    intent_data.setdefault("execution_constraints", {})
                    intent_data.setdefault("signature", None)
                    intent_data.setdefault("public_key_fingerprint", None)
                return HealingIntent.from_dict(intent_data)
            else:
                raise SerializationError(f"Unsupported version: {version}")
        except Exception as e:
            raise SerializationError(f"Failed to deserialize HealingIntent: {e}") from e

    @classmethod
    def to_json(cls, intent: HealingIntent, pretty: bool = False) -> str:
        return json.dumps(cls.serialize(intent), indent=2 if pretty else None, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> HealingIntent:
        return cls.deserialize(json.loads(json_str))

    @classmethod
    def to_enterprise_json(cls, intent: HealingIntent) -> str:
        return json.dumps(intent.to_enterprise_request(), default=str)

    @classmethod
    def validate_for_oss(cls, intent: HealingIntent) -> bool:
        try:
            if intent.oss_edition != OSS_EDITION:
                return False
            if intent.execution_allowed:
                return False
            if intent.similar_incidents and len(intent.similar_incidents) > HealingIntent.MAX_SIMILAR_INCIDENTS:
                return False
            if not intent.is_immutable():
                return False
            if intent.executed_at is not None or intent.execution_id is not None:
                return False
            return True
        except Exception:
            return False


# ============================================================================
# Factory functions (backward compatible)
# ============================================================================
def create_infrastructure_healing_intent(
    infrastructure_result: Any,
    action_mapping: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    parent_intent_id: Optional[str] = None,
    ancestor_chain: Optional[List[str]] = None,
    execution_constraints: Optional[Dict[str, Any]] = None,
) -> HealingIntent:
    if action_mapping is None:
        action_mapping = {"approve": "execute", "deny": "block", "escalate": "escalate", "defer": "defer"}
    recommended_action = getattr(infrastructure_result, 'recommended_action', None)
    action = action_mapping.get(recommended_action.value, "review") if recommended_action else "review"
    parameters = {
        "infrastructure_intent_id": getattr(infrastructure_result, 'intent_id', None),
        "risk_score": getattr(infrastructure_result, 'risk_score', None),
        "cost_projection": getattr(infrastructure_result, 'cost_projection', None),
        "policy_violations": getattr(infrastructure_result, 'policy_violations', []),
        "evaluation_details": getattr(infrastructure_result, 'evaluation_details', {})
    }
    justification_parts = [getattr(infrastructure_result, 'justification', "Infrastructure analysis completed")]
    policy_violations = getattr(infrastructure_result, 'policy_violations', [])
    if policy_violations:
        justification_parts.append(f"Policy violations: {'; '.join(policy_violations)}")
    cost_projection = getattr(infrastructure_result, 'cost_projection', None)
    return HealingIntent.from_infrastructure_intent(
        infrastructure_intent=getattr(infrastructure_result, 'infrastructure_intent', None),
        action=action,
        component="infrastructure",
        parameters=parameters,
        justification=" ".join(justification_parts),
        confidence=getattr(infrastructure_result, 'confidence_score', 0.85),
        risk_score=getattr(infrastructure_result, 'risk_score', None),
        cost_projection=cost_projection,
        policy_violations=policy_violations,
        recommended_action=recommended_action,
        source=IntentSource.INFRASTRUCTURE_ANALYSIS,
        metadata=metadata or {},
        parent_intent_id=parent_intent_id,
        ancestor_chain=ancestor_chain,
        execution_constraints=execution_constraints or {},
    ).mark_as_oss_advisory()


def create_rollback_intent(
    component: str,
    revision: str = "previous",
    justification: str = "",
    incident_id: str = "",
    similar_incidents: Optional[List[Dict[str, Any]]] = None,
    rag_similarity_score: Optional[float] = None,
    risk_score: Optional[float] = None,
    cost_projection: Optional[float] = None,
    parent_intent_id: Optional[str] = None,
    ancestor_chain: Optional[List[str]] = None,
) -> HealingIntent:
    if not justification:
        justification = f"Rollback {component} to {revision} revision"
    return HealingIntent.from_analysis(
        action="rollback",
        component=component,
        parameters={"revision": revision},
        justification=justification,
        confidence=0.9,
        similar_incidents=similar_incidents,
        incident_id=incident_id,
        rag_similarity_score=rag_similarity_score,
        risk_score=risk_score,
        cost_projection=cost_projection,
        parent_intent_id=parent_intent_id,
        ancestor_chain=ancestor_chain,
    ).mark_as_oss_advisory()


def create_restart_intent(
    component: str,
    container_id: Optional[str] = None,
    justification: str = "",
    incident_id: str = "",
    similar_incidents: Optional[List[Dict[str, Any]]] = None,
    rag_similarity_score: Optional[float] = None,
    risk_score: Optional[float] = None,
    cost_projection: Optional[float] = None,
    parent_intent_id: Optional[str] = None,
    ancestor_chain: Optional[List[str]] = None,
) -> HealingIntent:
    parameters = {}
    if container_id:
        parameters["container_id"] = container_id
    if not justification:
        justification = f"Restart container for {component}"
    return HealingIntent.from_analysis(
        action="restart_container",
        component=component,
        parameters=parameters,
        justification=justification,
        confidence=0.85,
        similar_incidents=similar_incidents,
        incident_id=incident_id,
        rag_similarity_score=rag_similarity_score,
        risk_score=risk_score,
        cost_projection=cost_projection,
        parent_intent_id=parent_intent_id,
        ancestor_chain=ancestor_chain,
    ).mark_as_oss_advisory()


def create_scale_out_intent(
    component: str,
    scale_factor: int = 2,
    justification: str = "",
    incident_id: str = "",
    similar_incidents: Optional[List[Dict[str, Any]]] = None,
    rag_similarity_score: Optional[float] = None,
    risk_score: Optional[float] = None,
    cost_projection: Optional[float] = None,
    parent_intent_id: Optional[str] = None,
    ancestor_chain: Optional[List[str]] = None,
) -> HealingIntent:
    if not justification:
        justification = f"Scale out {component} by factor {scale_factor}"
    return HealingIntent.from_analysis(
        action="scale_out",
        component=component,
        parameters={"scale_factor": scale_factor},
        justification=justification,
        confidence=0.8,
        similar_incidents=similar_incidents,
        incident_id=incident_id,
        rag_similarity_score=rag_similarity_score,
        risk_score=risk_score,
        cost_projection=cost_projection,
        parent_intent_id=parent_intent_id,
        ancestor_chain=ancestor_chain,
    ).mark_as_oss_advisory()


def create_oss_advisory_intent(
    action: str,
    component: str,
    parameters: Dict[str, Any],
    justification: str,
    confidence: float = 0.85,
    incident_id: str = "",
    risk_score: Optional[float] = None,
    cost_projection: Optional[float] = None,
    parent_intent_id: Optional[str] = None,
    ancestor_chain: Optional[List[str]] = None,
) -> HealingIntent:
    """
    Create a generic OSS advisory-only intent.

    If ancestor_chain is omitted but parent_intent_id is provided,
    ancestor_chain will be set to [parent_intent_id] to maintain validation.
    """
    if ancestor_chain is None:
        ancestor_chain = [parent_intent_id] if parent_intent_id else []
    return HealingIntent(
        action=action,
        component=component,
        parameters=parameters,
        justification=justification,
        confidence=confidence,
        incident_id=incident_id,
        risk_score=risk_score,
        cost_projection=cost_projection,
        parent_intent_id=parent_intent_id,
        ancestor_chain=tuple(ancestor_chain),
        root_intent_id=ancestor_chain[0] if ancestor_chain else None,
        oss_edition=OSS_EDITION,
        requires_enterprise=True,
        execution_allowed=False,
        status=IntentStatus.OSS_ADVISORY_ONLY,
    )


# ============================================================================
# Exports
# ============================================================================
__all__ = [
    "HealingIntent",
    "ConfidenceDistribution",
    "HealingIntentSerializer",
    "IntentSource",
    "IntentStatus",
    "RecommendedAction",
    "HealingIntentError",
    "SerializationError",
    "ValidationError",
    "IntegrityError",
    "create_infrastructure_healing_intent",
    "create_rollback_intent",
    "create_restart_intent",
    "create_scale_out_intent",
    "create_oss_advisory_intent",
]
