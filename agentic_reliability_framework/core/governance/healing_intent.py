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

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, ClassVar, Tuple, Union, Mapping
from datetime import datetime
import hashlib
import json
import time
import uuid
from enum import Enum
import numpy as np
from types import MappingProxyType
import hmac
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

    Instead of a single confidence score, this represents a distribution
    of possible confidence values, allowing for uncertainty quantification.
    Matches patterns from the risk_engine.py module.
    """

    def __init__(self, mean: float, std: float = 0.05, samples: Optional[List[float]] = None):
        self.mean = max(0.0, min(mean, 1.0))
        self.std = max(0.0, min(std, 0.5))
        if samples is None:
            # Deterministic RNG based on mean and std
            seed = int(hashlib.sha256(f"{self.mean}-{self.std}".encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            self._samples = list(rng.normal(self.mean, self.std, 1000).clip(0, 1))
        else:
            self._samples = samples

    @property
    def p5(self) -> float:
        """5th percentile (pessimistic)"""
        return float(np.percentile(self._samples, 5))

    @property
    def p50(self) -> float:
        """50th percentile (median)"""
        return float(np.percentile(self._samples, 50))

    @property
    def p95(self) -> float:
        """95th percentile (optimistic)"""
        return float(np.percentile(self._samples, 95))

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% confidence interval"""
        return (self.p5, self.p95)

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary"""
        return {
            "mean": self.mean,
            "std": self.std,
            "p5": self.p5,
            "p50": self.p50,
            "p95": self.p95
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ConfidenceDistribution":
        """Deserialize from dictionary"""
        return cls(
            mean=data["mean"],
            std=data.get("std", 0.05),
            samples=None  # Will regenerate on access
        )

    def __repr__(self) -> str:
        return f"ConfidenceDistribution(mean={self.mean:.3f}, 95% CI=[{self.p5:.3f}, {self.p95:.3f}])"


# ============================================================================
# Deep Freeze Utility
# ============================================================================
def _deep_freeze(obj: Any) -> Any:
    """
    Recursively freeze dictionaries, lists, and sets into immutable structures.
    Dictionaries become MappingProxyType, lists become tuples, sets become frozenset.
    """
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(v) for v in obj)
    elif isinstance(obj, set):
        return frozenset(_deep_freeze(v) for v in obj)
    else:
        return obj


# ============================================================================
# Healing Intent (Main Class)
# ============================================================================
@dataclass(frozen=True, slots=True)
class HealingIntent:
    """
    OSS-generated healing recommendation for Enterprise execution

    Enhanced with:
    - Probabilistic confidence distributions (deterministic)
    - Risk score and cost projection integration
    - Decision tree tracking for explainability
    - Human override audit trail
    - Partial execution support
    - Integration with infrastructure governance module
    - Full backward compatibility with old ARF patterns
    - Deep immutability for all fields
    - Causal linkage (parent/root IDs, causal chain)
    - Execution constraints (retries, timeout, blast radius)
    - Cryptographic signing for integrity
    - Expected value computation for decision policy
    """

    # === CORE ACTION FIELDS (Sent to Enterprise) ===
    action: str                          # Tool name, e.g., "restart_container", "provision_vm"
    component: str                       # Target component or resource
    parameters: Dict[str, Any] = field(default_factory=dict)  # Action parameters
    justification: str = ""              # OSS reasoning chain

    # === CONFIDENCE & METADATA ===
    confidence: float = 0.85             # OSS confidence score (0.0 to 1.0)
    confidence_distribution: Optional[Dict[str, float]] = None  # Probabilistic confidence
    incident_id: str = ""                # Source incident identifier
    detected_at: float = field(default_factory=time.time)  # When OSS detected

    # === RISK AND COST INTEGRATION ===
    risk_score: Optional[float] = None   # From risk engine (0-1)
    risk_factors: Optional[Dict[str, float]] = None  # Breakdown by factor
    cost_projection: Optional[float] = None  # Estimated cost impact
    cost_confidence_interval: Optional[Tuple[float, float]] = None  # 95% CI
    recommended_action: Optional[RecommendedAction] = None  # From risk engine

    # === DECISION TRACKING ===
    decision_tree: Optional[List[Dict[str, Any]]] = None  # How decision was reached
    alternative_actions: Optional[List[Dict[str, Any]]] = None  # Alternatives considered
    risk_profile: Optional[str] = None  # Risk tolerance used (conservative/moderate/aggressive)

    # === OSS ANALYSIS CONTEXT (Stays in OSS) ===
    reasoning_chain: Optional[List[Dict[str, Any]]] = None
    similar_incidents: Optional[List[Dict[str, Any]]] = None
    rag_similarity_score: Optional[float] = None
    source: IntentSource = IntentSource.OSS_ANALYSIS

    # === IMMUTABLE IDENTIFIERS ===
    intent_id: str = field(default_factory=lambda: f"intent_{uuid.uuid4().hex[:16]}")
    created_at: float = field(default_factory=time.time)

    # === EXECUTION METADATA (Set by Enterprise) ===
    status: IntentStatus = IntentStatus.CREATED
    execution_id: Optional[str] = None
    executed_at: Optional[float] = None
    execution_result: Optional[Dict[str, Any]] = None
    enterprise_metadata: Dict[str, Any] = field(default_factory=dict)

    # === HUMAN INTERACTION TRACKING ===
    human_overrides: List[Dict[str, Any]] = field(default_factory=list)  # Audit trail
    approvals: List[Dict[str, Any]] = field(default_factory=list)  # Who approved what
    comments: List[Dict[str, Any]] = field(default_factory=list)  # Human comments

    # === OSS EDITION METADATA ===
    oss_edition: str = OSS_EDITION
    oss_license: str = OSS_LICENSE
    requires_enterprise: bool = True  # Always True for OSS-generated intents
    execution_allowed: bool = EXECUTION_ALLOWED  # From OSS constants

    # === INFRASTRUCTURE INTEGRATION ===
    infrastructure_intent_id: Optional[str] = None  # Link to infrastructure intent if any
    policy_violations: List[str] = field(default_factory=list)  # From policy engine
    infrastructure_intent: Optional[Dict[str, Any]] = None  # Original infrastructure intent

    # === EXTENDED METADATA ===
    metadata: Dict[str, Any] = field(default_factory=dict)  # Will be deep-frozen

    # === CAUSAL LINKAGE (NEW) ===
    parent_intent_id: Optional[str] = None        # Parent intent that triggered this one
    root_intent_id: Optional[str] = None          # Root of the causal chain
    causal_chain: List[str] = field(default_factory=list)  # Ordered list of intent IDs

    # === EXECUTION CONSTRAINTS (NEW) ===
    execution_constraints: Dict[str, Any] = field(default_factory=dict)
    # Example: {"max_retries": 3, "timeout_seconds": 300, "allowed_regions": ["us-east-1"]}

    # === CRYPTOGRAPHIC INTEGRITY (NEW) ===
    signature: Optional[str] = None               # Base64-encoded signature
    public_key_fingerprint: Optional[str] = None  # For verification

    # Class constants for validation
    MIN_CONFIDENCE: ClassVar[float] = 0.0
    MAX_CONFIDENCE: ClassVar[float] = 1.0
    MAX_JUSTIFICATION_LENGTH: ClassVar[int] = 5000
    MAX_PARAMETERS_SIZE: ClassVar[int] = 100
    MAX_SIMILAR_INCIDENTS: ClassVar[int] = MAX_SIMILARITY_CACHE
    VERSION: ClassVar[str] = "2.1.0"  # Minor bump for new features
    MAX_INTENT_AGE_SECONDS: ClassVar[int] = 3600  # 1 hour default TTL

    def __post_init__(self) -> None:
        """Validate HealingIntent after initialization and deep-freeze mutable fields."""
        self._validate_oss_boundaries()
        self._validate_risk_integration()
        self._validate_causal_chain()
        self._validate_execution_constraints()

        # Deep freeze metadata and constraints
        if isinstance(self.metadata, dict):
            object.__setattr__(self, 'metadata', _deep_freeze(self.metadata))
        if isinstance(self.execution_constraints, dict):
            object.__setattr__(self, 'execution_constraints', _deep_freeze(self.execution_constraints))

        # Ensure causal_chain is a tuple (immutable)
        if isinstance(self.causal_chain, list):
            object.__setattr__(self, 'causal_chain', tuple(self.causal_chain))

        # Normalize the root intent ID if missing
        if self.root_intent_id is None and self.intent_id:
            object.__setattr__(self, 'root_intent_id', self.intent_id)

    # ------------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------------
    def _validate_oss_boundaries(self) -> None:
        errors: List[str] = []

        # Confidence range
        if not (self.MIN_CONFIDENCE <= self.confidence <= self.MAX_CONFIDENCE):
            errors.append(f"Confidence must be between {self.MIN_CONFIDENCE} and {self.MAX_CONFIDENCE}, got {self.confidence}")

        # Justification length
        if len(self.justification) > self.MAX_JUSTIFICATION_LENGTH:
            errors.append(f"Justification exceeds max length {self.MAX_JUSTIFICATION_LENGTH}")

        # Action and component
        if not self.action or not self.action.strip():
            errors.append("Action cannot be empty")
        if not self.component or not self.component.strip():
            errors.append("Component cannot be empty")

        # Parameters size
        if len(self.parameters) > self.MAX_PARAMETERS_SIZE:
            errors.append(f"Too many parameters: {len(self.parameters)} > {self.MAX_PARAMETERS_SIZE}")

        # JSON serializable check
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

        # Similar incidents
        if self.similar_incidents and len(self.similar_incidents) > self.MAX_SIMILAR_INCIDENTS:
            errors.append(f"Too many similar incidents: {len(self.similar_incidents)} > {self.MAX_SIMILAR_INCIDENTS}")

        # OSS edition restrictions
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
        if self.parent_intent_id and self.parent_intent_id not in self.causal_chain:
            raise ValidationError("parent_intent_id must be present in causal_chain")
        if self.root_intent_id and self.root_intent_id != (self.causal_chain[0] if self.causal_chain else self.intent_id):
            raise ValidationError("root_intent_id must match first element of causal_chain")

    def _validate_execution_constraints(self) -> None:
        # Basic type checks; deeper validation is application-specific
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
        """
        Deterministic ID for idempotency based on action + component + parameters (excluding temporal fields).
        This ensures the same intent yields the same ID regardless of creation time.
        """
        data = {
            "action": self.action,
            "component": self.component,
            "parameters": self._normalize_parameters(self.parameters),
            "incident_id": self.incident_id,
            # detected_at intentionally omitted for idempotency
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
        """Check if the intent has exceeded its TTL."""
        return self.age_seconds > self.MAX_INTENT_AGE_SECONDS

    @property
    def schema_hash(self) -> str:
        """
        Return a hash of the intent's serialized data (excluding OSS context) for contract verification.
        """
        data = self.to_dict(include_oss_context=False)
        # Remove fields that might change across versions (signature, IDs that may vary)
        data.pop("signature", None)
        data.pop("public_key_fingerprint", None)
        data.pop("execution_id", None)
        data.pop("executed_at", None)
        data.pop("execution_result", None)
        data.pop("enterprise_metadata", None)
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    @property
    def expected_value(self) -> float:
        """
        Compute a simple expected value for decision policy.
        Higher value indicates more desirable to approve (or execute).
        Formula: confidence * (1 - risk) - normalized_cost (cost scaled to [0,1]).
        """
        # Normalize cost projection: assume max cost $10,000 for now (configurable)
        max_cost = 10000.0
        normalized_cost = min(1.0, (self.cost_projection or 0.0) / max_cost)
        risk = self.risk_score or 0.0
        return self.confidence * (1 - risk) - normalized_cost

    @property
    def is_executable(self) -> bool:
        """Check if intent is ready for execution"""
        if self.oss_edition == OSS_EDITION:
            return False
        if self.is_stale:
            return False
        return self.status in [
            IntentStatus.CREATED,
            IntentStatus.PENDING_EXECUTION,
            IntentStatus.APPROVED
        ]

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

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------
    def sign(self, private_key: Any) -> "HealingIntent":
        """
        Sign the intent using the provided private key.
        Returns a new signed intent.
        """
        data_to_sign = self._get_signable_data()
        signature = private_key.sign(
            data_to_sign.encode(),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        import base64
        b64_sig = base64.b64encode(signature).decode()
        return HealingIntent(
            **{**asdict(self),
               "signature": b64_sig,
               "public_key_fingerprint": self._get_public_key_fingerprint(private_key.public_key())}
        )

    def verify(self, public_key: Any) -> bool:
        """Verify the signature using the provided public key."""
        if not self.signature:
            return False
        data_to_verify = self._get_signable_data()
        import base64
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
        """Return a canonical string of the intent's core fields for signing."""
        # Omit signature, public key fingerprint, and any execution ephemera
        data = {
            "intent_id": self.intent_id,
            "deterministic_id": self.deterministic_id,
            "action": self.action,
            "component": self.component,
            "parameters": self.parameters,
            "justification": self.justification,
            "incident_id": self.incident_id,
            "detected_at": self.detected_at,
            "created_at": self.created_at,
            "risk_score": self.risk_score,
            "cost_projection": self.cost_projection,
            "source": self.source.value,
            "policy_violations": self.policy_violations,
            "parent_intent_id": self.parent_intent_id,
            "root_intent_id": self.root_intent_id,
            "causal_chain": list(self.causal_chain),
            "execution_constraints": dict(self.execution_constraints),
            "metadata": dict(self.metadata),
        }
        # Sort keys for deterministic output
        json_str = json.dumps(data, sort_keys=True, default=str)
        return json_str

    def _get_public_key_fingerprint(self, public_key: Any) -> str:
        """Return a SHA-256 fingerprint of the public key."""
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(pem).hexdigest()[:16]

    # ------------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------------
    def to_enterprise_request(self) -> Dict[str, Any]:
        """Convert to Enterprise API request format (excludes OSS context)."""
        return {
            "intent_id": self.deterministic_id,
            "action": self.action,
            "component": self.component,
            "parameters": self.parameters,
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
            "causal_chain": list(self.causal_chain),
            "execution_constraints": dict(self.execution_constraints),
            "signature": self.signature,
            "public_key_fingerprint": self.public_key_fingerprint,
            "oss_metadata": {
                "similar_incidents_count": len(self.similar_incidents) if self.similar_incidents else 0,
                "rag_similarity_score": self.rag_similarity_score,
                "has_reasoning_chain": self.reasoning_chain is not None,
                "source": self.source.value,
                "is_oss_advisory": self.is_oss_advisory,
                "risk_factors": self.risk_factors,
                "policy_violations_count": len(self.policy_violations) if self.policy_violations else 0,
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
        data = asdict(self)
        # Convert enums to strings
        if "source" in data and isinstance(data["source"], IntentSource):
            data["source"] = self.source.value
        if "status" in data and isinstance(data["status"], IntentStatus):
            data["status"] = self.status.value
        if "recommended_action" in data and isinstance(data["recommended_action"], RecommendedAction):
            data["recommended_action"] = self.recommended_action.value if self.recommended_action else None

        # Convert proxy objects to dicts for serialization
        if "metadata" in data and isinstance(data["metadata"], Mapping):
            data["metadata"] = dict(data["metadata"])
        if "execution_constraints" in data and isinstance(data["execution_constraints"], Mapping):
            data["execution_constraints"] = dict(data["execution_constraints"])
        if "causal_chain" in data and isinstance(data["causal_chain"], tuple):
            data["causal_chain"] = list(data["causal_chain"])

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
        data["schema_hash"] = self.schema_hash

        return data

    # ------------------------------------------------------------------------
    # Builder methods (return new intents)
    # ------------------------------------------------------------------------
    def with_execution_result(
        self,
        execution_id: str,
        executed_at: float,
        result: Dict[str, Any],
        status: IntentStatus = IntentStatus.COMPLETED,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "HealingIntent":
        return HealingIntent(
            **{**asdict(self),
               "status": status,
               "execution_id": execution_id,
               "executed_at": executed_at,
               "execution_result": result,
               "enterprise_metadata": {**(self.enterprise_metadata or {}), **(metadata or {})}
            }
        )

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
        return HealingIntent(
            **{**asdict(self),
               "status": IntentStatus.APPROVED_WITH_OVERRIDES if overrides else IntentStatus.APPROVED,
               "human_overrides": new_overrides,
               "approvals": new_approvals,
               "comments": new_comments
            }
        )

    def mark_as_sent_to_enterprise(self) -> "HealingIntent":
        return HealingIntent(**{**asdict(self), "status": IntentStatus.PENDING_EXECUTION})

    def mark_as_oss_advisory(self) -> "HealingIntent":
        return HealingIntent(**{**asdict(self), "status": IntentStatus.OSS_ADVISORY_ONLY, "execution_allowed": False})

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
        execution_constraints: Optional[Dict[str, Any]] = None,
    ) -> "HealingIntent":
        infrastructure_intent_id = getattr(infrastructure_intent, 'intent_id', None)
        if hasattr(infrastructure_intent, 'model_dump'):
            intent_dict = infrastructure_intent.model_dump()
        elif hasattr(infrastructure_intent, 'to_dict'):
            intent_dict = infrastructure_intent.to_dict()
        else:
            intent_dict = {"type": str(type(infrastructure_intent))}

        causal_chain = []
        if parent_intent_id:
            causal_chain.append(parent_intent_id)
        causal_chain.append(cls.intent_id)  # This will be overwritten; we'll adjust after creation

        intent = cls(
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
            execution_constraints=execution_constraints or {},
            # root_intent_id will be set in __post_init__
            # causal_chain will be normalized later
        )
        # Fix causal_chain after we have the final intent_id
        new_chain = list(causal_chain)
        if new_chain and new_chain[-1] != intent.intent_id:
            new_chain.append(intent.intent_id)
        object.__setattr__(intent, 'causal_chain', tuple(new_chain))
        if parent_intent_id and not intent.root_intent_id:
            object.__setattr__(intent, 'root_intent_id', new_chain[0])
        return intent

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
        execution_constraints: Optional[Dict[str, Any]] = None,
    ) -> "HealingIntent":
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

        causal_chain = []
        if parent_intent_id:
            causal_chain.append(parent_intent_id)
        # The current intent ID will be appended later

        intent = cls(
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
            execution_constraints=execution_constraints or {},
        )
        # After creation, append own ID to causal chain
        new_chain = list(causal_chain)
        new_chain.append(intent.intent_id)
        object.__setattr__(intent, 'causal_chain', tuple(new_chain))
        if parent_intent_id and not intent.root_intent_id:
            object.__setattr__(intent, 'root_intent_id', new_chain[0])
        return intent

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
        clean_data = data.copy()
        if "source" in clean_data and isinstance(clean_data["source"], str):
            clean_data["source"] = IntentSource(clean_data["source"])
        if "status" in clean_data and isinstance(clean_data["status"], str):
            clean_data["status"] = IntentStatus(clean_data["status"])
        if "recommended_action" in clean_data and isinstance(clean_data["recommended_action"], str):
            try:
                clean_data["recommended_action"] = RecommendedAction(clean_data["recommended_action"])
            except ValueError:
                clean_data["recommended_action"] = None
        # Remove computed fields
        for field in ["deterministic_id", "age_seconds", "is_executable", "is_oss_advisory",
                      "requires_enterprise_upgrade", "version", "confidence_interval",
                      "is_stale", "expected_value", "schema_hash"]:
            clean_data.pop(field, None)
        return cls(**clean_data)

    # ------------------------------------------------------------------------
    # Helper: parameter normalization (preserve order for lists)
    # ------------------------------------------------------------------------
    def _normalize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in sorted(params.items()):
            normalized[key] = self._normalize_value(value)
        return normalized

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, set):
            # Sets are unordered -> sort
            return tuple(sorted(self._normalize_value(v) for v in value))
        elif isinstance(value, (list, tuple)):
            # Preserve order for sequences
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

    # ------------------------------------------------------------------------
    # OSS context and summary
    # ------------------------------------------------------------------------
    def get_oss_context(self) -> Dict[str, Any]:
        return {
            "reasoning_chain": self.reasoning_chain,
            "similar_incidents": self.similar_incidents,
            "rag_similarity_score": self.rag_similarity_score,
            "decision_tree": self.decision_tree,
            "alternative_actions": self.alternative_actions,
            "analysis_timestamp": datetime.fromtimestamp(self.detected_at).isoformat(),
            "source": self.source.value,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "oss_edition": self.oss_edition,
            "is_oss_advisory": self.is_oss_advisory,
            "infrastructure_intent": self.infrastructure_intent,
            "metadata": dict(self.metadata),
            "parent_intent_id": self.parent_intent_id,
            "root_intent_id": self.root_intent_id,
            "causal_chain": list(self.causal_chain),
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
            "causal_chain_length": len(self.causal_chain),
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

    def is_immutable(self) -> bool:
        try:
            object.__setattr__(self, '_test_immutable', True)
            return False
        except Exception:
            return True

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
                        "causal_chain_length": len(intent.causal_chain),
                        "expected_value": intent.expected_value,
                        "is_stale": intent.is_stale,
                    }
                }
            elif version in ("2.0.0", "1.1.0", "1.0.0"):
                # Fallback to earlier serialization (simplified)
                data = intent.to_dict(include_oss_context=True)
                for field in ["metadata", "parent_intent_id", "root_intent_id", "causal_chain",
                              "execution_constraints", "signature", "public_key_fingerprint"]:
                    data.pop(field, None)
                if version.startswith("1."):
                    data.pop("confidence_distribution", None)
                    data.pop("risk_score", None)
                    data.pop("risk_factors", None)
                    data.pop("cost_projection", None)
                    data.pop("cost_confidence_interval", None)
                    data.pop("recommended_action", None)
                    data.pop("decision_tree", None)
                    data.pop("alternative_actions", None)
                    data.pop("risk_profile", None)
                    data.pop("human_overrides", None)
                    data.pop("approvals", None)
                    data.pop("comments", None)
                    data.pop("infrastructure_intent_id", None)
                    data.pop("policy_violations", None)
                    data.pop("infrastructure_intent", None)
                    data.pop("confidence_interval", None)
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
                    # Add default values for v2 fields
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
                    intent_data.setdefault("causal_chain", [])
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
        serialized = cls.serialize(intent)
        return json.dumps(serialized, indent=2 if pretty else None, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> HealingIntent:
        data = json.loads(json_str)
        return cls.deserialize(data)

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
# Factory functions for common intents (unchanged except for new parameters)
# ============================================================================
def create_infrastructure_healing_intent(
    infrastructure_result: Any,
    action_mapping: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    parent_intent_id: Optional[str] = None,
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
) -> HealingIntent:
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
