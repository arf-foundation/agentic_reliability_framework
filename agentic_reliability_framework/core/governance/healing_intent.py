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


class HealingIntentError(Exception):
    """Base exception for HealingIntent errors"""
    pass


class SerializationError(HealingIntentError):
    """Error during serialization/deserialization"""
    pass


class ValidationError(HealingIntentError):
    """Error during intent validation"""
    pass


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


class ConfidenceDistribution:
    """
    Probabilistic confidence representation.

    Instead of a single confidence score, this represents a distribution
    of possible confidence values, allowing for uncertainty quantification.
    Matches patterns from the risk_engine.py module.
    """

    def __init__(self, mean: float, std: float = 0.05, samples: Optional[List[float]] = None):
        self.mean = max(0.0, min(mean, 1.0))
        self.std = max(0.0, min(std, 0.5))
        self._samples = samples or list(np.random.normal(self.mean, self.std, 1000).clip(0, 1))

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


@dataclass(frozen=True, slots=True)
class HealingIntent:
    """
    OSS-generated healing recommendation for Enterprise execution

    Enhanced with:
    - Probabilistic confidence distributions
    - Risk score and cost projection integration
    - Decision tree tracking for explainability
    - Human override audit trail
    - Partial execution support
    - Integration with infrastructure governance module
    - Full backward compatibility with old ARF patterns

    This is the clean boundary between OSS intelligence and Enterprise execution:
    - OSS creates HealingIntent through analysis (advisory only)
    - Enterprise executes HealingIntent through execution gateway
    - Immutable (frozen) to ensure consistency across OSS→Enterprise handoff
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

    # NEW: Extended metadata for governance loop outputs
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Class constants for validation
    MIN_CONFIDENCE: ClassVar[float] = 0.0
    MAX_CONFIDENCE: ClassVar[float] = 1.0
    MAX_JUSTIFICATION_LENGTH: ClassVar[int] = 5000
    MAX_PARAMETERS_SIZE: ClassVar[int] = 100
    MAX_SIMILAR_INCIDENTS: ClassVar[int] = MAX_SIMILARITY_CACHE
    VERSION: ClassVar[str] = "2.0.0"  # Major bump for probabilistic features

    def __post_init__(self) -> None:
        """Validate HealingIntent after initialization with OSS boundaries"""
        self._validate_oss_boundaries()
        self._validate_risk_integration()

        # NEW: Replace metadata dict with immutable proxy
        # We use object.__setattr__ because the dataclass is frozen
        if isinstance(self.metadata, dict):
            object.__setattr__(self, 'metadata', MappingProxyType(self.metadata))

    def _validate_oss_boundaries(self) -> None:
        """Validate all fields against OSS limits"""
        errors: List[str] = []

        # Validate confidence range
        if not (self.MIN_CONFIDENCE <= self.confidence <= self.MAX_CONFIDENCE):
            errors.append(
                f"Confidence must be between {self.MIN_CONFIDENCE} and "
                f"{self.MAX_CONFIDENCE}, got {self.confidence}"
            )

        # Validate justification length
        if len(self.justification) > self.MAX_JUSTIFICATION_LENGTH:
            errors.append(
                f"Justification exceeds max length {self.MAX_JUSTIFICATION_LENGTH}"
            )

        # Validate action and component
        if not self.action or not self.action.strip():
            errors.append("Action cannot be empty")

        if not self.component or not self.component.strip():
            errors.append("Component cannot be empty")

        # Validate parameters size
        if len(self.parameters) > self.MAX_PARAMETERS_SIZE:
            errors.append(
                f"Too many parameters: {len(self.parameters)} > {self.MAX_PARAMETERS_SIZE}"
            )

        # Validate parameters are JSON serializable
        try:
            json.dumps(self.parameters)
        except (TypeError, ValueError) as e:
            errors.append(f"Parameters must be JSON serializable: {e}")

        # Validate metadata is JSON serializable (proxy works same as dict for dumps)
        try:
            json.dumps(dict(self.metadata))
        except (TypeError, ValueError) as e:
            errors.append(f"Metadata must be JSON serializable: {e}")

        # Validate similar incidents
        if self.similar_incidents:
            if len(self.similar_incidents) > self.MAX_SIMILAR_INCIDENTS:
                errors.append(
                    f"Too many similar incidents: {len(self.similar_incidents)} > "
                    f"{self.MAX_SIMILAR_INCIDENTS}"
                )

        # Validate OSS edition restrictions
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
            raise ValidationError(
                f"HealingIntent validation failed:\n" +
                "\n".join(f"  • {error}" for error in errors)
            )

    def _validate_risk_integration(self) -> None:
        """Validate that risk and cost fields are consistent"""
        if self.risk_score is not None:
            if not (0.0 <= self.risk_score <= 1.0):
                raise ValidationError(f"Risk score must be between 0 and 1, got {self.risk_score}")

        if self.cost_projection is not None and self.cost_projection < 0:
            raise ValidationError(f"Cost projection cannot be negative, got {self.cost_projection}")

        if self.cost_confidence_interval is not None:
            low, high = self.cost_confidence_interval
            if low > high:
                raise ValidationError(f"Invalid confidence interval: [{low}, {high}]")

    @property
    def deterministic_id(self) -> str:
        """
        Deterministic ID for idempotency based on action + component + parameters

        This ensures the same action on the same component with the same parameters
        generates the same intent ID, preventing duplicate executions.
        """
        data = {
            "action": self.action,
            "component": self.component,
            "parameters": self._normalize_parameters(self.parameters),
            "incident_id": self.incident_id,
            "detected_at": int(self.detected_at),
            "oss_edition": self.oss_edition,
        }

        # Sort keys for deterministic JSON
        json_str = json.dumps(data, sort_keys=True, default=str)

        # Create hash-based ID
        hash_digest = hashlib.sha256(json_str.encode()).hexdigest()
        return f"intent_{hash_digest[:16]}"

    @property
    def age_seconds(self) -> float:
        """Get age of intent in seconds"""
        return time.time() - self.created_at

    @property
    def is_executable(self) -> bool:
        """Check if intent is ready for execution"""
        # In OSS edition, nothing is executable
        if self.oss_edition == OSS_EDITION:
            return False

        return self.status in [
            IntentStatus.CREATED,
            IntentStatus.PENDING_EXECUTION,
            IntentStatus.APPROVED
        ]

    @property
    def is_oss_advisory(self) -> bool:
        """Check if this is an OSS advisory-only intent"""
        return self.oss_edition == OSS_EDITION and not self.execution_allowed

    @property
    def requires_enterprise_upgrade(self) -> bool:
        """Check if intent requires Enterprise upgrade"""
        return self.requires_enterprise and self.oss_edition == OSS_EDITION

    @property
    def confidence_interval(self) -> Optional[Tuple[float, float]]:
        """Get confidence interval if distribution is available"""
        if self.confidence_distribution:
            return (self.confidence_distribution.get("p5", self.confidence),
                   self.confidence_distribution.get("p95", self.confidence))
        return None

    def to_enterprise_request(self) -> Dict[str, Any]:
        """
        Convert to Enterprise API request format

        Returns only the data needed for Enterprise execution.
        OSS analysis context stays in OSS.
        """
        return {
            # Core execution fields
            "intent_id": self.deterministic_id,
            "action": self.action,
            "component": self.component,
            "parameters": self.parameters,
            "justification": self.justification,

            # OSS metadata for Enterprise context
            "confidence": self.confidence,
            "confidence_interval": self.confidence_interval,
            "risk_score": self.risk_score,
            "cost_projection": self.cost_projection,
            "incident_id": self.incident_id,
            "detected_at": self.detected_at,
            "created_at": self.created_at,
            "source": self.source.value,
            "recommended_action": self.recommended_action.value if self.recommended_action else None,

            # OSS edition information
            "oss_edition": self.oss_edition,
            "oss_license": self.oss_license,
            "requires_enterprise": self.requires_enterprise,
            "execution_allowed": self.execution_allowed,
            "version": self.VERSION,

            # Minimal OSS context (for debugging only)
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

            # Upgrade information
            "upgrade_url": ENTERPRISE_UPGRADE_URL,
            "enterprise_features": [
                "autonomous_execution",
                "approval_workflows",
                "persistent_storage",
                "learning_engine",
                "audit_trails",
                "compliance_reports",
                "multi_tenant_support",
                "sso_integration",
                "24_7_support",
                "probabilistic_confidence",
                "risk_analytics",
                "cost_optimization"
            ]
        }

    def _get_confidence_basis(self) -> str:
        """Determine confidence basis based on available data"""
        if self.recommended_action == RecommendedAction.DENY and self.policy_violations:
            return "policy_violation"
        if self.rag_similarity_score and self.rag_similarity_score > SIMILARITY_THRESHOLD:
            return "historical_similarity"
        if self.risk_score is not None:
            return "risk_based"
        return "policy_only"

    def to_dict(self, include_oss_context: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization

        Args:
            include_oss_context: Whether to include OSS analysis context
                (should be False when sending to Enterprise)

        Returns:
            Dictionary representation of the intent
        """
        data = asdict(self)

        # Convert enums to strings
        if "source" in data and isinstance(data["source"], IntentSource):
            data["source"] = self.source.value
        if "status" in data and isinstance(data["status"], IntentStatus):
            data["status"] = self.status.value
        if "recommended_action" in data and isinstance(data["recommended_action"], RecommendedAction):
            data["recommended_action"] = self.recommended_action.value if self.recommended_action else None

        # Convert metadata proxy to dict for serialization
        if "metadata" in data and isinstance(data["metadata"], Mapping):
            data["metadata"] = dict(data["metadata"])

        # Remove OSS context if not needed
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

        return data

    def with_execution_result(
        self,
        execution_id: str,
        executed_at: float,
        result: Dict[str, Any],
        status: IntentStatus = IntentStatus.COMPLETED,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "HealingIntent":
        """
        Create a new HealingIntent with execution results (used by Enterprise)

        This is how Enterprise updates the intent after execution.
        Returns a new immutable intent with execution results.
        """
        # Create a new dataclass with updated fields
        return HealingIntent(
            # Core fields (copied)
            action=self.action,
            component=self.component,
            parameters=self.parameters,
            justification=self.justification,
            confidence=self.confidence,
            confidence_distribution=self.confidence_distribution,
            incident_id=self.incident_id,
            detected_at=self.detected_at,

            # Risk and cost fields (copied)
            risk_score=self.risk_score,
            risk_factors=self.risk_factors,
            cost_projection=self.cost_projection,
            cost_confidence_interval=self.cost_confidence_interval,
            recommended_action=self.recommended_action,

            # Decision tracking (copied)
            decision_tree=self.decision_tree,
            alternative_actions=self.alternative_actions,
            risk_profile=self.risk_profile,

            # OSS context (copied)
            reasoning_chain=self.reasoning_chain,
            similar_incidents=self.similar_incidents,
            rag_similarity_score=self.rag_similarity_score,
            source=self.source,

            # Identifiers (copied)
            intent_id=self.intent_id,
            created_at=self.created_at,

            # OSS metadata (copied)
            oss_edition=self.oss_edition,
            oss_license=self.oss_license,
            requires_enterprise=self.requires_enterprise,
            execution_allowed=self.execution_allowed,

            # Infrastructure integration (copied)
            infrastructure_intent_id=self.infrastructure_intent_id,
            policy_violations=self.policy_violations,
            infrastructure_intent=self.infrastructure_intent,

            # NEW: copy metadata
            metadata=self.metadata,

            # Updated execution fields
            status=status,
            execution_id=execution_id,
            executed_at=executed_at,
            execution_result=result,
            enterprise_metadata={**(self.enterprise_metadata or {}), **(metadata or {})},

            # Human interaction (copied)
            human_overrides=self.human_overrides,
            approvals=self.approvals,
            comments=self.comments
        )

    def with_human_approval(
        self,
        approver: str,
        approval_time: float,
        comments: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> "HealingIntent":
        """
        Record human approval with optional overrides.

        Returns a new intent with approval recorded and status updated.
        """
        approval_record = {
            "approver": approver,
            "timestamp": approval_time,
            "comments": comments,
            "overrides": overrides
        }

        new_overrides = list(self.human_overrides)
        if overrides:
            new_overrides.append({
                "overrider": approver,
                "timestamp": approval_time,
                "overrides": overrides,
                "reason": comments
            })

        new_approvals = list(self.approvals)
        new_approvals.append(approval_record)

        new_comments = list(self.comments)
        if comments:
            new_comments.append({
                "author": approver,
                "timestamp": approval_time,
                "comment": comments
            })

        return HealingIntent(
            # Copy all fields
            action=self.action,
            component=self.component,
            parameters=self.parameters,
            justification=self.justification,
            confidence=self.confidence,
            confidence_distribution=self.confidence_distribution,
            incident_id=self.incident_id,
            detected_at=self.detected_at,
            risk_score=self.risk_score,
            risk_factors=self.risk_factors,
            cost_projection=self.cost_projection,
            cost_confidence_interval=self.cost_confidence_interval,
            recommended_action=self.recommended_action,
            decision_tree=self.decision_tree,
            alternative_actions=self.alternative_actions,
            risk_profile=self.risk_profile,
            reasoning_chain=self.reasoning_chain,
            similar_incidents=self.similar_incidents,
            rag_similarity_score=self.rag_similarity_score,
            source=self.source,
            intent_id=self.intent_id,
            created_at=self.created_at,
            status=IntentStatus.APPROVED_WITH_OVERRIDES if overrides else IntentStatus.APPROVED,
            execution_id=self.execution_id,
            executed_at=self.executed_at,
            execution_result=self.execution_result,
            enterprise_metadata=self.enterprise_metadata,
            human_overrides=new_overrides,
            approvals=new_approvals,
            comments=new_comments,
            oss_edition=self.oss_edition,
            oss_license=self.oss_license,
            requires_enterprise=self.requires_enterprise,
            execution_allowed=self.execution_allowed,
            infrastructure_intent_id=self.infrastructure_intent_id,
            policy_violations=self.policy_violations,
            infrastructure_intent=self.infrastructure_intent,
            metadata=self.metadata  # NEW: copy metadata
        )

    def mark_as_sent_to_enterprise(self) -> "HealingIntent":
        """
        Mark intent as sent to Enterprise (used by OSS)

        Returns a new intent with status updated to PENDING_EXECUTION
        """
        return HealingIntent(
            # Copy all fields
            action=self.action,
            component=self.component,
            parameters=self.parameters,
            justification=self.justification,
            confidence=self.confidence,
            confidence_distribution=self.confidence_distribution,
            incident_id=self.incident_id,
            detected_at=self.detected_at,
            risk_score=self.risk_score,
            risk_factors=self.risk_factors,
            cost_projection=self.cost_projection,
            cost_confidence_interval=self.cost_confidence_interval,
            recommended_action=self.recommended_action,
            decision_tree=self.decision_tree,
            alternative_actions=self.alternative_actions,
            risk_profile=self.risk_profile,
            reasoning_chain=self.reasoning_chain,
            similar_incidents=self.similar_incidents,
            rag_similarity_score=self.rag_similarity_score,
            source=self.source,
            intent_id=self.intent_id,
            created_at=self.created_at,
            status=IntentStatus.PENDING_EXECUTION,
            execution_id=self.execution_id,
            executed_at=self.executed_at,
            execution_result=self.execution_result,
            enterprise_metadata=self.enterprise_metadata,
            human_overrides=self.human_overrides,
            approvals=self.approvals,
            comments=self.comments,
            oss_edition=self.oss_edition,
            oss_license=self.oss_license,
            requires_enterprise=self.requires_enterprise,
            execution_allowed=self.execution_allowed,
            infrastructure_intent_id=self.infrastructure_intent_id,
            policy_violations=self.policy_violations,
            infrastructure_intent=self.infrastructure_intent,
            metadata=self.metadata  # NEW: copy metadata
        )

    def mark_as_oss_advisory(self) -> "HealingIntent":
        """
        Mark intent as OSS advisory only

        Used when OSS creates an intent that can only be advisory
        """
        return HealingIntent(
            # Copy all fields
            action=self.action,
            component=self.component,
            parameters=self.parameters,
            justification=self.justification,
            confidence=self.confidence,
            confidence_distribution=self.confidence_distribution,
            incident_id=self.incident_id,
            detected_at=self.detected_at,
            risk_score=self.risk_score,
            risk_factors=self.risk_factors,
            cost_projection=self.cost_projection,
            cost_confidence_interval=self.cost_confidence_interval,
            recommended_action=self.recommended_action,
            decision_tree=self.decision_tree,
            alternative_actions=self.alternative_actions,
            risk_profile=self.risk_profile,
            reasoning_chain=self.reasoning_chain,
            similar_incidents=self.similar_incidents,
            rag_similarity_score=self.rag_similarity_score,
            source=self.source,
            intent_id=self.intent_id,
            created_at=self.created_at,
            status=IntentStatus.OSS_ADVISORY_ONLY,
            execution_id=self.execution_id,
            executed_at=self.executed_at,
            execution_result=self.execution_result,
            enterprise_metadata=self.enterprise_metadata,
            human_overrides=self.human_overrides,
            approvals=self.approvals,
            comments=self.comments,
            oss_edition=self.oss_edition,
            oss_license=self.oss_license,
            requires_enterprise=self.requires_enterprise,
            execution_allowed=False,  # Force no execution in OSS
            infrastructure_intent_id=self.infrastructure_intent_id,
            policy_violations=self.policy_violations,
            infrastructure_intent=self.infrastructure_intent,
            metadata=self.metadata  # NEW: copy metadata
        )

    @classmethod
    def from_infrastructure_intent(
        cls,
        infrastructure_intent: Any,  # InfrastructureIntent type
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
        metadata: Optional[Dict[str, Any]] = None,  # NEW: add metadata parameter
    ) -> "HealingIntent":
        """
        Create HealingIntent from infrastructure module analysis.

        This bridges the infrastructure governance module with the healing system.
        """
        # Extract intent_id if available
        infrastructure_intent_id = getattr(infrastructure_intent, 'intent_id', None)

        # Convert infrastructure intent to dict for storage
        if hasattr(infrastructure_intent, 'model_dump'):
            intent_dict = infrastructure_intent.model_dump()
        elif hasattr(infrastructure_intent, 'to_dict'):
            intent_dict = infrastructure_intent.to_dict()
        else:
            intent_dict = {"type": str(type(infrastructure_intent))}

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
            metadata=metadata or {},  # NEW: store metadata
            oss_edition=OSS_EDITION,
            requires_enterprise=True,
            execution_allowed=False
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
        # NEW: metadata and policy_violations parameters
        metadata: Optional[Dict[str, Any]] = None,
        policy_violations: Optional[List[str]] = None,
    ) -> "HealingIntent":
        """
        Factory method for creating HealingIntent from OSS analysis

        This is the primary way OSS creates intents.
        Enhanced with probabilistic confidence and risk integration.
        """
        # Apply OSS limits to similar incidents
        if similar_incidents and len(similar_incidents) > cls.MAX_SIMILAR_INCIDENTS:
            similar_incidents = similar_incidents[:cls.MAX_SIMILAR_INCIDENTS]

        # Create confidence distribution
        conf_dist = ConfidenceDistribution(confidence, confidence_std)

        # Calculate enhanced confidence based on similar incidents
        enhanced_confidence = confidence
        if similar_incidents:
            similarity_scores = [
                inc.get("similarity", 0.0)
                for inc in similar_incidents
                if "similarity" in inc
            ]
            if similarity_scores:
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                # Cap the boost to prevent overconfidence
                confidence_boost = min(0.2, avg_similarity * 0.3)
                enhanced_confidence = min(confidence * (1.0 + confidence_boost), cls.MAX_CONFIDENCE)

        # Use provided RAG score or calculate from similar incidents
        final_rag_score = rag_similarity_score
        if final_rag_score is None and similar_incidents and len(similar_incidents) > 0:
            # Take average of top 3 similarities
            top_similarities = [
                inc.get("similarity", 0.0)
                for inc in similar_incidents[:3]
                if "similarity" in inc
            ]
            if top_similarities:
                final_rag_score = sum(top_similarities) / len(top_similarities)

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
            # NEW: store metadata and policy_violations
            metadata=metadata or {},
            policy_violations=policy_violations or [],
            oss_edition=OSS_EDITION,
            requires_enterprise=True,
            execution_allowed=False,
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
        """
        Create HealingIntent from RAG graph recommendation

        Specialized factory for RAG-based recommendations
        """
        if not similar_incidents:
            raise ValidationError("RAG recommendation requires similar incidents")

        # Calculate success rate if not provided
        if success_rate is None:
            if len(similar_incidents) == 0:
                success_rate = 0.0
            else:
                successful = sum(1 for inc in similar_incidents if inc.get("success", False))
                success_rate = successful / len(similar_incidents)

        # Generate justification
        justification = justification_template.format(
            count=len(similar_incidents),
            success_rate=success_rate or 0.0,
            action=action,
            component=component,
        )

        # Calculate confidence based on RAG similarity
        base_confidence = rag_similarity_score * 0.8  # Scale similarity to confidence
        if success_rate:
            base_confidence = base_confidence * (0.7 + success_rate * 0.3)

        return cls.from_analysis(
            action=action,
            component=component,
            parameters=parameters,
            justification=justification,
            confidence=min(base_confidence, 0.95),  # Cap at 95%
            similar_incidents=similar_incidents,
            incident_id=similar_incidents[0].get("incident_id", "") if similar_incidents else "",
            source=IntentSource.RAG_SIMILARITY,
            rag_similarity_score=rag_similarity_score,
            risk_score=risk_score,
            cost_projection=cost_projection,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealingIntent":
        """
        Create from dictionary (deserialize)

        Handles versioning and field conversion
        """
        # Handle versioning
        version = data.get("version", "1.0.0")

        # Create a copy to avoid mutating input
        clean_data = data.copy()

        # Convert string enums back to Enum instances
        if "source" in clean_data and isinstance(clean_data["source"], str):
            clean_data["source"] = IntentSource(clean_data["source"])

        if "status" in clean_data and isinstance(clean_data["status"], str):
            clean_data["status"] = IntentStatus(clean_data["status"])

        if "recommended_action" in clean_data and isinstance(clean_data["recommended_action"], str):
            try:
                clean_data["recommended_action"] = RecommendedAction(clean_data["recommended_action"])
            except ValueError:
                clean_data["recommended_action"] = None

        # Remove computed fields that shouldn't be in constructor
        clean_data.pop("deterministic_id", None)
        clean_data.pop("age_seconds", None)
        clean_data.pop("is_executable", None)
        clean_data.pop("is_oss_advisory", None)
        clean_data.pop("requires_enterprise_upgrade", None)
        clean_data.pop("version", None)
        clean_data.pop("confidence_interval", None)

        return cls(**clean_data)

    def _normalize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameters for deterministic hashing

        Ensures that parameter order and minor format differences
        don't affect the deterministic ID.
        """
        normalized: Dict[str, Any] = {}

        for key, value in sorted(params.items()):
            normalized[key] = self._normalize_value(value)

        return normalized

    def _normalize_value(self, value: Any) -> Any:
        """Normalize a single value for hashing"""
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple, set)):
            # Convert all iterables to sorted tuples
            normalized_items = tuple(
                sorted(
                    self._normalize_value(v) for v in value
                )
            )
            return normalized_items
        elif isinstance(value, dict):
            # Recursively normalize dicts
            return self._normalize_parameters(value)
        elif hasattr(value, '__dict__'):
            # Handle objects with __dict__
            return self._normalize_parameters(value.__dict__)
        else:
            # Convert to string representation for other types
            try:
                return str(value)
            except Exception:
                # Fallback for objects that can't be stringified
                return f"<unserializable:{type(value).__name__}>"

    def get_oss_context(self) -> Dict[str, Any]:
        """
        Get OSS analysis context (stays in OSS)

        This data never leaves the OSS environment for privacy and IP protection.
        """
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
            "metadata": dict(self.metadata),  # Convert proxy to dict for output
        }

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get execution summary (public information)

        Safe to share externally
        """
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
            "policy_violations_count": len(self.policy_violations) if self.policy_violations else 0,
            "confidence_basis": self._get_confidence_basis(),
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
        """Check if the intent is truly immutable (frozen dataclass property)"""
        try:
            # Try to modify a field - should raise FrozenInstanceError
            object.__setattr__(self, '_test_immutable', True)
            return False
        except Exception:
            return True

    def __repr__(self) -> str:
        return (
            f"HealingIntent("
            f"id={self.deterministic_id[:8]}..., "
            f"action={self.action}, "
            f"component={self.component}, "
            f"confidence={self.confidence:.2f}, "
            f"risk={self.risk_score:.2f if self.risk_score else 'N/A'}, "
            f"status={self.status.value}"
            f")"
        )


class HealingIntentSerializer:
    """
    Versioned serialization for HealingIntent

    Enhanced with:
    - Probabilistic confidence distribution support
    - Risk and cost field serialization
    - Backward compatibility with v1.x
    - OSS/Enterprise edition detection
    """

    SCHEMA_VERSION: ClassVar[str] = "2.0.0"

    @classmethod
    def serialize(cls, intent: HealingIntent, version: str = "2.0.0") -> Dict[str, Any]:
        """
        Serialize HealingIntent with versioning

        Args:
            intent: HealingIntent to serialize
            version: Schema version to use

        Returns:
            Versioned serialization dictionary

        Raises:
            SerializationError: If serialization fails
        """
        try:
            if version == "2.0.0":
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
                        # NEW: add metadata snapshot (keys only to keep size small)
                        "metadata_keys": list(intent.metadata.keys()) if intent.metadata else [],
                    }
                }
            elif version == "1.1.0" or version == "1.0.0":
                # Backward compatibility with v1.x
                data = intent.to_dict(include_oss_context=True)

                # Remove v2.0.0 fields for compatibility
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
                data.pop("confidence_interval", None)  # computed field
                data.pop("metadata", None)             # NEW: remove metadata for v1

                # Ensure status is compatible
                if data.get("status") in [
                    IntentStatus.EXECUTING_PARTIAL.value,
                    IntentStatus.COMPLETED_PARTIAL.value,
                    IntentStatus.ROLLED_BACK.value,
                    IntentStatus.PENDING_APPROVAL.value,
                    IntentStatus.APPROVED.value,
                    IntentStatus.APPROVED_WITH_OVERRIDES.value
                ]:
                    data["status"] = IntentStatus.PENDING_EXECUTION.value

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
        """
        Deserialize HealingIntent with version detection

        Args:
            data: Serialized data

        Returns:
            Deserialized HealingIntent

        Raises:
            SerializationError: If deserialization fails
        """
        try:
            version = data.get("version", "1.0.0")
            intent_data = data.get("data", data)  # Handle both wrapped and unwrapped

            if version in ["2.0.0", "1.1.0", "1.0.0"]:
                # Handle version differences
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
                    intent_data.setdefault("metadata", {})  # NEW: default metadata

                return HealingIntent.from_dict(intent_data)
            else:
                raise SerializationError(f"Unsupported version: {version}")

        except KeyError as e:
            raise SerializationError(f"Missing required field in serialized data: {e}") from e
        except Exception as e:
            raise SerializationError(f"Failed to deserialize HealingIntent: {e}") from e

    @classmethod
    def to_json(cls, intent: HealingIntent, pretty: bool = False) -> str:
        """Convert HealingIntent to JSON string"""
        try:
            serialized = cls.serialize(intent)
            if pretty:
                return json.dumps(serialized, indent=2, default=str)
            else:
                return json.dumps(serialized, default=str)
        except Exception as e:
            raise SerializationError(f"Failed to convert to JSON: {e}") from e

    @classmethod
    def from_json(cls, json_str: str) -> HealingIntent:
        """Create HealingIntent from JSON string"""
        try:
            data = json.loads(json_str)
            return cls.deserialize(data)
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON: {e}") from e
        except Exception as e:
            raise SerializationError(f"Failed to parse JSON: {e}") from e

    @classmethod
    def to_enterprise_json(cls, intent: HealingIntent) -> str:
        """
        Convert to Enterprise-ready JSON (excludes OSS context)

        This is what should be sent to the Enterprise API
        """
        try:
            enterprise_request = intent.to_enterprise_request()
            return json.dumps(enterprise_request, default=str)
        except Exception as e:
            raise SerializationError(f"Failed to create Enterprise JSON: {e}") from e

    @classmethod
    def validate_for_oss(cls, intent: HealingIntent) -> bool:
        """
        Validate that HealingIntent complies with OSS boundaries

        Returns:
            True if intent is valid for OSS edition
        """
        try:
            # Check OSS edition
            if intent.oss_edition != OSS_EDITION:
                return False

            # Check execution restrictions
            if intent.execution_allowed:
                return False

            # Check similar incidents limit
            if intent.similar_incidents and len(intent.similar_incidents) > HealingIntent.MAX_SIMILAR_INCIDENTS:
                return False

            # Check that frozen dataclass property is preserved
            if not intent.is_immutable():
                return False

            # Check that no execution fields are set
            if intent.executed_at is not None or intent.execution_id is not None:
                return False

            return True

        except Exception:
            return False


# Factory functions for common use cases
def create_infrastructure_healing_intent(
    infrastructure_result: Any,  # HealingIntent from infrastructure module
    action_mapping: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None,  # NEW: pass through metadata
) -> HealingIntent:
    """
    Create a healing intent from infrastructure module analysis result.

    This bridges the infrastructure governance module with the main healing system.

    Args:
        infrastructure_result: The HealingIntent from infrastructure.evaluate()
        action_mapping: Optional mapping from infrastructure actions to healing actions
        metadata: Optional metadata to attach to the intent

    Returns:
        HealingIntent ready for the healing system
    """
    # Default action mapping
    if action_mapping is None:
        action_mapping = {
            "approve": "execute",
            "deny": "block",
            "escalate": "escalate",
            "defer": "defer"
        }

    # Extract fields from infrastructure result
    recommended_action = getattr(infrastructure_result, 'recommended_action', None)
    if recommended_action and hasattr(recommended_action, 'value'):
        action = action_mapping.get(recommended_action.value, "review")
    else:
        action = "review"

    # Build parameters
    parameters = {
        "infrastructure_intent_id": getattr(infrastructure_result, 'intent_id', None),
        "risk_score": getattr(infrastructure_result, 'risk_score', None),
        "cost_projection": getattr(infrastructure_result, 'cost_projection', None),
        "policy_violations": getattr(infrastructure_result, 'policy_violations', []),
        "evaluation_details": getattr(infrastructure_result, 'evaluation_details', {})
    }

    # Build justification
    justification_parts = [
        getattr(infrastructure_result, 'justification', "Infrastructure analysis completed"),
    ]

    policy_violations = getattr(infrastructure_result, 'policy_violations', [])
    if policy_violations:
        justification_parts.append(f"Policy violations: {'; '.join(policy_violations)}")

    # Extract cost projection for top-level field
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
        metadata=metadata or {},  # NEW
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
) -> HealingIntent:
    """Create a rollback healing intent with OSS limits"""
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
) -> HealingIntent:
    """Create a container restart healing intent with OSS limits"""
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
) -> HealingIntent:
    """Create a scale-out healing intent with OSS limits"""
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
) -> HealingIntent:
    """
    Create a generic OSS advisory-only intent

    Used when OSS wants to recommend an action without execution capability
    """
    return HealingIntent(
        action=action,
        component=component,
        parameters=parameters,
        justification=justification,
        confidence=confidence,
        incident_id=incident_id,
        risk_score=risk_score,
        cost_projection=cost_projection,
        oss_edition=OSS_EDITION,
        requires_enterprise=True,
        execution_allowed=False,
        status=IntentStatus.OSS_ADVISORY_ONLY,
    )


# Export
__all__ = [
    # Main class
    "HealingIntent",

    # Supporting classes
    "ConfidenceDistribution",
    "HealingIntentSerializer",

    # Enums
    "IntentSource",
    "IntentStatus",
    "RecommendedAction",

    # Exceptions
    "HealingIntentError",
    "SerializationError",
    "ValidationError",

    # Factory functions
    "create_infrastructure_healing_intent",
    "create_rollback_intent",
    "create_restart_intent",
    "create_scale_out_intent",
    "create_oss_advisory_intent",
]
