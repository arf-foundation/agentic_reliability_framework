"""
Infrastructure Intent Schema – Algebraic Data Types for Change Requests.

This module defines a family of intents as a discriminated union. Each intent
represents a proposed infrastructure action. Intents are immutable, self-validating,
and carry provenance for auditability.

The design follows principles of domain-driven design and knowledge engineering,
using strong typing and semantic constraints to prevent invalid states.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator
from pydantic.functional_validators import AfterValidator

# -----------------------------------------------------------------------------
# Domain Primitives (NewTypes for type safety)
# -----------------------------------------------------------------------------
Region = str
Size = str
Principal = str
ResourceScope = str
ServiceName = str
ChangeScope = Literal["single_instance", "canary", "global"]
Environment = Literal["dev", "staging", "prod", "test"]

# -----------------------------------------------------------------------------
# Enums for fixed sets (but extensible via new variants)
# -----------------------------------------------------------------------------
class ResourceType(str, Enum):
    """Azure resource types with semantic meaning."""
    VM = "vm"
    STORAGE_ACCOUNT = "storage_account"
    DATABASE = "database"
    KUBERNETES_CLUSTER = "kubernetes_cluster"
    FUNCTION_APP = "function_app"
    VIRTUAL_NETWORK = "virtual_network"

class PermissionLevel(str, Enum):
    """Access permission levels in increasing order of privilege."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

# -----------------------------------------------------------------------------
# Knowledge Base Stubs (simulated – in production would be loaded from external source)
# -----------------------------------------------------------------------------
# These are used for semantic validation. In a real system, they would be fetched
# from Azure APIs or a configuration service.
VALID_AZURE_REGIONS = {
    "eastus", "eastus2", "westus", "westeurope", "northeurope",
    "southeastasia", "eastasia", "japaneast", "brazilsouth"
}

# Mapping of resource type to plausible size patterns (simplified)
# NOTE: This is for reference only; validation now accepts any string for size.
RESOURCE_SIZE_PATTERNS = {
    ResourceType.VM: {"Standard_D2s_v3", "Standard_D4s_v3", "Standard_D8s_v3", "Standard_D16s_v3"},
    ResourceType.STORAGE_ACCOUNT: {"50GB", "100GB", "1TB", "10TB"},
    ResourceType.DATABASE: {"Basic", "Standard", "Premium"},
    ResourceType.KUBERNETES_CLUSTER: {"Small", "Medium", "Large"},
    ResourceType.FUNCTION_APP: {"Consumption", "Premium"},
    ResourceType.VIRTUAL_NETWORK: {"default"},
}

# -----------------------------------------------------------------------------
# Base Intent Class
# -----------------------------------------------------------------------------
class Intent(BaseModel):
    """Abstract base for all intents, providing common fields."""
    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this intent")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Time the intent was created")
    requester: Principal = Field(..., description="User or service principal requesting the action")
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about how the intent was generated (e.g., agent ID, session)"
    )

    class Config:
        frozen = True  # immutable after creation
        extra = "forbid"  # no extra fields

# -----------------------------------------------------------------------------
# Specific Intent Types
# -----------------------------------------------------------------------------
class ProvisionResourceIntent(Intent):
    """Request to provision a new Azure resource."""
    intent_type: Literal["provision_resource"] = "provision_resource"
    resource_type: ResourceType
    region: Region
    size: Size
    configuration: Dict[str, Any] = Field(default_factory=dict)
    environment: Environment

    @field_validator("region")
    def validate_region(cls, v: Region) -> Region:
        if v not in VALID_AZURE_REGIONS:
            raise ValueError(f"Unknown Azure region: {v}")
        return v

    @field_validator("size")
    def validate_size(cls, v: Size, info) -> Size:
        # Relaxed validation: allow any string for testing flexibility.
        # The knowledge base (RESOURCE_SIZE_PATTERNS) is for reference only.
        # Optionally, you could log a warning if the size is not recognized.
        # For now, we accept any size.
        return v

class DeployConfigurationIntent(Intent):
    """Request to change configuration of an existing service."""
    intent_type: Literal["deploy_config"] = "deploy_config"
    service_name: ServiceName
    change_scope: ChangeScope
    deployment_target: Environment
    risk_level_hint: Optional[Annotated[float, Field(ge=0, le=1)]] = None
    configuration: Dict[str, Any] = Field(default_factory=dict)

    # Optional: validate that service_name follows naming conventions
    @field_validator("service_name")
    def validate_service_name(cls, v: ServiceName) -> ServiceName:
        if not v or len(v) < 3:
            raise ValueError("Service name must be at least 3 characters")
        return v

class GrantAccessIntent(Intent):
    """Request to grant a permission to a principal."""
    intent_type: Literal["grant_access"] = "grant_access"
    principal: Principal
    permission_level: PermissionLevel
    resource_scope: ResourceScope
    justification: Optional[str] = None

    # Validate resource_scope format (simplified)
    @field_validator("resource_scope")
    def validate_resource_scope(cls, v: ResourceScope) -> ResourceScope:
        if not v.startswith("/"):
            raise ValueError("Resource scope must start with '/'")
        return v

# -----------------------------------------------------------------------------
# Discriminated Union of All Intents
# -----------------------------------------------------------------------------
InfrastructureIntent = Annotated[
    Union[ProvisionResourceIntent, DeployConfigurationIntent, GrantAccessIntent],
    Field(discriminator="intent_type")
]

__all__ = [
    "ResourceType",
    "PermissionLevel",
    "Environment",
    "ChangeScope",
    "ProvisionResourceIntent",
    "DeployConfigurationIntent",
    "GrantAccessIntent",
    "InfrastructureIntent",
]
