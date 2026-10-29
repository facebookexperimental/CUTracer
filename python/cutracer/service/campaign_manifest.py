"""Integrated, auditable planning for scheduled diagnosis campaigns."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from typing import Any

from cutracer.service.campaign import (
    AdmissionDecision,
    AdmissionDisposition,
    AdmissionPlan,
    apply_campaign_budget,
    assign_catalog_shards,
    CadenceDecision,
    CadenceDisposition,
    CampaignBudget,
    CampaignMode,
    CampaignTool,
    plan_catalog_sampling,
    plan_tool_cadence,
    SamplingDisposition,
    SamplingPlan,
    ToolCadence,
    ToolCadencePlan,
)
from cutracer.service.campaign_budget import (
    apply_resource_budget,
    CampaignResourceBudget,
    CampaignResourceEstimate,
    CampaignResourceUsage,
    ResourceAdmissionDecision,
    ResourceAdmissionPlan,
)
from cutracer.service.catalog import WorkloadCatalog
from cutracer.service.catalog_adapters import CatalogWorkUnit, expand_workload_catalog


CAMPAIGN_MANIFEST_SCHEMA_VERSION = 1


class ManifestOutcome(str, Enum):
    ADMITTED = "admitted"
    DISABLED = "disabled"
    SAMPLED_OUT = "sampled_out"
    OTHER_SHARD = "other_shard"
    NOT_DUE = "not_due"
    NOT_APPLICABLE = "not_applicable"
    DEFERRED_BUDGET = "deferred_budget"


@dataclasses.dataclass(frozen=True)
class CampaignPlannerConfig:
    mode: CampaignMode
    as_of: datetime.datetime
    shard_index: int
    shard_count: int
    shard_seed: str
    tool_cadences: tuple[ToolCadence, ...]
    execution_budget: CampaignBudget = dataclasses.field(default_factory=CampaignBudget)
    resource_budget: CampaignResourceBudget = dataclasses.field(
        default_factory=CampaignResourceBudget
    )

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("campaign planner as_of must be timezone-aware")
        if self.shard_count <= 0 or not 0 <= self.shard_index < self.shard_count:
            raise ValueError("campaign planner shard bounds are invalid")
        if not self.shard_seed.strip():
            raise ValueError("campaign planner shard seed must not be empty")
        if not self.tool_cadences:
            raise ValueError("campaign planner requires at least one tool cadence")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "as_of": self.as_of.isoformat(),
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "shard_seed": self.shard_seed,
            "tool_cadences": [
                {"tool": item.tool.value, "cadence_days": item.cadence_days}
                for item in self.tool_cadences
            ],
            "execution_budget": dataclasses.asdict(self.execution_budget),
            "resource_budget": dataclasses.asdict(self.resource_budget),
        }


@dataclasses.dataclass(frozen=True)
class CampaignManifestItem:
    target: CatalogWorkUnit
    tool: CampaignTool
    outcome: ManifestOutcome
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target.to_dict(),
            "tool": self.tool.value,
            "outcome": self.outcome.value,
            "reasons": list(self.reasons),
        }


@dataclasses.dataclass(frozen=True)
class PlannedCampaignSession:
    target: CatalogWorkUnit
    sanitizer_tools: tuple[str, ...]
    enable_stress: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enable_stress, bool):
            raise ValueError("planned enable_stress must be boolean")
        supported = {
            tool.value
            for tool in CampaignTool
            if tool != CampaignTool.RANDOM_DELAY_STRESS
        }
        if len(self.sanitizer_tools) != len(set(self.sanitizer_tools)):
            raise ValueError("planned sanitizer tools must be unique")
        if set(self.sanitizer_tools) - supported:
            raise ValueError("planned campaign contains unsupported sanitizer tools")
        if not self.sanitizer_tools and not self.enable_stress:
            raise ValueError("planned campaign session requires an evidence source")

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target.to_dict(),
            "sanitizer_tools": list(self.sanitizer_tools),
            "enable_stress": self.enable_stress,
        }


@dataclasses.dataclass(frozen=True)
class CampaignManifest:
    catalog_revision: str
    catalog_digest: str
    config: CampaignPlannerConfig
    items: tuple[CampaignManifestItem, ...]
    sessions: tuple[PlannedCampaignSession, ...]
    schema_version: int = CAMPAIGN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported campaign manifest schema_version: {self.schema_version}"
            )
        if not self.catalog_revision.strip() or len(self.catalog_digest) != 64:
            raise ValueError("campaign manifest requires catalog identity")
        unit_ids = [session.target.unit.unit_id for session in self.sessions]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("campaign manifest sessions must have unique unit ids")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "catalog_revision": self.catalog_revision,
            "catalog_digest": self.catalog_digest,
            "config": self.config.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "sessions": [session.to_dict() for session in self.sessions],
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _scheduled_sample(
    targets: tuple[CatalogWorkUnit, ...],
    catalog: WorkloadCatalog,
    mode: CampaignMode,
) -> tuple[SamplingPlan, tuple[CatalogWorkUnit, ...]]:
    plan = plan_catalog_sampling(targets, catalog_revision=catalog.catalog_revision)
    selected = targets if mode == CampaignMode.ON_DEMAND else plan.selected
    return plan, selected


def _shard_selection(
    targets: tuple[CatalogWorkUnit, ...],
    config: CampaignPlannerConfig,
) -> tuple[dict[str, int], tuple[CatalogWorkUnit, ...]]:
    assignments = assign_catalog_shards(
        targets,
        shard_count=config.shard_count,
        seed=config.shard_seed,
    )
    by_unit = {
        assignment.target.unit.unit_id: assignment.shard_index
        for assignment in assignments
    }
    if config.mode == CampaignMode.ON_DEMAND:
        return by_unit, targets
    return (
        by_unit,
        tuple(
            assignment.target
            for assignment in assignments
            if assignment.shard_index == config.shard_index
        ),
    )


def _due_targets(
    decisions: tuple[CadenceDecision, ...],
) -> tuple[CatalogWorkUnit, ...]:
    by_unit: dict[str, CatalogWorkUnit] = {}
    for decision in decisions:
        if decision.disposition == CadenceDisposition.DUE:
            by_unit.setdefault(decision.target.unit.unit_id, decision.target)
    return tuple(by_unit.values())


def _outcome(
    target: CatalogWorkUnit,
    tool: CampaignTool,
    *,
    mode: CampaignMode,
    shard_index: int,
    sample: Mapping[str, SamplingDisposition],
    shards: Mapping[str, int],
    cadence: Mapping[tuple[str, CampaignTool], CadenceDecision],
    execution: Mapping[str, AdmissionDecision],
    resources: Mapping[str, ResourceAdmissionDecision],
) -> tuple[ManifestOutcome, tuple[str, ...]]:
    unit_id = target.unit.unit_id
    if not target.enabled:
        return ManifestOutcome.DISABLED, ("catalog entry is disabled",)
    if (
        mode == CampaignMode.SCHEDULED
        and sample[unit_id] == SamplingDisposition.SAMPLED_OUT
    ):
        return ManifestOutcome.SAMPLED_OUT, ("selector excluded by sampling policy",)
    if mode == CampaignMode.SCHEDULED and shards[unit_id] != shard_index:
        return ManifestOutcome.OTHER_SHARD, (f"assigned to shard {shards[unit_id]}",)
    cadence_decision = cadence.get((unit_id, tool))
    if cadence_decision is None:
        # Cadence decisions are planned only for the in-shard subset; a pair
        # outside that subset has no applicable cadence rather than aborting
        # the whole manifest.
        return ManifestOutcome.NOT_APPLICABLE, ("no cadence decision for this tool",)
    if cadence_decision.disposition == CadenceDisposition.NOT_APPLICABLE:
        return ManifestOutcome.NOT_APPLICABLE, (cadence_decision.reason,)
    if cadence_decision.disposition == CadenceDisposition.NOT_DUE:
        return ManifestOutcome.NOT_DUE, (cadence_decision.reason,)
    execution_decision = execution[unit_id]
    if execution_decision.disposition == AdmissionDisposition.DEFERRED:
        reasons = tuple(
            f"execution:{item.value}" for item in execution_decision.constraints
        )
        return ManifestOutcome.DEFERRED_BUDGET, reasons
    resource_decision = resources[unit_id]
    if resource_decision.disposition == AdmissionDisposition.DEFERRED:
        reasons = tuple(
            f"resource:{item.value}" for item in resource_decision.constraints
        )
        return ManifestOutcome.DEFERRED_BUDGET, reasons
    return ManifestOutcome.ADMITTED, ("all campaign gates passed",)


def _manifest_items(
    targets: tuple[CatalogWorkUnit, ...],
    config: CampaignPlannerConfig,
    sample_plan: SamplingPlan,
    shards: Mapping[str, int],
    cadence_decisions: tuple[CadenceDecision, ...],
    execution_decisions: tuple[AdmissionDecision, ...],
    resource_decisions: tuple[ResourceAdmissionDecision, ...],
) -> tuple[CampaignManifestItem, ...]:
    sample = {
        decision.target.unit.unit_id: decision.disposition
        for decision in sample_plan.decisions
    }
    cadence = {
        (decision.target.unit.unit_id, decision.tool): decision
        for decision in cadence_decisions
    }
    execution = {
        decision.target.unit.unit_id: decision for decision in execution_decisions
    }
    resources = {
        decision.target.unit.unit_id: decision for decision in resource_decisions
    }
    items: list[CampaignManifestItem] = []
    for target in targets:
        for tool_cadence in config.tool_cadences:
            outcome, reasons = _outcome(
                target,
                tool_cadence.tool,
                mode=config.mode,
                shard_index=config.shard_index,
                sample=sample,
                shards=shards,
                cadence=cadence,
                execution=execution,
                resources=resources,
            )
            items.append(
                CampaignManifestItem(target, tool_cadence.tool, outcome, reasons)
            )
    return tuple(items)


def _planned_sessions(
    targets: tuple[CatalogWorkUnit, ...],
    due_tools_by_unit: Mapping[str, tuple[CampaignTool, ...]],
) -> tuple[PlannedCampaignSession, ...]:
    sessions: list[PlannedCampaignSession] = []
    for target in targets:
        tools = due_tools_by_unit[target.unit.unit_id]
        sanitizer_tools = tuple(
            tool.value for tool in tools if tool != CampaignTool.RANDOM_DELAY_STRESS
        )
        sessions.append(
            PlannedCampaignSession(
                target=target,
                sanitizer_tools=sanitizer_tools,
                enable_stress=CampaignTool.RANDOM_DELAY_STRESS in tools,
            )
        )
    return tuple(sessions)


def _execution_admission(
    targets: tuple[CatalogWorkUnit, ...],
    config: CampaignPlannerConfig,
    last_completed: Mapping[tuple[str, CampaignTool], datetime.datetime],
) -> tuple[ToolCadencePlan, AdmissionPlan]:
    cadence_plan = plan_tool_cadence(
        targets,
        config.tool_cadences,
        mode=config.mode,
        as_of=config.as_of,
        last_completed=last_completed,
    )
    execution_plan = apply_campaign_budget(
        _due_targets(cadence_plan.decisions), config.execution_budget
    )
    return cadence_plan, execution_plan


def _resource_admission(
    execution_plan: AdmissionPlan,
    config: CampaignPlannerConfig,
    resource_estimates: Mapping[str, CampaignResourceEstimate],
    sandcastle_submitted_at: tuple[datetime.datetime, ...],
    initial_resource_usage: CampaignResourceUsage | None,
) -> ResourceAdmissionPlan:
    return apply_resource_budget(
        execution_plan.admitted,
        resource_estimates,
        config.resource_budget,
        as_of=config.as_of,
        sandcastle_submitted_at=sandcastle_submitted_at,
        initial_usage=initial_resource_usage,
    )


def plan_campaign(
    catalog: WorkloadCatalog,
    config: CampaignPlannerConfig,
    *,
    resource_estimates: Mapping[str, CampaignResourceEstimate],
    last_completed: Mapping[tuple[str, CampaignTool], datetime.datetime] | None = None,
    sandcastle_submitted_at: tuple[datetime.datetime, ...] = (),
    initial_resource_usage: CampaignResourceUsage | None = None,
) -> CampaignManifest:
    all_targets = expand_workload_catalog(catalog, include_disabled=True)
    active = tuple(target for target in all_targets if target.enabled)
    sample_plan, sampled = _scheduled_sample(active, catalog, config.mode)
    shards, in_shard = _shard_selection(sampled, config)
    cadence_plan, execution_plan = _execution_admission(
        in_shard,
        config,
        {} if last_completed is None else last_completed,
    )
    resource_plan = _resource_admission(
        execution_plan,
        config,
        resource_estimates,
        sandcastle_submitted_at,
        initial_resource_usage,
    )
    all_shards = {
        **{target.unit.unit_id: -1 for target in all_targets},
        **shards,
    }
    due_by_unit = {
        target.unit.unit_id: cadence_plan.due_tools(target.unit.unit_id)
        for target in resource_plan.admitted
    }
    items = _manifest_items(
        all_targets,
        config,
        sample_plan,
        all_shards,
        cadence_plan.decisions,
        execution_plan.decisions,
        resource_plan.decisions,
    )
    return CampaignManifest(
        catalog_revision=catalog.catalog_revision,
        catalog_digest=catalog.digest,
        config=config,
        items=items,
        sessions=_planned_sessions(resource_plan.admitted, due_by_unit),
    )
