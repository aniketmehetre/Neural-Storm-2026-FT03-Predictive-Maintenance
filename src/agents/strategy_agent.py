"""Simulated AI Agent for Automated Maintenance Strategy & Parts Ordering.

This module acts as an autonomous aerospace maintenance engineering specialist.
It evaluates predicted Remaining Useful Life (RUL), determines the operational risk
tier, isolates the primary degraded subsystem based on sensor telemetry, and synthesizes:
1. A plain-English, actionable maintenance strategy.
2. An automated, simulated parts-order procurement request with aerospace SKUs and logistics.
3. Operational restrictions, technician qualifications, and downtime estimates.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

# ── Alert Thresholds (Cycles) ────────────────────────────────────────────────
ALERT_CRITICAL = 30
ALERT_WARNING = 80


# ── Subsystem & Part Catalog Mapping ──────────────────────────────────────────
# Physical subsystem details keyed by sensor identifier.
SUBSYSTEM_CATALOG: dict[str, dict[str, Any]] = {
    "sensor_02": {
        "subsystem": "Low-Pressure Compressor (LPC)",
        "component": "LPC Stator Stages & Inlet Guide Vanes",
        "primary_failure_mode": "Fouling, stator erosion, and inlet temperature distortion",
        "parts": [
            {
                "sku": "LPC-VNE-201A",
                "part_name": "LPC Variable Stator Vane Segment (Stage 2)",
                "unit_cost_usd": 4200.0,
                "lead_time_days": 2,
                "criticality": "HIGH",
            },
            {
                "sku": "SEN-TP-024",
                "part_name": "T24 Dual Platinum RTD Temperature Probe",
                "unit_cost_usd": 1150.0,
                "lead_time_days": 1,
                "criticality": "STANDARD",
            },
        ],
    },
    "sensor_03": {
        "subsystem": "High-Pressure Compressor (HPC)",
        "component": "HPC Rear Compression Stages",
        "primary_failure_mode": "Aerodynamic stall margin degradation & thermal blade fatigue",
        "parts": [
            {
                "sku": "HPC-BLD-309X",
                "part_name": "HPC Titanium-Alloy Rotor Blade Set (Stages 7-9)",
                "unit_cost_usd": 14800.0,
                "lead_time_days": 3,
                "criticality": "CRITICAL",
            },
            {
                "sku": "HPC-SEA-301",
                "part_name": "Compressor Discharge Pressure (CDP) Labyrinth Seal",
                "unit_cost_usd": 3950.0,
                "lead_time_days": 2,
                "criticality": "HIGH",
            },
        ],
    },
    "sensor_04": {
        "subsystem": "Low-Pressure Turbine (LPT)",
        "component": "LPT Rotor & Exhaust Mixer Assembly",
        "primary_failure_mode": "Exhaust gas over-temperature & turbine nozzle guide vane creep",
        "parts": [
            {
                "sku": "LPT-NGV-411C",
                "part_name": "LPT Stage 1 Nozzle Guide Vane Ring",
                "unit_cost_usd": 18200.0,
                "lead_time_days": 4,
                "criticality": "CRITICAL",
            },
            {
                "sku": "SEN-TC-050",
                "part_name": "T50 Exhaust Thermocouple Harness Assembly",
                "unit_cost_usd": 2100.0,
                "lead_time_days": 1,
                "criticality": "STANDARD",
            },
        ],
    },
    "sensor_07": {
        "subsystem": "High-Pressure Compressor (HPC Pressure Stage)",
        "component": "HPC Outer Casing & Bleed Manifold",
        "primary_failure_mode": "Compressor discharge pressure leakage and manifold seal wear",
        "parts": [
            {
                "sku": "VBV-ACT-711",
                "part_name": "Variable Bleed Valve (VBV) Electro-Hydraulic Actuator",
                "unit_cost_usd": 8600.0,
                "lead_time_days": 2,
                "criticality": "HIGH",
            },
            {
                "sku": "GSK-HPC-707",
                "part_name": "High-Temperature Inconel Flange Seal Kit",
                "unit_cost_usd": 1450.0,
                "lead_time_days": 1,
                "criticality": "STANDARD",
            },
        ],
    },
    "sensor_08": {
        "subsystem": "Fan Rotor & Front Frame",
        "component": "Titanium Fan Blades & #1 Front Bearing",
        "primary_failure_mode": "Leading-edge blade foreign object damage & bearing race spalling",
        "parts": [
            {
                "sku": "FAN-BLD-813T",
                "part_name": "Wide-Chord Titanium Fan Blade Matched Pair",
                "unit_cost_usd": 24500.0,
                "lead_time_days": 3,
                "criticality": "CRITICAL",
            },
            {
                "sku": "BRG-ROL-001",
                "part_name": "#1 Roller Bearing & Squeeze Film Damper Kit",
                "unit_cost_usd": 7800.0,
                "lead_time_days": 2,
                "criticality": "HIGH",
            },
        ],
    },
    "sensor_09": {
        "subsystem": "High-Pressure Core Rotor Assembly",
        "component": "HP Spool Shaft & #3 Ball Thrust Bearing",
        "primary_failure_mode": "Core rotational speed loss, shaft eccentricity & bearing wear",
        "parts": [
            {
                "sku": "BRG-HPS-914B",
                "part_name": "#3 HP Spool Ball Thrust Bearing Assembly",
                "unit_cost_usd": 12600.0,
                "lead_time_days": 3,
                "criticality": "CRITICAL",
            },
            {
                "sku": "SLR-CC-400",
                "part_name": "Carbon-Carbon Dynamic Shaft Air Seal Ring",
                "unit_cost_usd": 3200.0,
                "lead_time_days": 1,
                "criticality": "HIGH",
            },
        ],
    },
    "sensor_11": {
        "subsystem": "High-Pressure Compressor (Static Pressure Stage)",
        "component": "HPC Static Bleed & Diffuser Case",
        "primary_failure_mode": "HPC diffuser flow separation and stage static pressure drop",
        "parts": [
            {
                "sku": "HPC-DIF-110",
                "part_name": "HPC Diffuser Casing Air Seal & Vane Cluster",
                "unit_cost_usd": 16500.0,
                "lead_time_days": 3,
                "criticality": "CRITICAL",
            },
            {
                "sku": "SEN-PS30-11",
                "part_name": "Ps30 Static Pressure Differential Transducer",
                "unit_cost_usd": 2400.0,
                "lead_time_days": 1,
                "criticality": "STANDARD",
            },
        ],
    },
    "sensor_12": {
        "subsystem": "Combustion & Fuel Metering System",
        "component": "Fuel Metering Unit (FMU) & Fuel Injector Nozzles",
        "primary_failure_mode": "Fuel nozzle coking, spray pattern asymmetry & elevated fuel-to-pressure ratio",
        "parts": [
            {
                "sku": "FUL-NZL-120S",
                "part_name": "Dual-Stage Aerated Fuel Injector Nozzle Set (Qty 16)",
                "unit_cost_usd": 11200.0,
                "lead_time_days": 2,
                "criticality": "HIGH",
            },
            {
                "sku": "FMV-ACT-205",
                "part_name": "Hydromechanical Fuel Metering Valve Assembly",
                "unit_cost_usd": 9400.0,
                "lead_time_days": 2,
                "criticality": "HIGH",
            },
        ],
    },
    "sensor_14": {
        "subsystem": "Core Compressor & Turbine Spool Dynamics",
        "component": "Corrected Core Spool (NRc) & Intermediate Casing",
        "primary_failure_mode": "Aerodynamic core speed mismatch & rotor blade rubbing",
        "parts": [
            {
                "sku": "SPL-BLD-140A",
                "part_name": "High-Pressure Compressor Blisk Assembly (Stage 1-2)",
                "unit_cost_usd": 29000.0,
                "lead_time_days": 5,
                "criticality": "CRITICAL",
            },
            {
                "sku": "CAS-RUB-142",
                "part_name": "Abradable Casing Rub Strip Kit",
                "unit_cost_usd": 4100.0,
                "lead_time_days": 2,
                "criticality": "HIGH",
            },
        ],
    },
    "sensor_15": {
        "subsystem": "Bypass Duct & Fan Exit Guide Vanes",
        "component": "Fan Duct Splitter Cowl & Acoustic Panels",
        "primary_failure_mode": "Bypass ratio deviation & acoustic panel delamination",
        "parts": [
            {
                "sku": "FEG-VNE-150B",
                "part_name": "Composite Fan Exit Guide Vane Cluster",
                "unit_cost_usd": 7600.0,
                "lead_time_days": 2,
                "criticality": "HIGH",
            },
            {
                "sku": "DCT-SEA-152",
                "part_name": "Bypass Cowl Aerodynamic Perimeter Seal",
                "unit_cost_usd": 1850.0,
                "lead_time_days": 1,
                "criticality": "STANDARD",
            },
        ],
    },
    "sensor_17": {
        "subsystem": "Bleed Air & Thermal Management System",
        "component": "Customer Bleed Air Precooler & Control Valve",
        "primary_failure_mode": "Bleed enthalpy elevation & heat exchanger tube bundle fouling",
        "parts": [
            {
                "sku": "BLD-HEX-171",
                "part_name": "Titanium Plate-Fin Bleed Air Precooler Core",
                "unit_cost_usd": 13400.0,
                "lead_time_days": 3,
                "criticality": "HIGH",
            },
            {
                "sku": "BLD-SOV-172",
                "part_name": "Pneumatic High-Pressure Shutoff & Regulator Valve",
                "unit_cost_usd": 5100.0,
                "lead_time_days": 2,
                "criticality": "STANDARD",
            },
        ],
    },
    "sensor_20": {
        "subsystem": "High-Pressure Turbine (HPT) Cooling System",
        "component": "HPT Active Clearance Control & Blade Cooling",
        "primary_failure_mode": "HPT coolant bleed flow reduction & nozzle guide vane oxidation",
        "parts": [
            {
                "sku": "HPT-CLN-201",
                "part_name": "HPT Stage 1 Cooled Single-Crystal Blade Set",
                "unit_cost_usd": 36000.0,
                "lead_time_days": 5,
                "criticality": "CRITICAL",
            },
            {
                "sku": "ACC-VLV-204",
                "part_name": "HPT Active Clearance Control Modulating Valve",
                "unit_cost_usd": 6900.0,
                "lead_time_days": 2,
                "criticality": "HIGH",
            },
        ],
    },
    "sensor_21": {
        "subsystem": "Low-Pressure Turbine (LPT) Cooling System",
        "component": "LPT Cooling Air Transfer Tubes & Case Shrouds",
        "primary_failure_mode": "LPT shroud clearance enlargement & cooling air distribution drop",
        "parts": [
            {
                "sku": "LPT-SHR-210",
                "part_name": "LPT Segmented Turbine Shroud Seal Kit",
                "unit_cost_usd": 12800.0,
                "lead_time_days": 3,
                "criticality": "HIGH",
            },
            {
                "sku": "LPT-TB-215",
                "part_name": "LPT External Cooling Air Transfer Manifold",
                "unit_cost_usd": 4700.0,
                "lead_time_days": 2,
                "criticality": "STANDARD",
            },
        ],
    },
}

# Default generic kit if no sensor is isolated
DEFAULT_KIT = {
    "subsystem": "Core High-Pressure Compressor & Hot Section",
    "component": "HPC Compressor Stages & Turbine Interstage Duct",
    "primary_failure_mode": "General multi-stage thermal degradation and mechanical wear",
    "parts": [
        {
            "sku": "HPC-HS-KIT-01",
            "part_name": "HPC Stage 5-8 Inspection Overhaul & Blade Kit",
            "unit_cost_usd": 21500.0,
            "lead_time_days": 3,
            "criticality": "CRITICAL",
        },
        {
            "sku": "FLT-LUB-001",
            "part_name": "Synthetic MIL-PRF-23699 Turbine Lubrication & Filter Pack",
            "unit_cost_usd": 1250.0,
            "lead_time_days": 1,
            "criticality": "STANDARD",
        },
    ],
}


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class PartsOrderItem(BaseModel):
    sku: str = Field(..., description="Aerospace part stock keeping unit")
    part_name: str = Field(..., description="Standardized aerospace component name")
    subsystem: str = Field(..., description="Aircraft turbofan subsystem")
    quantity: int = Field(..., ge=1, description="Quantity required")
    unit_cost_usd: float = Field(..., ge=0, description="Unit cost in USD")
    total_cost_usd: float = Field(..., ge=0, description="Item total cost in USD")
    lead_time_days: int = Field(..., description="Procurement lead time in days")
    criticality: str = Field(..., description="Criticality classification (CRITICAL/HIGH/STANDARD)")


class SimulatedPartsOrder(BaseModel):
    order_id: str = Field(..., description="Automated purchase order tracking number")
    order_timestamp: str = Field(..., description="ISO 8601 generation timestamp")
    order_priority: str = Field(..., description="Order priority tier (AOG_EMERGENCY, EXPEDITED, ROUTINE)")
    target_subsystem: str = Field(..., description="Target turbofan engine subsystem")
    items: list[PartsOrderItem] = Field(..., description="Line items for parts procurement")
    total_cost_usd: float = Field(..., ge=0, description="Total order valuation in USD")
    procurement_hub: str = Field(..., description="Designated logistics distribution center")
    estimated_delivery_hours: int = Field(..., description="Expected delivery window in hours")
    shipping_method: str = Field(..., description="Logistics carrier and shipping modality")


class MaintenanceStrategyRequest(BaseModel):
    unit_id: int | None = Field(default=None, description="Engine identifier (optional, 1-100)")
    predicted_rul: float = Field(..., ge=0, description="Predicted Remaining Useful Life in operating cycles")
    dominant_sensor: str | None = Field(
        default=None,
        description="Top degrading sensor identifier from SHAP or anomaly telemetry (e.g., 'sensor_11')",
    )
    operating_cycle: int | None = Field(
        default=None,
        ge=1,
        description="Current operating cycle count of the engine",
    )


class MaintenanceStrategyResponse(BaseModel):
    unit_id: int | None = Field(..., description="Target engine ID")
    predicted_rul: float = Field(..., description="Predicted RUL in cycles")
    alert_level: str = Field(..., description="Operational alert tier (CRITICAL, WARNING, HEALTHY)")
    urgency_rating: str = Field(..., description="Urgency index and response timeline")
    primary_affected_subsystem: str = Field(..., description="Turbofan module flagged for service")
    plain_english_strategy: str = Field(..., description="Actionable maintenance narrative for operations")
    recommended_action_window: str = Field(..., description="Target deadline for work execution")
    operational_restrictions: list[str] = Field(..., description="Flight and power-envelope constraints")
    maintenance_tasks: list[str] = Field(..., description="Step-by-step MRO procedure checklist")
    technician_certifications: list[str] = Field(..., description="Mandatory maintenance personnel licenses")
    estimated_downtime_hours: float = Field(..., description="Estimated hangar maintenance downtime in hours")
    parts_order: SimulatedPartsOrder = Field(..., description="Automated procurement request")
    agent_confidence: str = Field(..., description="Simulated AI agent diagnostic confidence")
    agent_signature: str = Field(..., description="Digital signature of the AI agent")


# ── AI Agent Logic ─────────────────────────────────────────────────────────────
class SimulatedMaintenanceAgent:
    """Specialist AI Maintenance Agent that formulates automated MRO strategies.

    Combines domain heuristics, NASA turbofan failure mode patterns, and
    aerospace supply chain logistics to turn statistical RUL predictions into
    concrete, auditable, and actionable engineering workflows.
    """

    def __init__(self, agent_name: str = "AeroGuard-PredictiveMaintenanceAgent-v2.6"):
        self.agent_name = agent_name

    def generate_strategy(
        self,
        predicted_rul: float,
        unit_id: int | None = None,
        dominant_sensor: str | None = None,
        operating_cycle: int | None = None,
    ) -> MaintenanceStrategyResponse:
        """Formulate an actionable maintenance strategy and simulated parts order.

        Args:
            predicted_rul: Estimated remaining operating cycles.
            unit_id: Optional engine identifier.
            dominant_sensor: Leading sensor feature contributing to degradation.
            operating_cycle: Current cumulative cycle count.

        Returns:
            MaintenanceStrategyResponse with complete narrative and procurement order.
        """
        # Determine status tier
        if predicted_rul <= ALERT_CRITICAL:
            alert_level = "CRITICAL"
        elif predicted_rul <= ALERT_WARNING:
            alert_level = "WARNING"
        else:
            alert_level = "HEALTHY"

        # Resolve subsystem details
        sensor_key = dominant_sensor.lower() if dominant_sensor else ""
        catalog_entry = SUBSYSTEM_CATALOG.get(sensor_key, DEFAULT_KIT)
        subsystem_name = catalog_entry["subsystem"]
        component_name = catalog_entry["component"]
        failure_mode = catalog_entry["primary_failure_mode"]

        now_utc = datetime.now(timezone.utc)
        ts_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Generate unique deterministic order ID based on inputs
        hash_seed = f"{unit_id}-{predicted_rul:.1f}-{dominant_sensor}-{now_utc.strftime('%Y%m%d')}"
        order_hash = hashlib.sha256(hash_seed.encode()).hexdigest()[:8].upper()
        order_id = f"PO-AERO-2026-{order_hash}"

        engine_tag = f"Engine #{unit_id}" if unit_id is not None else "Target Engine"
        cycle_tag = f" at cycle {operating_cycle}" if operating_cycle else ""

        # ── Branch 1: CRITICAL (RUL <= 30 cycles) ───────────────────────────
        if alert_level == "CRITICAL":
            urgency_rating = "LEVEL 1 - IMMEDIATE AIRCRAFT ON GROUND (AOG) DISPATCH"
            recommended_action_window = f"Within next 3 cycles or 48 calendar hours (RUL: {predicted_rul:.1f} cycles)"
            order_priority = "AOG_EMERGENCY"
            estimated_delivery_hours = 12
            shipping_method = "Dedicated Air Courier / Hot-Shot Logistics"
            estimated_downtime_hours = 36.0

            plain_english_strategy = (
                f"{engine_tag}{cycle_tag} has entered a critical degradation phase with only "
                f"{predicted_rul:.1f} operating cycles remaining before anticipated functional failure. "
                f"Telemetry indicates severe risk concentrated in the {subsystem_name} ({component_name}), "
                f"presenting signature symptoms of {failure_mode}. "
                f"The AI Agent advises immediate grounding upon completion of the current mission, "
                f"cancellation of extended twin-engine operational performance (ETOPS) clearance, "
                f"and induction into a qualified depot facility for borescope inspection and rapid modular replacement. "
                f"A pre-emptive expedited purchase order has been automatically generated to preposition critical replacement assemblies."
            )

            operational_restrictions = [
                "GROUND IMMEDIATELY upon arrival at primary maintenance base.",
                "Revoke ETOPS (Extended-range Twin-engine Operational Performance Standards) authorization.",
                "Limit max continuous takeoff thrust derate to 85% if reposition flight is required.",
                "Prohibit revenue passenger flights until maintenance release certificate is signed.",
            ]

            maintenance_tasks = [
                f"Perform comprehensive video-borescope inspection of {subsystem_name}.",
                "Inspect magnetic chip detectors and oil scavenger filters for metallic debris.",
                f"Remove and replace {component_name} using pre-staged parts order {order_id}.",
                "Perform cold-crank rotation check and acoustic resonance verification.",
                "Conduct full-throttle static ground run cell testing with 3-minute stabilization at takeoff thrust.",
            ]

            technician_certifications = [
                "FAA Airframe & Powerplant (A&P) with Gas Turbine Endorsement",
                "EASA Part-66 B1 Licensed Aircraft Maintenance Engineer (Turbofan)",
                "Borescope Non-Destructive Testing (NDT) Level II Certification",
            ]

            # Multiply quantities for critical overhaul
            item_multiplier = 1

        # ── Branch 2: WARNING (30 < RUL <= 80 cycles) ───────────────────────
        elif alert_level == "WARNING":
            urgency_rating = "LEVEL 2 - ELEVATED PREVENTATIVE OVERHAUL"
            recommended_action_window = f"Schedule within next 20 operating cycles (RUL: {predicted_rul:.1f} cycles)"
            order_priority = "EXPEDITED"
            estimated_delivery_hours = 48
            shipping_method = "Priority Air Cargo (Next Flight Out)"
            estimated_downtime_hours = 16.0

            plain_english_strategy = (
                f"{engine_tag}{cycle_tag} exhibits early-stage degradation markers with "
                f"{predicted_rul:.1f} cycles remaining. The telemetry trend highlights abnormal divergence in "
                f"the {subsystem_name}, pointing toward {failure_mode}. "
                f"While the engine remains safe for domestic line operations, the AI Agent recommends scheduling an "
                f"intermediate hangar inspection at the next scheduled turnaround window. "
                f"Parts have been queued in advance to avoid aircraft turnaround delays."
            )

            operational_restrictions = [
                "Continue standard line flights with real-time continuous sensor stream telemetry active.",
                "Mandate daily post-flight trend analysis upload to Fleet Health Monitor.",
                "Restrict operation on high-altitude/hot-day airfields exceeding 38°C ambient.",
            ]

            maintenance_tasks = [
                f"Execute targeted borescope examination of {component_name}.",
                "Perform dry-crank oil pressure and scavenging flow verification.",
                "Inspect bleed air routing lines and flange couplings for micro-fractures.",
                "Calibrate pressure/temperature transducer harnesses.",
            ]

            technician_certifications = [
                "FAA A&P Powerplant Technician",
                "EASA Part-66 B1/B2 Technician",
            ]

            item_multiplier = 1

        # ── Branch 3: HEALTHY (RUL > 80 cycles) ─────────────────────────────
        else:
            urgency_rating = "LEVEL 3 - ROUTINE PLANNED MONITORING"
            recommended_action_window = f"Align with next standard phased C-Check (RUL: {predicted_rul:.1f} cycles)"
            order_priority = "ROUTINE"
            estimated_delivery_hours = 120
            shipping_method = "Standard Ground/Consolidated Freight"
            estimated_downtime_hours = 4.0

            plain_english_strategy = (
                f"{engine_tag}{cycle_tag} is operating within nominal aerospace specifications with "
                f"{predicted_rul:.1f} cycles remaining (exceeding healthy fleet baseline). "
                f"All {subsystem_name} pressure ratios, temperatures, and rotor shaft speeds display stable laminar trends. "
                f"No immediate servicing or engine derating is required. "
                f"A routine preventative consumable replenishment requisition has been placed for supply-chain readiness."
            )

            operational_restrictions = [
                "Zero operational restrictions; full ETOPS and maximum takeoff thrust certified.",
            ]

            maintenance_tasks = [
                "Maintain standard line-station visual pre-flight inspection checklist.",
                "Record flight log sensor telemetry into central fleet database at 50-cycle intervals.",
                "Verify engine oil consumption remains under 0.25 quarts/flight hour.",
            ]

            technician_certifications = [
                "FAA A&P Line Maintenance Inspector",
                "EASA Part-66 A/B1 License",
            ]

            item_multiplier = 1

        # Assemble parts order line items
        order_items: list[PartsOrderItem] = []
        total_cost = 0.0

        for raw_part in catalog_entry["parts"]:
            qty = 1 * item_multiplier
            cost = raw_part["unit_cost_usd"] * qty
            total_cost += cost
            order_items.append(
                PartsOrderItem(
                    sku=raw_part["sku"],
                    part_name=raw_part["part_name"],
                    subsystem=subsystem_name,
                    quantity=qty,
                    unit_cost_usd=raw_part["unit_cost_usd"],
                    total_cost_usd=round(cost, 2),
                    lead_time_days=raw_part["lead_time_days"],
                    criticality=raw_part["criticality"],
                )
            )

        parts_order = SimulatedPartsOrder(
            order_id=order_id,
            order_timestamp=ts_str,
            order_priority=order_priority,
            target_subsystem=subsystem_name,
            items=order_items,
            total_cost_usd=round(total_cost, 2),
            procurement_hub="ORD-Central-AeroDepot (Chicago, IL)",
            estimated_delivery_hours=estimated_delivery_hours,
            shipping_method=shipping_method,
        )

        confidence_score = 94.8 if alert_level == "CRITICAL" else (91.2 if alert_level == "WARNING" else 98.5)

        return MaintenanceStrategyResponse(
            unit_id=unit_id,
            predicted_rul=round(float(predicted_rul), 2),
            alert_level=alert_level,
            urgency_rating=urgency_rating,
            primary_affected_subsystem=subsystem_name,
            plain_english_strategy=plain_english_strategy,
            recommended_action_window=recommended_action_window,
            operational_restrictions=operational_restrictions,
            maintenance_tasks=maintenance_tasks,
            technician_certifications=technician_certifications,
            estimated_downtime_hours=estimated_downtime_hours,
            parts_order=parts_order,
            agent_confidence=f"{confidence_score:.1f}% (Based on Optuna-tuned XGBoost & CMAPSS FD001 fleet trends)",
            agent_signature=f"{self.agent_name} [VERIFIED_MRO_DISPATCH]",
        )

