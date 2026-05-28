"""Project-wide constants: paths, schema, and feature groups."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOK_RUN_DIR = RAW_DATA_DIR / "notebook_run"
# Общий parquet, куда дописывается каждый прогон парсера (дедуп по url).
AGGREGATE_PARQUET = RAW_DATA_DIR / "autoru_raw_all.parquet"

AUTORU_SCHEMA = [
    # базовые (со страницы объявления)
    "brand",
    "model",
    "generation",
    "year",
    "price",
    "mileage",
    "body_type",
    "color",
    "engine_volume",
    "engine_power_hp",
    "fuel_type",
    "transmission",
    "drive_type",
    "steering_wheel",
    "condition",
    "vehicle_state",
    "owners_count",
    "pts_type",
    "customs",
    "region",
    "seller_type",
    # offer-only
    "vin",
    "complectation",
    "modification",
    "tax_amount",
    "service_book",
    "warranty_until",
    # геометрия и динамика (catalog/specs)
    "doors_count",
    "seats_count",
    "acceleration_0_100",
    "max_speed",
    "fuel_consumption_city",
    "fuel_consumption_highway",
    "fuel_consumption_mixed",
    "eco_class",
    "trunk_volume",
    "clearance",
    "wheelbase",
    "length",
    "width",
    "height",
    "weight_curb",
    "weight_gross",
    "fuel_tank_volume",
    "front_track_width",
    "rear_track_width",
    "wheel_size",
    "bolt_pattern",
    "disc_size",
    # двигатель и трансмиссия (catalog/specs)
    "boost_type",
    "max_torque_nm",
    "cylinder_layout",
    "cylinders_count",
    "valves_per_cylinder",
    "gears_count",
    "engine_position",
    "fuel_brand",
    "co2_emissions",
    # подвеска и тормоза
    "front_suspension",
    "rear_suspension",
    "front_brakes",
    "rear_brakes",
    # электрокары
    "battery_capacity_kwh",
    "battery_type",
    "electric_range_km",
    "max_charging_power_kw",
    "fast_charging_time_min",
    "fast_charging_description",
    "charging_connector",
    "power_consumption_kwh_100",
    "consumption_method",
    # маркетинговые
    "country_brand",
    "car_class",
    # статус объявления
    "is_sold",
    # технические
    "description_text",
    "url",
    "parsed_at",
]

NUMERIC_FEATURES = [
    "year",
    "price",
    "mileage",
    "engine_volume",
    "engine_power_hp",
    "owners_count",
    "doors_count",
    "seats_count",
    "acceleration_0_100",
    "max_speed",
    "fuel_consumption_city",
    "fuel_consumption_highway",
    "fuel_consumption_mixed",
    "trunk_volume",
    "clearance",
    "wheelbase",
    "length",
    "width",
    "height",
    "weight_curb",
    "weight_gross",
    "fuel_tank_volume",
    "front_track_width",
    "rear_track_width",
    "max_torque_nm",
    "cylinders_count",
    "valves_per_cylinder",
    "gears_count",
    "co2_emissions",
    "tax_amount",
    # электрокары
    "battery_capacity_kwh",
    "electric_range_km",
    "max_charging_power_kw",
    "fast_charging_time_min",
    "power_consumption_kwh_100",
    # derived EV-to-ICE semantic equivalents
    "ev_consumption_l_100km_energy_eq",
    "ev_tank_equiv_l_energy_eq",
    "consumption_l_100km_semantic",
    "tank_volume_l_semantic",
    "ev_consumption_l_100km_cost_eq",
    "ev_tank_equiv_l_cost_eq",
    "consumption_l_100km_cost_eq",
]

CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "generation",
    "body_type",
    "color",
    "fuel_type",
    "transmission",
    "drive_type",
    "steering_wheel",
    "condition",
    "vehicle_state",
    "pts_type",
    "customs",
    "region",
    "seller_type",
]

DEFAULT_USER_AGENT = "Mozilla/5.0"
