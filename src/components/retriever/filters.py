# filters.py — define valid values for each filterable field declared in params.cfg.
#
# Rules:
# - Keys must match field names in params.cfg [metadata_filters] filterable_fields exactly.
# - Every field in filterable_fields MUST have an entry here — missing entries raise a
#   ValueError at startup.
# - Values must match what is stored in Qdrant payload metadata for that field.
# - Extra keys here that are not in filterable_fields are silently ignored.

FILTER_VALUES: dict[str, list] = {
    "crop_type": ["wheat", "maize", "strawberry", "rice"],
    "title": ["protection_from_insects", "irrigation", "soil_management"],
}
