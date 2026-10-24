"""Indian meal library. Carbs (g) per typical home serving and a glycaemic-index class.

GI values are approximate, drawn from the Indian GI literature
(e.g. Henry et al. 2008; Mohan et al., Madras Diabetes Research Foundation GI work).
`kabs` (1/min) is the gut absorption rate used by the physiology model:
high-GI foods empty faster -> sharper, earlier spikes.
"""

MEALS = {
    # breakfast
    "poha":            {"carbs": 45, "gi": 70, "label": "Poha (1 plate)"},
    "aloo_paratha":    {"carbs": 60, "gi": 65, "label": "Aloo paratha x2 + curd"},
    "idli_sambar":     {"carbs": 50, "gi": 77, "label": "Idli x3 + sambar"},
    "masala_dosa":     {"carbs": 65, "gi": 75, "label": "Masala dosa"},
    "upma":            {"carbs": 45, "gi": 68, "label": "Upma (1 bowl)"},
    "besan_chilla":    {"carbs": 25, "gi": 40, "label": "Besan chilla x2"},
    "oats_dalia":      {"carbs": 30, "gi": 50, "label": "Dalia / oats (1 bowl)"},
    "eggs_toast":      {"carbs": 25, "gi": 55, "label": "2 eggs + 1 toast"},
    # lunch / dinner
    "rice_dal":        {"carbs": 85, "gi": 72, "label": "Rice + dal + sabzi"},
    "roti_sabzi":      {"carbs": 55, "gi": 60, "label": "Roti x3 + sabzi + dal"},
    "rajma_chawal":    {"carbs": 90, "gi": 70, "label": "Rajma chawal"},
    "chole_bhature":   {"carbs": 110, "gi": 75, "label": "Chole bhature"},
    "biryani":         {"carbs": 100, "gi": 74, "label": "Chicken biryani (1 plate)"},
    "millet_roti":     {"carbs": 45, "gi": 52, "label": "Jowar/bajra roti x2 + sabzi"},
    "brown_rice_dal":  {"carbs": 60, "gi": 55, "label": "Brown rice (small) + dal + salad"},
    "salad_paneer":    {"carbs": 15, "gi": 35, "label": "Paneer tikka + salad"},
    # snacks
    "chai_biscuit":    {"carbs": 25, "gi": 70, "label": "Chai with sugar + 2 biscuits"},
    "samosa":          {"carbs": 35, "gi": 70, "label": "Samosa x1"},
    "fruit":           {"carbs": 20, "gi": 45, "label": "Apple / guava"},
    "sweets":          {"carbs": 45, "gi": 80, "label": "Gulab jamun x2"},
    "roasted_chana":   {"carbs": 15, "gi": 30, "label": "Roasted chana (handful)"},
    "none":            {"carbs": 0, "gi": 50, "label": "Skipped"},
}


def kabs_from_gi(gi: float) -> float:
    """Map GI (30..90) to gut absorption rate (1/min). ~0.008 (slow) .. ~0.028 (fast)."""
    gi = max(30.0, min(90.0, float(gi)))
    return 0.008 + (gi - 30.0) / 60.0 * 0.020


def meal(key: str, scale: float = 1.0) -> dict:
    m = MEALS.get(key, MEALS["none"])
    return {"key": key, "carbs": m["carbs"] * scale, "gi": m["gi"], "label": m["label"]}
