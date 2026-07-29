"""RFMiD label definitions and the official RIADD evaluation grouping.

The CSVs ship 46 target columns: `Disease_Risk` (binary screening label) plus
45 disease columns. The RIADD/ISBI-2021 challenge scored the multi-disease task
over **28** categories: the 27 most frequent diseases plus a single `OTHER`
bucket that absorbs the remaining 18 rare classes.
"""

from __future__ import annotations

BINARY_LABEL = "Disease_Risk"

# All 45 disease columns, in CSV order.
DISEASE_LABELS = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
    "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA",
    "VS", "BRAO", "PLQ", "HPED", "CL",
]

# Model output order: index 0 is the binary head, 1..45 are the diseases.
ALL_LABELS = [BINARY_LABEL] + DISEASE_LABELS
NUM_CLASSES = len(ALL_LABELS)  # 46

# The 27 diseases scored individually by the challenge.
SCORED_LABELS = DISEASE_LABELS[:27]
# Everything else is folded into OTHER (logical OR of the raw columns).
OTHER_LABELS = DISEASE_LABELS[27:]
# Official multi-disease target set: 28 categories.
RIADD_ML_LABELS = SCORED_LABELS + ["OTHER"]

LABEL_TO_INDEX = {name: i for i, name in enumerate(ALL_LABELS)}

# Full names, used for human-readable reports.
LABEL_FULL_NAME = {
    "Disease_Risk": "Any abnormality present",
    "DR": "Diabetic retinopathy",
    "ARMD": "Age-related macular degeneration",
    "MH": "Media haze",
    "DN": "Drusen",
    "MYA": "Myopia",
    "BRVO": "Branch retinal vein occlusion",
    "TSLN": "Tessellation",
    "ERM": "Epiretinal membrane",
    "LS": "Laser scars",
    "MS": "Macular scar",
    "CSR": "Central serous retinopathy",
    "ODC": "Optic disc cupping",
    "CRVO": "Central retinal vein occlusion",
    "TV": "Tortuous vessels",
    "AH": "Asteroid hyalosis",
    "ODP": "Optic disc pallor",
    "ODE": "Optic disc edema",
    "ST": "Optociliary shunt",
    "AION": "Anterior ischemic optic neuropathy",
    "PT": "Parafoveal telangiectasia",
    "RT": "Retinal traction",
    "RS": "Retinitis",
    "CRS": "Chorioretinitis",
    "EDN": "Exudation",
    "RPEC": "Retinal pigment epithelium changes",
    "MHL": "Macular hole",
    "RP": "Retinitis pigmentosa",
    "CWS": "Cotton wool spots",
    "CB": "Coloboma",
    "ODPM": "Optic disc pit maculopathy",
    "PRH": "Preretinal haemorrhage",
    "MNF": "Myelinated nerve fibers",
    "HR": "Hemorrhagic retinopathy",
    "CRAO": "Central retinal artery occlusion",
    "TD": "Tilted disc",
    "CME": "Cystoid macular edema",
    "PTCR": "Post-traumatic choroidal rupture",
    "CF": "Choroidal folds",
    "VH": "Vitreous haemorrhage",
    "MCA": "Macroaneurysm",
    "VS": "Vasculitis",
    "BRAO": "Branch retinal artery occlusion",
    "PLQ": "Plaque",
    "HPED": "Haemorrhagic pigment epithelial detachment",
    "CL": "Collateral",
    "OTHER": "Other / rare findings",
}
