# 성분명은 INCI_Ingredient_mapping.csv 기준 INCI 표준명으로 통일
CONFLICT_RULES = {
        "Vitamin C": {
        "good" : ["Hyaluronic Acid", "Panthenol", "Vitamin E", "Ferulic Acid",
                     "Peptide", "Niacinamide", "Mineral Sunscreen", "Chemical Sunscreen"],
        "bad": ["Copper Peptide", "AHA", "BHA", "PHA", "LHA", "Retinol"],
    },
    "Niacinamide": {
        "good" : ["Hyaluronic Acid", "Panthenol", "Zinc", "Retinol", "Salicylic Acid",
                     "Vitamin C Derivatives", "Magnesium Ascorbyl Phosphate",
                     "Ascorbyl Glucoside", "Ethyl Ascorbic Ether", "Ascorbyl Tetraisopalmitate"],
        "bad": [],
    },
    "Retinol": {
        "good" : ["Hyaluronic Acid", "Panthenol", "Peptide"],
        "bad": ["AHA", "Azelaic Acid", "Salicylic Acid", "Benzoyl Peroxide"],
    },
    "Panthenol": {
        "good" : ["Vitamin C", "Niacinamide", "Hyaluronic Acid", "Peptide"],
        "bad": [],
    },
    "Hyaluronic Acid": {
        "good" : ["Vitamin C", "Niacinamide", "Panthenol", "Peptide", "Retinol", "Ceramide"],
        "bad": [],
    },
    "Peptide": {
        "good" : ["Hyaluronic Acid", "Niacinamide", "Panthenol"],
        "bad": ["Vitamin C"],
    },
    "Copper Peptide": {
        "good" : ["Hyaluronic Acid", "Panthenol"],
        "bad": ["Vitamin C", "Retinol"],
    },
    "Ceramide": {
        "good" : ["Hyaluronic Acid", "Panthenol", "Cholesterol", "Fatty Acids"],
        "bad": [],
    },
    "Benzoyl Peroxide": {
        "good" : ["Niacinamide"],
        "bad": ["Retinol", "Vitamin C"],
    },
    "Salicylic Acid": {
        "good" : ["Niacinamide", "Panthenol"],
        "bad": ["Vitamin C", "Retinol"],
    },
    "AHA": {
        "good" : [],
        "bad": ["Vitamin C", "Retinol"],
    },
    "PHA": {
        "good" : [],
        "bad": ["Vitamin C", "Retinol"],
    },
    "LHA": {
        "good" : [],
        "bad": ["Vitamin C", "Retinol"],
    },
}