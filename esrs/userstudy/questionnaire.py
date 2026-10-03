"""Post-study questionnaire."""

LIKERT = [
    ("PI1", "PI",           "I would consider buying one of these products."),
    ("PI2", "PI",           "I found the product suggestions appealing."),
    ("PI3", "PI",           "I would return to a store that recommends like this."),
    ("T1",  "Trust",        "I trust the recommendations I was shown."),
    ("T2",  "Trust",        "The recommendations seemed fair and unbiased."),
    ("T3",  "Trust",        "I believe the system has my interests in mind."),
    ("Tr1", "Transparency", "The system explained why each item was shown."),
    ("Tr2", "Transparency", "I understood the reason behind each recommendation."),
    ("Tr3", "Transparency", "The recommendation logic was clear to me."),
]

C_ONLY = [
    ("Pe", "Persuasiveness", "The explanations were persuasive."),
    ("Sp", "Specificity",    "The explanations gave specific, verifiable information."),
    ("Co", "Coherence",      "The explanations matched the products shown."),
    ("Fi", "Fidelity",       "The explanations reflected how the products were ranked."),
]

DEMOGRAPHICS = [
    ("age",                     "integer", "Age (years)"),
    ("gender",                  "choice",  "Gender",
     ["Female", "Male", "Other", "Prefer not to say"]),
    ("country",                 "text",    "Country of residence"),
    ("shopping_freq",           "choice",  "Online shopping frequency",
     ["Several times a week", "Weekly", "Monthly", "Rarely", "Never"]),
    ("sustainability_attitude", "likert",  "I try to choose sustainable products when I shop."),
    ("device",                  "choice",  "Device used",
     ["Desktop", "Laptop", "Tablet", "Phone"]),
    ("prior_esrs_exposure",     "choice",  "Have you seen this type of study before?",
     ["Yes", "No", "Not sure"]),
]


def likert_items_for_condition(condition):
    """PI/Trust/Transparency (all conditions) + Pe/Sp/Co/Fi (condition C only)."""
    items = list(LIKERT)
    if condition == "C":
        items += list(C_ONLY)
    return items