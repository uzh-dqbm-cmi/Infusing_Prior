from pathlib import Path

# Paths
HOME_DIR = Path.home()
PARSING_MODEL_DIR = HOME_DIR / ".local/share/bllipparser/GENIA+PubMed"

# Observation constants
NO_FINDING = "No Finding"
OBSERVATION = "observation"

CATEGORIES = ["Keyword", "Label"]

PHRASES = ["previous", "prior", "previously", "comparison", "preceding"]

# Numeric constants
POSITIVE = 1
NEGATIVE = 0

# Misc. constants
UNCERTAINTY = "uncertainty"
NEGATION = "negation"
REPORTS = "Reports"
KEYWORD = "Keyword"