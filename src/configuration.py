from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs" / "figures"

MADRID_FILE = DATA_DIR / "madrid_quantumVR_learningimpact.xlsx"
SEGOVIA_FILE = DATA_DIR / "segovia_quantumVR_learningimpact.xlsx"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRE_POST_METRICS = {
    "Test score": ("Score", "Score 2"),
    "Interest": (
        "I'm interested in learning quantum computing",
        "I'm interested in learning quantum computing 2",
    ),
    "Relevance": (
        "Quantum computing feels relevant to my studies and/or career",
        "Quantum computing feels relevant to my studies and/or career 2",
    ),
    "Future work": (
        "I would like to work on something related to quantum computing in the future",
        "I would like to work on something related to quantum computing in the future 2",
    ),
    "Confidence": (
        "I feel confident I can learn the basics of quantum computing",
        "I feel confident I can learn the basics of quantum computing 2",
    ),
}

VR_METRICS = [
    "The VR experience helped me understand qubits, superposition, entanglement, how measurement affects a quantum system, and the fragility of quantum systems",
    "Completing the VR game increased my motivation to learn quantum computing",
    "The VR game's difficulty was appropriate",
    "I would recommend this VR activity to a friend that is curious about quantum computing",
]

MADRID_SEMINAR = {
    "madrid0002", "madrid0007", "madrid0011", "madrid0014", "madrid0015",
    "madrid0028", "madrid0029", "madrid0030", "madrid0031", "madrid0032",
    "madrid0033", "madrid0034", "madrid0035", "madrid0037", "madrid0038",
    "madrid0039", "madrid0040", "madrid0041", "madrid0043", "madrid0048",
    "madrid0049", "madrid0051", "madrid0052",
}

SEGOVIA_SEMINAR = {
    "segovia0001", "segovia0002", "segovia0003", "segovia0004", "segovia0005",
    "segovia0006", "segovia0008", "segovia0009", "segovia0010", "segovia0011",
    "segovia0012", "segovia0013", "segovia0014", "segovia0016", "segovia0017",
    "segovia0018", "segovia0020",
}