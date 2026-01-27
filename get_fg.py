import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from fg_core import get_fgi_estimation
import warnings

try:
    from pandas.errors import Pandas4Warning
except ImportError:
    Pandas4Warning = FutureWarning

warnings.filterwarnings(
    "ignore",
    category=Pandas4Warning,
    message="Timestamp.utcnow is deprecated and will be removed in a future version. Use Timestamp.now\\('UTC'\\) instead.",
)

# Constants
# 
CALIB_MODEL_PATH = Path("models") / "fg_linear_weights.pkl"
load_dotenv()
API_KEY = os.getenv("API_KEY", "")

def load_model(
    path: str | Path,
) -> tuple[dict[str, float] | None, float]:
    """
    Charge un modèle linéaire scikit-learn sauvegardé en pickle
    et en extrait l'intercept et les poids par composante.

    Hypothèses :
    - modèle avec attributs `coef_` et `intercept_`
    - entraîné sur des colonnes de type 'score_momentum_spx', etc.
      => on retire le préfixe 'score_' pour revenir aux noms bruts
         attendus par `compute_fear_greed`.
    """
    path = Path(path)
    if not path.exists():
        # pas de modèle : on utilisera la moyenne simple
        return None, 0.0

    with open(path, "rb") as f:
        model = pickle.load(f)

    intercept = float(model.intercept_)

    # Récupération des noms de features utilisés à l'entraînement
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        # fallback : à adapter si tu stockes les noms ailleurs
        raise ValueError(
            "The loaded model has no `feature_names_in_`. "
            "Save the model with this attribute or adapt the loader."
        )

    coefs = pd.Series(model.coef_, index=feature_names, dtype=float)

    # 'score_momentum_spx' -> 'momentum_spx', etc.
    weights = {}
    for col, w in coefs.items():
        base = col.replace("score_", "")
        weights[base] = float(w)

    return weights, intercept


if __name__ == "__main__":
    # Input
    start = input("Start date (YYYY-MM-DD) [default: 1 year ago]: ").strip()
    end = input("End date (YYYY-MM-DD) [default: today]: ").strip()
    with_components_input = input("Include component components? (y/n) [default: y]: ").strip().lower()
    with_components = with_components_input != "n"
    start_date = start if start else None
    end_date = end if end else None
    
    # Estimation
    fg_df = get_fgi_estimation(
        start_date=start_date,
        end_date=end_date,
        with_components=with_components,
        use_calibrated_model=True,
    )
    print(fg_df.head())