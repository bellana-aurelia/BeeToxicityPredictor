import base64
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy import stats
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATASET_PATH = APP_DIR / "apistox_dataset.csv"
MODEL_PATH = APP_DIR / "best_model.pkl"


def smiles_to_features(smiles: str) -> Dict[str, float]:
    """Generate RDKit molecular descriptors from a SMILES string."""
    if not smiles or not isinstance(smiles, str):
        return {
            "MolWt": 0.0,
            "TPSA": 0.0,
            "NumHDonors": 0,
            "NumHAcceptors": 0,
            "MolLogP": 0.0,
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "MolWt": 0.0,
            "TPSA": 0.0,
            "NumHDonors": 0,
            "NumHAcceptors": 0,
            "MolLogP": 0.0,
        }

    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "NumHDonors": int(Descriptors.NumHDonors(mol)),
        "NumHAcceptors": int(Descriptors.NumHAcceptors(mol)),
        "MolLogP": float(Descriptors.MolLogP(mol)),
    }


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used in the notebook."""
    df = df.copy().drop_duplicates().reset_index(drop=True)

    features = df["SMILES"].fillna("").astype(str).apply(smiles_to_features)
    features_df = pd.DataFrame(features.tolist())
    df = pd.concat([df, features_df], axis=1)
    df = df.drop(columns=["SMILES"], errors="ignore")

    # One-hot encode toxicity_type as in the notebook while dropping the reference class.
    df = pd.get_dummies(df, columns=["toxicity_type"], drop_first=True)

    # Keep only training columns used by the final model.
    training_cols = [
        "herbicide",
        "fungicide",
        "insecticide",
        "other_agrochemical",
        "MolWt",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "MolLogP",
        "toxicity_type_Oral",
        "toxicity_type_Other",
        "label",
    ]

    for col in training_cols:
        if col not in df.columns:
            df[col] = 0

    return df[training_cols]


def build_model() -> Pipeline:
    """Train the notebook's Random Forest classifier using the notebook pipeline."""
    df = pd.read_csv(DATASET_PATH)
    prepared = prepare_dataframe(df)

    X = prepared.drop(columns=["label"])
    y = prepared["label"]

    numeric_cols = [
        "herbicide",
        "fungicide",
        "insecticide",
        "other_agrochemical",
        "MolWt",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "MolLogP",
        "toxicity_type_Oral",
        "toxicity_type_Other",
    ]

    preprocessor = ColumnTransformer(
        transformers=[("numeric", StandardScaler(), numeric_cols)],
        remainder="drop",
    )

    model = RandomForestClassifier(
        max_depth=5,
        max_features="sqrt",
        min_samples_split=5,
        n_estimators=100,
        random_state=21,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def build_prediction_row(smiles: str, toxicity_type: str, herbicide: int, fungicide: int, insecticide: int, other_agrochemical: int) -> pd.DataFrame:
    """Build a single-row DataFrame matching the training feature columns."""
    mol_features = smiles_to_features(smiles)
    row = {
        "herbicide": int(herbicide),
        "fungicide": int(fungicide),
        "insecticide": int(insecticide),
        "other_agrochemical": int(other_agrochemical),
        "MolWt": mol_features["MolWt"],
        "TPSA": mol_features["TPSA"],
        "NumHDonors": mol_features["NumHDonors"],
        "NumHAcceptors": mol_features["NumHAcceptors"],
        "MolLogP": mol_features["MolLogP"],
        "toxicity_type_Oral": 1 if toxicity_type == "Oral" else 0,
        "toxicity_type_Other": 1 if toxicity_type == "Other" else 0,
    }

    columns = [
        "herbicide",
        "fungicide",
        "insecticide",
        "other_agrochemical",
        "MolWt",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "MolLogP",
        "toxicity_type_Oral",
        "toxicity_type_Other",
    ]
    return pd.DataFrame([row], columns=columns)

DESC_COLS = [
    "MolWt",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "MolLogP"
]

DESC_LABELS = {
    "MolWt": "Molecular Weight",
    "TPSA": "Topological Polar Surface Area",
    "NumHDonors": "Number of Hydrogen Donors",
    "NumHAcceptors": "Number of Hydrogen Acceptors",
    "MolLogP": "Molecular LogP"
}

EXAMPLES = {
    "Imidacloprid": {
        "smiles": "CC1=CC(=C(C=C1)N2C=NC=N2)C3=CN=CN3",
        "toxicity_type": "Oral",
        "herbicide": 0,
        "fungicide": 0,
        "insecticide": 1,
        "other_agrochemical": 0
    },
    "Glyphosate": {
        "smiles": "C(C(=O)O)NCP(=O)(O)O",
        "toxicity_type": "Other",
        "herbicide": 1,
        "fungicide": 0,
        "insecticide": 0,
        "other_agrochemical": 0
    },
    "Tebuconazole": {
        "smiles": "CC(C)N1C=NC2=C1C(=CC=C2)C3=CC(=C(C=C3)Cl)Cl",
        "toxicity_type": "Oral",
        "herbicide": 0,
        "fungicide": 1,
        "insecticide": 0,
        "other_agrochemical": 0
    },
}

st.set_page_config(page_title="Bee Toxicity Predictor", page_icon="🐝", layout="wide")

def load_css(path: str = "style.css") -> None:
    """Load the stylesheet from a separate file."""
    css_file = APP_DIR / path
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

@st.cache_resource(show_spinner = "Loading the trained Bee Toxicity model...")
def get_model() -> Pipeline:
    """Load the saved model if it exists, otherwise train and persist the deployed model."""
    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as fh:
            return pickle.load(fh)

    model = build_model()
    with MODEL_PATH.open("wb") as fh:
        pickle.dump(model, fh)
    return model

@st.cache_data
def get_stats() -> dict:
    """Dataset summary for the sidebar and the out-of-range warning."""
    prepared = prepare_dataframe(pd.read_csv(DATASET_PATH))
    return {
        "n": int(len(prepared)),
        "toxic_pct": float(prepared["label"].mean() * 100),
        "ranges":{c: (float(prepared[c].min()), float(prepared[c].max())) for c in DESC_COLS}
    }

def mol_to_svg(smiles: str) -> str:
    """Convert a SMILES string to an SVG image."""
    mol = Chem.MolFromSmiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

def apply_example(name: str) -> None:
    ex = EXAMPLES[name]
    for k in ("smiles", "toxicity_type", "herbicide", "fungicide", "insecticide", "other_agrochemical"):
        st.session_state[k] = ex[k]
    st.session_state["result"] = None

load_css()
model = get_model()
stats = get_stats()

defaults = {
    "smiles": EXAMPLES["Imidacloprid"]["smiles"],
    "toxicity_type": "Oral",
    "herbicide": False,
    "fungicide": False,
    "insecticide": False,
    "other_agrochemical": False,
    "result": None
}

for k, v in defaults.items():
    st.session_state.setdefault(k, v)


with st.sidebar:
    st.markdown("## 🐝 About")
    st.write(
        "This app predicts wheter an agrochemical is **toxic to honey bees**, "
        "using a Random Forest trained on the ApisTox dataset."
    )
    c1, c2 = st.columns(2)
    c1.metric("Training samples", f"{stats['n']:,}")
    c2.metric("Toxicity rate", f"{stats['toxic_pct']:.2f}%")
    
    st.markdown("## Try an example")
    for name in EXAMPLES:
        st.button(name, key=f"example_{name}", on_click=apply_example, args=(name,), use_container_width=True)
    
    st.markdown("## Settings")
    threshold = st.slider(
        "Toxic threshold", 0.05, 0.95, 0.5, 0.05, 
        help="The probability threshold above which a chemical is considered toxic."
    )
    st.caption("For educational purposes only. This app is not a substitute for laboratory toxicity testing.")
    
st.markdown(
    '<div class="hero"><h1>🐝 Bee Toxicity Predictor</h1>'
    "<p>Predict whether an agrochemical compound is toxic to honey bees from its "
    "molecular structure and agrochemical category.</p>"
    '<span class="pill">🧬 RDKit descriptors</span>'
    '<span class="pill">🌲 Random Forest</span>'
    '<span class="pill">📊 ApisTox dataset</span></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap = "large")

with left:
    st.markdown("## 🧪 Compound input")
    with st.form("prediction_form"):
        smiles = st.text_input("SMILES string", key="smiles", help="Example: COO (ethanol)")
        toxicity_type = st.radio("Toxicity type", ["Contact", "Oral", "Other"], horizontal=True, key="toxicity_type")
        st.markdown("**Agrochemical category**")
        a, b = st.columns(2)
        herbicide = int(a.toggle("🌿 Herbicide", key="herbicide"))
        fungicide = int(b.toggle("🍄 Fungicide", key="fungicide"))
        insecticide = int(a.toggle("🐜 Insecticide", key="insecticide"))
        other_agrochemical = int(b.toggle("🧴 Other", key="other_agrochemical"))
        submitted = st.form_submit_button("Predict toxicity", use_container_width=True, type="primary")
        
    if submitted:
        if Chem.MolFromSmiles(smiles) is None:
            st.session_state["result"] = None
            st.error("Invalid SMILES string. Please enter a valid chemical structure.")
        else:
            row = build_prediction_row(smiles, toxicity_type, herbicide, fungicide, insecticide, other_agrochemical)
            st.session_state["result"] = {
                "prob": float(model.predict_proba(row)[0][1]),
                "desc": {c: float(row[c].iloc[0]) for c in DESC_COLS},
                "svg": mol_to_svg(smiles)
            }

with right:
    st.markdown("## 📋  Prediction result")
    res = st.session_state["result"]
    if res is None:
        st.markdown(
            '<div class="placeholder"><span>🧪</span><br><b>No prediction yet</b></br>'
            "Enter a SMILES string and press <i>Predict toxicity</i><div>",
            unsafe_allow_html=True
        )
    else:
        prob = res["prob"]
        toxic = prob >= threshold
        color = "#C0392B" if toxic else "#2E8B57"
        css = "result-toxic" if toxic else "result-safe"
        title = "☠️ Toxic to bees" if toxic else "✅ Non-toxic to bees"
        st.markdown(
            f'<div class="result-card {css}">'
            f'<div class="gauge" style="background: conic-gradient({color} {prob * 100:.1f}%, #e9e2d0 0);">'
            f'<div class="gauge-inner"><b>{prob * 100:.0f}%</b><small>toxic prob.</small></div></div>'
            f"<div><h2>{title}</h2>"
            f"<p>Toxic {prob * 100:.1f}% &nbsp;|&nbsp; Non-toxic {(1 - prob) * 100:.1f}%<br>"
            f"Threshold: {threshold:.2f}</p></div></div>",
            unsafe_allow_html=True,
        )

res = st.session_state["result"]
if res is not None:
    st.markdown("## 🧬 Molecular descriptors")
    cols = st.columns(5)
    for col, key in zip(cols, DESC_COLS):
        val = res["desc"][key]
        col.metric(DESC_LABELS[key], f"{val:.2f}", help=f"{DESC_LABELS[key]}: {val:.2f}")
        
    outside = [DESC_LABELS[c] for c in DESC_COLS
               if not (stats["ranges"][c][0] <= res["desc"][c] <= stats["ranges"][c][1])]
    if outside:
        st.warning(
            "⚠️ Some molecular descriptors are outside the range of the training dataset: "
            + ", ".join(outside)
        )

st.markdown("---")
st.caption(
    "This app is for educational purposes only. It is not a substitute for laboratory toxicity testing. "
    "The model was trained on the ApisTox dataset and may not generalize to all agrochemical compounds."
)