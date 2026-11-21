import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cervical Cancer Risk Prediction", layout="centered"
)

# 1. Load the saved model bundle from Colab
artifact = joblib.load("cervical_cancer_model.pkl")
model = artifact["model"]
feature_names = artifact["features"]
threshold = artifact["threshold"]
model_name = artifact.get("model_name", "Boosting Model")
config_name = artifact.get("config_name", "Selected Features")
medians = artifact.get("medians", {})

# 2. Page Header
st.title("Cervical Cancer Biopsy Risk Prediction")
st.caption(
    f"Winning Model: **{model_name}** ({config_name}) | Decision Threshold:"
    f" **{threshold:.4f}**"
)
st.write(
    "Enter patient clinical and behavioral details below to predict biopsy"
    " risk."
)

# 3. Columns that are Yes/No (1/0) questions in your dataset
binary_flags = {
    "Smokes",
    "Hormonal Contraceptives",
    "IUD",
    "STDs",
    "STDs:condylomatosis",
    "STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis",
    "STDs:pelvic inflammatory disease",
    "STDs:genital herpes",
    "STDs:molluscum contagiosum",
    "STDs:AIDS",
    "STDs:HIV",
    "STDs:Hepatitis B",
    "STDs:HPV",
}

# 4. Build the input form dynamically from your saved best_cols
user_inputs = {}
col1, col2 = st.columns(2)

for idx, feat in enumerate(feature_names):
  target_col = col1 if idx % 2 == 0 else col2
  with target_col:
    if feat in binary_flags:
      choice = st.selectbox(
          feat,
          options=["No (0)", "Yes (1)", "Unknown"],
          index=0,
          key=f"input_{idx}",
      )
      if choice == "Yes (1)":
        user_inputs[feat] = 1.0
      elif choice == "No (0)":
        user_inputs[feat] = 0.0
      else:
        user_inputs[feat] = np.nan
    else:
      default_val = float(medians.get(feat, 0.0))
      if np.isnan(default_val):
        default_val = 0.0
      user_inputs[feat] = st.number_input(
          feat,
          min_value=0.0,
          value=default_val,
          step=1.0,
          key=f"input_{idx}",
      )

# 5. Predict when button is clicked
if st.button("Predict Biopsy Risk", type="primary", use_container_width=True):
  input_df = pd.DataFrame([user_inputs])[feature_names]

  prob = float(model.predict_proba(input_df)[0, 1])
  prediction = int(prob >= threshold)

  st.divider()
  st.metric(label="Estimated Positive Biopsy Probability", value=f"{prob:.2%}")

  if prediction == 1:
    st.error(
        f"**High Risk (Positive Biopsy Prediction)** — Probability ({prob:.2%})"
        f" is at or above the cutoff ({threshold:.2%})."
    )
  else:
    st.success(
        f"**Low Risk (Negative Biopsy Prediction)** — Probability ({prob:.2%})"
        f" is below the cutoff ({threshold:.2%})."
    )

  st.info(
      "**Disclaimer:** This research prototype is trained on the UCI Cervical"
      " Cancer Risk Factors dataset and is not a clinically validated"
      " diagnostic tool."
  )