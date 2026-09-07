import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Prediksi Tingkat Stres Mahasiswa",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# LOAD MODEL DAN FILE PENDUKUNG
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("svm_model_final.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_names = joblib.load("feature_names.pkl")
    background = joblib.load("shap_background.pkl")

    return model, label_encoder, feature_names, background


model, label_encoder, feature_names, background = load_model()


# ============================================================
# VALIDASI FILE MODEL
# ============================================================

expected_features = ["PASS", "PSQI"]

if list(feature_names) != expected_features:
    st.error(
        f"Urutan fitur model tidak sesuai. "
        f"Ditemukan: {list(feature_names)} | "
        f"Seharusnya: {expected_features}"
    )
    st.stop()


# ============================================================
# FUNGSI PREDIKSI UNTUK SHAP
# ============================================================

def predict_proba_for_shap(data):
    data_df = pd.DataFrame(
        data,
        columns=feature_names
    )
    return model.predict_proba(data_df)


@st.cache_resource
def create_shap_explainer():
    return shap.KernelExplainer(
        predict_proba_for_shap,
        background
    )


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Prediksi Tingkat Stres Mahasiswa Semester Akhir")

st.markdown(
    """
    Aplikasi ini menggunakan **Support Vector Machine (SVM)**
    untuk memprediksi tingkat stres mahasiswa berdasarkan:

    - **PASS** — skor faktor akademik
    - **PSQI** — skor kualitas tidur

    Pendekatan **Explainable AI (SHAP)** digunakan untuk memberikan
    penjelasan terhadap kontribusi fitur pada hasil prediksi.
    """
)

st.divider()


# ============================================================
# INPUT
# ============================================================

st.subheader("Input Data Mahasiswa")

col1, col2 = st.columns(2)

with col1:
    pass_score = st.number_input(
        "Skor PASS",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0,
        help="Masukkan skor PASS yang diperoleh mahasiswa."
    )

with col2:
    psqi_score = st.number_input(
        "Skor PSQI",
        min_value=0.0,
        max_value=21.0,
        value=5.0,
        step=1.0,
        help="Skor global PSQI berada pada rentang 0–21."
    )


predict_button = st.button(
    "🔍 Prediksi Tingkat Stres",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDIKSI
# ============================================================

if predict_button:

    input_data = pd.DataFrame(
        [[pass_score, psqi_score]],
        columns=feature_names
    )

    # Prediksi kelas
    prediction_encoded = model.predict(input_data)

    prediction_label = label_encoder.inverse_transform(
        prediction_encoded
    )[0]

    # Probabilitas
    probabilities = model.predict_proba(input_data)[0]

    classes = label_encoder.classes_

    probability_df = pd.DataFrame({
        "Tingkat Stres": classes,
        "Probabilitas": probabilities
    })

    # Index kelas hasil prediksi
    predicted_class_index = list(classes).index(
        prediction_label
    )

    # ========================================================
    # HASIL PREDIKSI
    # ========================================================

    st.divider()
    st.subheader("Hasil Prediksi")

    result_col, prob_col = st.columns([1, 1])

    with result_col:
        st.metric(
            "Prediksi Tingkat Stres",
            prediction_label
        )

    with prob_col:
        st.metric(
            "Probabilitas Prediksi",
            f"{probabilities[predicted_class_index] * 100:.2f}%"
        )

    # ========================================================
    # PROBABILITAS SETIAP KELAS
    # ========================================================

    st.subheader("Probabilitas Setiap Kelas")

    probability_display = probability_df.copy()
    probability_display["Probabilitas"] = (
        probability_display["Probabilitas"] * 100
    )

    st.dataframe(
        probability_display.style.format(
            {"Probabilitas": "{:.2f}%"}
        ),
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        probability_df.set_index("Tingkat Stres"),
        y="Probabilitas"
    )

    # ========================================================
    # SHAP LOCAL EXPLANATION
    # ========================================================

    st.divider()
    st.subheader("Penjelasan Prediksi dengan SHAP")

    with st.spinner("Menghitung penjelasan SHAP..."):

        explainer = create_shap_explainer()

        shap_values = explainer.shap_values(
            input_data,
            nsamples=100
        )

    shap_array = np.asarray(shap_values)

    # Kompatibilitas dengan beberapa bentuk output SHAP
    if isinstance(shap_values, list):
        local_shap = np.asarray(
            shap_values[predicted_class_index]
        )[0]

    elif shap_array.ndim == 3:
        # shape: (jumlah_data, jumlah_fitur, jumlah_kelas)
        local_shap = shap_array[
            0,
            :,
            predicted_class_index
        ]

    elif shap_array.ndim == 2:
        # shape: (jumlah_data, jumlah_fitur)
        local_shap = shap_array[0]

    else:
        st.error(
            f"Bentuk SHAP Values tidak dikenali: "
            f"{shap_array.shape}"
        )
        st.stop()

    shap_df = pd.DataFrame({
        "Fitur": feature_names,
        "SHAP Value": local_shap,
        "Nilai Input": [
            pass_score,
            psqi_score
        ]
    })

    shap_df["Kontribusi Absolut"] = (
        shap_df["SHAP Value"].abs()
    )

    shap_df = shap_df.sort_values(
        "Kontribusi Absolut",
        ascending=False
    )

    st.write(
        "Nilai absolut SHAP yang lebih besar menunjukkan "
        "kontribusi yang lebih besar terhadap output kelas "
        "yang diprediksi."
    )

    st.dataframe(
        shap_df[
            ["Fitur", "Nilai Input", "SHAP Value"]
        ].style.format({
            "SHAP Value": "{:.6f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    shap_chart = shap_df[
        ["Fitur", "SHAP Value"]
    ].set_index("Fitur")

    st.bar_chart(shap_chart)

    # ========================================================
    # INTERPRETASI SEDERHANA
    # ========================================================

    dominant_feature = shap_df.iloc[0]["Fitur"]
    dominant_value = shap_df.iloc[0]["SHAP Value"]

    if dominant_value > 0:
        direction = "mendorong output model ke kelas yang diprediksi"
    elif dominant_value < 0:
        direction = "menurunkan output model untuk kelas yang diprediksi"
    else:
        direction = "tidak memberikan kontribusi berarti pada output kelas yang diprediksi"

    st.info(
        f"Pada input ini, fitur dengan kontribusi absolut terbesar "
        f"adalah **{dominant_feature}**. Nilai SHAP sebesar "
        f"**{dominant_value:.6f}** menunjukkan bahwa fitur tersebut "
        f"{direction}."
    )


# ============================================================
# CATATAN
# ============================================================

st.divider()

st.caption(
    "Model SVM final menggunakan fitur PASS dan PSQI. "
    "Penjelasan SHAP menunjukkan kontribusi fitur terhadap "
    "prediksi model dan tidak dimaksudkan sebagai hubungan sebab-akibat."
)
