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
# LOAD MODEL
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
# VALIDASI FITUR
# ============================================================

expected_features = ["PASS", "PSQI"]

if list(feature_names) != expected_features:
    st.error(
        f"Urutan fitur tidak sesuai. "
        f"Ditemukan: {list(feature_names)} | "
        f"Seharusnya: {expected_features}"
    )
    st.stop()


# ============================================================
# SHAP
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
    Aplikasi ini digunakan untuk memprediksi tingkat stres mahasiswa
    semester akhir berdasarkan faktor akademik dan kualitas tidur
    menggunakan **Support Vector Machine (SVM)**.

    **Variabel input:**
    - **PASS** — skor faktor akademik
    - **PSQI** — skor kualitas tidur

    Pendekatan **Explainable AI (SHAP)** digunakan untuk menjelaskan
    kontribusi fitur terhadap hasil prediksi model.
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
        step=1.0
    )

with col2:
    psqi_score = st.number_input(
        "Skor PSQI",
        min_value=0.0,
        max_value=21.0,
        value=5.0,
        step=1.0
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

    # Prediksi utama SVM
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

    predicted_class_index = list(classes).index(
        prediction_label
    )

    highest_probability_index = np.argmax(probabilities)

    highest_probability_class = classes[
        highest_probability_index
    ]

    # ========================================================
    # HASIL
    # ========================================================

    st.divider()
    st.subheader("Hasil Prediksi")

    result_col, prob_col = st.columns(2)

    with result_col:
        st.metric(
            "Prediksi Model SVM",
            prediction_label
        )

    with prob_col:
        st.metric(
            "Probabilitas Kelas Prediksi",
            f"{probabilities[predicted_class_index] * 100:.2f}%"
        )

    # Jika probabilitas tertinggi berbeda dari predict()
    if highest_probability_class != prediction_label:
        st.warning(
            f"Prediksi utama SVM adalah **{prediction_label}**, "
            f"sedangkan probabilitas terbesar dari `predict_proba()` "
            f"adalah **{highest_probability_class}** "
            f"({probabilities[highest_probability_index] * 100:.2f}%). "
            f"Prediksi utama aplikasi tetap mengikuti `model.predict()`."
        )

    # ========================================================
    # PROBABILITAS
    # ========================================================

    st.subheader("Probabilitas Setiap Kelas")

    probability_display = probability_df.copy()

    probability_display["Probabilitas (%)"] = (
        probability_display["Probabilitas"] * 100
    )

    probability_display = probability_display[
        ["Tingkat Stres", "Probabilitas (%)"]
    ]

    st.dataframe(
        probability_display.style.format({
            "Probabilitas (%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Grafik menggunakan persen
    chart_data = probability_display.set_index(
        "Tingkat Stres"
    )

    st.bar_chart(
        chart_data,
        y="Probabilitas (%)"
    )

    # ========================================================
    # SHAP LOCAL
    # ========================================================

    st.divider()
    st.subheader("Penjelasan Prediksi dengan SHAP")

    st.write(
        "SHAP digunakan untuk melihat kontribusi masing-masing fitur "
        "terhadap output kelas yang diprediksi."
    )

    with st.spinner("Menghitung penjelasan SHAP..."):

        explainer = create_shap_explainer()

        shap_values = explainer.shap_values(
            input_data,
            nsamples=100
        )

    shap_array = np.asarray(shap_values)

    if isinstance(shap_values, list):

        local_shap = np.asarray(
            shap_values[predicted_class_index]
        )[0]

    elif shap_array.ndim == 3:

        local_shap = shap_array[
            0,
            :,
            predicted_class_index
        ]

    elif shap_array.ndim == 2:

        local_shap = shap_array[0]

    else:

        st.error(
            f"Bentuk SHAP Values tidak dikenali: "
            f"{shap_array.shape}"
        )

        st.stop()


    shap_df = pd.DataFrame({
        "Fitur": feature_names,
        "Nilai Input": [
            pass_score,
            psqi_score
        ],
        "SHAP Value": local_shap
    })

    shap_df["Kontribusi Absolut"] = (
        shap_df["SHAP Value"].abs()
    )

    shap_df = shap_df.sort_values(
        "Kontribusi Absolut",
        ascending=False
    )


    st.dataframe(
        shap_df[
            ["Fitur", "Nilai Input", "SHAP Value"]
        ].style.format({
            "Nilai Input": "{:.2f}",
            "SHAP Value": "{:.6f}"
        }),
        use_container_width=True,
        hide_index=True
    )


    # Grafik SHAP
    shap_chart = shap_df[
        ["Fitur", "SHAP Value"]
    ].set_index("Fitur")

    st.bar_chart(
        shap_chart,
        y="SHAP Value"
    )


    # ========================================================
    # INTERPRETASI SHAP
    # ========================================================

    dominant_feature = shap_df.iloc[0]["Fitur"]
    dominant_value = shap_df.iloc[0]["SHAP Value"]

    if dominant_value > 0:

        direction = (
            "memberikan kontribusi positif terhadap "
            "output kelas yang diprediksi"
        )

    elif dominant_value < 0:

        direction = (
            "memberikan kontribusi negatif terhadap "
            "output kelas yang diprediksi"
        )

    else:

        direction = (
            "tidak memberikan kontribusi berarti "
            "terhadap output kelas yang diprediksi"
        )


    st.info(
        f"Fitur dengan kontribusi absolut terbesar pada input ini "
        f"adalah **{dominant_feature}** dengan nilai SHAP "
        f"**{dominant_value:.6f}**. Fitur tersebut "
        f"{direction}."
    )


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

st.divider()

st.subheader("Global Feature Importance SHAP")

st.write(
    "Berdasarkan model final deployment, kontribusi global fitur "
    "dihitung menggunakan rata-rata nilai absolut SHAP."
)

global_shap = pd.DataFrame({
    "Fitur": ["PSQI", "PASS"],
    "Mean Absolute SHAP": [
        0.027112,
        0.021735
    ]
})

total_shap = global_shap[
    "Mean Absolute SHAP"
].sum()

global_shap["Kontribusi (%)"] = (
    global_shap["Mean Absolute SHAP"]
    / total_shap
    * 100
)

global_shap = global_shap.sort_values(
    "Mean Absolute SHAP",
    ascending=False
)

st.dataframe(
    global_shap.style.format({
        "Mean Absolute SHAP": "{:.6f}",
        "Kontribusi (%)": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)

global_chart = global_shap[
    ["Fitur", "Kontribusi (%)"]
].set_index("Fitur")

st.bar_chart(
    global_chart,
    y="Kontribusi (%)"
)


# ============================================================
# CATATAN
# ============================================================

st.divider()

st.caption(
    "Model SVM final menggunakan fitur PASS dan PSQI. "
    "Nilai SHAP menunjukkan kontribusi fitur terhadap output model "
    "dan tidak dimaksudkan sebagai hubungan sebab-akibat."
)
