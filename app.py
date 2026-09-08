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
        f"Urutan fitur tidak sesuai.\n\n"
        f"Ditemukan: {list(feature_names)}\n\n"
        f"Seharusnya: {expected_features}"
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Prediksi Tingkat Stres Mahasiswa Semester Akhir")
st.caption("Universitas Sam Ratulangi • Support Vector Machine (SVM) • Explainable AI (SHAP)")

st.markdown(
    """
Aplikasi ini digunakan untuk memprediksi tingkat stres mahasiswa
semester akhir Universitas Sam Ratulangi menggunakan **Support Vector
Machine (SVM)**.

Model menggunakan:

- **PASS** — faktor akademik
- **PSQI** — kualitas pola tidur

Tingkat stres sebagai variabel target diklasifikasikan berdasarkan
**PSS-10** ke dalam tiga kategori:

- Rendah
- Sedang
- Tinggi

Pendekatan **Explainable AI (SHAP)** digunakan untuk menjelaskan
kontribusi masing-masing variabel prediktor terhadap hasil prediksi.
"""
)

st.divider()


# ============================================================
# INPUT DATA
# ============================================================

st.header("1. Prediksi Tingkat Stres Mahasiswa")

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
# FUNGSI SHAP
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
# PROSES PREDIKSI
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # DATA INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[pass_score, psqi_score]],
        columns=feature_names
    )


    # --------------------------------------------------------
    # PREDIKSI MODEL SVM
    # --------------------------------------------------------

    prediction_encoded = model.predict(input_data)

    prediction_label = label_encoder.inverse_transform(
        prediction_encoded
    )[0]


    # --------------------------------------------------------
    # PROBABILITAS SETIAP KELAS
    # --------------------------------------------------------

    probabilities = model.predict_proba(input_data)[0]

    classes = label_encoder.classes_


    probability_df = pd.DataFrame({
        "Tingkat Stres": classes,
        "Probabilitas (%)": probabilities * 100
    })


    predicted_class_index = list(classes).index(
        prediction_label
    )


    # ========================================================
    # HASIL PREDIKSI
    # ========================================================

    st.divider()

    st.subheader("Hasil Prediksi Model")

    result_col, prob_col = st.columns(2)

    with result_col:

        st.metric(
            "Prediksi Model SVM",
            prediction_label
        )

    with prob_col:

        st.metric(
            "Probabilitas Kelas Hasil Prediksi",
            f"{probabilities[predicted_class_index] * 100:.2f}%"
        )


    # --------------------------------------------------------
    # CATATAN PROBABILITAS
    # --------------------------------------------------------

    st.caption(
        "Prediksi tingkat stres ditentukan berdasarkan hasil "
        "model.predict(). Nilai probabilitas ditampilkan "
        "berdasarkan model.predict_proba()."
    )


    # ========================================================
    # PROBABILITAS SETIAP KELAS
    # ========================================================

    st.subheader("Probabilitas Setiap Kelas")

    st.dataframe(
        probability_df.style.format({
            "Probabilitas (%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


    # Grafik probabilitas

    chart_data = probability_df.set_index(
        "Tingkat Stres"
    )

    st.bar_chart(
        chart_data["Probabilitas (%)"]
    )


    # ========================================================
    # SHAP LOCAL
    # ========================================================

    st.divider()

    st.header("2. Explainable AI — SHAP Lokal")

    st.write(
        "SHAP digunakan untuk melihat kontribusi masing-masing "
        "fitur terhadap output kelas yang diprediksi."
    )


    with st.spinner("Menghitung penjelasan SHAP..."):

        explainer = create_shap_explainer()

        shap_values = explainer.shap_values(
            input_data,
            nsamples=100
        )


    # --------------------------------------------------------
    # KONVERSI SHAP
    # --------------------------------------------------------

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


    # ========================================================
    # TABEL SHAP
    # ========================================================

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
            [
                "Fitur",
                "Nilai Input",
                "SHAP Value"
            ]
        ].style.format({

            "Nilai Input": "{:.2f}",

            "SHAP Value": "{:.6f}"

        }),

        use_container_width=True,

        hide_index=True
    )


    # ========================================================
    # GRAFIK SHAP
    # ========================================================

    st.subheader("Kontribusi Fitur terhadap Prediksi")

    shap_chart = shap_df[
        ["Fitur", "SHAP Value"]
    ].set_index("Fitur")


    st.bar_chart(
        shap_chart["SHAP Value"]
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

        f"Fitur dengan kontribusi absolut terbesar pada "
        f"input ini adalah **{dominant_feature}** dengan "
        f"nilai SHAP **{dominant_value:.6f}**. "
        f"Fitur tersebut {direction}."
    )


# ============================================================
# GLOBAL FEATURE IMPORTANCE SHAP
# ============================================================

st.divider()

st.header("5. Global Feature Importance — SHAP")

st.write(
    "Berdasarkan hasil analisis SHAP pada model final, "
    "kontribusi global fitur dihitung menggunakan "
    "rata-rata nilai absolut SHAP."
)

# ============================================================
# DISTRIBUSI PREDIKSI SELURUH UNSRAT
# ============================================================

st.divider()

st.header("3. Distribusi Prediksi Tingkat Stres Mahasiswa Semester Akhir UNSRAT")

st.write(
    "Distribusi berikut merupakan hasil prediksi model SVM "
    "terhadap 150 responden penelitian yang telah melalui proses "
    "screening dan digunakan dalam dataset penelitian."
)


@st.cache_data
def load_prediction_data():
    return pd.read_csv("hasil_prediksi_final.csv")


hasil_prediksi_final = load_prediction_data()


# ------------------------------------------------------------
# CEK KOLOM
# ------------------------------------------------------------

required_columns = [
    "Fakultas",
    "Prediksi"
]

missing_columns = [
    col for col in required_columns
    if col not in hasil_prediksi_final.columns
]

if missing_columns:

    st.error(
        f"Kolom berikut tidak ditemukan pada "
        f"hasil_prediksi_final.csv: {missing_columns}"
    )

    st.stop()


st.caption(f"Jumlah responden yang digunakan pada distribusi: **{len(hasil_prediksi_final)}**")


# ------------------------------------------------------------
# DISTRIBUSI PREDIKSI
# ------------------------------------------------------------

distribusi_unsrat = (
    hasil_prediksi_final["Prediksi"]
    .value_counts()
    .reindex(
        ["Rendah", "Sedang", "Tinggi"],
        fill_value=0
    )
    .reset_index()
)

distribusi_unsrat.columns = [
    "Tingkat Stres",
    "Jumlah"
]


distribusi_unsrat["Persentase (%)"] = (
    distribusi_unsrat["Jumlah"]
    / len(hasil_prediksi_final)
    * 100
)


# ------------------------------------------------------------
# TABEL
# ------------------------------------------------------------

st.dataframe(
    distribusi_unsrat.style.format({
        "Persentase (%)": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# GRAFIK
# ------------------------------------------------------------

grafik_unsrat = distribusi_unsrat.set_index(
    "Tingkat Stres"
)

st.bar_chart(
    grafik_unsrat["Persentase (%)"]
)

st.caption(
    "Interpretasi distribusi ini terbatas pada responden penelitian yang "
    "terdapat dalam hasil_prediksi_final.csv dan bukan estimasi prevalensi "
    "untuk seluruh populasi mahasiswa UNSRAT."
)

# ============================================================
# HASIL SHAP FINAL PENELITIAN
# ============================================================

global_shap = pd.DataFrame({

    "Fitur": [
        "PSQI",
        "PASS"
    ],

    "Mean Absolute SHAP": [
        0.027112,
        0.021735
    ]

})


# Menghitung kontribusi persentase
# berdasarkan hasil SHAP final

total_shap = global_shap[
    "Mean Absolute SHAP"
].sum()


global_shap["Kontribusi (%)"] = (

    global_shap["Mean Absolute SHAP"]

    / total_shap

    * 100
)


# Urutkan berdasarkan nilai SHAP
global_shap = global_shap.sort_values(
    "Mean Absolute SHAP",
    ascending=False
)

# ============================================================
# DISTRIBUSI PREDIKSI BERDASARKAN FAKULTAS
# ============================================================

st.divider()

st.header("4. Distribusi Prediksi Tingkat Stres Berdasarkan Fakultas")

st.write(
    "Tabel berikut menunjukkan distribusi hasil prediksi "
    "tingkat stres berdasarkan fakultas pada responden "
    "penelitian."
)


# ------------------------------------------------------------
# TABULASI FAKULTAS
# ------------------------------------------------------------

prediksi_fakultas = pd.crosstab(
    hasil_prediksi_final["Fakultas"],
    hasil_prediksi_final["Prediksi"]
)


# Pastikan ketiga kelas selalu tersedia
for kelas in ["Rendah", "Sedang", "Tinggi"]:

    if kelas not in prediksi_fakultas.columns:
        prediksi_fakultas[kelas] = 0


prediksi_fakultas = prediksi_fakultas[
    ["Rendah", "Sedang", "Tinggi"]
]


# ------------------------------------------------------------
# JUMLAH RESPONDEN PER FAKULTAS
# ------------------------------------------------------------

prediksi_fakultas["Total"] = (
    prediksi_fakultas[
        ["Rendah", "Sedang", "Tinggi"]
    ].sum(axis=1)
)


# ------------------------------------------------------------
# PERSENTASE PER FAKULTAS
# ------------------------------------------------------------

persentase_fakultas = (
    prediksi_fakultas[
        ["Rendah", "Sedang", "Tinggi"]
    ]
    .div(prediksi_fakultas["Total"], axis=0)
    * 100
)


persentase_fakultas = (
    persentase_fakultas
    .reset_index()
)


# ------------------------------------------------------------
# TAMPILKAN TABEL
# ------------------------------------------------------------

st.dataframe(
    persentase_fakultas.style.format({
        "Rendah": "{:.2f}%",
        "Sedang": "{:.2f}%",
        "Tinggi": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)

# ============================================================
# TABEL GLOBAL SHAP
# ============================================================

st.dataframe(

    global_shap.style.format({

        "Mean Absolute SHAP": "{:.6f}",

        "Kontribusi (%)": "{:.2f}%"

    }),

    use_container_width=True,

    hide_index=True
)


# ============================================================
# GRAFIK GLOBAL SHAP
# ============================================================

global_chart = global_shap[
    ["Fitur", "Kontribusi (%)"]
].set_index("Fitur")


st.bar_chart(
    global_chart["Kontribusi (%)"]
)


# ============================================================
# INTERPRETASI GLOBAL
# ============================================================

dominant_global_feature = global_shap.iloc[0]["Fitur"]

dominant_global_value = global_shap.iloc[0][
    "Mean Absolute SHAP"
]

dominant_global_percentage = global_shap.iloc[0][
    "Kontribusi (%)"
]


st.info(

    f"Berdasarkan analisis global SHAP, fitur yang memiliki "
    f"kontribusi terbesar terhadap output model adalah "
    f"**{dominant_global_feature}** dengan Mean Absolute SHAP "
    f"sebesar **{dominant_global_value:.6f}** atau sekitar "
    f"**{dominant_global_percentage:.2f}%** dari total kontribusi "
    f"fitur yang dianalisis."
)


# ============================================================
# CATATAN PENELITIAN
# ============================================================

st.divider()

st.caption(
    "Model SVM final menggunakan fitur PASS dan PSQI dengan "
    "Tingkat_Stres sebagai variabel target. "
    "PSS-10_Score tidak digunakan sebagai fitur masukan model. "
    "Nilai SHAP menunjukkan kontribusi fitur terhadap output model "
    "dan tidak dimaksudkan sebagai hubungan sebab-akibat."
)
