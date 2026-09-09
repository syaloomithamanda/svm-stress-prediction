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

st.markdown(
    """
    <h1 style="
        font-size: 2.35rem;
        line-height: 1.2;
        margin-bottom: 0.4rem;
    ">
        🧠 Prediksi Tingkat Stres Mahasiswa Semester Akhir
        Universitas Sam Ratulangi Menggunakan Algoritma
        Support Vector Machine dengan Pendekatan Explainable AI
    </h1>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Universitas Sam Ratulangi • Support Vector Machine (SVM) • Explainable AI (SHAP)"
)


# ============================================================
# INPUT DATA
# ============================================================

st.header("1. Prediksi Tingkat Stres Mahasiswa")

st.write(
    "Masukkan skor variabel prediktor penelitian, yaitu faktor akademik "
    "(PASS) dan kualitas pola tidur (PSQI). PSS-10 tidak dimasukkan sebagai "
    "input karena digunakan sebagai dasar pembentukan target tingkat stres."
)

col1, col2 = st.columns(2)

with col1:
    pass_score = st.number_input(
        "Skor PASS (Faktor Akademik)",
        min_value=40.0,
        max_value=68.0,
        value=None,
        step=1.0,
        help="Masukkan skor PASS. Rentang 40–68 merupakan rentang skor PASS yang terdapat pada 150 responden dalam dataset penelitian."
    )
    st.caption("Rentang pada dataset penelitian: 40–68")

with col2:
    psqi_score = st.number_input(
        "Skor PSQI (Kualitas Pola Tidur)",
        min_value=2.0,
        max_value=18.0,
        value=None,
        step=1.0,
        help="Masukkan skor PSQI. Rentang 2–18 merupakan rentang skor PSQI yang terdapat pada 150 responden dalam dataset penelitian."
    )
    st.caption("Rentang pada dataset penelitian: 2–18")


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
    if pass_score is None or psqi_score is None:
        st.warning("Silakan masukkan skor PASS dan PSQI terlebih dahulu sebelum melakukan prediksi.")
        st.stop()


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
    # PROBABILITAS PREDIKSI
    # --------------------------------------------------------

    # predict_proba() menghasilkan probabilitas untuk setiap
    # kelas berdasarkan urutan model.classes_.
    probabilities = model.predict_proba(input_data)[0]

    # Urutan kelas pada model
    model_class_encoded = model.classes_

    # Konversi label encoded menjadi label asli
    model_class_labels = label_encoder.inverse_transform(
        model_class_encoded
    )

    # DataFrame probabilitas
    probability_df = pd.DataFrame({
        "Tingkat Stres": model_class_labels,
        "Probabilitas (%)": probabilities * 100
    })

    # --------------------------------------------------------
    # MENENTUKAN POSISI KELAS HASIL PREDIKSI
    # --------------------------------------------------------

    # prediction_encoded[0] adalah kelas yang dihasilkan
    # oleh model.predict().
    #
    # predicted_class_index adalah posisi kelas tersebut
    # pada output predict_proba().
    predicted_class_index = np.where(
        model_class_encoded == prediction_encoded[0]
    )[0][0]

    # Probabilitas kelas yang diprediksi
    predicted_class_probability = probabilities[
        predicted_class_index
    ] * 100

    # ========================================================
    # HASIL PREDIKSI MODEL
    # ========================================================

    st.subheader("Hasil Prediksi Model")

    st.caption(
        "Hasil utama prediksi ditentukan menggunakan "
        "`model.predict()`. Nilai probabilitas pada bagian "
        "berikut berasal dari `model.predict_proba()` dan "
        "digunakan sebagai informasi tambahan."
    )

    result_col, prob_col = st.columns(2)

    with result_col:

        st.markdown("**Prediksi Model SVM (`predict()`)**")

        st.markdown(
            f"<h2>{prediction_label}</h2>",
            unsafe_allow_html=True
        )

        st.caption(
            "Kategori tingkat stres yang dihasilkan langsung "
            "oleh model SVM."
        )

    with prob_col:

        st.markdown(
            "**Probabilitas Kelas Hasil Prediksi (`predict_proba()`)**"
        )

        st.markdown(
            f"<h2>{predicted_class_probability:.2f}%</h2>",
            unsafe_allow_html=True
        )

        st.caption(
            f"Probabilitas untuk kelas **{prediction_label}** "
            "berdasarkan `predict_proba()`."
        )

    # --------------------------------------------------------
    # PENJELASAN predict() DAN predict_proba()
    # --------------------------------------------------------

    st.info(
        """
        **Perbedaan `predict()` dan `predict_proba()`**

        - **`model.predict()`** digunakan sebagai **hasil
          prediksi utama model SVM**, yaitu menentukan satu
          kategori tingkat stres: Rendah, Sedang, atau Tinggi.
        - **`model.predict_proba()`** digunakan untuk menampilkan
          **probabilitas masing-masing kelas** sebagai informasi
          tambahan mengenai keluaran model.

        Oleh karena itu, kategori yang ditampilkan pada
        **Prediksi Model SVM** tetap mengikuti hasil
        `model.predict()`, sedangkan probabilitas digunakan
        untuk melihat distribusi keyakinan model terhadap
        masing-masing kelas.
        """
    )

    # ========================================================
    # PROBABILITAS SETIAP KELAS
    # ========================================================

    st.subheader("Probabilitas Prediksi Setiap Kelas")

    st.caption(
        "Probabilitas berikut merupakan keluaran "
        "`model.predict_proba()` untuk masing-masing kategori "
        "tingkat stres."
    )

    st.dataframe(
        probability_df.style.format({
            "Probabilitas (%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # GRAFIK PROBABILITAS
    # --------------------------------------------------------

    st.markdown("#### Grafik Probabilitas Prediksi")

    st.caption(
        "Grafik menunjukkan probabilitas masing-masing kelas "
        "tingkat stres dalam persen (%). Nilai pada grafik "
        "merupakan keluaran `model.predict_proba()`."
    )

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
        "SHAP digunakan untuk menjelaskan kontribusi masing-masing "
        "fitur terhadap output kelas yang diprediksi oleh model."
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

    # Pastikan jumlah SHAP sesuai dengan jumlah fitur
    if len(local_shap) != len(feature_names):

        st.error(
            "Jumlah nilai SHAP tidak sesuai dengan jumlah fitur. "
            f"Jumlah SHAP: {len(local_shap)}, "
            f"jumlah fitur: {len(feature_names)}."
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

    st.subheader("Nilai SHAP Lokal")

    st.caption(
        f"Nilai SHAP menjelaskan kontribusi fitur terhadap "
        f"kelas **{prediction_label}**, yaitu kelas yang "
        f"dihasilkan oleh `model.predict()`."
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
    # INTERPRETASI SHAP LOKAL
    # ========================================================

    st.markdown("#### Interpretasi SHAP Lokal")

    # Fitur dengan kontribusi absolut terbesar
    dominant_feature = shap_df.iloc[0]["Fitur"]

    dominant_value = shap_df.iloc[0]["SHAP Value"]

    dominant_abs_value = shap_df.iloc[0][
        "Kontribusi Absolut"
    ]

    # Arah kontribusi SHAP
    if dominant_value > 0:

        direction = (
            "memberikan kontribusi positif terhadap "
            f"output kelas **{prediction_label}**"
        )

    elif dominant_value < 0:

        direction = (
            "memberikan kontribusi negatif terhadap "
            f"output kelas **{prediction_label}**"
        )

    else:

        direction = (
            "tidak memberikan kontribusi positif maupun "
            "negatif yang berarti terhadap output kelas "
            f"**{prediction_label}**"
        )

    # Interpretasi utama
    st.info(
        f"""
        Pada input yang diberikan, fitur dengan kontribusi
        absolut terbesar adalah **{dominant_feature}** dengan
        nilai SHAP **{dominant_value:.6f}**.

        Nilai absolut SHAP sebesar **{dominant_abs_value:.6f}**
        menunjukkan bahwa fitur tersebut merupakan fitur yang
        paling besar kontribusinya dalam menjelaskan output
        model pada input ini.

        Secara arah, fitur tersebut {direction}.
        """
    )

    # --------------------------------------------------------
    # CARA MEMBACA SHAP
    # --------------------------------------------------------

    st.caption(
        """
        **Cara membaca SHAP:** nilai SHAP positif menunjukkan
        kontribusi yang meningkatkan output kelas yang sedang
        dijelaskan relatif terhadap nilai dasar (baseline) model.
        Nilai SHAP negatif menunjukkan kontribusi yang menurunkan
        output kelas tersebut. Semakin besar nilai absolut SHAP,
        semakin besar kontribusi fitur pada input yang dianalisis.

        Nilai SHAP menjelaskan kontribusi fitur terhadap output
        model dan **tidak menunjukkan hubungan sebab-akibat**.
        """
    )


# ============================================================
# DISTRIBUSI PREDIKSI SELURUH RESPONDEN PENELITIAN
# ============================================================

st.divider()

st.header("3. Distribusi Prediksi Tingkat Stres pada 150 Responden Penelitian")

st.write(
    "Distribusi berikut merupakan hasil prediksi model SVM terhadap "
    "responden penelitian yang telah melalui proses screening dan "
    "digunakan dalam dataset penelitian."
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
    "PASS",
    "PSQI"
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

# ============================================================
# BATCH PREDICTION — 150 RESPONDEN PENELITIAN
# ============================================================

st.subheader("Prediksi Tingkat Stres pada Seluruh 150 Responden Penelitian")

st.write(
    "Fitur ini menerapkan model SVM secara langsung pada seluruh "
    "150 responden penelitian sekaligus menggunakan skor PASS dan "
    "PSQI yang terdapat pada dataset penelitian. Responden tidak "
    "perlu dimasukkan satu per satu."
)

st.caption(
    "Hasil batch prediction berlaku untuk 150 responden sampel penelitian "
    "yang dipilih secara proporsional, bukan prediksi individual untuk "
    "seluruh populasi mahasiswa UNSRAT."
)

batch_required = ["PASS", "PSQI"]
batch_missing = [
    col for col in batch_required
    if col not in hasil_prediksi_final.columns
]

if batch_missing:
    st.warning(
        "Batch prediction tidak dapat dijalankan karena kolom berikut "
        f"tidak tersedia pada hasil_prediksi_final.csv: {batch_missing}"
    )
else:
    batch_data = hasil_prediksi_final[batch_required].copy()

    # Pastikan seluruh input batch berupa numerik.
    for col in batch_required:
        batch_data[col] = pd.to_numeric(
            batch_data[col],
            errors="coerce"
        )

    invalid_batch = batch_data.isna().any(axis=1)

    if invalid_batch.any():
        st.warning(
            f"Terdapat {invalid_batch.sum()} baris dengan nilai PASS/PSQI "
            "yang tidak valid. Batch prediction tidak dijalankan agar "
            "hasil penelitian tidak berubah atau diisi secara otomatis."
        )
    else:
        batch_encoded = model.predict(batch_data)

        batch_labels = label_encoder.inverse_transform(
            batch_encoded
        )

        if batch_missing:
    st.warning(
        "Batch prediction tidak dapat dijalankan karena kolom berikut "
        f"tidak tersedia pada hasil_prediksi_final.csv: {batch_missing}"
    )
else:
    batch_data = hasil_prediksi_final[batch_required].copy()

    for col in batch_required:
        batch_data[col] = pd.to_numeric(
            batch_data[col],
            errors="coerce"
        )

    invalid_batch = batch_data.isna().any(axis=1)

    if invalid_batch.any():
        st.warning(
            f"Terdapat {invalid_batch.sum()} baris dengan nilai PASS/PSQI "
            "yang tidak valid. Batch prediction tidak dijalankan agar "
            "hasil penelitian tidak berubah atau diisi secara otomatis."
        )
    else:
        batch_encoded = model.predict(batch_data)

        batch_labels = label_encoder.inverse_transform(
            batch_encoded
        )

        batch_result = hasil_prediksi_final.copy()
        batch_result["Prediksi_Batch"] = batch_labels

        # semua analisis distribusi yang menggunakan
        # Prediksi_Batch diletakkan di sini
        batch_result["Prediksi_Batch"] = batch_labels

        st.success(
            f"Batch prediction berhasil dijalankan pada "
            f"{len(batch_result)} responden penelitian."
        )

        batch_display_cols = [
            col for col in [
                "Fakultas",
                "PASS",
                "PSQI",
                "Tingkat_Stres",
                "Prediksi_Batch"
            ]
            if col in batch_result.columns
        ]

        st.dataframe(
            batch_result[batch_display_cols],
            use_container_width=True,
            hide_index=True
        )

        batch_distribution = (
            batch_result["Prediksi_Batch"]
            .value_counts()
            .reindex(
                ["Rendah", "Sedang", "Tinggi"],
                fill_value=0
            )
            .reset_index()
        )

        batch_distribution.columns = [
            "Tingkat Stres",
            "Jumlah"
        ]

        batch_distribution["Persentase (%)"] = (
            batch_distribution["Jumlah"]
            / len(batch_result)
            * 100
        )

        st.markdown("#### Ringkasan Hasil Batch Prediction")

        st.dataframe(
            batch_distribution.style.format({
                "Persentase (%)": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            batch_distribution.set_index(
                "Tingkat Stres"
            )["Persentase (%)"]
        )


# ------------------------------------------------------------
# DISTRIBUSI PREDIKSI
# ------------------------------------------------------------

distribusi_unsrat = (
    batch_result["Prediksi_Batch"]
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
    / len(batch_result)
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
    batch_result["Fakultas"],
    batch_result["Prediksi_Batch"]
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

# Tambahkan jumlah responden (n) agar persentase tiap fakultas
# dapat dibaca bersama ukuran sampelnya.
persentase_fakultas["Jumlah Responden"] = (
    prediksi_fakultas["Total"]
    .values
)

persentase_fakultas = persentase_fakultas[
    ["Fakultas", "Jumlah Responden", "Rendah", "Sedang", "Tinggi"]
]


# ------------------------------------------------------------
# TAMPILKAN TABEL
# ------------------------------------------------------------

st.dataframe(
    persentase_fakultas.style.format({
        "Jumlah Responden": "{:.0f}",
        "Rendah": "{:.2f}%",
        "Sedang": "{:.2f}%",
        "Tinggi": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Persentase pada setiap fakultas dihitung berdasarkan jumlah responden "
    "dalam fakultas tersebut, sehingga persentase setiap baris berjumlah 100%."
)

# ============================================================
# GLOBAL FEATURE IMPORTANCE SHAP
# ============================================================

st.divider()

st.header("5. Global Feature Importance — SHAP")

st.write(
    "Berdasarkan hasil analisis SHAP pada model final, kontribusi global "
    "fitur dihitung menggunakan rata-rata nilai absolut SHAP."
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

st.caption(
    "Kontribusi (%) dihitung dari proporsi Mean Absolute SHAP masing-masing "
    "fitur terhadap total Mean Absolute SHAP fitur yang dianalisis."
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
    "Catatan penelitian: Model SVM final menggunakan PASS dan PSQI sebagai "
    "fitur masukan dengan Tingkat_Stres sebagai variabel target. PSS-10_Score "
    "digunakan untuk pembentukan target dan tidak digunakan sebagai fitur masukan. "
    "Nilai SHAP menjelaskan kontribusi fitur terhadap output model dan tidak "
    "dimaksudkan sebagai bukti hubungan sebab-akibat."
)
