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
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLE — UI ONLY
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.14);
        }

        .hero-kicker {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.72;
            margin-bottom: 0.7rem;
        }

        .hero-title {
            font-size: 2.25rem;
            line-height: 1.15;
            font-weight: 750;
            margin: 0 0 0.7rem 0;
        }

        .hero-subtitle {
            font-size: 1rem;
            line-height: 1.6;
            opacity: 0.86;
            margin: 0;
        }

        .section-intro {
            color: #64748b;
            line-height: 1.65;
            margin-top: -0.35rem;
            margin-bottom: 1.25rem;
        }

        .input-card {
            padding: 1.1rem 1.2rem 0.8rem 1.2rem;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
        }

        .metric-card {
            padding: 1.25rem 1.35rem;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            background: #ffffff;
            min-height: 125px;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
        }

        .metric-label {
            color: #64748b;
            font-size: 0.84rem;
            font-weight: 650;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: #0f172a;
            font-size: 1.85rem;
            line-height: 1.1;
            font-weight: 760;
            margin-bottom: 0.35rem;
        }

        .metric-note {
            color: #94a3b8;
            font-size: 0.78rem;
            line-height: 1.4;
        }

        .result-card {
            padding: 1.4rem 1.5rem;
            border-radius: 18px;
            border: 1px solid #cbd5e1;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            min-height: 155px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .result-label {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.55rem;
        }

        .result-value {
            color: #0f172a;
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.45rem;
        }

        .result-note {
            color: #64748b;
            font-size: 0.82rem;
            line-height: 1.5;
        }

        .feature-pill {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            background: #f1f5f9;
            color: #334155;
            font-size: 0.78rem;
            font-weight: 650;
            margin-right: 0.35rem;
        }

        .footer-note {
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
            color: #64748b;
            font-size: 0.78rem;
            line-height: 1.6;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #e2e8f0;
            padding: 0.85rem 1rem;
            border-radius: 14px;
            background: #ffffff;
        }

        @media (max-width: 768px) {
            .hero-title {
                font-size: 1.7rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load("svm_model_final.pkl")
    feature_names = joblib.load("feature_names.pkl")
    background = joblib.load("shap_background.pkl")

    return model, feature_names, background


try:
    model, feature_names, background = load_model()

except Exception as e:
    st.error("Model atau file pendukung tidak dapat dimuat.")
    st.code(str(e))
    st.stop()


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
# VALIDASI MODEL
# ============================================================
try:
    model_classes = model.classes_

except AttributeError:
    st.error(
        "Model tidak memiliki atribut classes_. "
        "Pastikan svm_model_final.pkl merupakan model SVM final penelitian."
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Sistem Prediksi Berbasis Machine Learning</div>

        <div class="hero-title">
            🧠 Prediksi Tingkat Stres Mahasiswa Semester Akhir
        </div>

        <p class="hero-subtitle">
            Universitas Sam Ratulangi menggunakan algoritma
            <strong>Support Vector Machine (SVM)</strong> dengan pendekatan
            <strong>Explainable AI (SHAP)</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<span class="feature-pill">PASS · Faktor Akademik</span>'
    '<span class="feature-pill">PSQI · Kualitas Pola Tidur</span>'
    '<span class="feature-pill">PSS-10 · Target Tingkat Stres</span>',
    unsafe_allow_html=True
)

st.caption(
    "Teknik Informatika • Universitas Sam Ratulangi • Model SVM final penelitian"
)


# ============================================================
# SIDEBAR INFORMASI
# ============================================================
with st.sidebar:

    st.markdown("### 🧠 Informasi Sistem")

    st.markdown(
        "Aplikasi ini merupakan implementasi model SVM final penelitian "
        "untuk memprediksi kategori tingkat stres berdasarkan skor PASS dan PSQI."
    )

    st.divider()

    st.markdown("**Variabel masukan**")
    st.write("• PASS — faktor akademik")
    st.write("• PSQI — kualitas pola tidur")

    st.markdown("**Variabel target**")
    st.write("• Tingkat Stres berdasarkan PSS-10")

    st.divider()

    st.markdown("**Model SVM**")
    st.write("• Kernel: RBF")
    st.write("• C: 100")
    st.write("• Gamma: 0.1")
    st.write("• Class weight: balanced")
    st.write("• StandardScaler: Ya")

    st.divider()

    st.caption(
        "PSS-10 tidak digunakan sebagai input model. "
        "Skor PSS-10 digunakan untuk membentuk kategori target tingkat stres."
    )


# ============================================================
# TABS UTAMA
# ============================================================
tab_prediction, tab_batch, tab_faculty, tab_shap = st.tabs(
    [
        "🔍 Prediksi Individu",
        "👥 Seluruh Responden",
        "🏫 Berdasarkan Fakultas",
        "🧩 Global SHAP"
    ]
)


# ============================================================
# FUNGSI MODEL UNTUK SHAP
# ============================================================
def model_decision_for_shap(data):

    data_df = pd.DataFrame(
        data,
        columns=feature_names
    )

    return model.decision_function(data_df)


# ============================================================
# SHAP EXPLAINER
# ============================================================
@st.cache_resource
def create_shap_explainer():

    return shap.KernelExplainer(
        model_decision_for_shap,
        background
    )


# ============================================================
# FUNGSI MENGAMBIL SHAP LOCAL
# ============================================================
def extract_local_shap(shap_values, predicted_class_index):

    shap_array = np.asarray(shap_values)

    # --------------------------------------------------------
    # Bentuk list
    # --------------------------------------------------------
    if isinstance(shap_values, list):

        values = np.asarray(
            shap_values[predicted_class_index]
        )

        if values.ndim == 2:
            return values[0]

        return values

    # --------------------------------------------------------
    # Bentuk 3 dimensi
    # --------------------------------------------------------
    if shap_array.ndim == 3:

        # (samples, features, classes)
        if shap_array.shape[0] == 1:

            return shap_array[
                0,
                :,
                predicted_class_index
            ]

        # (samples, classes, features)
        if shap_array.shape[1] == len(model_classes):

            return shap_array[
                0,
                predicted_class_index,
                :
            ]

    # --------------------------------------------------------
    # Bentuk 2 dimensi
    # --------------------------------------------------------
    if shap_array.ndim == 2:

        return shap_array[0]

    # --------------------------------------------------------
    # Jika tidak dikenali
    # --------------------------------------------------------
    raise ValueError(
        f"Bentuk SHAP Values tidak dikenali: {shap_array.shape}"
    )


# ============================================================
# TAB 1 — PREDIKSI INDIVIDU
# ============================================================
with tab_prediction:

    st.header("Prediksi Tingkat Stres Mahasiswa")

    st.markdown(
        '<p class="section-intro">'
        'Masukkan skor variabel prediktor penelitian. '
        'Model menggunakan <strong>PASS</strong> sebagai faktor akademik '
        'dan <strong>PSQI</strong> sebagai kualitas pola tidur. '
        'PSS-10 tidak dimasukkan sebagai input karena digunakan sebagai '
        'dasar pembentukan target tingkat stres.'
        '</p>',
        unsafe_allow_html=True
    )

    input_col1, input_col2 = st.columns(2, gap="large")


    # ========================================================
    # INPUT PASS
    # ========================================================
    with input_col1:

        st.markdown(
            '<div class="input-card">',
            unsafe_allow_html=True
        )

        pass_score = st.number_input(
            "Skor PASS (Faktor Akademik)",
            min_value=40.0,
            max_value=68.0,
            value=None,
            step=1.0,
            help=(
                "Masukkan skor PASS. Rentang 40–68 merupakan "
                "rentang skor PASS yang terdapat pada dataset penelitian."
            )
        )

        st.caption(
            "Rentang pada dataset penelitian: 40–68"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # INPUT PSQI
    # ========================================================
    with input_col2:

        st.markdown(
            '<div class="input-card">',
            unsafe_allow_html=True
        )

        psqi_score = st.number_input(
            "Skor PSQI (Kualitas Pola Tidur)",
            min_value=2.0,
            max_value=18.0,
            value=None,
            step=1.0,
            help=(
                "Masukkan skor PSQI. Rentang 2–18 merupakan "
                "rentang skor PSQI yang terdapat pada dataset penelitian."
            )
        )

        st.caption(
            "Rentang pada dataset penelitian: 2–18"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    st.write("")


    # ========================================================
    # BUTTON PREDIKSI
    # ========================================================
    predict_button = st.button(
        "🔍  Prediksi Tingkat Stres",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        if pass_score is None or psqi_score is None:

            st.warning(
                "Silakan masukkan skor PASS dan PSQI terlebih dahulu "
                "sebelum melakukan prediksi."
            )

            st.stop()


        # ====================================================
        # DATA INPUT
        # ====================================================
        input_data = pd.DataFrame(
            [[pass_score, psqi_score]],
            columns=feature_names
        )


        # ====================================================
        # PREDIKSI SVM
        # ====================================================
        prediction = model.predict(input_data)

        prediction_label = prediction[0]


        # ====================================================
        # POSISI KELAS
        # ====================================================
        predicted_class_index = np.where(
            model_classes == prediction_label
        )[0][0]


        # ====================================================
        # HASIL PREDIKSI
        # ====================================================
        st.divider()

        st.subheader("Hasil Prediksi Model")


        result_col, info_col = st.columns(
            2,
            gap="large"
        )


        with result_col:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        Prediksi Model SVM
                    </div>

                    <div class="result-value">
                        {prediction_label}
                    </div>

                    <div class="result-note">
                        Kategori tingkat stres yang dihasilkan langsung
                        oleh <strong>model.predict()</strong>.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with info_col:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        Input Model
                    </div>

                    <div class="result-value">
                        PASS + PSQI
                    </div>

                    <div class="result-note">
                        PASS = {pass_score:.0f} &nbsp; | &nbsp;
                        PSQI = {psqi_score:.0f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # INTERPRETASI HASIL
        # ====================================================
        st.markdown("### 🧠 Interpretasi Hasil")

        stress_category_info = {
            "Rendah": "0–13",
            "Sedang": "14–26",
            "Tinggi": "27–40"
        }

        predicted_range = stress_category_info.get(
            prediction_label,
            "-"
        )

        st.info(
            f"""
            **Hasil prediksi model: {prediction_label}**

            Berdasarkan kategorisasi tingkat stres yang digunakan
            dalam penelitian:

            - **Rendah:** skor PSS-10 0–13
            - **Sedang:** skor PSS-10 14–26
            - **Tinggi:** skor PSS-10 27–40

            Model memprediksi input berada pada kategori
            **{prediction_label}** berdasarkan skor **PASS** dan **PSQI**.

            Kategori tersebut berada pada rentang skor PSS-10
            **{predicted_range}** yang digunakan sebagai dasar
            pembentukan target penelitian.

            **Catatan:** model tidak mengetahui skor PSS-10 individu
            secara langsung. PSS-10 merupakan variabel target,
            sedangkan input model adalah PASS dan PSQI.
            """
        )


        # ====================================================
        # SHAP LOCAL
        # ====================================================
        st.divider()

        st.subheader(
            "🧩 Explainable AI — SHAP Lokal"
        )

        st.markdown(
            '<p class="section-intro">'
            'SHAP digunakan untuk menjelaskan kontribusi masing-masing '
            'fitur terhadap output <strong>decision function</strong> '
            'kelas yang diprediksi oleh model SVM.'
            '</p>',
            unsafe_allow_html=True
        )


        # ====================================================
        # HITUNG SHAP
        # ====================================================
        with st.spinner(
            "Menghitung penjelasan SHAP..."
        ):

            try:

                explainer = create_shap_explainer()

                shap_values = explainer.shap_values(
                    input_data,
                    nsamples=100
                )

                local_shap = extract_local_shap(
                    shap_values,
                    predicted_class_index
                )

            except Exception as e:

                st.error(
                    "SHAP tidak dapat dihitung untuk input ini."
                )

                st.code(str(e))

                st.stop()


        # ====================================================
        # VALIDASI SHAP
        # ====================================================
        local_shap = np.asarray(
            local_shap,
            dtype=float
        )


        if len(local_shap) != len(feature_names):

            st.error(
                "Jumlah nilai SHAP tidak sesuai dengan jumlah fitur."
            )

            st.write(
                "Jumlah SHAP:",
                len(local_shap)
            )

            st.write(
                "Jumlah fitur:",
                len(feature_names)
            )

            st.stop()


        # ====================================================
        # TABEL SHAP
        # ====================================================
        shap_df = pd.DataFrame(
            {
                "Fitur": feature_names,

                "Nilai Input": [
                    pass_score,
                    psqi_score
                ],

                "SHAP Value": local_shap
            }
        )


        shap_df["Kontribusi Absolut"] = (
            shap_df["SHAP Value"].abs()
        )


        shap_df = shap_df.sort_values(
            "Kontribusi Absolut",
            ascending=False
        ).reset_index(drop=True)


        # ====================================================
        # FITUR DOMINAN
        # ====================================================
        dominant_feature = (
            shap_df.iloc[0]["Fitur"]
        )

        dominant_value = (
            shap_df.iloc[0]["SHAP Value"]
        )

        dominant_abs_value = (
            shap_df.iloc[0]["Kontribusi Absolut"]
        )


        if dominant_value > 0:

            direction = (
                f"memberikan kontribusi positif terhadap "
                f"output kelas {prediction_label}"
            )

        elif dominant_value < 0:

            direction = (
                f"memberikan kontribusi negatif terhadap "
                f"output kelas {prediction_label}"
            )

        else:

            direction = (
                f"tidak menunjukkan kontribusi positif maupun "
                f"negatif terhadap output kelas {prediction_label}"
            )


        # ====================================================
        # TAMPILKAN SHAP
        # ====================================================
        shap_col1, shap_col2 = st.columns(
            [1, 1.2],
            gap="large"
        )


        with shap_col1:

            st.markdown(
                "**Nilai SHAP Lokal**"
            )

            st.dataframe(
                shap_df[
                    [
                        "Fitur",
                        "Nilai Input",
                        "SHAP Value"
                    ]
                ].style.format(
                    {
                        "Nilai Input": "{:.2f}",
                        "SHAP Value": "{:.6f}"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


        with shap_col2:

            st.markdown(
                "**Kontribusi Fitur terhadap Prediksi**"
            )

            shap_chart = (
                shap_df[
                    ["Fitur", "SHAP Value"]
                ]
                .set_index("Fitur")
            )

            st.bar_chart(
                shap_chart["SHAP Value"],
                height=260
            )


        # ====================================================
        # INTERPRETASI SHAP
        # ====================================================
        st.info(
            f"""
            Pada input yang diberikan, fitur dengan kontribusi
            absolut terbesar adalah **{dominant_feature}**
            dengan nilai SHAP **{dominant_value:.6f}**.

            Nilai absolut SHAP sebesar **{dominant_abs_value:.6f}**
            menunjukkan bahwa fitur tersebut memiliki kontribusi
            terbesar dalam menjelaskan output model pada input ini.

            Secara arah, fitur tersebut {direction}.

            **Catatan:** nilai SHAP menjelaskan perilaku model
            dan tidak menunjukkan hubungan sebab-akibat.
            """
        )


        # ====================================================
        # PENJELASAN SHAP
        # ====================================================
        with st.expander(
            "📖 Cara membaca nilai SHAP"
        ):

            st.markdown(
                """
                **Nilai SHAP positif** menunjukkan kontribusi yang
                meningkatkan output decision function kelas yang
                sedang dijelaskan relatif terhadap nilai dasar model.

                **Nilai SHAP negatif** menunjukkan kontribusi yang
                menurunkan output decision function kelas tersebut.

                Semakin besar nilai absolut SHAP, semakin besar
                kontribusi fitur pada input yang dianalisis.

                **Catatan penting:** SHAP menjelaskan kontribusi
                fitur terhadap output model dan tidak menunjukkan
                hubungan sebab-akibat.
                """
            )


# ============================================================
# LOAD DATA 150 RESPONDEN
# ============================================================
@st.cache_data
def load_prediction_data():

    return pd.read_csv(
        "hasil_prediksi_final.csv"
    )


try:

    hasil_prediksi_final = load_prediction_data()

except Exception as e:

    hasil_prediksi_final = None

    st.warning(
        "File hasil_prediksi_final.csv tidak dapat dimuat."
    )


# ============================================================
# VALIDASI DATA BATCH
# ============================================================
batch_result = None

if hasil_prediksi_final is not None:

    required_columns = [
        "Fakultas",
        "PASS",
        "PSQI"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in hasil_prediksi_final.columns
    ]

    if missing_columns:

        batch_result = None

    else:

        batch_data = (
            hasil_prediksi_final[
                ["PASS", "PSQI"]
            ]
            .copy()
        )

        for col in ["PASS", "PSQI"]:

            batch_data[col] = pd.to_numeric(
                batch_data[col],
                errors="coerce"
            )


        invalid_batch = (
            batch_data.isna().any(axis=1)
        )


        if not invalid_batch.any():

            batch_encoded = model.predict(
                batch_data
            )

            batch_labels = np.asarray(
                batch_encoded
            )

            batch_result = (
                hasil_prediksi_final.copy()
            )

            batch_result["Prediksi_Batch"] = (
                batch_labels
            )


# ============================================================
# TAB 2 — 150 RESPONDEN
# ============================================================
with tab_batch:

    st.header(
        "Prediksi Tingkat Stres pada 150 Responden Penelitian"
    )

    st.markdown(
        '<p class="section-intro">'
        'Model SVM final diterapkan pada 150 responden '
        'sampel penelitian menggunakan skor PASS dan PSQI.'
        '</p>',
        unsafe_allow_html=True
    )


    if batch_result is None:

        st.warning(
            "Batch prediction tidak dapat dijalankan."
        )


    else:

        st.caption(
            "Hasil batch prediction berlaku untuk 150 responden "
            "sampel penelitian dan bukan prediksi untuk seluruh "
            "populasi mahasiswa UNSRAT."
        )


        # ====================================================
        # DISTRIBUSI
        # ====================================================
        batch_distribution = (
            batch_result[
                "Prediksi_Batch"
            ]
            .value_counts()
            .reindex(
                [
                    "Rendah",
                    "Sedang",
                    "Tinggi"
                ],
                fill_value=0
            )
            .reset_index()
        )


        batch_distribution.columns = [
            "Tingkat Stres",
            "Jumlah"
        ]


        batch_distribution[
            "Persentase (%)"
        ] = (
            batch_distribution["Jumlah"]
            / len(batch_result)
            * 100
        )


        # ====================================================
        # METRIC
        # ====================================================
        metric_cols = st.columns(
            3,
            gap="medium"
        )


        for col, kelas in zip(
            metric_cols,
            [
                "Rendah",
                "Sedang",
                "Tinggi"
            ]
        ):

            row = batch_distribution[
                batch_distribution[
                    "Tingkat Stres"
                ] == kelas
            ].iloc[0]


            with col:

                st.markdown(
                    f"""
                    <div class="metric-card">

                        <div class="metric-label">
                            {kelas}
                        </div>

                        <div class="metric-value">
                            {int(row['Jumlah'])}
                        </div>

                        <div class="metric-note">
                            {row['Persentase (%)']:.2f}%
                            dari 150 responden
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        st.write("")

        st.markdown(
            "### 📊 Ringkasan Distribusi"
        )


        distribution_col1, distribution_col2 = st.columns(
            [1, 1.35],
            gap="large"
        )


        with distribution_col1:

            st.dataframe(
                batch_distribution.style.format(
                    {
                        "Persentase (%)": "{:.2f}%"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


        with distribution_col2:

            st.bar_chart(
                batch_distribution.set_index(
                    "Tingkat Stres"
                )[
                    "Persentase (%)"
                ],
                height=280
            )


        st.caption(
            "Distribusi ini terbatas pada 150 responden "
            "sampel penelitian dan bukan estimasi prevalensi "
            "tingkat stres seluruh mahasiswa UNSRAT."
        )


        # ====================================================
        # DATA RESPONDEN
        # ====================================================
        with st.expander(
            "📋 Lihat data prediksi seluruh responden"
        ):

            batch_display_cols = [
                col
                for col in [
                    "Fakultas",
                    "PASS",
                    "PSQI",
                    "Tingkat_Stres",
                    "Prediksi_Batch"
                ]
                if col in batch_result.columns
            ]


            st.dataframe(
                batch_result[
                    batch_display_cols
                ],
                use_container_width=True,
                hide_index=True
            )


            st.caption(
                f"Menampilkan {len(batch_result)} "
                "responden penelitian."
            )


# ============================================================
# TAB 3 — FAKULTAS
# ============================================================
with tab_faculty:

    st.header(
        "Distribusi Prediksi Tingkat Stres Berdasarkan Fakultas"
    )

    st.markdown(
        '<p class="section-intro">'
        'Distribusi berikut menunjukkan hasil prediksi '
        'tingkat stres berdasarkan fakultas pada responden penelitian.'
        '</p>',
        unsafe_allow_html=True
    )


    if batch_result is None:

        st.warning(
            "Data prediksi fakultas belum tersedia."
        )


    else:

        prediksi_fakultas = pd.crosstab(
            batch_result["Fakultas"],
            batch_result["Prediksi_Batch"]
        )


        for kelas in [
            "Rendah",
            "Sedang",
            "Tinggi"
        ]:

            if kelas not in prediksi_fakultas.columns:

                prediksi_fakultas[
                    kelas
                ] = 0


        prediksi_fakultas = (
            prediksi_fakultas[
                [
                    "Rendah",
                    "Sedang",
                    "Tinggi"
                ]
            ]
        )


        prediksi_fakultas["Total"] = (
            prediksi_fakultas[
                [
                    "Rendah",
                    "Sedang",
                    "Tinggi"
                ]
            ]
            .sum(axis=1)
        )


        persentase_fakultas = (
            prediksi_fakultas[
                [
                    "Rendah",
                    "Sedang",
                    "Tinggi"
                ]
            ]
            .div(
                prediksi_fakultas["Total"],
                axis=0
            )
            * 100
        )


        persentase_fakultas = (
            persentase_fakultas
            .reset_index()
        )


        persentase_fakultas[
            "Jumlah Responden"
        ] = (
            prediksi_fakultas[
                "Total"
            ].values
        )


        persentase_fakultas = (
            persentase_fakultas[
                [
                    "Fakultas",
                    "Jumlah Responden",
                    "Rendah",
                    "Sedang",
                    "Tinggi"
                ]
            ]
        )


        st.dataframe(
            persentase_fakultas.style.format(
                {
                    "Jumlah Responden": "{:.0f}",
                    "Rendah": "{:.2f}%",
                    "Sedang": "{:.2f}%",
                    "Tinggi": "{:.2f}%"
                }
            ),
            use_container_width=True,
            hide_index=True
        )


        st.caption(
            "Persentase pada setiap fakultas dihitung "
            "berdasarkan jumlah responden pada fakultas tersebut."
        )


        st.info(
            "Persentase fakultas menggunakan responden penelitian "
            "sebagai dasar perhitungan dan bukan total populasi "
            "mahasiswa UNSRAT."
        )


# ============================================================
# GLOBAL SHAP FINAL
# ============================================================
# Nilai berasal dari analisis SHAP final pada model penelitian.
#
# Mean Absolute SHAP per kelas:
#
# Rendah:
# PASS = 0.555542
# PSQI = 0.580899
#
# Sedang:
# PASS = 0.269581
# PSQI = 0.285503
#
# Tinggi:
# PASS = 0.648638
# PSQI = 0.580207
# ============================================================

global_shap_per_class = pd.DataFrame(
    {
        "Kelas": [
            "Rendah",
            "Rendah",
            "Sedang",
            "Sedang",
            "Tinggi",
            "Tinggi"
        ],

        "Fitur": [
            "PASS",
            "PSQI",
            "PASS",
            "PSQI",
            "PASS",
            "PSQI"
        ],

        "Mean Absolute SHAP": [
            0.555542,
            0.580899,
            0.269581,
            0.285503,
            0.648638,
            0.580207
        ]
    }
)


# ============================================================
# TAB 4 — GLOBAL SHAP
# ============================================================
with tab_shap:

    st.header(
        "Global Feature Importance — SHAP"
    )

    st.markdown(
        '<p class="section-intro">'
        'Global SHAP merangkum besarnya kontribusi absolut '
        'fitur terhadap output model pada data pengujian.'
        '</p>',
        unsafe_allow_html=True
    )


    # ========================================================
    # TABEL GLOBAL SHAP
    # ========================================================
    st.markdown(
        "### 📋 Mean Absolute SHAP Berdasarkan Kelas"
    )


    shap_pivot = (
        global_shap_per_class
        .pivot(
            index="Fitur",
            columns="Kelas",
            values="Mean Absolute SHAP"
        )
        .reset_index()
    )


    shap_pivot = shap_pivot[
        [
            "Fitur",
            "Rendah",
            "Sedang",
            "Tinggi"
        ]
    ]


    st.dataframe(
        shap_pivot.style.format(
            {
                "Rendah": "{:.6f}",
                "Sedang": "{:.6f}",
                "Tinggi": "{:.6f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    st.caption(
        "Nilai Mean Absolute SHAP yang lebih besar menunjukkan "
        "kontribusi absolut yang lebih besar dalam menjelaskan "
        "output kelas model pada data yang dianalisis."
    )


    # ========================================================
    # AGREGASI GLOBAL
    # ========================================================
    aggregated_global = (
        global_shap_per_class
        .groupby("Fitur")[
            "Mean Absolute SHAP"
        ]
        .mean()
        .reset_index()
    )


    total_global_shap = (
        aggregated_global[
            "Mean Absolute SHAP"
        ].sum()
    )


    aggregated_global[
        "Kontribusi Relatif (%)"
    ] = (
        aggregated_global[
            "Mean Absolute SHAP"
        ]
        / total_global_shap
        * 100
    )


    aggregated_global = (
        aggregated_global
        .sort_values(
            "Mean Absolute SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )


    # ========================================================
    # GRAFIK GLOBAL
    # ========================================================
    st.markdown(
        "### 📊 Agregasi Mean Absolute SHAP"
    )


    global_chart = (
        aggregated_global[
            [
                "Fitur",
                "Kontribusi Relatif (%)"
            ]
        ]
        .set_index("Fitur")
    )


    st.bar_chart(
        global_chart[
            "Kontribusi Relatif (%)"
        ],
        height=280
    )


    st.dataframe(
        aggregated_global.style.format(
            {
                "Mean Absolute SHAP": "{:.6f}",
                "Kontribusi Relatif (%)": "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # INTERPRETASI GLOBAL
    # ========================================================
    dominant_global_feature = (
        aggregated_global.iloc[0]["Fitur"]
    )

    dominant_global_value = (
        aggregated_global.iloc[0][
            "Mean Absolute SHAP"
        ]
    )

    dominant_global_percentage = (
        aggregated_global.iloc[0][
            "Kontribusi Relatif (%)"
        ]
    )


    st.info(
        f"""
        Berdasarkan agregasi Mean Absolute SHAP dari tiga
        output kelas model, fitur **{dominant_global_feature}**
        memiliki nilai rata-rata kontribusi absolut sebesar
        **{dominant_global_value:.6f}**, atau sekitar
        **{dominant_global_percentage:.2f}%** dari total kontribusi
        absolut kedua fitur.

        Interpretasi ini menunjukkan besarnya kontribusi fitur
        dalam menjelaskan output model, bukan hubungan sebab-akibat.
        """
    )


    # ========================================================
    # PENJELASAN GLOBAL SHAP
    # ========================================================
    with st.expander(
        "📖 Tentang interpretasi Global SHAP"
    ):

        st.markdown(
            """
            **Mean Absolute SHAP** digunakan untuk melihat besarnya
            kontribusi absolut fitur terhadap output model.

            Nilai absolut digunakan karena tanda positif dan negatif
            menunjukkan arah kontribusi terhadap output kelas,
            sedangkan analisis global berfokus pada besarnya
            kontribusi.

            Hasil SHAP menjelaskan perilaku model dalam menggunakan
            fitur untuk menghasilkan prediksi dan **tidak menunjukkan
            hubungan sebab-akibat**.
            """
        )


# ============================================================
# CATATAN PENELITIAN
# ============================================================
st.divider()

st.markdown(
    """
    <div class="footer-note">

        <strong>Catatan penelitian:</strong>

        Model SVM final menggunakan <strong>PASS</strong> dan
        <strong>PSQI</strong> sebagai fitur masukan dengan
        <em>Tingkat_Stres</em> sebagai variabel target.

        <em>PSS-10_Score</em> digunakan untuk pembentukan target
        dan tidak digunakan sebagai fitur masukan.

        Model menggunakan StandardScaler dan SVM kernel RBF
        dengan C = 100, gamma = 0.1, dan class weight = balanced.

        Nilai SHAP menjelaskan kontribusi fitur terhadap output
        model dan tidak dimaksudkan sebagai bukti hubungan
        sebab-akibat.

    </div>
    """,
    unsafe_allow_html=True
)
