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
    <div class="hero">
        <div class="hero-kicker">Sistem Prediksi Berbasis Machine Learning</div>
        <div class="hero-title">🧠 Prediksi Tingkat Stres Mahasiswa Semester Akhir</div>
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

    st.caption(
        "PSS-10 tidak digunakan sebagai input model. Skor PSS-10 digunakan "
        "untuk membentuk kategori target tingkat stres."
    )


# ============================================================
# TABS UTAMA
# ============================================================
tab_prediction, tab_batch, tab_faculty, tab_shap = st.tabs(
    [
        "🔍 Prediksi Individu",
        "👥 150 Responden",
        "🏫 Berdasarkan Fakultas",
        "🧩 Global SHAP"
    ]
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
# TAB 1 — PREDIKSI INDIVIDU
# ============================================================
with tab_prediction:
    st.header("Prediksi Tingkat Stres Mahasiswa")
    st.markdown(
        '<p class="section-intro">Masukkan skor variabel prediktor penelitian. '
        'Model menggunakan <strong>PASS</strong> sebagai faktor akademik dan '
        '<strong>PSQI</strong> sebagai kualitas pola tidur. PSS-10 tidak dimasukkan '
        'sebagai input karena digunakan sebagai dasar pembentukan target tingkat stres.</p>',
        unsafe_allow_html=True
    )

    input_col1, input_col2 = st.columns(2, gap="large")

    with input_col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        pass_score = st.number_input(
            "Skor PASS (Faktor Akademik)",
            min_value=40.0,
            max_value=68.0,
            value=None,
            step=1.0,
            help=(
                "Masukkan skor PASS. Rentang 40–68 merupakan rentang skor PASS "
                "yang terdapat pada 150 responden dalam dataset penelitian."
            )
        )
        st.caption("Rentang pada dataset penelitian: 40–68")
        st.markdown('</div>', unsafe_allow_html=True)

    with input_col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        psqi_score = st.number_input(
            "Skor PSQI (Kualitas Pola Tidur)",
            min_value=2.0,
            max_value=18.0,
            value=None,
            step=1.0,
            help=(
                "Masukkan skor PSQI. Rentang 2–18 merupakan rentang skor PSQI "
                "yang terdapat pada 150 responden dalam dataset penelitian."
            )
        )
        st.caption("Rentang pada dataset penelitian: 2–18")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    predict_button = st.button(
        "🔍  Prediksi Tingkat Stres",
        type="primary",
        use_container_width=True
    )

    if predict_button:
        if pass_score is None or psqi_score is None:
            st.warning(
                "Silakan masukkan skor PASS dan PSQI terlebih dahulu sebelum melakukan prediksi."
            )
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
        probabilities = model.predict_proba(input_data)[0]
        model_class_encoded = model.classes_
        model_class_labels = label_encoder.inverse_transform(
            model_class_encoded
        )

        probability_df = pd.DataFrame({
            "Tingkat Stres": model_class_labels,
            "Probabilitas (%)": probabilities * 100
        })

        # Posisi kelas hasil prediksi pada output predict_proba()
        predicted_class_index = np.where(
            model_class_encoded == prediction_encoded[0]
        )[0][0]

        predicted_class_probability = probabilities[
            predicted_class_index
        ] * 100

        # --------------------------------------------------------
        # HASIL PREDIKSI UTAMA
        # --------------------------------------------------------
        st.divider()
        st.subheader("Hasil Prediksi Model")
        st.caption(
            "Hasil utama ditentukan menggunakan `model.predict()`. "
            "Nilai `predict_proba()` ditampilkan sebagai informasi tambahan."
        )

        result_col, prob_col = st.columns(2, gap="large")

        with result_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Prediksi Model SVM</div>
                    <div class="result-value">{prediction_label}</div>
                    <div class="result-note">
                        Kategori tingkat stres yang dihasilkan langsung oleh
                        <strong>model.predict()</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with prob_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Probabilitas Kelas Hasil Prediksi</div>
                    <div class="result-value">{predicted_class_probability:.2f}%</div>
                    <div class="result-note">
                        Probabilitas kelas <strong>{prediction_label}</strong>
                        berdasarkan <strong>predict_proba()</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # --------------------------------------------------------
        # INTERPRETASI HASIL PREDIKSI
        # --------------------------------------------------------
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

            Berdasarkan kategori tingkat stres yang digunakan dalam penelitian:

            - **Rendah:** skor PSS-10 0–13
            - **Sedang:** skor PSS-10 14–26
            - **Tinggi:** skor PSS-10 27–40

            Model memprediksi responden berada pada kategori **{prediction_label}**,
            yang dalam kategorisasi PSS-10 berada pada rentang **{predicted_range}**.

            **Catatan:** model tidak mengetahui skor PSS-10 individu secara langsung.
            PSS-10 merupakan variabel target, sedangkan input model adalah
            **PASS** sebagai faktor akademik dan **PSQI** sebagai kualitas pola tidur.
            Oleh karena itu, hasil tersebut merupakan **prediksi kategori tingkat stres**,
            bukan pengukuran langsung skor PSS-10 individu.
            """
        )

        # --------------------------------------------------------
        # PROBABILITAS SETIAP KELAS
        # --------------------------------------------------------
        st.markdown("### 📊 Probabilitas Prediksi Setiap Kelas")
        st.caption(
            "Probabilitas merupakan keluaran `model.predict_proba()` untuk masing-masing "
            "kategori tingkat stres."
        )

        probability_col1, probability_col2 = st.columns([1, 1.35], gap="large")

        with probability_col1:
            st.dataframe(
                probability_df.style.format({
                    "Probabilitas (%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )

        with probability_col2:
            chart_data = probability_df.set_index("Tingkat Stres")
            st.bar_chart(chart_data["Probabilitas (%)"], height=260)

        st.caption(
            "Probabilitas menunjukkan tingkat keyakinan model SVM terhadap masing-masing "
            "kategori kelas, bukan persentase tingkat stres individu."
        )

        # --------------------------------------------------------
        # PENJELASAN predict() DAN predict_proba()
        # --------------------------------------------------------
        with st.expander("ℹ️ Mengapa hasil `predict()` dan `predict_proba()` ditampilkan terpisah?"):
            st.markdown(
                """
                - **`model.predict()`** digunakan sebagai **hasil prediksi utama model SVM**, """
                """
                  yaitu menentukan satu kategori tingkat stres: Rendah, Sedang, atau Tinggi.
                - **`model.predict_proba()`** digunakan untuk menampilkan **probabilitas masing-masing
                  kelas** sebagai informasi tambahan mengenai keluaran model.

                Oleh karena itu, kategori yang ditampilkan pada **Prediksi Model SVM** tetap
                mengikuti hasil `model.predict()`, sedangkan probabilitas digunakan untuk melihat
                distribusi keyakinan model terhadap masing-masing kelas.
                """
            )

        # --------------------------------------------------------
        # SHAP LOCAL
        # --------------------------------------------------------
        st.divider()
        st.subheader("🧩 Explainable AI — SHAP Lokal")
        st.markdown(
            '<p class="section-intro">SHAP digunakan untuk menjelaskan kontribusi '
            'masing-masing fitur terhadap output kelas yang diprediksi oleh model.</p>',
            unsafe_allow_html=True
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
                f"Bentuk SHAP Values tidak dikenali: {shap_array.shape}"
            )
            st.stop()

        if len(local_shap) != len(feature_names):
            st.error(
                "Jumlah nilai SHAP tidak sesuai dengan jumlah fitur. "
                f"Jumlah SHAP: {len(local_shap)}, "
                f"jumlah fitur: {len(feature_names)}."
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

        shap_df["Kontribusi Absolut"] = shap_df["SHAP Value"].abs()
        shap_df = shap_df.sort_values(
            "Kontribusi Absolut",
            ascending=False
        )

        dominant_feature = shap_df.iloc[0]["Fitur"]
        dominant_value = shap_df.iloc[0]["SHAP Value"]
        dominant_abs_value = shap_df.iloc[0]["Kontribusi Absolut"]

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
                "tidak memberikan kontribusi positif maupun negatif yang berarti "
                f"terhadap output kelas **{prediction_label}**"
            )

        shap_col1, shap_col2 = st.columns([1, 1.2], gap="large")

        with shap_col1:
            st.markdown("**Nilai SHAP Lokal**")
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

        with shap_col2:
            st.markdown("**Kontribusi Fitur terhadap Prediksi**")
            shap_chart = shap_df[
                ["Fitur", "SHAP Value"]
            ].set_index("Fitur")
            st.bar_chart(shap_chart["SHAP Value"], height=260)

        st.info(
            f"""
            Pada input yang diberikan, fitur dengan kontribusi absolut terbesar adalah
            **{dominant_feature}** dengan nilai SHAP **{dominant_value:.6f}**.

            Nilai absolut SHAP sebesar **{dominant_abs_value:.6f}** menunjukkan bahwa
            fitur tersebut merupakan fitur yang paling besar kontribusinya dalam menjelaskan
            output model pada input ini.

            Secara arah, fitur tersebut {direction}.
            """
        )

        with st.expander("📖 Cara membaca nilai SHAP"):
            st.markdown(
                """
                Nilai SHAP positif menunjukkan kontribusi yang meningkatkan output kelas
                yang sedang dijelaskan relatif terhadap nilai dasar (baseline) model.
                Nilai SHAP negatif menunjukkan kontribusi yang menurunkan output kelas tersebut.
                Semakin besar nilai absolut SHAP, semakin besar kontribusi fitur pada input yang dianalisis.

                **Catatan:** nilai SHAP menjelaskan kontribusi fitur terhadap output model
                dan **tidak menunjukkan hubungan sebab-akibat**.
                """
            )


# ============================================================
# DATA PREDIKSI 150 RESPONDEN
# ============================================================
@st.cache_data
def load_prediction_data():
    return pd.read_csv("hasil_prediksi_final.csv")


hasil_prediksi_final = load_prediction_data()

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
    batch_result = None
else:
    batch_required = ["PASS", "PSQI"]
    batch_data = hasil_prediksi_final[batch_required].copy()

    for col in batch_required:
        batch_data[col] = pd.to_numeric(
            batch_data[col],
            errors="coerce"
        )

    invalid_batch = batch_data.isna().any(axis=1)

    if invalid_batch.any():
        batch_result = None
    else:
        batch_encoded = model.predict(batch_data)
        batch_labels = label_encoder.inverse_transform(batch_encoded)

        batch_result = hasil_prediksi_final.copy()
        batch_result["Prediksi_Batch"] = batch_labels


# ============================================================
# TAB 2 — 150 RESPONDEN
# ============================================================
with tab_batch:
    st.header("Distribusi Prediksi pada 150 Responden Penelitian")
    st.markdown(
        '<p class="section-intro">Model SVM final diterapkan secara langsung pada '
        '150 responden penelitian menggunakan skor PASS dan PSQI yang terdapat dalam '
        'dataset penelitian.</p>',
        unsafe_allow_html=True
    )

    if missing_columns:
        st.error(
            "Data batch prediction tidak dapat ditampilkan karena kolom berikut tidak "
            f"tersedia pada hasil_prediksi_final.csv: {missing_columns}"
        )
    elif batch_result is None:
        st.warning(
            "Batch prediction tidak dapat dijalankan karena terdapat nilai PASS/PSQI "
            "yang tidak valid pada dataset."
        )
    else:
        st.caption(
            "Hasil batch prediction berlaku untuk 150 responden sampel penelitian yang "
            "dipilih secara proporsional, bukan prediksi untuk seluruh populasi mahasiswa UNSRAT."
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

        metric_cols = st.columns(3, gap="medium")
        for col, kelas in zip(
            metric_cols,
            ["Rendah", "Sedang", "Tinggi"]
        ):
            row = batch_distribution[
                batch_distribution["Tingkat Stres"] == kelas
            ].iloc[0]
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{kelas}</div>
                        <div class="metric-value">{int(row['Jumlah'])}</div>
                        <div class="metric-note">{row['Persentase (%)']:.2f}% dari 150 responden</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.write("")
        st.markdown("### 📊 Ringkasan Distribusi")

        distribution_col1, distribution_col2 = st.columns([1, 1.35], gap="large")

        with distribution_col1:
            st.dataframe(
                batch_distribution.style.format({
                    "Persentase (%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )

        with distribution_col2:
            st.bar_chart(
                batch_distribution.set_index("Tingkat Stres")["Persentase (%)"],
                height=280
            )

        st.caption(
            "Interpretasi distribusi ini terbatas pada 150 responden penelitian yang digunakan "
            "dalam dataset dan bukan merupakan estimasi prevalensi tingkat stres untuk seluruh "
            "populasi mahasiswa UNSRAT."
        )

        with st.expander("📋 Lihat data prediksi seluruh responden"):
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

            st.caption(
                f"Menampilkan {len(batch_result)} responden penelitian. "
                "Prediksi dihitung menggunakan model SVM final."
            )


# ============================================================
# TAB 3 — FAKULTAS
# ============================================================
with tab_faculty:
    st.header("Distribusi Prediksi Tingkat Stres Berdasarkan Fakultas")
    st.markdown(
        '<p class="section-intro">Distribusi berikut menunjukkan hasil prediksi '
        'tingkat stres berdasarkan fakultas pada responden penelitian.</p>',
        unsafe_allow_html=True
    )

    if batch_result is None:
        st.warning("Data prediksi fakultas belum dapat ditampilkan karena batch prediction tidak tersedia.")
    else:
        prediksi_fakultas = pd.crosstab(
            batch_result["Fakultas"],
            batch_result["Prediksi_Batch"]
        )

        for kelas in ["Rendah", "Sedang", "Tinggi"]:
            if kelas not in prediksi_fakultas.columns:
                prediksi_fakultas[kelas] = 0

        prediksi_fakultas = prediksi_fakultas[
            ["Rendah", "Sedang", "Tinggi"]
        ]

        prediksi_fakultas["Total"] = (
            prediksi_fakultas[
                ["Rendah", "Sedang", "Tinggi"]
            ].sum(axis=1)
        )

        persentase_fakultas = (
            prediksi_fakultas[
                ["Rendah", "Sedang", "Tinggi"]
            ]
            .div(prediksi_fakultas["Total"], axis=0)
            * 100
        )

        persentase_fakultas = persentase_fakultas.reset_index()

        persentase_fakultas["Jumlah Responden"] = (
            prediksi_fakultas["Total"].values
        )

        persentase_fakultas = persentase_fakultas[
            [
                "Fakultas",
                "Jumlah Responden",
                "Rendah",
                "Sedang",
                "Tinggi"
            ]
        ]

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
            "Persentase pada setiap fakultas dihitung berdasarkan jumlah responden dalam "
            "fakultas tersebut, sehingga persentase setiap baris berjumlah 100%."
        )

        st.info(
            "Persentase fakultas tidak menggunakan total populasi mahasiswa UNSRAT sebagai "
            "penyebut. Perhitungan dilakukan berdasarkan responden penelitian pada masing-masing fakultas."
        )


# ============================================================
# GLOBAL SHAP FINAL — DATA AKTUAL COLAB
# ============================================================

# Nilai berikut berasal dari hasil analisis SHAP final pada notebook Google Colab
# menggunakan model final penelitian.
# Mean Absolute SHAP: PSQI = 0.027112; PASS = 0.021735.

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
).reset_index(drop=True)


# ============================================================
# TAB 4 — GLOBAL SHAP
# ============================================================
with tab_shap:
    st.header("Global Feature Importance — SHAP")
    st.markdown(
        '<p class="section-intro">Berdasarkan hasil analisis SHAP pada model final penelitian, '
        'kontribusi global fitur dihitung menggunakan rata-rata nilai absolut SHAP.</p>',
        unsafe_allow_html=True
    )

    shap_metric_cols = st.columns(2, gap="large")

    for col, feature in zip(shap_metric_cols, ["PSQI", "PASS"]):
        row = global_shap[
            global_shap["Fitur"] == feature
        ].iloc[0]
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{feature}</div>
                    <div class="metric-value">{row['Kontribusi (%)']:.2f}%</div>
                    <div class="metric-note">Mean Absolute SHAP = {row['Mean Absolute SHAP']:.6f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    global_col1, global_col2 = st.columns([1, 1.35], gap="large")

    with global_col1:
        st.markdown("### 📋 Nilai Global SHAP")
        st.dataframe(
            global_shap.style.format({
                "Mean Absolute SHAP": "{:.6f}",
                "Kontribusi (%)": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    with global_col2:
        st.markdown("### 📊 Kontribusi Relatif Fitur")
        global_chart = global_shap[
            ["Fitur", "Kontribusi (%)"]
        ].set_index("Fitur")
        st.bar_chart(global_chart["Kontribusi (%)"], height=280)

    st.caption(
        "Kontribusi (%) dihitung dari proporsi Mean Absolute SHAP masing-masing fitur "
        "terhadap total Mean Absolute SHAP fitur yang dianalisis."
    )

    dominant_global_feature = global_shap.iloc[0]["Fitur"]
    dominant_global_value = global_shap.iloc[0]["Mean Absolute SHAP"]
    dominant_global_percentage = global_shap.iloc[0]["Kontribusi (%)"]

    st.info(
        f"""
        Berdasarkan analisis global SHAP, fitur yang memiliki kontribusi relatif terbesar
        terhadap output model adalah **{dominant_global_feature}** dengan Mean Absolute SHAP
        sebesar **{dominant_global_value:.6f}** atau sekitar **{dominant_global_percentage:.2f}%**
        dari total kontribusi fitur yang dianalisis.

        Hal ini menunjukkan bahwa **{dominant_global_feature}** memiliki kontribusi relatif
        lebih besar dalam menjelaskan keluaran model pada data penelitian.

        **Catatan:** hasil SHAP menjelaskan perilaku model dalam menggunakan fitur untuk
        menghasilkan prediksi dan tidak menunjukkan hubungan sebab-akibat.
        """
    )

    with st.expander("📖 Tentang interpretasi Global SHAP"):
        st.markdown(
            """
            Global SHAP merangkum kontribusi fitur pada seluruh data yang dianalisis.
            Nilai yang digunakan adalah **Mean Absolute SHAP**, sehingga tanda positif/negatif
            tidak menjadi fokus pada ringkasan global ini; yang dibandingkan adalah besarnya
            kontribusi absolut fitur terhadap output model.

            Hasil ini digunakan untuk menjelaskan perilaku model SVM dan **bukan untuk menyatakan
            hubungan sebab-akibat** antara variabel prediktor dan tingkat stres.
            """
        )


# ============================================================
# CATATAN PENELITIAN
# ============================================================
st.divider()

st.markdown(
    """
    <div class="footer-note">
        <strong>Catatan penelitian:</strong> Model SVM final menggunakan PASS dan PSQI
        sebagai fitur masukan dengan <em>Tingkat_Stres</em> sebagai variabel target.
        <em>PSS-10_Score</em> digunakan untuk pembentukan target dan tidak digunakan sebagai
        fitur masukan. Nilai SHAP menjelaskan kontribusi fitur terhadap output model dan
        tidak dimaksudkan sebagai bukti hubungan sebab-akibat.
    </div>
    """,
    unsafe_allow_html=True
)
