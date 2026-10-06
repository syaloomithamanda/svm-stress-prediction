import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Prediksi Tingkat Stres Mahasiswa UNSRAT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .hero-box {
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 25px;
        background-color: #f5f8fc;
        border: 1px solid #dce5f2;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 700;
        color: #17365d;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #536273;
        line-height: 1.6;
    }

    .research-box {
        padding: 20px;
        border-radius: 14px;
        background-color: #f8fafc;
        border-left: 5px solid #4f81bd;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 14px;
        background-color: white;
        border: 1px solid #e2e8f0;
        text-align: center;
        min-height: 145px;
    }

    .metric-label {
        font-size: 16px;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #17365d;
        margin-bottom: 5px;
    }

    .metric-note {
        font-size: 14px;
        color: #64748b;
    }

    .result-card {
        padding: 20px;
        border-radius: 14px;
        background-color: #f8fafc;
        border: 1px solid #dbe4ee;
        margin-top: 15px;
    }

    .result-label {
        font-size: 14px;
        color: #64748b;
    }

    .result-value {
        font-size: 32px;
        font-weight: 700;
        color: #17365d;
    }

    .footer-box {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        padding: 20px 0;
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

    st.error("Model tidak dapat dimuat.")

    st.write(
        "Pastikan file berikut tersedia di repository:"
    )

    st.code(
        """
svm_model_final.pkl
feature_names.pkl
shap_background.pkl
hasil_prediksi_final.csv
        """
    )

    st.exception(e)
    st.stop()


# ============================================================
# VALIDATE FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "PASS",
    "PSQI"
]

if list(feature_names) != EXPECTED_FEATURES:

    st.error(
        f"Fitur model tidak sesuai. "
        f"Fitur yang ditemukan: {feature_names}"
    )

    st.stop()


# ============================================================
# SHAP FUNCTION
# ============================================================

def model_decision_for_shap(data):

    data_df = pd.DataFrame(
        data,
        columns=feature_names
    )

    return model.decision_function(data_df)


@st.cache_resource
def create_shap_explainer():

    return shap.KernelExplainer(
        model_decision_for_shap,
        background
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-box">

        <div class="hero-title">
            🧠 Prediksi Tingkat Stres Mahasiswa Semester Akhir
        </div>

        <div class="hero-subtitle">
            Universitas Sam Ratulangi menggunakan algoritma
            Support Vector Machine (SVM) dengan pendekatan
            Explainable AI (SHAP).
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RESEARCH NOTE
# ============================================================

with st.container():

    st.info(
        """
        **Catatan penelitian**

        Model SVM final menggunakan **PASS** dan **PSQI** sebagai
        fitur masukan dengan **Tingkat_Stres** sebagai variabel target.

        **PSS-10_Score** digunakan untuk pembentukan target dan tidak
        digunakan sebagai fitur masukan.

        Model menggunakan StandardScaler dan SVM kernel RBF dengan
        **C = 100**, **gamma = 0.1**, dan **class weight = balanced**.

        Nilai SHAP menjelaskan kontribusi fitur terhadap output model
        dan tidak dimaksudkan sebagai bukti hubungan sebab-akibat.
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Informasi Model")

    st.write("**Algoritma:**")
    st.write("Support Vector Machine")

    st.write("**Kernel:**")
    st.write("RBF")

    st.write("**C:**")
    st.write("100")

    st.write("**Gamma:**")
    st.write("0.1")

    st.write("**Class Weight:**")
    st.write("balanced")

    st.write("**Fitur:**")
    st.write("PASS + PSQI")

    st.divider()

    st.caption(
        "Model memprediksi kategori tingkat stres berdasarkan "
        "fitur PASS dan PSQI."
    )


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = pd.read_csv(
        "hasil_prediksi_final.csv"
    )

except Exception as e:

    st.error(
        "File hasil_prediksi_final.csv tidak ditemukan."
    )

    st.exception(e)
    st.stop()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "PASS",
    "PSQI",
    "Tingkat_Stres"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Kolom berikut tidak ditemukan: {missing_columns}"
    )

    st.stop()


# ============================================================
# CREATE PREDICTION COLUMN
# ============================================================

if "Prediksi_Batch" not in df.columns:

    df["Prediksi_Batch"] = model.predict(
        df[EXPECTED_FEATURES]
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔎 Prediksi Individu",
        "📊 150 Responden",
        "🏫 Analisis Fakultas",
        "💡 Explainable AI (SHAP)"
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.header(
        "Prediksi Tingkat Stres Individu"
    )

    st.write(
        "Masukkan nilai PASS dan PSQI untuk memperoleh "
        "kategori tingkat stres yang diprediksi model."
    )

    col1, col2 = st.columns(2)

    with col1:

        pass_value = st.number_input(
            "Nilai PASS",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

    with col2:

        psqi_value = st.number_input(
            "Nilai PSQI",
            min_value=0.0,
            max_value=21.0,
            value=9.0,
            step=1.0
        )

    predict_button = st.button(
        "🔍 Prediksi Tingkat Stres",
        use_container_width=True
    )

    if predict_button:

        input_data = pd.DataFrame(
            [[pass_value, psqi_value]],
            columns=EXPECTED_FEATURES
        )

        prediction = model.predict(
            input_data
        )[0]

        st.success(
            f"Hasil prediksi model: **{prediction}**"
        )

        st.info(
            "Model memprediksi kategori Tingkat_Stres "
            "(Rendah, Sedang, atau Tinggi), bukan nilai PSS-10 secara langsung."
        )

        # ====================================================
        # LOCAL SHAP
        # ====================================================

        st.subheader(
            "Penjelasan SHAP"
        )

        try:

            explainer = create_shap_explainer()

            shap_result = explainer.shap_values(
                input_data,
                nsamples=100
            )

            classes = list(
                model.classes_
            )

            predicted_class_index = classes.index(
                prediction
            )

            # ------------------------------------------------
            # HANDLE SHAP OUTPUT
            # ------------------------------------------------

            if isinstance(
                shap_result,
                list
            ):

                shap_array = np.array(
                    shap_result
                )

                if shap_array.ndim == 3:

                    shap_array = np.transpose(
                        shap_array,
                        (1, 2, 0)
                    )

            else:

                shap_array = np.asarray(
                    shap_result
                )

                if shap_array.ndim == 2:

                    shap_array = (
                        shap_array[:, :, np.newaxis]
                    )

            local_values = shap_array[
                0,
                :,
                predicted_class_index
            ]

            local_table = pd.DataFrame(
                {
                    "Fitur": EXPECTED_FEATURES,
                    "Nilai": [
                        pass_value,
                        psqi_value
                    ],
                    "SHAP": local_values
                }
            )

            st.dataframe(
                local_table,
                use_container_width=True,
                hide_index=True
            )

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            ax.barh(
                local_table["Fitur"],
                local_table["SHAP"]
            )

            ax.axvline(
                0,
                linewidth=1
            )

            ax.set_xlabel(
                "Nilai SHAP"
            )

            ax.set_title(
                f"Kontribusi Fitur terhadap Kelas {prediction}"
            )

            plt.tight_layout()

            st.pyplot(
                fig
            )

            st.caption(
                "Nilai SHAP menjelaskan kontribusi fitur terhadap "
                "keputusan model dan bukan hubungan sebab-akibat."
            )

        except Exception as e:

            st.warning(
                "Penjelasan SHAP tidak dapat ditampilkan."
            )

            st.exception(e)


# ============================================================
# TAB 2 — 150 RESPONDENTS
# ============================================================

with tab2:

    st.header(
        "Distribusi Tingkat Stres 150 Responden"
    )

    st.write(
        "Distribusi berikut menunjukkan kategori tingkat stres "
        "aktual berdasarkan PSS-10 yang digunakan sebagai variabel target."
    )

    # --------------------------------------------------------
    # ACTUAL DISTRIBUTION
    # --------------------------------------------------------

    actual_counts = (
        df["Tingkat_Stres"]
        .value_counts()
    )

    rendah = int(
        actual_counts.get(
            "Rendah",
            0
        )
    )

    sedang = int(
        actual_counts.get(
            "Sedang",
            0
        )
    )

    tinggi = int(
        actual_counts.get(
            "Tinggi",
            0
        )
    )

    total = len(df)

    # --------------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Rendah
                </div>

                <div class="metric-value">
                    {rendah}
                </div>

                <div class="metric-note">
                    {rendah / total * 100:.2f}% dari {total} responden
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Sedang
                </div>

                <div class="metric-value">
                    {sedang}
                </div>

                <div class="metric-note">
                    {sedang / total * 100:.2f}% dari {total} responden
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Tinggi
                </div>

                <div class="metric-value">
                    {tinggi}
                </div>

                <div class="metric-note">
                    {tinggi / total * 100:.2f}% dari {total} responden
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # ACTUAL TABLE
    # --------------------------------------------------------

    st.subheader(
        "Distribusi Aktual"
    )

    actual_table = pd.DataFrame(
        {
            "Kategori Tingkat Stres": [
                "Rendah",
                "Sedang",
                "Tinggi"
            ],
            "Jumlah": [
                rendah,
                sedang,
                tinggi
            ],
            "Persentase": [
                f"{rendah / total * 100:.2f}%",
                f"{sedang / total * 100:.2f}%",
                f"{tinggi / total * 100:.2f}%"
            ]
        }
    )

    st.dataframe(
        actual_table,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Distribusi Hasil Prediksi Model"
    )

    st.write(
        "Bagian ini menunjukkan hasil prediksi SVM terhadap "
        "150 responden. Nilainya berbeda dari kategori aktual."
    )

    prediction_counts = (
        df["Prediksi_Batch"]
        .value_counts()
    )

    prediction_table = pd.DataFrame(
        {
            "Kategori Prediksi": [
                "Rendah",
                "Sedang",
                "Tinggi"
            ],
            "Jumlah": [
                int(
                    prediction_counts.get(
                        "Rendah",
                        0
                    )
                ),
                int(
                    prediction_counts.get(
                        "Sedang",
                        0
                    )
                ),
                int(
                    prediction_counts.get(
                        "Tinggi",
                        0
                    )
                )
            ]
        }
    )

    prediction_table["Persentase"] = (
        prediction_table["Jumlah"]
        / total
        * 100
    ).round(2)

    prediction_table["Persentase"] = (
        prediction_table["Persentase"].astype(str)
        + "%"
    )

    st.dataframe(
        prediction_table,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # RESPONDENT DATA
    # --------------------------------------------------------

    st.subheader(
        "Data Responden"
    )

    display_columns = [
        col
        for col in [
            "ID",
            "Jenis_Kelamin",
            "Usia",
            "Fakultas",
            "Angkatan",
            "PASS",
            "PSQI",
            "PSS-10_Score",
            "Tingkat_Stres",
            "Prediksi_Batch"
        ]
        if col in df.columns
    ]

    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 3 — FACULTY
# ============================================================

with tab3:

    st.header(
        "Analisis Berdasarkan Fakultas"
    )

    if "Fakultas" not in df.columns:

        st.warning(
            "Kolom Fakultas tidak tersedia."
        )

    else:

        faculty_actual = pd.crosstab(
            df["Fakultas"],
            df["Tingkat_Stres"]
        )

        for category in [
            "Rendah",
            "Sedang",
            "Tinggi"
        ]:

            if category not in faculty_actual.columns:

                faculty_actual[category] = 0

        faculty_actual = faculty_actual[
            [
                "Rendah",
                "Sedang",
                "Tinggi"
            ]
        ]

        st.subheader(
            "Jumlah Responden berdasarkan Kategori"
        )

        st.dataframe(
            faculty_actual,
            use_container_width=True
        )

        faculty_total = (
            faculty_actual.sum(
                axis=1
            )
        )

        faculty_percentage = (
            faculty_actual
            .div(
                faculty_total,
                axis=0
            )
            * 100
        ).round(2)

        st.subheader(
            "Persentase dalam Masing-masing Fakultas"
        )

        st.dataframe(
            faculty_percentage,
            use_container_width=True
        )

        st.caption(
            "Persentase dihitung di dalam masing-masing fakultas."
        )


# ============================================================
# TAB 4 — SHAP
# ============================================================

with tab4:

    st.header(
        "Explainable AI dengan SHAP"
    )

    st.write(
        "SHAP digunakan untuk menjelaskan kontribusi fitur "
        "PASS dan PSQI terhadap output keputusan model SVM."
    )

    st.info(
        "Model SVM final menggunakan probability=False. "
        "Karena itu, SHAP menjelaskan output decision_function "
        "dan bukan probabilitas prediksi."
    )

    # --------------------------------------------------------
    # GLOBAL SHAP
    # --------------------------------------------------------

    st.subheader(
        "Global Feature Importance"
    )

    st.write(
        "Nilai berikut merupakan mean absolute SHAP "
        "dari 30 data uji penelitian."
    )

    shap_global = pd.DataFrame(
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
            "Mean |SHAP|": [
                0.555542,
                0.580899,
                0.269581,
                0.285503,
                0.648638,
                0.580207
            ],
            "Mean SHAP": [
                0.211535,
                0.116239,
                -0.037854,
                0.036628,
                -0.166991,
                -0.155836
            ],
            "SHAP Min": [
                -0.774064,
                -0.789384,
                -0.431665,
                -0.491609,
                -1.062037,
                -1.059518
            ],
            "SHAP Max": [
                1.275040,
                1.367352,
                0.406396,
                0.495525,
                0.960726,
                0.803173
            ]
        }
    )

    st.dataframe(
        shap_global.style.format(
            {
                "Mean |SHAP|": "{:.6f}",
                "Mean SHAP": "{:.6f}",
                "SHAP Min": "{:.6f}",
                "SHAP Max": "{:.6f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # SHAP CHART
    # --------------------------------------------------------

    st.subheader(
        "Mean Absolute SHAP"
    )

    shap_chart = shap_global.copy()

    shap_chart["Label"] = (
        shap_chart["Kelas"]
        + " — "
        + shap_chart["Fitur"]
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.barh(
        shap_chart["Label"],
        shap_chart["Mean |SHAP|"]
    )

    ax.set_xlabel(
        "Mean |SHAP|"
    )

    ax.set_ylabel(
        "Kelas dan fitur"
    )

    ax.set_title(
        "Mean Absolute SHAP berdasarkan kelas"
    )

    ax.invert_yaxis()

    plt.tight_layout()

    st.pyplot(
        fig
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "Interpretasi"
    )

    st.write(
        """
        **Kelas Rendah:** PSQI memiliki mean absolute SHAP
        sebesar 0.580899, sedangkan PASS sebesar 0.555542.

        **Kelas Sedang:** PSQI memiliki mean absolute SHAP
        sebesar 0.285503, sedangkan PASS sebesar 0.269581.

        **Kelas Tinggi:** PASS memiliki mean absolute SHAP
        sebesar 0.648638, sedangkan PSQI sebesar 0.580207.

        Nilai tersebut menunjukkan kontribusi fitur terhadap
        keputusan model pada data yang dianalisis. Nilai SHAP
        tidak digunakan sebagai bukti hubungan sebab-akibat.
        """
    )

    # --------------------------------------------------------
    # LOCAL EXAMPLES
    # --------------------------------------------------------

    st.subheader(
        "Contoh Penjelasan Individu"
    )

    local_examples = pd.DataFrame(
        {
            "ID": [
                5,
                33,
                124
            ],
            "Aktual": [
                "Sedang",
                "Tinggi",
                "Sedang"
            ],
            "Prediksi": [
                "Sedang",
                "Tinggi",
                "Rendah"
            ],
            "PASS": [
                59,
                46,
                55
            ],
            "PSQI": [
                9,
                11,
                8
            ]
        }
    )

    st.dataframe(
        local_examples,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        """
        **ID 5 — Sedang → Sedang**

        PASS = 59 dan PSQI = 9. Untuk output kelas Sedang,
        kontribusi SHAP PASS sebesar +0.406396 dan PSQI sebesar
        +0.153212.

        **ID 33 — Tinggi → Tinggi**

        PASS = 46 dan PSQI = 11. Untuk output kelas Tinggi,
        kontribusi SHAP PASS sebesar +0.465401 dan PSQI sebesar
        +0.311273.

        **ID 124 — Sedang → Rendah**

        PASS = 55 dan PSQI = 8. Untuk output kelas Rendah,
        kontribusi SHAP PASS sebesar +1.275040 dan PSQI sebesar
        +0.989094. Model menghasilkan prediksi Rendah sedangkan
        kategori aktualnya adalah Sedang.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Prediksi Tingkat Stres Mahasiswa Semester Akhir UNSRAT"
)

st.caption(
    "Support Vector Machine (SVM) + Explainable AI (SHAP)"
)
