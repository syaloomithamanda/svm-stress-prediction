import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

st.set_page_config(
    page_title="Prediksi Tingkat Stres Mahasiswa Semester Akhir Universitas Sam Ratulangi",      
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# UI THEME — Dark + Navy (visual only)
# -----------------------------
st.markdown("""
<style>
:root {
  --navy-950: #07111f; --navy-900: #0a1628; --navy-800: #10233d;
  --navy-700: #173554; --cyan-400: #38bdf8; --teal-400: #2dd4bf;
  --text-main: #e8eef8; --text-muted: #9fb0c7; --panel: #101e31;
}
.stApp { background: radial-gradient(ellipse at 10% 0%, #142b49 0%, #0a1628 38%, #070f1b 100%); color: var(--text-main); }
[data-testid="stHeader"] { background: rgba(7,17,31,.78); }
[data-testid="stToolbar"] { right: 1rem; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1c30 0%, #081321 100%); border-right: 1px solid rgba(125,168,215,.16); }
section[data-testid="stSidebar"] * { color: #dce8f8; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aabbd1; }
h1, h2, h3 { color: #f1f6ff !important; letter-spacing: -0.025em; }
p, label, .stCaption, [data-testid="stMarkdownContainer"] { color: #d4deec; }
[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(21,43,70,.96), rgba(13,29,49,.96)); border: 1px solid rgba(91,153,207,.22); border-radius: 16px; padding: 17px 18px; box-shadow: 0 8px 24px rgba(0,0,0,.15); }
[data-testid="stMetricLabel"] { color: #9fb7d3 !important; font-size: .88rem; }
[data-testid="stMetricValue"] { color: #f2f7ff !important; font-weight: 750; }
[data-testid="stMetricDelta"] { font-weight: 600; }
[data-testid="stTabs"] [role="tablist"] { gap: 8px; border-bottom: 1px solid rgba(125,168,215,.18); }
[data-testid="stTabs"] button[role="tab"] { background: rgba(15,31,51,.68); border: 1px solid rgba(125,168,215,.14); border-radius: 11px 11px 0 0; padding: 11px 15px; color: #aebfd5; transition: all .18s ease; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background: linear-gradient(135deg, #173b60, #124a61); color: #f4fbff; border-color: rgba(56,189,248,.5); box-shadow: inset 0 -2px 0 #38bdf8; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: rgba(125,168,215,.20) !important; }
[data-testid="stAlert"] { border-radius: 12px; }
.stButton > button, .stDownloadButton > button { border-radius: 11px; border: 1px solid rgba(56,189,248,.35); font-weight: 650; transition: transform .15s ease, box-shadow .15s ease; }
.stButton > button[kind="primary"] { background: linear-gradient(100deg, #087eaa, #155e91); color: white; box-shadow: 0 6px 18px rgba(8,126,170,.18); }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #38bdf8; transform: translateY(-1px); box-shadow: 0 7px 20px rgba(56,189,248,.12); }
[data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="textarea"] { background: #0b192b; border-color: #29415e; border-radius: 10px; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid rgba(125,168,215,.18); border-radius: 12px; overflow: hidden; }
hr { border-color: rgba(125,168,215,.18); }
.hero-panel { padding: 28px 30px; margin: 4px 0 22px; border-radius: 22px; border: 1px solid rgba(80,169,220,.30); background: linear-gradient(115deg, rgba(17,48,79,.97), rgba(11,28,48,.95) 58%, rgba(10,47,63,.92)); box-shadow: 0 16px 42px rgba(0,0,0,.22); position: relative; overflow: hidden; }
.hero-panel:after { content: ""; position: absolute; width: 230px; height: 230px; border-radius: 50%; right: -55px; top: -115px; background: rgba(45,212,191,.09); border: 1px solid rgba(56,189,248,.16); }
.hero-eyebrow { color: #5eead4; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 800; margin-bottom: 10px; }
.hero-title { color: #f4f8ff; font-size: clamp(1.65rem, 3vw, 2.45rem); font-weight: 800; letter-spacing: -.045em; line-height: 1.14; margin: 0 0 10px; }
.hero-subtitle { color: #b6c9df; font-size: 1rem; line-height: 1.6; max-width: 850px; margin: 0; }
.hero-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.hero-tag { border: 1px solid rgba(125,211,252,.24); background: rgba(8,24,41,.48); color: #d5f3ff; padding: 6px 10px; border-radius: 999px; font-size: .78rem; font-weight: 650; }
@media (max-width: 700px) { .hero-panel { padding: 22px 20px; } [data-testid="stTabs"] [role="tablist"] { gap: 4px; } [data-testid="stTabs"] button[role="tab"] { padding: 8px 10px; font-size: .78rem; } }
</style>
""", unsafe_allow_html=True)

# Plot themes affect chart appearance only; values and model logic remain unchanged.
px.defaults.template = "plotly_dark"
plt.rcParams.update({
    "figure.facecolor": "#0d1a2b", "axes.facecolor": "#0d1a2b",
    "axes.edgecolor": "#49617c", "axes.labelcolor": "#d8e4f2",
    "xtick.color": "#b6c6da", "ytick.color": "#b6c6da",
    "text.color": "#e6eef8", "axes.titlecolor": "#f1f6ff",
    "grid.color": "#2a3e56", "savefig.facecolor": "#0d1a2b",
})

EXPECTED_FEATURES = ["PASS", "PSQI"]
STRESS_CLASSES = ["Rendah", "Sedang", "Tinggi"]
STRESS_COLORS = {"Rendah": "#22C55E", "Sedang": "#F59E0B", "Tinggi": "#EF4444"}
STRESS_ICONS = {"Rendah": "🟢", "Sedang": "🟡", "Tinggi": "🔴"}

# -----------------------------
# Data & model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("svm_model_final.joblib")
    feature_names = joblib.load("feature_names.joblib")
    class_names = joblib.load("class_names.joblib")

    background = pd.read_csv("shap_background.csv")
    background = background[EXPECTED_FEATURES]

    return model, feature_names, class_names, background


try:
    model, feature_names, class_names, background = load_model()
except Exception as e:
    st.error("Model atau artefak SHAP tidak dapat dimuat.")
    st.write("Pastikan file berikut tersedia di repository:")
    st.code(
        "svm_model_final.joblib\n"
        "feature_names.joblib\n"
        "class_names.joblib\n"
        "shap_background.csv"
    )
    st.exception(e)
    st.stop()


if list(feature_names) != EXPECTED_FEATURES:
    st.error(f"Fitur model tidak sesuai: {feature_names}")
    st.stop()

if list(class_names) != list(model.classes_):
    st.error(
        "Urutan kelas pada class_names.joblib tidak sesuai "
        "dengan kelas model."
    )
    st.stop()


@st.cache_resource
def create_shap_explainer():
    """Buat explainer untuk output decision_function model SVM final."""
    def predict_decision(data):
        if isinstance(data, pd.DataFrame):
            frame = data.loc[:, EXPECTED_FEATURES]
        else:
            frame = pd.DataFrame(data, columns=EXPECTED_FEATURES)
        return model.decision_function(frame)

    return shap.KernelExplainer(
        predict_decision,
        background.loc[:, EXPECTED_FEATURES]
    )

# -----------------------------
# Memuat dataset seluruh responden
# -----------------------------
try:
    df = pd.read_csv("dataset_responden_app.csv")

    required_columns = [
        "PASS",
        "PSQI",
        "Fakultas",
        "Tingkat_Stres",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            f"Kolom wajib tidak ditemukan: {missing_columns}"
        )
        st.stop()

    # Pastikan skor fitur dapat dibaca sebagai angka.
    df["PASS"] = pd.to_numeric(
        df["PASS"], errors="raise"
    )
    df["PSQI"] = pd.to_numeric(
        df["PSQI"], errors="raise"
    )

    # Selalu hitung ulang prediksi batch menggunakan model final dari Colab.
    # Ini mencegah kolom Prediksi_Batch lama di CSV menampilkan hasil model berbeda.
    df["Prediksi_Batch"] = model.predict(df[EXPECTED_FEATURES])

except FileNotFoundError:
    st.error(
        "File dataset_responden_app.csv tidak ditemukan. "
        "Unggah file tersebut ke root repository GitHub."
    )
    st.stop()

except Exception as e:
    st.error("Terjadi kesalahan saat memuat dataset.")
    st.exception(e)
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🧠 _Support Vector Machine + Explainable AI_ ")
    st.caption("Aplikasi Penelitian")
    st.divider()

    st.subheader("⚙️ Konfigurasi Model")
    st.write("**Algoritma:** SVM")
    st.write("**Kernel:** RBF")
    st.write("**C:** 100")
    st.write("**Gamma:** 0.1")
    st.write("**_Class weight:_** _balanced_")
    st.write("**Fitur:** PASS + PSQI")

    st.divider()
    st.subheader("🎨 Kategori Stres")
    st.success("🟢 **Rendah** — PSS-10 0–13")
    st.warning("🟡 **Sedang** — PSS-10 14–26")
    st.error("🔴 **Tinggi** — PSS-10 27–40")

    st.divider()
    st.caption("_Perceived Stress Scale-10_ (PSS-10) digunakan untuk membentuk target. _The Perceptions of Academic Stress Scale_ (PASS) dan _Pittsburgh Sleep Quality Index_ (PSQI) digunakan sebagai fitur model. PSS-10_Score tidak digunakan sebagai fitur untuk mencegah target leakage. _SHapley Additive exPlanations_ (SHAP) menjelaskan keputusan model dan bukan hubungan sebab-akibat.")

# -----------------------------
# Header — visual redesign only
# -----------------------------
st.markdown("""
<div class="hero-panel">
  <div class="hero-eyebrow">Dashboard Penelitian · Universitas Sam Ratulangi</div>
  <div class="hero-title">Prediksi Tingkat Stres<br>Mahasiswa Semester Akhir</div>
  <p class="hero-subtitle">Sistem klasifikasi berbasis <b>Support Vector Machine (SVM)</b> dengan pendekatan <b>Explainable AI (SHAP)</b> untuk menampilkan hasil prediksi dan penjelasan kontribusi fitur.</p>
  <div class="hero-tags"><span class="hero-tag">SVM · RBF</span><span class="hero-tag">Fitur: PASS + PSQI</span><span class="hero-tag">Explainable AI</span><span class="hero-tag">UNSRAT</span></div>
</div>
""", unsafe_allow_html=True)

# Compact overview cards using native metrics
counts = df["Tingkat_Stres"].value_counts()
total = len(df)
rendah = int(counts.get("Rendah", 0))
sedang = int(counts.get("Sedang", 0))
tinggi = int(counts.get("Tinggi", 0))

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("👥 Responden", total)
with m2:
    st.metric("🟢 Rendah", rendah, f"{rendah / total * 100:.2f}%" if total else "0.00%")
with m3:
    st.metric("🟡 Sedang", sedang, f"{sedang / total * 100:.2f}%" if total else "0.00%")
with m4:
    st.metric("🔴 Tinggi", tinggi, f"{tinggi / total * 100:.2f}%" if total else "0.00%")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔎 Prediksi Individu",
    "📊 Distribusi 150 Responden",
    "🏫 Analisis Fakultas",
    "💡 _Explainable AI_ (SHAP)",
    "📈 Performa Model",
])

# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    st.header("Prediksi Tingkat Stres Individu")
    st.write("Masukkan skor **_The Perceptions of Academic Stress Scale_ (PASS)** dan **_Pittsburgh Sleep Quality Index_ (PSQI)** untuk melihat kategori yang diprediksi oleh model SVM terbaru.")

    c1, c2 = st.columns(2)
    with c1:
        pass_text = st.text_input(
            "📘 Nilai PASS",
            value="",
            placeholder="Masukkan skor bulat 18–90",
            help="PASS Total dihitung dari 18 item berskala 1–5; rentang teoretis total 18–90."
        )
    with c2:
        psqi_text = st.text_input(
            "🌙 Nilai PSQI",
            value="",
            placeholder="Masukkan skor bulat 0–21",
            help="Masukkan skor global PSQI (jumlah 7 komponen), dengan rentang teoretis 0–21."
        )

    # Skor instrumen harus berupa angka bulat yang sah. Nilai di luar rentang
    # teoretis tidak diproses sebagai skor kuesioner yang valid.
    pass_value = None
    psqi_value = None
    pass_format_invalid = False
    psqi_format_invalid = False

    if pass_text.strip():
        try:
            pass_value = float(pass_text.strip().replace(",", "."))
            if not np.isfinite(pass_value):
                pass_format_invalid = True
                pass_value = None
        except ValueError:
            pass_format_invalid = True

    if psqi_text.strip():
        try:
            psqi_value = float(psqi_text.strip().replace(",", "."))
            if not np.isfinite(psqi_value):
                psqi_format_invalid = True
                psqi_value = None
        except ValueError:
            psqi_format_invalid = True

    pass_observed_min = float(df["PASS"].min())
    pass_observed_max = float(df["PASS"].max())
    psqi_observed_min = float(df["PSQI"].min())
    psqi_observed_max = float(df["PSQI"].max())

    pass_not_integer = (
        pass_value is not None and not float(pass_value).is_integer()
    )
    psqi_not_integer = (
        psqi_value is not None and not float(psqi_value).is_integer()
    )
    pass_outside_theoretical = (
        pass_value is not None and not (18 <= pass_value <= 90)
    )
    psqi_outside_theoretical = (
        psqi_value is not None and not (0 <= psqi_value <= 21)
    )
    pass_outside_observed = (
        pass_value is not None
        and not (pass_observed_min <= pass_value <= pass_observed_max)
    )
    psqi_outside_observed = (
        psqi_value is not None
        and not (psqi_observed_min <= psqi_value <= psqi_observed_max)
    )

    input_invalid = (
        pass_format_invalid or psqi_format_invalid
        or pass_not_integer or psqi_not_integer
        or pass_outside_theoretical or psqi_outside_theoretical
    )

    if pass_format_invalid or psqi_format_invalid:
        invalid_messages = []
        if pass_format_invalid:
            invalid_messages.append("PASS harus berupa angka yang valid dan terbatas (finite).")
        if psqi_format_invalid:
            invalid_messages.append("PSQI harus berupa angka yang valid dan terbatas (finite).")
        st.error("Input tidak dapat dibaca. " + " ".join(invalid_messages))

    if pass_not_integer:
        st.error("PASS harus berupa skor bulat karena merupakan total skor item kuesioner.")
    if psqi_not_integer:
        st.error("PSQI harus berupa skor bulat karena merupakan skor global tujuh komponen.")

    if pass_outside_theoretical:
        st.error(
            f"PASS {pass_value:g} berada di luar rentang teoretis 18–90. "
            "Periksa kembali skor total PASS."
        )
    if psqi_outside_theoretical:
        st.error(
            f"PSQI {psqi_value:g} berada di luar rentang teoretis 0–21. "
            "Periksa kembali skor global PSQI."
        )

    range_messages = []
    if pass_outside_observed and not pass_outside_theoretical:
        range_messages.append(
            f"PASS {pass_value:g} berada di luar rentang pengamatan data "
            f"({pass_observed_min:.0f}–{pass_observed_max:.0f})."
        )
    if psqi_outside_observed and not psqi_outside_theoretical:
        range_messages.append(
            f"PSQI {psqi_value:g} berada di luar rentang pengamatan data "
            f"({psqi_observed_min:.0f}–{psqi_observed_max:.0f})."
        )
    if range_messages:
        st.warning(
            "Skor masih berada dalam rentang teoretis instrumen, tetapi di luar "
            "rentang yang diamati pada dataset. Model tetap dapat menghitung "
            "prediksi dan SHAP, namun hasil perlu ditafsirkan dengan hati-hati. "
            + " ".join(range_messages)
        )

    st.caption(
        "Input harus berupa skor bulat yang sah menurut rentang teoretis instrumen. "
        "Skor valid yang berada di luar rentang pengamatan dataset tetap dapat "
        "diproses, dengan peringatan."
    )

    if st.button(
        "🔍 Prediksi Tingkat Stres",
        type="primary",
        use_container_width=True,
        disabled=(
            input_invalid or pass_value is None or psqi_value is None
        ),
    ):
        if pass_value is None or psqi_value is None:
            st.warning("Masukkan skor PASS dan PSQI terlebih dahulu.")
        else:
            input_data = pd.DataFrame(

                [[pass_value, psqi_value]],
                columns=EXPECTED_FEATURES,
            )
        
            prediction = model.predict(input_data)[0]
        
            st.divider()
            st.subheader("Hasil Prediksi")
        
            if prediction == "Rendah":
                st.success(f"🟢 **{prediction.upper()}**")
            elif prediction == "Sedang":
                st.warning(f"🟡 **{prediction.upper()}**")
            else:
                st.error(f"🔴 **{prediction.upper()}**")
        
            st.write(f"**PASS:** {pass_value:g}")
            st.write(f"**PSQI:** {psqi_value:g}")
        
            st.info(
                "Hasil ini merupakan prediksi model SVM, "
                "bukan hasil pengukuran PSS-10 secara langsung."
            )

            st.subheader("Penjelasan SHAP")
            st.caption(
                "Grafik ini menjelaskan kontribusi PASS dan PSQI terhadap "
                "output keputusan kelas yang diprediksi. Nilai SHAP bukan "
                "probabilitas dan bukan bukti sebab-akibat."
            )

            try:
                explainer = create_shap_explainer()

                with st.spinner("Menghitung SHAP untuk input ini..."):
                    raw_shap = explainer.shap_values(
                        input_data,
                        nsamples=100
                    )

                class_index = list(model.classes_).index(prediction)

                # SHAP versions may return a list of per-class arrays
                # or a NumPy array with different axis arrangements.
                if isinstance(raw_shap, list):
                    if len(raw_shap) == len(model.classes_):
                        contributions = np.asarray(raw_shap[class_index])[0].reshape(-1)
                    elif len(raw_shap) == 1:
                        contributions = np.asarray(raw_shap[0])[0].reshape(-1)
                    else:
                        raise ValueError(
                            f"Jumlah array SHAP ({len(raw_shap)}) tidak sesuai "
                            f"dengan jumlah kelas ({len(model.classes_)})."
                        )
                else:
                    values = np.asarray(raw_shap)

                    if values.ndim == 3:
                        if values.shape == (1, len(EXPECTED_FEATURES), len(model.classes_)):
                            contributions = values[0, :, class_index]
                        elif values.shape == (len(model.classes_), 1, len(EXPECTED_FEATURES)):
                            contributions = values[class_index, 0, :]
                        else:
                            raise ValueError(
                                f"Bentuk array SHAP 3D tidak dikenali: {values.shape}"
                            )
                    elif values.ndim == 2 and values.shape == (1, len(EXPECTED_FEATURES)):
                        contributions = values[0, :]
                    else:
                        raise ValueError(
                            f"Bentuk hasil SHAP tidak dikenali: {values.shape}"
                        )

                contributions = np.asarray(contributions, dtype=float).reshape(-1)
                if len(contributions) != len(EXPECTED_FEATURES):
                    raise ValueError(
                        "Jumlah nilai SHAP tidak sama dengan jumlah fitur."
                    )

                shap_df = pd.DataFrame({
                    "Fitur": EXPECTED_FEATURES,
                    "Nilai Input": [
                        float(input_data.iloc[0][feature])
                        for feature in EXPECTED_FEATURES
                    ],
                    "Nilai SHAP": contributions,
                }).sort_values("Nilai SHAP")

                fig, ax = plt.subplots(figsize=(7, 3.5))
                colors = [
                    "#EF4444" if value < 0 else "#22C55E"
                    for value in shap_df["Nilai SHAP"]
                ]
                ax.barh(
                    shap_df["Fitur"],
                    shap_df["Nilai SHAP"],
                    color=colors
                )
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Kontribusi SHAP terhadap output keputusan")
                ax.set_title(f"Penjelasan untuk kelas {prediction}")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.dataframe(
                    shap_df,
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(
                    "Nilai SHAP positif mendorong output keputusan kelas yang "
                    "dijelaskan ke arah lebih tinggi; nilai negatif mendorongnya "
                    "ke arah lebih rendah. Ini menjelaskan perilaku model, bukan "
                    "penyebab stres atau probabilitas kelas."
                )

            except Exception as e:
                st.error("Penjelasan SHAP gagal dihitung.")
                st.exception(e)

# -----------------------------
# Tab 2
# -----------------------------
with tab2:
    st.header("Distribusi Tingkat Stres")
    st.write("Kategori aktual dibentuk dari skor _Perceived Stress Scale-10_ (PSS-10) pada responden penelitian.")

    # Hitung langsung dari label aktual; tidak memakai hasil prediksi model.
    actual_counts = (
        df["Tingkat_Stres"].astype(str).str.strip()
        .value_counts()
        .reindex(STRESS_CLASSES, fill_value=0)
        .astype(int)
    )
    actual_total = int(actual_counts.sum())
    if actual_total == 0:
        st.warning("Dataset tidak berisi kategori stres yang dapat dianalisis.")
        st.stop()
    actual_pct = actual_counts / actual_total * 100

    # Validasi angka yang ditampilkan.
    if actual_total != len(df):
        st.error("Terjadi ketidaksesuaian perhitungan jumlah responden.")
        st.stop()

    if actual_total == 150:
        expected = {"Rendah": 7, "Sedang": 104, "Tinggi": 39}
        if actual_counts.to_dict() != expected:
            st.warning(
                "Distribusi pada file CSV saat ini berbeda dari distribusi penelitian "
                "yang digunakan pada hasil analisis (7/104/39). Periksa kembali CSV."
            )

    # Kartu ringkasan.
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("👥 Total responden", f"{actual_total}")
    with c2:
        st.metric("🟢 Rendah", f"{actual_counts['Rendah']}", f"{actual_pct['Rendah']:.2f}%")
    with c3:
        st.metric("🟡 Sedang", f"{actual_counts['Sedang']}", f"{actual_pct['Sedang']:.2f}%")
    with c4:
        st.metric("🔴 Tinggi", f"{actual_counts['Tinggi']}", f"{actual_pct['Tinggi']:.2f}%")

    st.markdown("### Grafik Distribusi Aktual")

    chart_df = pd.DataFrame({
        "Kategori": STRESS_CLASSES,
        "Jumlah": [int(actual_counts[x]) for x in STRESS_CLASSES],
        "Persentase": [float(actual_pct[x]) for x in STRESS_CLASSES],
    })

    fig = px.bar(
        chart_df,
        x="Kategori",
        y="Jumlah",
        color="Kategori",
        text="Jumlah",
        category_orders={"Kategori": STRESS_CLASSES},
        color_discrete_map=STRESS_COLORS,
        custom_data=["Persentase"],
    )
    fig.update_traces(
        texttemplate="%{y} responden<br>(%{customdata[0]:.2f}%)",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Jumlah: %{y} responden<br>"
            "Persentase: %{customdata[0]:.2f}%"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        showlegend=False,
        height=460,
        xaxis_title="Tingkat Stres",
        yaxis_title="Jumlah Responden",
        margin=dict(l=25, r=25, t=25, b=25),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Angka pada grafik dihitung langsung dari kolom Tingkat_Stres. "
        "Persentase = jumlah kategori / total responden × 100%."
    )

    actual_table = pd.DataFrame({
        "Kategori": [f"{STRESS_ICONS[x]} {x}" for x in STRESS_CLASSES],
        "Jumlah": [int(actual_counts[x]) for x in STRESS_CLASSES],
        "Persentase": [f"{actual_pct[x]:.2f}%" for x in STRESS_CLASSES],
    })
    st.subheader("Distribusi Aktual")
    st.dataframe(actual_table, use_container_width=True, hide_index=True)

    pred_counts = (
        df["Prediksi_Batch"].astype(str).str.strip()
        .value_counts()
        .reindex(STRESS_CLASSES, fill_value=0)
        .astype(int)
    )
    pred_pct = pred_counts / actual_total * 100
    pred_table = pd.DataFrame({
        "Kategori Prediksi": [f"{STRESS_ICONS[x]} {x}" for x in STRESS_CLASSES],
        "Jumlah": [int(pred_counts[x]) for x in STRESS_CLASSES],
        "Persentase": [f"{pred_pct[x]:.2f}%" for x in STRESS_CLASSES],
    })
    st.subheader("Distribusi Hasil Prediksi Model")
    st.write("Distribusi prediksi model dapat berbeda dari distribusi kategori aktual.")
    st.dataframe(pred_table, use_container_width=True, hide_index=True)

    st.subheader("Data Responden")
    display_columns = [
        c for c in [
            "ID", "Jenis_Kelamin", "Usia", "Fakultas", "Angkatan",
            "PASS", "PSQI", "PSS-10_Score", "Tingkat_Stres", "Prediksi_Batch"
        ] if c in df.columns
    ]
    st.dataframe(df[display_columns], use_container_width=True, hide_index=True, height=420)

    st.download_button(
        "⬇️ Unduh hasil prediksi 150 responden (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset_responden_app.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if actual_total != 150:
        st.warning(
            f"File saat ini berisi {actual_total} responden, bukan 150. "
            "Periksa kembali dataset_responden_app.csv dan data penelitian."
        )

# -----------------------------
# Tab 3
# -----------------------------
with tab3:
    st.header("Analisis Berdasarkan Fakultas")
    st.write("Tabel berikut menggambarkan distribusi kategori stres aktual di dalam masing-masing fakultas pada data penelitian.")

    if "Fakultas" not in df.columns:
        st.warning("Kolom Fakultas tidak tersedia.")
    else:
        faculty = pd.crosstab(df["Fakultas"], df["Tingkat_Stres"])
        for cat in STRESS_CLASSES:
            if cat not in faculty.columns:
                faculty[cat] = 0
        faculty = faculty[STRESS_CLASSES]

        # Native table first
        st.subheader("Jumlah Responden")
        st.dataframe(faculty, use_container_width=True)

        # Stacked colored chart
        fig, ax = plt.subplots(figsize=(11, max(4.8, len(faculty) * 0.42)))
        bottom = np.zeros(len(faculty))
        for cat in STRESS_CLASSES:
            vals = faculty[cat].values
            ax.barh(faculty.index.astype(str), vals, left=bottom, label=cat, color=STRESS_COLORS[cat])
            bottom += vals
        ax.set_xlabel("Jumlah responden")
        ax.set_title("Komposisi Kategori Stres per Fakultas", fontweight="bold")
        ax.legend(title="Kategori", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        pct = (faculty.div(faculty.sum(axis=1), axis=0) * 100).round(2)
        st.subheader("Persentase dalam Masing-masing Fakultas")
        st.dataframe(pct, use_container_width=True)
        st.caption("Persentase dihitung di dalam masing-masing fakultas.")

# -----------------------------
# Tab 4
# -----------------------------
with tab4:
    st.header("_Explainable AI_ (_SHapley Additive exPlanations_)")
    st.write("SHAP digunakan untuk menjelaskan kontribusi fitur **PASS** dan **PSQI** terhadap output keputusan model SVM.")
    st.info("Model SVM final menggunakan probability=False. Karena itu, SHAP menjelaskan output **decision_function**, bukan probabilitas prediksi.")

    # Ringkasan global dihitung langsung dari nilai SHAP pada data uji.
    shap_file = "shap_values_testing.csv"
    expected_shap_columns = {
        "SHAP_PASS_Rendah", "SHAP_PSQI_Rendah",
        "SHAP_PASS_Sedang", "SHAP_PSQI_Sedang",
        "SHAP_PASS_Tinggi", "SHAP_PSQI_Tinggi",
    }
    try:
        shap_test = pd.read_csv(shap_file)
        missing_shap_columns = sorted(expected_shap_columns - set(shap_test.columns))
        if missing_shap_columns:
            raise ValueError(
                "Kolom SHAP yang diperlukan tidak ditemukan: "
                + ", ".join(missing_shap_columns)
            )
        if shap_test.empty:
            raise ValueError("File SHAP tidak berisi baris data.")

        shap_rows = []
        for class_name in STRESS_CLASSES:
            for feature in EXPECTED_FEATURES:
                column = f"SHAP_{feature}_{class_name}"
                values = pd.to_numeric(shap_test[column], errors="coerce")
                if values.isna().any() or not np.isfinite(values.to_numpy()).all():
                    raise ValueError(f"Kolom {column} berisi nilai kosong atau nonnumerik.")
                shap_rows.append({
                    "Kelas": class_name,
                    "Fitur": feature,
                    "Mean |SHAP|": float(values.abs().mean()),
                    "Mean SHAP": float(values.mean()),
                    "SHAP Min": float(values.min()),
                    "SHAP Max": float(values.max()),
                })
        shap_global = pd.DataFrame(shap_rows)
        st.caption(f"Ringkasan dihitung dari {len(shap_test)} baris pada {shap_file}.")
    except FileNotFoundError:
        st.error(
            f"File {shap_file} tidak ditemukan. Tambahkan file tersebut ke root "
            "repository agar ringkasan SHAP global dapat ditampilkan."
        )
        shap_global = pd.DataFrame(columns=[
            "Kelas", "Fitur", "Mean |SHAP|", "Mean SHAP", "SHAP Min", "SHAP Max"
        ])
    except Exception as e:
        st.error("Ringkasan SHAP global tidak dapat dihitung dari file SHAP.")
        st.exception(e)
        shap_global = pd.DataFrame(columns=[
            "Kelas", "Fitur", "Mean |SHAP|", "Mean SHAP", "SHAP Min", "SHAP Max"
        ])

    st.subheader("_Global Feature Importance_")
    st.caption("Mean absolute SHAP dari 30 data uji penelitian.")

    if not shap_global.empty:
        # Kartu ringkasan per kelas
        r1, r2, r3 = st.columns(3)
        for col, cat in zip([r1, r2, r3], STRESS_CLASSES):
            row = shap_global[shap_global["Kelas"] == cat].set_index("Fitur")["Mean |SHAP|"]
            with col:
                if cat == "Rendah":
                    st.success(f"{STRESS_ICONS[cat]} **{cat}**")
                elif cat == "Sedang":
                    st.warning(f"{STRESS_ICONS[cat]} **{cat}**")
                else:
                    st.error(f"{STRESS_ICONS[cat]} **{cat}**")
                st.metric("PASS", f"{row['PASS']:.6f}")
                st.metric("PSQI", f"{row['PSQI']:.6f}")

        st.subheader("Tabel Nilai SHAP")
        st.dataframe(
            shap_global.style.format({c: "{:.6f}" for c in ["Mean |SHAP|", "Mean SHAP", "SHAP Min", "SHAP Max"]}),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Perbandingan Mean Absolute SHAP")
        pivot = shap_global.pivot(index="Kelas", columns="Fitur", values="Mean |SHAP|").loc[STRESS_CLASSES]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(STRESS_CLASSES))
        width = 0.34
        ax.bar(x - width / 2, pivot["PASS"], width, label="PASS", color="#4F46E5")
        ax.bar(x + width / 2, pivot["PSQI"], width, label="PSQI", color="#06B6D4")
        ax.set_xticks(x)
        ax.set_xticklabels(STRESS_CLASSES)
        ax.set_ylabel("Mean |SHAP|")
        ax.set_title("Kontribusi Rata-rata Fitur berdasarkan Kelas", fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Interpretasi")
        for cat in STRESS_CLASSES:
            class_rows = shap_global[shap_global["Kelas"] == cat].set_index("Fitur")
            pass_mean = class_rows.loc["PASS", "Mean |SHAP|"]
            psqi_mean = class_rows.loc["PSQI", "Mean |SHAP|"]
            if pass_mean > psqi_mean:
                lead_feature, lead_value = "PASS", pass_mean
                other_feature, other_value = "PSQI", psqi_mean
            else:
                lead_feature, lead_value = "PSQI", psqi_mean
                other_feature, other_value = "PASS", pass_mean
            st.write(
                f"**Kelas {cat}:** {lead_feature} memiliki mean absolute SHAP "
                f"{lead_value:.6f}, sedangkan {other_feature} {other_value:.6f}."
            )
        st.caption("Nilai tersebut menunjukkan besarnya kontribusi fitur terhadap output keputusan model pada data yang dianalisis. Nilai SHAP tidak membuktikan hubungan sebab-akibat.")

    st.info(
        "Penjelasan SHAP untuk satu input tersedia pada tab Prediksi Individu. "
        "Contoh statis tidak ditampilkan agar tidak tertukar dengan hasil "
        "prediksi aktual yang dihitung dari model saat aplikasi dijalankan."
    )

st.divider()
st.caption("Prediksi Tingkat Stres Mahasiswa Semester Akhir Universitas Sam Ratulangi  •  _Support Vector Machine_ (SVM) + _Explainable AI_ (SHAP)")


# -----------------------------
# Tab 5 — Evaluasi model dari artefak Google Colab
# -----------------------------
with tab5:
    st.header("Performa Model SVM")
    st.write(
        "Metrik dihitung dari `hasil_prediksi_testing.csv` yang diekspor oleh notebook Google Colab. "
        "Dengan demikian, tabel evaluasi tidak bergantung pada angka yang diketik manual di aplikasi."
    )

    test_path = "hasil_prediksi_testing.csv"
    metrics_path = "metrik_model.json"

    try:
        test_df = pd.read_csv(test_path)
        required_test_columns = ["Tingkat_Stres_Aktual", "Tingkat_Stres_Prediksi"]
        missing_test_columns = [c for c in required_test_columns if c not in test_df.columns]
        if missing_test_columns:
            raise ValueError(
                "Kolom wajib tidak ditemukan pada hasil pengujian: "
                + ", ".join(missing_test_columns)
            )
        if test_df.empty:
            raise ValueError("File hasil pengujian tidak berisi data.")

        y_true = test_df["Tingkat_Stres_Aktual"].astype(str)
        y_pred_test = test_df["Tingkat_Stres_Prediksi"].astype(str)
        invalid_labels = (set(y_true) | set(y_pred_test)) - set(STRESS_CLASSES)
        if invalid_labels:
            raise ValueError(f"Label kelas tidak dikenali: {sorted(invalid_labels)}")

        cm = confusion_matrix(y_true, y_pred_test, labels=STRESS_CLASSES)
        calculated_metrics = {
            "Accuracy": accuracy_score(y_true, y_pred_test),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred_test),
            "Macro Precision": precision_score(
                y_true, y_pred_test, labels=STRESS_CLASSES, average="macro", zero_division=0
            ),
            "Macro Recall": recall_score(
                y_true, y_pred_test, labels=STRESS_CLASSES, average="macro", zero_division=0
            ),
            "Macro F1": f1_score(
                y_true, y_pred_test, labels=STRESS_CLASSES, average="macro", zero_division=0
            ),
            "Weighted F1": f1_score(
                y_true, y_pred_test, labels=STRESS_CLASSES, average="weighted", zero_division=0
            ),
        }

        st.caption(f"Sumber: `{test_path}` · jumlah data uji: {len(test_df)}")
        st.subheader("Metrik Evaluasi")
        metric_order = [
            "Accuracy", "Balanced Accuracy", "Macro Precision",
            "Macro Recall", "Macro F1", "Weighted F1"
        ]
        metric_cols = st.columns(3)
        for i, metric_name in enumerate(metric_order):
            with metric_cols[i % 3]:
                st.metric(metric_name, f"{calculated_metrics[metric_name] * 100:.2f}%")

        st.subheader("_Confusion Matrix_")
        cm_df = pd.DataFrame(
            cm,
            index=[f"Aktual {name}" for name in STRESS_CLASSES],
            columns=[f"Prediksi {name}" for name in STRESS_CLASSES],
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption(
            "Baris menunjukkan kelas aktual dan kolom menunjukkan kelas prediksi. "
            "Diagonal menunjukkan jumlah klasifikasi yang benar."
        )

        st.subheader("Laporan Klasifikasi per Kelas")
        report = classification_report(
            y_true,
            y_pred_test,
            labels=STRESS_CLASSES,
            output_dict=True,
            zero_division=0,
        )
        report_rows = []
        for label in STRESS_CLASSES:
            item = report[label]
            report_rows.append({
                "Kelas": label,
                "Precision": item["precision"],
                "Recall": item["recall"],
                "F1-score": item["f1-score"],
                "Support": int(item["support"]),
            })
        report_df = pd.DataFrame(report_rows)
        st.dataframe(
            report_df.style.format({
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-score": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Cocokkan metrik yang dihitung dari CSV dengan metrik yang diekspor notebook.
        try:
            with open(metrics_path, "r", encoding="utf-8") as metrics_file:
                experiment = json.load(metrics_file)
            saved_metrics = experiment.get("metrics", {})
            comparison_keys = [
                "Accuracy",
                "Balanced Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
                "Weighted F1",
            ]
            discrepancies = []
            for key in comparison_keys:
                if key not in saved_metrics:
                    discrepancies.append(f"{key}: tidak tersedia di {metrics_path}")
                elif abs(float(saved_metrics[key]) - calculated_metrics[key]) > 1e-6:
                    discrepancies.append(
                        f"{key}: JSON={float(saved_metrics[key]):.6f}, "
                        f"CSV={calculated_metrics[key]:.6f}"
                    )
            if discrepancies:
                st.warning("Metrik JSON dan CSV berbeda:\n- " + "\n- ".join(discrepancies))
            else:
                st.success(
                    "Audit berhasil: metrik dari hasil_prediksi_testing.csv sesuai dengan "
                    "metrik yang tersimpan di metrik_model.json."
                )
        except FileNotFoundError:
            st.warning(f"`{metrics_path}` tidak ditemukan; metrik tetap dihitung dari CSV pengujian.")
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            st.warning(f"File `{metrics_path}` tidak dapat dibaca untuk audit: {e}")

        st.subheader("Konfigurasi Model dari Notebook")
        config_items = [
            ("Algoritma", "Support Vector Machine"),
            ("Kernel", "RBF"),
            ("Standardisasi", "StandardScaler"),
            ("Fitur", " + ".join(feature_names)),
            ("Jumlah data uji", str(len(test_df))),
        ]
        try:
            with open(metrics_path, "r", encoding="utf-8") as metrics_file:
                experiment = json.load(metrics_file)
            best_params = experiment.get("best_params", {})
            param_labels = {
                "svc__C": "C",
                "svc__gamma": "Gamma",
                "svc__class_weight": "Class weight",
            }
            for key, label in param_labels.items():
                if key in best_params:
                    config_items.append((label, str(best_params[key])))
            if "best_cv_macro_f1" in experiment:
                config_items.append((
                    "CV Macro F1 (GridSearchCV)",
                    f"{float(experiment['best_cv_macro_f1']):.6f}",
                ))
            if "random_state" in experiment:
                config_items.append(("Random state", str(experiment["random_state"])))
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            config_items.extend([
                ("C", "100"),
                ("Gamma", "0.1"),
                ("Class weight", "balanced"),
            ])
        config_df = pd.DataFrame(config_items, columns=["Komponen", "Nilai"])
        st.dataframe(config_df, use_container_width=True, hide_index=True)

        st.info(
            "Balanced accuracy merata-ratakan recall antarkelas, sedangkan accuracy "
            "mengukur proporsi seluruh prediksi yang benar. Keduanya dapat berbeda "
            "ketika distribusi kelas tidak seimbang."
        )
        st.warning(
            "Data uji hanya berisi 30 responden dan kelas Rendah memiliki jumlah sampel "
            "yang sangat kecil. Karena itu, interpretasikan metrik per kelas secara hati-hati; "
            "hasil ini belum menjamin performa yang sama pada populasi mahasiswa secara umum."
        )

    except FileNotFoundError:
        st.error(
            f"File `{test_path}` tidak ditemukan. Tambahkan artefak hasil ekspor notebook "
            "Google Colab ke direktori yang sama dengan app.py."
        )
    except Exception as e:
        st.error("Evaluasi model tidak dapat dihitung dari artefak notebook.")
        st.exception(e)
