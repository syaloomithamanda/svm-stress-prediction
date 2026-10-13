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
    page_title="Prediksi Tingkat Stres UNSRAT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    st.title("🧠 SVM Stress App")
    st.caption("Aplikasi penelitian — UNSRAT")
    st.divider()

    st.subheader("⚙️ Konfigurasi Model")
    st.write("**Algoritma:** SVM")
    st.write("**Kernel:** RBF")
    st.write("**C:** 100")
    st.write("**Gamma:** 0.1")
    st.write("**Class weight:** balanced")
    st.write("**Fitur:** PASS + PSQI")

    st.divider()
    st.subheader("🎨 Kategori Stres")
    st.success("🟢 **Rendah** — PSS-10 0–13")
    st.warning("🟡 **Sedang** — PSS-10 14–26")
    st.error("🔴 **Tinggi** — PSS-10 27–40")

    st.divider()
    st.caption("PSS-10 digunakan untuk membentuk target. PASS dan PSQI digunakan sebagai fitur model. PSS-10_Score tidak digunakan sebagai fitur untuk mencegah target leakage. SHAP menjelaskan keputusan model dan bukan hubungan sebab-akibat.")

# -----------------------------
# Header
# -----------------------------
st.title("🧠 Prediksi Tingkat Stres Mahasiswa")
st.subheader("Semester Akhir — Universitas Sam Ratulangi")
st.write("Sistem klasifikasi berbasis **Support Vector Machine (SVM)** dengan pendekatan **Explainable AI (SHAP)**.")

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
    "💡 Explainable AI (SHAP)",
    "📈 Performa Model",
])

# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    st.header("Prediksi Tingkat Stres Individu")
    st.write("Masukkan skor **PASS** dan **PSQI** untuk melihat kategori yang diprediksi oleh model SVM terbaru.")

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
    st.write("Kategori aktual dibentuk dari skor PSS-10 pada responden penelitian.")

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
    st.header("Explainable AI dengan SHAP")
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

    st.subheader("Global Feature Importance")
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
st.caption("Prediksi Tingkat Stres Mahasiswa Semester Akhir UNSRAT  •  Support Vector Machine (SVM) + Explainable AI (SHAP)")


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

        st.subheader("Confusion Matrix")
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
                "Accuracy", "Balanced Accuracy", "Macro Precision", "Macro Recall", "Macro F1"
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
