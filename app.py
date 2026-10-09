import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

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
    model = joblib.load("svm_model_final.pkl")
    feature_names = joblib.load("feature_names.pkl")
    background = joblib.load("shap_background.pkl")
    return model, feature_names, background

try:
    model, feature_names, background = load_model()
except Exception as e:
    st.error("Model tidak dapat dimuat.")
    st.write("Pastikan file berikut tersedia di repository:")
    st.code("svm_model_final.pkl\nfeature_names.pkl\nshap_background.pkl\nhasil_prediksi_testing.csv")
    st.exception(e)
    st.stop()

if list(feature_names) != EXPECTED_FEATURES:
    st.error(f"Fitur model tidak sesuai. Fitur yang ditemukan: {feature_names}")
    st.stop()


def model_decision_for_shap(data):
    return model.decision_function(pd.DataFrame(data, columns=feature_names))


@st.cache_resource
def create_shap_explainer():
    return shap.KernelExplainer(model_decision_for_shap, background)


def normalize_shap_values(shap_result, n_rows, n_features, n_classes):
    arr = np.asarray(shap_result)
    if isinstance(shap_result, list):
        if arr.ndim == 3 and arr.shape == (n_classes, n_rows, n_features):
            return np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape == (n_rows, n_features, n_classes):
            return arr
    if arr.ndim == 2 and arr.shape == (n_rows, n_features):
        return arr[:, :, np.newaxis]
    if arr.ndim == 3:
        if arr.shape == (n_rows, n_features, n_classes):
            return arr
        if arr.shape == (n_rows, n_classes, n_features):
            return np.transpose(arr, (0, 2, 1))
        if arr.shape == (n_classes, n_rows, n_features):
            return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Bentuk keluaran SHAP tidak dikenali: {arr.shape}")


try:
    df = pd.read_csv("hasil_prediksi_final.csv")
except Exception as e:
    st.error("File hasil_prediksi_final.csv tidak ditemukan.")
    st.exception(e)
    st.stop()

missing_columns = [c for c in ["PASS", "PSQI", "Tingkat_Stres"] if c not in df.columns]
if missing_columns:
    st.error(f"Kolom berikut tidak ditemukan: {missing_columns}")
    st.stop()

if "Prediksi_Batch" not in df.columns:
    df["Prediksi_Batch"] = model.predict(df[EXPECTED_FEATURES])

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
    st.metric("🟢 Rendah", rendah, f"{rendah / total * 100:.2f}%")
with m3:
    st.metric("🟡 Sedang", sedang, f"{sedang / total * 100:.2f}%")
with m4:
    st.metric("🔴 Tinggi", tinggi, f"{tinggi / total * 100:.2f}%")

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
    st.write("Masukkan skor **PASS** dan **PSQI** untuk melihat kategori yang diprediksi oleh model SVM.")

    c1, c2 = st.columns(2)
    with c1:
        pass_value = st.number_input(
            "📘 Nilai PASS",
            min_value=0.0,
            max_value=90.0,
            value=50.0,
            step=1.0,
            help="Masukkan skor total PASS.",
        )
    with c2:
        psqi_value = st.number_input(
            "🌙 Nilai PSQI",
            min_value=0.0,
            max_value=21.0,
            value=9.0,
            step=1.0,
            help="Masukkan skor global PSQI (0–21).",
        )

    st.caption("Rentang PSQI: 0–21. Skor PASS pada aplikasi dibatasi 0–90 untuk memberi ruang input.")

    if st.button("🔍  Prediksi Tingkat Stres", type="primary", use_container_width=True):
        input_data = pd.DataFrame([[pass_value, psqi_value]], columns=EXPECTED_FEATURES)
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Hasil Prediksi")
        result_col, detail_col = st.columns([1, 2])

        with result_col:
            if prediction == "Rendah":
                st.success(f"{STRESS_ICONS[prediction]} **{prediction.upper()}**")
            elif prediction == "Sedang":
                st.warning(f"{STRESS_ICONS[prediction]} **{prediction.upper()}**")
            else:
                st.error(f"{STRESS_ICONS[prediction]} **{prediction.upper()}**")
            st.caption("Kategori yang dipilih oleh model SVM")

        with detail_col:
            st.write(f"**PASS:** {pass_value:.0f}")
            st.write(f"**PSQI:** {psqi_value:.0f}")
            st.info("Hasil ini adalah prediksi kategori model, bukan nilai PSS-10 secara langsung.")

        st.subheader("💡 Penjelasan SHAP")
        try:
            explainer = create_shap_explainer()
            with st.spinner("Menghitung kontribusi SHAP..."):
                shap_result = explainer.shap_values(input_data, nsamples=100)

            classes = list(model.classes_)
            class_idx = classes.index(prediction)
            shap_array = normalize_shap_values(
                shap_result, 1, len(EXPECTED_FEATURES), len(classes)
            )
            local_values = shap_array[0, :, class_idx]

            local_table = pd.DataFrame(
                {
                    "Fitur": EXPECTED_FEATURES,
                    "Nilai Input": [pass_value, psqi_value],
                    "SHAP": local_values,
                }
            )
            st.dataframe(
                local_table.style.format({"Nilai Input": "{:.0f}", "SHAP": "{:+.6f}"}),
                use_container_width=True,
                hide_index=True,
            )

            fig, ax = plt.subplots(figsize=(8, 3.8))
            bars = ax.barh(
                local_table["Fitur"],
                local_table["SHAP"],
                color=["#4F46E5" if v >= 0 else "#94A3B8" for v in local_values],
            )
            ax.axvline(0, linewidth=1.2, color="#334155")
            ax.set_xlabel("Nilai SHAP")
            ax.set_title(f"Kontribusi fitur terhadap kelas {prediction}", fontweight="bold")
            ax.grid(axis="x", alpha=0.18)
            for bar, val in zip(bars, local_values):
                ax.text(
                    val + (0.02 if val >= 0 else -0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}",
                    va="center",
                    ha="left" if val >= 0 else "right",
                )
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("SHAP positif menunjukkan kontribusi menuju kelas yang dijelaskan; SHAP negatif menunjukkan kontribusi berlawanan. Ini bukan bukti hubungan sebab-akibat.")
        except Exception as e:
            st.warning("Penjelasan SHAP tidak dapat ditampilkan untuk input ini.")
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
        file_name="hasil_prediksi_final.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if actual_total != 150:
        st.warning(
            f"File saat ini berisi {actual_total} responden, bukan 150. "
            "Periksa kembali hasil_prediksi_final.csv."
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

    shap_global = pd.DataFrame(
        {
            "Kelas": ["Rendah", "Rendah", "Sedang", "Sedang", "Tinggi", "Tinggi"],
            "Fitur": ["PASS", "PSQI", "PASS", "PSQI", "PASS", "PSQI"],
            "Mean |SHAP|": [0.555542, 0.580899, 0.269581, 0.285503, 0.648638, 0.580207],
            "Mean SHAP": [0.211535, 0.116239, -0.037854, 0.036628, -0.166991, -0.155836],
            "SHAP Min": [-0.774064, -0.789384, -0.431665, -0.491609, -1.062037, -1.059518],
            "SHAP Max": [1.275040, 1.367352, 0.406396, 0.495525, 0.960726, 0.803173],
        }
    )

    st.subheader("Global Feature Importance")
    st.caption("Mean absolute SHAP dari 30 data uji penelitian.")

    # Visual summary cards by class
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
            st.metric("PASS", f"{row['PASS']:.4f}")
            st.metric("PSQI", f"{row['PSQI']:.4f}")

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
    st.write("**Kelas Rendah:** PSQI memiliki mean absolute SHAP 0.580899, sedangkan PASS 0.555542.")
    st.write("**Kelas Sedang:** PSQI memiliki mean absolute SHAP 0.285503, sedangkan PASS 0.269581.")
    st.write("**Kelas Tinggi:** PASS memiliki mean absolute SHAP 0.648638, sedangkan PSQI 0.580207.")
    st.caption("Nilai tersebut menunjukkan kontribusi fitur terhadap keputusan model pada data yang dianalisis. Nilai SHAP tidak digunakan sebagai bukti hubungan sebab-akibat.")

    st.subheader("Contoh Penjelasan Individu")
    examples = pd.DataFrame(
        {
            "ID": [5, 33, 124],
            "Aktual": ["Sedang", "Tinggi", "Sedang"],
            "Prediksi": ["Sedang", "Tinggi", "Rendah"],
            "PASS": [59, 46, 55],
            "PSQI": [9, 11, 8],
        }
    )
    st.dataframe(examples, use_container_width=True, hide_index=True)
    st.write("**ID 5 — Sedang → Sedang:** PASS = 59 dan PSQI = 9. Untuk output kelas Sedang, kontribusi SHAP PASS +0.406396 dan PSQI +0.153212.")
    st.write("**ID 33 — Tinggi → Tinggi:** PASS = 46 dan PSQI = 11. Untuk output kelas Tinggi, kontribusi SHAP PASS +0.465401 dan PSQI +0.311273.")
    st.write("**ID 124 — Sedang → Rendah:** PASS = 55 dan PSQI = 8. Untuk output kelas Rendah, kontribusi SHAP PASS +1.275040 dan PSQI +0.989094. Model menghasilkan prediksi Rendah sedangkan kategori aktualnya Sedang.")

st.divider()
st.caption("Prediksi Tingkat Stres Mahasiswa Semester Akhir UNSRAT  •  Support Vector Machine (SVM) + Explainable AI (SHAP)")


# -----------------------------
# Tab 5
# -----------------------------
with tab5:
    st.header("Performa Model SVM")
    st.write(
        "Metrik berikut berasal dari evaluasi model final pada data pengujian "
        "(30 responden) menggunakan pembagian data 80:20 secara stratified."
    )

    metrics = {
        "Accuracy": 0.5000,
        "Balanced Accuracy": 0.6587,
        "Macro Precision": 0.4339,
        "Macro Recall": 0.6587,
        "Macro F1": 0.4307,
    }

    a, b, c, d, e = st.columns(5)
    for col, (label, value) in zip(
        [a, b, c, d, e], metrics.items()
    ):
        with col:
            st.metric(label, f"{value * 100:.2f}%")

    st.subheader("Confusion Matrix")
    cm = np.array([
        [1, 0, 0],
        [6, 10, 5],
        [0, 4, 4],
    ])
    cm_df = pd.DataFrame(
        cm,
        index=[f"Aktual {x}" for x in STRESS_CLASSES],
        columns=[f"Prediksi {x}" for x in STRESS_CLASSES],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.caption(
        "Baris menunjukkan kelas aktual dan kolom menunjukkan kelas prediksi. "
        "Diagonal menunjukkan jumlah klasifikasi yang sesuai."
    )

    st.subheader("Konfigurasi Model Final")
    config_df = pd.DataFrame({
        "Komponen": [
            "Algoritma", "Kernel", "C", "Gamma",
            "Class Weight", "Standardisasi", "Fitur"
        ],
        "Nilai": [
            "Support Vector Machine", "RBF", "100", "0.1",
            "balanced", "StandardScaler", "PASS + PSQI"
        ],
    })
    st.dataframe(config_df, use_container_width=True, hide_index=True)

    st.info(
        "Accuracy dan balanced accuracy menggambarkan aspek yang berbeda. "
        "Balanced accuracy memperhitungkan recall masing-masing kelas sehingga "
        "lebih informatif ketika jumlah anggota kelas tidak seimbang."
    )
