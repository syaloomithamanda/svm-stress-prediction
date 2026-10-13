import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import shap

st.set_page_config(page_title="Prediksi Tingkat Stres UNSRAT", page_icon="🧠", layout="wide")

ROOT = Path(__file__).resolve().parent
FEATURES_EXPECTED = ["PASS", "PSQI"]
CLASSES_EXPECTED = ["Rendah", "Sedang", "Tinggi"]
COLORS = {"Rendah": "#2E8B57", "Sedang": "#E5A000", "Tinggi": "#D64545"}

@st.cache_resource
def load_artifacts():
    required = ["svm_model_final.joblib", "feature_names.joblib", "class_names.joblib", "metrik_model.json"]
    missing = [f for f in required if not (ROOT / f).exists()]
    if missing:
        raise FileNotFoundError("Artefak belum lengkap: " + ", ".join(missing))
    model = joblib.load(ROOT / "svm_model_final.joblib")
    features = list(joblib.load(ROOT / "feature_names.joblib"))
    classes = list(joblib.load(ROOT / "class_names.joblib"))
    metrics = json.loads((ROOT / "metrik_model.json").read_text(encoding="utf-8"))
    background_path = ROOT / "shap_background.csv"
    background = pd.read_csv(background_path) if background_path.exists() else None
    if background is not None:
        background = background[features].apply(pd.to_numeric, errors="raise")
    return model, features, classes, metrics, background

try:
    model, features, classes, metrics, background = load_artifacts()
except Exception as exc:
    st.error("Artefak model tidak berhasil dimuat.")
    st.markdown("Pastikan file hasil ekspor Colab berada satu folder dengan `app.py`.")
    st.code("svm_model_final.joblib\nfeature_names.joblib\nclass_names.joblib\nmetrik_model.json\nshap_background.csv (disarankan)\ndataset_responden_app.csv (opsional)")
    st.exception(exc)
    st.stop()

if features != FEATURES_EXPECTED or classes != list(model.classes_):
    st.error(f"Ketidaksesuaian artefak. Fitur: {features}; kelas file: {classes}; kelas model: {list(model.classes_)}")
    st.stop()

@st.cache_data
def load_dataset():
    path = ROOT / "dataset_responden_app.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    required = FEATURES_EXPECTED + ["Tingkat_Stres"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Kolom dataset tidak ditemukan: {missing}")
    for col in FEATURES_EXPECTED:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["Tingkat_Stres"] = data["Tingkat_Stres"].astype(str).str.strip().str.title()
    data = data.dropna(subset=required).copy()
    data["Prediksi_Model"] = model.predict(data[FEATURES_EXPECTED])
    return data

try:
    df = load_dataset()
except Exception as exc:
    st.error("Dataset opsional tidak dapat dibaca.")
    st.exception(exc)
    df = None

st.title("🧠 Prediksi Tingkat Stres Mahasiswa")
st.subheader("Semester Akhir — Universitas Sam Ratulangi")
st.write("Klasifikasi menggunakan **Support Vector Machine (SVM) kernel RBF** dengan fitur **PASS** dan **PSQI**. Pendekatan **SHAP** digunakan untuk menjelaskan output model.")

with st.sidebar:
    st.header("Konfigurasi")
    st.write("**Algoritma:** SVM RBF")
    st.write("**Fitur:** PASS, PSQI")
    st.write("**Target:** Tingkat_Stres dari PSS-10")
    st.caption("PSS-10 tidak dimasukkan sebagai fitur prediktor untuk menghindari target leakage.")
    st.divider()
    st.markdown("**Kategori target**")
    st.write("- Rendah: skor PSS-10 0–13")
    st.write("- Sedang: skor PSS-10 14–26")
    st.write("- Tinggi: skor PSS-10 27–40")

if df is not None:
    counts = df["Tingkat_Stres"].value_counts().reindex(CLASSES_EXPECTED, fill_value=0)
    a,b,c,d = st.columns(4)
    a.metric("Responden", len(df))
    b.metric("Rendah", int(counts["Rendah"]), f"{counts['Rendah']/len(df)*100:.2f}%" if len(df) else "0%")
    c.metric("Sedang", int(counts["Sedang"]), f"{counts['Sedang']/len(df)*100:.2f}%" if len(df) else "0%")
    d.metric("Tinggi", int(counts["Tinggi"]), f"{counts['Tinggi']/len(df)*100:.2f}%" if len(df) else "0%")
else:
    st.info("Dataset responden tidak disertakan. Prediksi individu dan ringkasan metrik tetap dapat digunakan; unggah `dataset_responden_app.csv` ke repositori untuk menampilkan distribusi aktual.")

predict_tab, data_tab, eval_tab, shap_tab = st.tabs(["🔎 Prediksi Individu", "📊 Data & Distribusi", "📈 Evaluasi Model", "💡 Tentang SHAP"])

with predict_tab:
    st.header("Prediksi skor individu")
    left, right = st.columns(2)
    with left:
        pass_score = st.number_input("Total PASS", min_value=18, max_value=90, value=50, step=1, help="Rentang teoretis total PASS 18–90.")
    with right:
        psqi_score = st.number_input("Skor global PSQI", min_value=0, max_value=21, value=9, step=1, help="Rentang teoretis skor global PSQI 0–21.")
    if st.button("Prediksi tingkat stres", type="primary", use_container_width=True):
        x_one = pd.DataFrame([[pass_score, psqi_score]], columns=features)
        pred = str(model.predict(x_one)[0])
        st.subheader("Hasil prediksi")
        if pred == "Rendah": st.success(f"Kategori prediksi: **{pred}**")
        elif pred == "Sedang": st.warning(f"Kategori prediksi: **{pred}**")
        else: st.error(f"Kategori prediksi: **{pred}**")
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_one)[0]
            prob_df = pd.DataFrame({"Kategori": model.classes_, "Probabilitas model (%)": np.asarray(probs)*100})
            st.dataframe(prob_df.round(2), hide_index=True, use_container_width=True)
            st.caption("Probabilitas adalah output model terkalibrasi internal SVM, bukan hasil diagnosis klinis.")
        st.info("Prediksi ini merupakan estimasi model berdasarkan PASS dan PSQI, bukan pengukuran PSS-10 langsung atau diagnosis klinis.")
        if background is not None:
            try:
                with st.spinner("Menghitung penjelasan SHAP untuk input ini..."):
                    bg = background.sample(n=min(20, len(background)), random_state=42)
                    # Fungsi pembungkus menerima array, menghindari mutasi feature_names_in_ pada Pipeline.
                    def predict_proba_for_shap(values):
                        frame = pd.DataFrame(np.asarray(values), columns=features)
                        return model.predict_proba(frame)
                    explainer = shap.KernelExplainer(predict_proba_for_shap, bg.to_numpy())
                    raw = explainer.shap_values(x_one.to_numpy(), nsamples=100)
                idx = list(model.classes_).index(pred)
                if isinstance(raw, list):
                    contrib = np.asarray(raw[idx]).reshape(-1)
                else:
                    arr = np.asarray(raw)
                    if arr.ndim == 3 and arr.shape == (1, len(features), len(model.classes_)):
                        contrib = arr[0, :, idx]
                    elif arr.ndim == 3 and arr.shape == (len(model.classes_), 1, len(features)):
                        contrib = arr[idx, 0, :]
                    elif arr.ndim == 2 and arr.shape == (1, len(features)):
                        contrib = arr[0]
                    else:
                        raise ValueError(f"Format SHAP tidak dikenali: {arr.shape}")
                shap_df = pd.DataFrame({"Fitur": features, "Nilai input": [pass_score, psqi_score], "Nilai SHAP": contrib})
                st.markdown("#### Kontribusi SHAP untuk kelas prediksi")
                fig = px.bar(shap_df, x="Nilai SHAP", y="Fitur", orientation="h", color="Nilai SHAP", color_continuous_scale="RdBu")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(shap_df.round(5), hide_index=True, use_container_width=True)
                st.caption("SHAP menjelaskan kontribusi fitur terhadap output probabilitas kelas model; bukan bukti sebab-akibat.")
            except Exception as exc:
                st.warning("Prediksi berhasil, tetapi perhitungan SHAP gagal. Periksa versi shap/scikit-learn pada requirements.")
                st.exception(exc)
        else:
            st.caption("Penjelasan SHAP tersedia setelah `shap_background.csv` dari notebook Colab ditambahkan ke repositori.")

with data_tab:
    st.header("Dataset dan distribusi aktual")
    if df is None:
        st.warning("File `dataset_responden_app.csv` belum tersedia.")
    else:
        dist = df["Tingkat_Stres"].value_counts().reindex(CLASSES_EXPECTED, fill_value=0).rename_axis("Kategori").reset_index(name="Jumlah")
        dist["Persentase"] = dist["Jumlah"] / max(1, len(df))*100
        fig = px.bar(dist, x="Kategori", y="Jumlah", color="Kategori", text="Jumlah", category_orders={"Kategori": CLASSES_EXPECTED}, color_discrete_map=COLORS)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dist.round(2), hide_index=True, use_container_width=True)
        show_cols = [c for c in ["PASS", "PSQI", "Tingkat_Stres", "Prediksi_Model"] if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
        st.download_button("Unduh tabel responden (CSV)", df.to_csv(index=False).encode("utf-8"), "hasil_dataset_prediksi.csv", "text/csv")

with eval_tab:
    st.header("Evaluasi model dari Google Colab")
    key_map = [("test_accuracy", "Accuracy"), ("test_balanced_accuracy", "Balanced accuracy"), ("test_macro_f1", "Macro-F1"), ("best_cv_macro_f1", "Best CV macro-F1")]
    cols = st.columns(4)
    for col, (key, label) in zip(cols, key_map):
        val = metrics.get(key)
        col.metric(label, f"{float(val)*100:.2f}%" if val is not None else "Tidak tersedia")
    st.write("**Parameter terbaik:**")
    st.json(metrics.get("best_params", {}))
    pred_path = ROOT / "hasil_prediksi_testing.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        if {"Aktual", "Prediksi"}.issubset(pred_df.columns):
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(pred_df["Aktual"], pred_df["Prediksi"], labels=CLASSES_EXPECTED)
            cm_df = pd.DataFrame(cm, index=[f"Aktual {x}" for x in CLASSES_EXPECTED], columns=[f"Prediksi {x}" for x in CLASSES_EXPECTED])
            st.subheader("Confusion matrix — test set")
            st.dataframe(cm_df, use_container_width=True)
            st.subheader("Laporan klasifikasi")
            st.code(classification_report(pred_df["Aktual"], pred_df["Prediksi"], labels=CLASSES_EXPECTED, zero_division=0, digits=4))
        else:
            st.info("File prediksi testing tersedia, tetapi kolom Aktual/Prediksi tidak ditemukan.")
    else:
        st.info("Tambahkan `hasil_prediksi_testing.csv` hasil ekspor Colab untuk menampilkan confusion matrix dan laporan klasifikasi.")
    st.caption("Metrik di atas harus berasal dari file ekspor run Colab yang sama dengan model yang dipasang.")

with shap_tab:
    st.header("Explainable AI (SHAP)")
    st.write("SHAP memperkirakan kontribusi fitur PASS dan PSQI terhadap output model. Penjelasan tidak menunjukkan sebab-akibat dan bukan diagnosis.")
    if background is None:
        st.warning("`shap_background.csv` belum tersedia. Tambahkan file background dari hasil ekspor Colab agar penjelasan individu aktif.")
    else:
        st.success(f"Background SHAP berhasil dimuat: {len(background)} baris, {len(features)} fitur.")
    st.markdown("**Catatan interpretasi:** nilai positif/negatif bergantung pada kelas output yang sedang dijelaskan. Besarnya nilai SHAP tidak boleh ditafsirkan sebagai persentase probabilitas atau kontribusi kausal.")
