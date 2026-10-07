import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Tingkat Stres Mahasiswa UNSRAT", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
EXPECTED_FEATURES = ["PASS", "PSQI"]
STRESS_CLASSES = ["Rendah", "Sedang", "Tinggi"]

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
    st.code("svm_model_final.pkl\nfeature_names.pkl\nshap_background.pkl\nhasil_prediksi_final.csv")
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

with st.sidebar:
    st.header("⚙️ Informasi Model")
    st.write("**Algoritma:** Support Vector Machine")
    st.write("**Kernel:** RBF")
    st.write("**C:** 100")
    st.write("**Gamma:** 0.1")
    st.write("**Class Weight:** balanced")
    st.write("**Fitur:** PASS + PSQI")
    st.write("**Output:** Rendah / Sedang / Tinggi")
    st.divider()
    st.caption("Model memprediksi kategori tingkat stres berdasarkan fitur PASS dan PSQI.")

st.markdown('<div class="section-kicker">Sistem Prediksi Penelitian</div>', unsafe_allow_html=True)

st.title("🧠 Prediksi Tingkat Stres Mahasiswa Semester Akhir")
st.subheader("Universitas Sam Ratulangi")
st.write("Aplikasi penelitian menggunakan algoritma **Support Vector Machine (SVM)** dengan pendekatan **Explainable AI (SHAP)**.")
st.info("**Catatan penelitian**\n\nModel SVM final menggunakan **PASS** dan **PSQI** sebagai fitur masukan dengan **Tingkat_Stres** sebagai variabel target. **PSS-10_Score** digunakan untuk pembentukan target dan tidak digunakan sebagai fitur masukan.\n\nModel menggunakan StandardScaler dan SVM kernel RBF dengan **C = 100**, **gamma = 0.1**, dan **class weight = balanced**.\n\nNilai SHAP menjelaskan kontribusi fitur terhadap output model dan tidak dimaksudkan sebagai bukti hubungan sebab-akibat.")

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Prediksi Individu", "📊 150 Responden", "🏫 Analisis Fakultas", "💡 Explainable AI (SHAP)"])

with tab1:
    st.header("Prediksi Tingkat Stres Individu")
    st.write("Masukkan nilai PASS dan PSQI untuk memperoleh kategori tingkat stres yang diprediksi model.")
    c1, c2 = st.columns(2)
    with c1:
        pass_value = st.number_input("Nilai PASS", min_value=0.0, max_value=90.0, value=50.0, step=1.0)
    with c2:
        psqi_value = st.number_input("Nilai PSQI", min_value=0.0, max_value=21.0, value=9.0, step=1.0)
    if st.button("🔍 Prediksi Tingkat Stres", use_container_width=True):
        input_data = pd.DataFrame([[pass_value, psqi_value]], columns=EXPECTED_FEATURES)
        prediction = model.predict(input_data)[0]
        st.success(f"Hasil prediksi model: **{prediction}**")
        st.info("Model memprediksi kategori Tingkat_Stres (Rendah, Sedang, atau Tinggi), bukan nilai PSS-10 secara langsung.")
        st.subheader("Penjelasan SHAP")
        try:
            explainer = create_shap_explainer()
            with st.spinner("Menghitung penjelasan SHAP..."):
                shap_result = explainer.shap_values(input_data, nsamples=100)
            classes = list(model.classes_)
            class_idx = classes.index(prediction)
            shap_array = normalize_shap_values(shap_result, 1, len(EXPECTED_FEATURES), len(classes))
            local_values = shap_array[0, :, class_idx]
            local_table = pd.DataFrame({"Fitur": EXPECTED_FEATURES, "Nilai": [pass_value, psqi_value], "SHAP": local_values})
            st.dataframe(local_table.style.format({"Nilai": "{:.2f}", "SHAP": "{:.6f}"}), use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(local_table["Fitur"], local_table["SHAP"])
            ax.axvline(0, linewidth=1)
            ax.set_xlabel("Nilai SHAP")
            ax.set_title(f"Kontribusi Fitur terhadap Kelas {prediction}")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.caption("SHAP positif menunjukkan kontribusi menuju output kelas yang sedang dijelaskan; SHAP negatif menunjukkan kontribusi berlawanan. Ini bukan bukti hubungan sebab-akibat.")
        except Exception as e:
            st.warning("Penjelasan SHAP tidak dapat ditampilkan.")
            st.exception(e)

with tab2:
    st.header("Distribusi Tingkat Stres 150 Responden")
    st.write("Distribusi berikut menunjukkan kategori tingkat stres aktual berdasarkan PSS-10 yang digunakan sebagai variabel target.")
    counts = df["Tingkat_Stres"].value_counts()
    rendah, sedang, tinggi, total = [int(counts.get(x, 0)) for x in STRESS_CLASSES] + [len(df)]
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Rendah", rendah, f"{rendah / total * 100:.2f}%")
    with c2: st.metric("Sedang", sedang, f"{sedang / total * 100:.2f}%")
    with c3: st.metric("Tinggi", tinggi, f"{tinggi / total * 100:.2f}%")
    actual_table = pd.DataFrame({"Kategori Tingkat Stres": STRESS_CLASSES, "Jumlah": [rendah, sedang, tinggi], "Persentase": [f"{x / total * 100:.2f}%" for x in [rendah, sedang, tinggi]]})
    st.subheader("Distribusi Aktual")
    st.dataframe(actual_table, use_container_width=True, hide_index=True)
    pred_counts = df["Prediksi_Batch"].value_counts()
    pred_table = pd.DataFrame({"Kategori Prediksi": STRESS_CLASSES, "Jumlah": [int(pred_counts.get(x, 0)) for x in STRESS_CLASSES]})
    pred_table["Persentase"] = (pred_table["Jumlah"] / total * 100).round(2).astype(str) + "%"
    st.subheader("Distribusi Hasil Prediksi Model")
    st.write("Distribusi prediksi dapat berbeda dari distribusi kategori aktual.")
    st.dataframe(pred_table, use_container_width=True, hide_index=True)
    st.subheader("Data Responden")
    display_columns = [c for c in ["ID", "Jenis_Kelamin", "Usia", "Fakultas", "Angkatan", "PASS", "PSQI", "PSS-10_Score", "Tingkat_Stres", "Prediksi_Batch"] if c in df.columns]
    st.dataframe(df[display_columns], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Unduh hasil prediksi 150 responden (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name="hasil_prediksi_final.csv", mime="text/csv")
    if total != 150: st.warning(f"File saat ini berisi {total} responden, bukan 150. Periksa kembali hasil_prediksi_final.csv.")

with tab3:
    st.header("Analisis Berdasarkan Fakultas")
    if "Fakultas" not in df.columns:
        st.warning("Kolom Fakultas tidak tersedia.")
    else:
        faculty = pd.crosstab(df["Fakultas"], df["Tingkat_Stres"])
        for cat in STRESS_CLASSES:
            if cat not in faculty.columns: faculty[cat] = 0
        faculty = faculty[STRESS_CLASSES]
        st.subheader("Jumlah Responden berdasarkan Kategori")
        st.dataframe(faculty, use_container_width=True)
        pct = (faculty.div(faculty.sum(axis=1), axis=0) * 100).round(2)
        st.subheader("Persentase dalam Masing-masing Fakultas")
        st.dataframe(pct, use_container_width=True)
        st.caption("Persentase dihitung di dalam masing-masing fakultas.")

with tab4:
    st.header("Explainable AI dengan SHAP")
    st.write("SHAP digunakan untuk menjelaskan kontribusi fitur PASS dan PSQI terhadap output keputusan model SVM.")
    st.info("Model SVM final menggunakan probability=False. Karena itu, SHAP menjelaskan output decision_function dan bukan probabilitas prediksi.")
    shap_global = pd.DataFrame({"Kelas": ["Rendah","Rendah","Sedang","Sedang","Tinggi","Tinggi"], "Fitur": ["PASS","PSQI","PASS","PSQI","PASS","PSQI"], "Mean |SHAP|": [0.555542,0.580899,0.269581,0.285503,0.648638,0.580207], "Mean SHAP": [0.211535,0.116239,-0.037854,0.036628,-0.166991,-0.155836], "SHAP Min": [-0.774064,-0.789384,-0.431665,-0.491609,-1.062037,-1.059518], "SHAP Max": [1.275040,1.367352,0.406396,0.495525,0.960726,0.803173]})
    st.subheader("Global Feature Importance")
    st.write("Nilai berikut merupakan mean absolute SHAP dari 30 data uji penelitian.")
    st.dataframe(shap_global.style.format({c: "{:.6f}" for c in ["Mean |SHAP|","Mean SHAP","SHAP Min","SHAP Max"]}), use_container_width=True, hide_index=True)
    st.subheader("Mean Absolute SHAP")
    chart = shap_global.copy(); chart["Label"] = chart["Kelas"] + " — " + chart["Fitur"]
    fig, ax = plt.subplots(figsize=(9,5)); ax.barh(chart["Label"], chart["Mean |SHAP|"]); ax.set_xlabel("Mean |SHAP|"); ax.set_ylabel("Kelas dan fitur"); ax.set_title("Mean Absolute SHAP berdasarkan kelas"); ax.invert_yaxis(); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("Interpretasi")
    st.write("**Kelas Rendah:** PSQI memiliki mean absolute SHAP 0.580899, sedangkan PASS 0.555542.\n\n**Kelas Sedang:** PSQI memiliki mean absolute SHAP 0.285503, sedangkan PASS 0.269581.\n\n**Kelas Tinggi:** PASS memiliki mean absolute SHAP 0.648638, sedangkan PSQI 0.580207.\n\nNilai tersebut menunjukkan kontribusi fitur terhadap keputusan model pada data yang dianalisis. Nilai SHAP tidak digunakan sebagai bukti hubungan sebab-akibat.")
    st.subheader("Contoh Penjelasan Individu")
    st.dataframe(pd.DataFrame({"ID":[5,33,124],"Aktual":["Sedang","Tinggi","Sedang"],"Prediksi":["Sedang","Tinggi","Rendah"],"PASS":[59,46,55],"PSQI":[9,11,8]}), use_container_width=True, hide_index=True)
    st.write("**ID 5 — Sedang → Sedang:** PASS = 59 dan PSQI = 9. Untuk output kelas Sedang, kontribusi SHAP PASS +0.406396 dan PSQI +0.153212.\n\n**ID 33 — Tinggi → Tinggi:** PASS = 46 dan PSQI = 11. Untuk output kelas Tinggi, kontribusi SHAP PASS +0.465401 dan PSQI +0.311273.\n\n**ID 124 — Sedang → Rendah:** PASS = 55 dan PSQI = 8. Untuk output kelas Rendah, kontribusi SHAP PASS +1.275040 dan PSQI +0.989094. Model menghasilkan prediksi Rendah sedangkan kategori aktualnya Sedang.")

st.divider()
st.caption("Prediksi Tingkat Stres Mahasiswa Semester Akhir UNSRAT")
st.caption("Support Vector Machine (SVM) + Explainable AI (SHAP)")
