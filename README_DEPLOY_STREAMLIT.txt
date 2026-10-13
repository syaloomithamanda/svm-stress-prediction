# Deployment Streamlit — SVM + SHAP UNSRAT

## Isi root repository GitHub
- `app.py`
- `requirements.txt`
- `svm_model_final.joblib` (diunduh dari ZIP hasil Google Colab)
- `feature_names.joblib`
- `class_names.joblib`
- `metrik_model.json`
- `hasil_prediksi_testing.csv` (untuk confusion matrix dan laporan klasifikasi)
- `shap_background.csv` (untuk penjelasan SHAP pada prediksi individu)
- `dataset_responden_app.csv` (opsional, untuk distribusi aktual dan tabel data)

## Langkah
1. Jalankan semua sel notebook `SVM_SHAP_Google_Colab_dari_scoring_final_app_ready.ipynb` di Google Colab.
2. Unduh `hasil_svm_shap.zip` dari sel terakhir, lalu ekstrak di komputer.
3. Unggah file-file hasil ekstraksi ke root repository yang sama dengan `app.py` dan `requirements.txt`.
4. Pastikan file `requirements.txt` benar-benar bernama `requirements.txt`, bukan `requirements_streamlit.txt` atau `requirements(1).txt`.
5. Pada Streamlit Community Cloud, pilih repository, branch `main`, dan main file `app.py`. Reboot setelah commit.

## Catatan kompatibilitas
Notebook mematok versi pustaka inti agar selaras dengan `requirements.txt`. Jalankan ulang notebook dari awal setelah mengganti versi instalasi; jangan gunakan model `.joblib` lama yang dilatih dengan versi scikit-learn berbeda tanpa menguji kompatibilitas.
