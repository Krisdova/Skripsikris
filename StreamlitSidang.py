import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
import os

# Load data dan model
data_kaggle = pd.read_excel("diabetes_kaggle.xlsx")  # Ganti sesuai nama file
model_terbaik = joblib.load("adaboost_model.pkl")

# Sidebar
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Tampilkan Data", "Preprocessing", "Split Data", "SMOTE", "Modeling", "Prediksi"])

st.title("Aplikasi Prediksi Diabetes")

# 0. Halaman Beranda
if page == "Beranda":
    st.subheader("Tentang Diabetes")
    st.write("""
    Diabetes, yang sering kali dikenal sebagai penyakit pembunuh yang tidak terlihat, disebabkan oleh tingginya kadar insulin dalam darah. Pankreas bertugas memproduksi insulin untuk menurunkan kadar gula darah yang tinggi, tetapi ketika kadar gula darah menjadi tidak normal, hal ini dapat berdampak negatif pada berbagai organ internal dan sistem saraf tubuh manusia.

    Penyakit ini tidak hanya mengancam kesehatan individu, tetapi juga menjadi perhatian di tingkat global karena dampaknya yang signifikan dan luas. Dengan meningkatnya jumlah penderita diabetes di seluruh dunia, kesadaran akan pentingnya pencegahan dan penanganan penyakit ini semakin mendesak, menjadikannya sebagai masalah kesehatan serius yang perlu diatasi dengan segera dan efektif.

    Diabetes terbagi menjadi dua tipe utama, yaitu **diabetes melitus tipe 1** dan **tipe 2**. Tipe 1 terjadi karena reaksi autoimun, di mana tubuh menyerang sel-sel di pankreas yang memproduksi insulin, sehingga kadar insulin berkurang drastis.

    Sebaliknya, **diabetes melitus tipe 2** muncul akibat kombinasi faktor genetik dan lingkungan. Faktor genetik melibatkan gangguan sekresi insulin serta resistensi insulin, sementara faktor lingkungan seperti obesitas, pola makan berlebihan, kurang olahraga, stres, dan penuaan memperburuk kondisi ini. Tipe 2 lebih sering dikaitkan dengan gaya hidup tidak sehat dan menjadi lebih umum di masyarakat modern.
    """)

# 1. Tampilkan Data
elif page == "Tampilkan Data":
    st.subheader("Dataset Mentah")
    st.write(data_kaggle)

# 2. Preprocessing
elif page == "Preprocessing":
    st.subheader("Menghapus Duplikat")
    kaggle_baru1 = data_kaggle.drop_duplicates()
    st.write(f"Jumlah data setelah menghapus duplikat: {kaggle_baru1.shape[0]}")
    
    st.subheader("Encoding Kolom Kategorikal")
    label_encoder = LabelEncoder()
    kaggle_baru1['gender'] = label_encoder.fit_transform(kaggle_baru1['gender'])
    kaggle_baru1['smoking_history'] = label_encoder.fit_transform(kaggle_baru1['smoking_history'])
    st.write(kaggle_baru1[['gender', 'smoking_history']].head())

    st.subheader("Menghapus Outlier (Z-Score > 4)")
    z_scores = np.abs(zscore(kaggle_baru1.select_dtypes(include=np.number)))
    mask = (z_scores < 4).all(axis=1)
    kaggle_no_outliers = kaggle_baru1[mask]
    st.write(f"Jumlah data setelah outlier dihapus: {kaggle_no_outliers.shape[0]}")

    st.subheader("Menghapus Kolom Tidak Diperlukan")
    kolom_hapus = ['hypertension','heart_disease','smoking_history','bmi']
    kaggle_baru = kaggle_no_outliers.drop(columns=kolom_hapus)
    st.write(kaggle_baru.head())

# 3. Split Data
elif page == "Split Data":
    st.subheader("Split Data (Train & Test)")
    kaggle_baru1 = data_kaggle.drop_duplicates()
    kaggle_baru1['gender'] = LabelEncoder().fit_transform(kaggle_baru1['gender'])
    kaggle_baru1['smoking_history'] = LabelEncoder().fit_transform(kaggle_baru1['smoking_history'])
    z_scores = np.abs(zscore(kaggle_baru1.select_dtypes(include=np.number)))
    mask = (z_scores < 4).all(axis=1)
    kaggle_no_outliers = kaggle_baru1[mask]
    kaggle_baru = kaggle_no_outliers.drop(columns=['hypertension','heart_disease','smoking_history','bmi'])

    X = kaggle_baru.drop(columns='diabetes')
    y = kaggle_baru['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write("Jumlah Data:", X.shape[0])
    st.write("Data Latih:", X_train.shape[0])
    st.write("Data Uji:", X_test.shape[0])

# 4. SMOTE
elif page == "SMOTE":
    st.subheader("SMOTE Oversampling")
    kaggle_baru1 = data_kaggle.drop_duplicates()
    kaggle_baru1['gender'] = LabelEncoder().fit_transform(kaggle_baru1['gender'])
    kaggle_baru1['smoking_history'] = LabelEncoder().fit_transform(kaggle_baru1['smoking_history'])
    z_scores = np.abs(zscore(kaggle_baru1.select_dtypes(include=np.number)))
    mask = (z_scores < 4).all(axis=1)
    kaggle_no_outliers = kaggle_baru1[mask]
    kaggle_baru = kaggle_no_outliers.drop(columns=['hypertension','heart_disease','smoking_history','bmi'])

    X = kaggle_baru.drop(columns='diabetes')
    y = kaggle_baru['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    st.write("Sebelum SMOTE:")
    st.write(y_train.value_counts())
    st.write("Setelah SMOTE:")
    st.write(y_train_smote.value_counts())

# 5. Modeling
elif page == "Modeling":
    st.subheader("Modeling - Confusion Matrix")
    image_path = "Model 2.png"
    
    if os.path.exists(image_path):
        st.image(image_path, caption="Confusion Matrix", use_column_width=True)
    else:
        st.warning(f"Gambar '{image_path}' tidak ditemukan di folder ini.")

# 6. Prediksi
elif page == "Prediksi":
    st.subheader("Masukkan Data untuk Prediksi")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki", "Lainnya"])
        age = st.number_input("Umur", min_value=0, step=1)

    with col2:
        HbA1c_levelc = st.number_input("HbA1c", min_value=0.0, step=0.1)
        blood_glucose_level = st.number_input("Gula Darah", min_value=0.0, step=0.1)

    gender_map = {"Perempuan": 0, "Laki-laki": 1, "Lainnya": 2}
    gender_numerik = gender_map[gender]
    new_data = np.array([[gender_numerik, age, HbA1c_levelc, blood_glucose_level]])

    if st.button("Prediksi"):
        result = model_terbaik.predict(new_data)[0]
        st.subheader("Hasil Prediksi")
        if result == 1:
            st.success("Pasien diprediksi memiliki diabetes.")
        else:
            st.success("Pasien diprediksi tidak memiliki diabetes.")