
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from streamlit_option_menu import option_menu

# Konfigurasi halaman utama
st.set_page_config(page_title="Prediksi Kriminalitas", page_icon="⚖️", layout="wide")

# --- NAVIGASI SIDEBAR ---
with st.sidebar:
    selected = option_menu(
        menu_title="Navigasi",
        options=["Dashboard", "Prediksi Baru", "Tentang"],
        icons=["bar-chart", "cpu", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# --- UNGGAH DATASET ---
uploaded_file = st.sidebar.file_uploader("Unggah Dataset (.csv)", type=["csv"])

# --- PROSES UTAMA ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Pra-pemrosesan: Hapus baris 'INDONESIA' jika ada
    if 'Kepolisian Daerah' in df.columns and 'INDONESIA' in df['Kepolisian Daerah'].values:
        df = df[df['Kepolisian Daerah'] != 'INDONESIA'].reset_index(drop=True)

    # Deteksi kolom fitur dan target secara dinamis
    jumlah_cols = sorted([col for col in df.columns if 'Jumlah Tindak Pidana' in col])
    selesai_cols = sorted([col for col in df.columns if 'Penyelesaian tindak pidana' in col])

    # Pastikan dataset memiliki kolom yang dibutuhkan
    if len(jumlah_cols) >= 2 and len(selesai_cols) >= 2:
        # Ambil 2 tahun terakhir untuk fitur dan target
        jumlah_terakhir = jumlah_cols[-2:]
        selesai_terakhir = selesai_cols[-2:]

        # --- FEATURE ENGINEERING ---
        df['Rata_Rata_Penyelesaian(%)'] = df[selesai_terakhir].mean(axis=1)

        def klasifikasi(p):
            if p > 70:
                return 'Tinggi'
            elif p >= 55:
                return 'Sedang'
            return 'Rendah'

        df['Tingkat_Penanganan'] = df['Rata_Rata_Penyelesaian(%)'].apply(klasifikasi)

        features = jumlah_terakhir
        target = 'Tingkat_Penanganan'
        X = df[features]
        y = df[target]

        if len(y.unique()) < 2:
            st.error("Error: Data target hanya memiliki satu kelas. Model tidak dapat dilatih. Mohon gunakan data yang lebih bervariasi.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = DecisionTreeClassifier(max_depth=3, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            metrics = {
                "Akurasi": accuracy_score(y_test, y_pred),
                "Presisi": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            class_names = model.classes_.tolist()

            if selected == "Dashboard":
                st.title("⚖️ Dashboard Prediksi Penanganan Kriminalitas")
                st.markdown(f"""
                Analisis performa model dan distribusi data berdasarkan **Jumlah Tindak Pidana** tahun **{jumlah_terakhir[0][-4:]}** dan **{jumlah_terakhir[1][-4:]}**.
                """)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("Performa Model")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Akurasi", f"{metrics['Akurasi']:.2%}")
                    m2.metric("Presisi", f"{metrics['Presisi']:.2%}")
                    m3.metric("Recall", f"{metrics['Recall']:.2%}")
                    m4.metric("F1-Score", f"{metrics['F1-Score']:.2%}")

                    st.subheader("Visualisasi Decision Tree")
                    fig, ax = plt.subplots(figsize=(15, 8))
                    plot_tree(model, feature_names=features, class_names=class_names, filled=True, rounded=True, fontsize=10)
                    st.pyplot(fig)

                with col2:
                    st.subheader("Distribusi Penanganan")
                    st.bar_chart(df['Tingkat_Penanganan'].value_counts())

                    with st.expander("Lihat Data Lengkap"):
                        st.dataframe(df)

            elif selected == "Prediksi Baru":
                st.title("📊 Input untuk Prediksi Baru")
                st.markdown("Masukkan estimasi jumlah kasus untuk memprediksi tingkat penanganan.")

                st.info(f"""
                Rata-rata dari data Anda:
                - Rata-rata untuk `{features[0]}`: **{int(df[features[0]].mean())}**
                - Rata-rata untuk `{features[1]}`: **{int(df[features[1]].mean())}**
                """)

                with st.form("prediction_form"):
                    nilai1 = st.number_input(f"Jumlah Kasus ({features[0][-4:]})", min_value=0, value=int(df[features[0]].mean()))
                    nilai2 = st.number_input(f"Jumlah Kasus ({features[1][-4:]})", min_value=0, value=int(df[features[1]].mean()))
                    submit_button = st.form_submit_button("Prediksi Sekarang", use_container_width=True)

                if submit_button:
                    input_data = np.array([[nilai1, nilai2]])
                    input_scaled = scaler.transform(input_data)
                    hasil = model.predict(input_scaled)
                    proba = model.predict_proba(input_scaled)

                    st.success(f"**Hasil Prediksi Tingkat Penanganan: {hasil[0]}**")
                    st.subheader("Probabilitas Prediksi:")
                    proba_df = pd.DataFrame(proba, columns=class_names, index=["Probabilitas"])
                    st.dataframe(proba_df)

                    st.subheader("Visualisasi Probabilitas:")
                    st.bar_chart(proba_df.T)

            elif selected == "Tentang":
                st.title("ℹ️ Tentang Aplikasi")
                st.markdown("""
                Aplikasi ini menggunakan Machine Learning untuk memprediksi **Tingkat Penanganan Kriminalitas** (Tinggi, Sedang, Rendah) 
                berdasarkan data jumlah tindak pidana dari dua tahun terakhir.

                - **Model**: `DecisionTreeClassifier` dari Scikit-Learn.
                - **Metodologi**: Data jumlah kasus dari dua tahun terakhir digunakan sebagai fitur untuk melatih model. Tingkat penanganan diklasifikasikan berdasarkan rata-rata persentase penyelesaian kasus.

                ---
                - **Dibuat oleh**: *Mohammad Azmi Abdussyukur*
                - **Sumber Data**: Badan Pusat Statistik (BPS) atau sumber Open Data lainnya.
                """)

    else:
        st.error("Dataset tidak valid. Pastikan dataset Anda memiliki minimal 2 kolom yang mengandung 'Jumlah Tindak Pidana' dan 2 kolom yang mengandung 'Penyelesaian tindak pidana'.")

else:
    st.warning("Silakan unggah file dataset (.csv) melalui sidebar untuk memulai.")
    st.markdown("""
    #### Format Dataset yang Diperlukan:
    | Kepolisian Daerah | Jumlah Tindak Pidana 2023 | Jumlah Tindak Pidana 2024 | Penyelesaian tindak pidana 2023(%) | Penyelesaian tindak pidana 2024(%) |
    |-------------------|---------------------------|---------------------------|--------------------------------------|--------------------------------------|
    | POLDA ACEH        | 5500                      | 5800                      | 65.5                                 | 68.2                                 |
    | POLDA SUMUT       | 25000                     | 26100                     | 72.1                                 | 75.0                                 |
    """)
