import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Prediksi Kriminalitas", page_icon="⚖️", layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Navigasi",
        options=["Dashboard", "Prediksi Baru", "Tentang"],
        icons=["bar-chart", "cpu", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

uploaded_file = st.sidebar.file_uploader("Unggah Dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'INDONESIA' in df['Kepolisian Daerah'].values:
        df = df[df['Kepolisian Daerah'] != 'INDONESIA'].reset_index(drop=True)

    # Deteksi kolom 'Jumlah Tindak Pidana' dan 'Penyelesaian tindak pidana' otomatis
    jumlah_cols = sorted([col for col in df.columns if 'Jumlah Tindak Pidana' in col])
    selesai_cols = sorted([col for col in df.columns if 'Penyelesaian tindak pidana' in col])

    if len(jumlah_cols) >= 2 and len(selesai_cols) >= 2:
        jumlah_terakhir = jumlah_cols[-2:]
        selesai_terakhir = selesai_cols[-2:]

        # Hitung rata-rata penyelesaian
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
            st.error("Data hanya memiliki satu kelas target. Mohon tambahkan variasi kelas.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            metrics = {
                "Akurasi": accuracy_score(y_test, y_pred),
                "Presisi": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "F1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }

            class_names = model.classes_

            if selected == "Dashboard":
                st.title("⚖️ Dashboard Prediksi Penanganan Kriminalitas")
                st.markdown(f"""
                Menampilkan analisis performa model dan distribusi berdasarkan data tahun {jumlah_terakhir[0][-4:]} dan {jumlah_terakhir[1][-4:]}.
                """)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("Performa Model")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Akurasi", f"{metrics['Akurasi']:.2%}")
                    m2.metric("Presisi", f"{metrics['Presisi']:.2%}")
                    m3.metric("Recall", f"{metrics['Recall']:.2%}")
                    m4.metric("F1-Score", f"{metrics['F1']:.2%}")

                    with st.expander("Lihat Data"):
                        st.dataframe(df)

                    st.subheader("Visualisasi Decision Tree")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plot_tree(model, feature_names=features, class_names=class_names, filled=True, rounded=True, fontsize=10)
                    st.pyplot(fig)

                with col2:
                    st.subheader("Distribusi Penanganan")
                    st.bar_chart(df['Tingkat_Penanganan'].value_counts())

            elif selected == "Prediksi Baru":
                st.title("📊 Input Prediksi")
                st.markdown("Masukkan estimasi jumlah kasus untuk dua tahun terakhir")

                nilai1 = st.number_input(f"{features[0]}", min_value=0, value=5000)
                nilai2 = st.number_input(f"{features[1]}", min_value=0, value=7000)

                if st.button("Prediksi Sekarang", use_container_width=True):
                    input_data = np.array([[nilai1, nilai2]])
                    input_scaled = scaler.transform(input_data)
                    hasil = model.predict(input_scaled)
                    proba = model.predict_proba(input_scaled)

                    st.success(f"Tingkat Penanganan: {hasil[0]}")
                    st.subheader("Probabilitas Prediksi:")
                    proba_df = pd.DataFrame(proba, columns=class_names)
                    proba_df.index = ["Probabilitas"]
                    st.dataframe(proba_df.T)

                    st.subheader("Visualisasi Probabilitas:")
                    st.bar_chart(proba_df.T)

            elif selected == "Tentang":
                st.title("ℹ️ Tentang Aplikasi")
                st.markdown("""
                Aplikasi ini dibuat untuk memprediksi tingkat penyelesaian kasus kriminalitas berdasarkan dua tahun terakhir
                menggunakan algoritma Machine Learning Decision Tree.

                Dibuat oleh: *Mohammad Azmi Abdussyukur*  
                Sumber Data: BPS / Open Data
                """)
    else:
        st.error("Dataset harus memiliki minimal 2 kolom 'Jumlah Tindak Pidana' dan 'Penyelesaian tindak pidana'.")
else:
    st.warning("Silakan unggah file dataset terlebih dahulu.")
    st.markdown("""
    **Format Dataset yang Diperlukan**:

    | Kolom                                | Keterangan                          |
    | ------------------------------------ | ----------------------------------- |
    | `Kepolisian Daerah`                  | Nama wilayah kepolisian             |
    | `Jumlah Tindak Pidana 2023`          | Jumlah kasus tahun tertentu         |
    | `Jumlah Tindak Pidana 2024`          | Jumlah kasus tahun tertentu         |
    | `Penyelesaian tindak pidana 2023(%)` | Persentase kasus selesai tahun tsb  |
    | `Penyelesaian tindak pidana 2024(%)` | Persentase kasus selesai tahun tsb  |

    Pastikan nama kolom sesuai format di atas dan tersedia minimal 2 tahun.
    """)
