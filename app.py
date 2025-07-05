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

    # Membersihkan nama kolom dari spasi tersembunyi
    df.columns = df.columns.str.strip()

    if 'INDONESIA' in df['Kepolisian Daerah'].values:
        df = df[df['Kepolisian Daerah'] != 'INDONESIA'].reset_index(drop=True)

    # Validasi kolom yang dibutuhkan
    required_cols = ['Jumlah Tindak Pidana', 'Penyelesaian tindak pidana (%)']
    if not all(col in df.columns for col in required_cols):
        st.error("Kolom tidak lengkap. Harus ada: 'Jumlah Tindak Pidana' dan 'Penyelesaian tindak pidana (%)'")
    else:
        # Klasifikasi berdasarkan penyelesaian
        def klasifikasi(p):
            if p > 70:
                return 'Tinggi'
            elif p >= 55:
                return 'Sedang'
            return 'Rendah'

        df['Tingkat_Penanganan'] = df['Penyelesaian tindak pidana (%)'].apply(klasifikasi)

        features = ['Jumlah Tindak Pidana']
        target = 'Tingkat_Penanganan'

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)

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
            st.title("⚖️ Dashboard Prediksi Kriminalitas")
            st.markdown("Analisis dan visualisasi berdasarkan jumlah kasus dan tingkat penyelesaian secara umum.")

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
                plot_tree(
                    model,
                    feature_names=features,
                    class_names=class_names,
                    filled=True,
                    rounded=True,
                    fontsize=10
                )
                st.pyplot(fig)

            with col2:
                st.subheader("Distribusi Penanganan")
                st.bar_chart(df['Tingkat_Penanganan'].value_counts())

        elif selected == "Prediksi Baru":
            st.title("📊 Input Prediksi")
            st.markdown("Masukkan estimasi jumlah kasus untuk memprediksi tingkat penyelesaian kriminalitas.")

            jumlah_kasus = st.number_input("Jumlah Tindak Pidana", min_value=0, value=5000)

            if st.button("Prediksi Sekarang", use_container_width=True):
                input_data = np.array([[jumlah_kasus]])
                input_scaled = scaler.transform(input_data)
                hasil = model.predict(input_scaled)
                proba = model.predict_proba(input_scaled)

                st.success(f"Tingkat Penanganan: {hasil[0]}")
                st.subheader("Probabilitas:")
                st.dataframe(pd.DataFrame(proba, columns=class_names))

        elif selected == "Tentang":
            st.title("ℹ️ Tentang Aplikasi")
            st.markdown("""
            Aplikasi ini dibuat untuk memprediksi tingkat penyelesaian kasus kriminalitas berdasarkan jumlah kasus menggunakan model Machine Learning Decision Tree.

            Dibuat oleh: *Mohammad Azmi Abdussyukur*  
            Sumber Data: BPS / Open Data
            """)

else:
    st.warning("Silakan unggah file dataset terlebih dahulu untuk melanjutkan.")
    st.markdown("""
    Format dataset yang sesuai:
    | Kolom                          | Keterangan                         |
    | ----------------------------- | ---------------------------------- |
    | `Kepolisian Daerah`           | Nama wilayah kepolisian            |
    | `Jumlah Tindak Pidana`        | Jumlah kasus di wilayah tsb        |
    | `Penyelesaian tindak pidana (%)` | Persentase penyelesaian kasus    |
    """)
