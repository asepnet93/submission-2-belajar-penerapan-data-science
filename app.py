import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================
# LOAD MODEL, SCALER, & ENCODER
# ============================================
try:
    model = joblib.load('model_rf.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    
    # Load nama kolom kategorikal (dari Notebook)
    with open('categorical_cols.pkl', 'rb') as f:
        categorical_cols = pickle.load(f)
except FileNotFoundError:
    st.error("File model/tools tidak ditemukan. Pastikan semua file .pkl ada.")
    st.stop()

# ============================================
# UI STREAMLIT
# ============================================
st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="wide")

st.title("Sistem Prediksi Status Mahasiswa")
st.markdown("""
Aplikasi ini membantu institusi pendidikan untuk memprediksi kemungkinan status mahasiswa 
berdasarkan data akademik dan demografis. 
**Masukkan data mahasiswa di bawah ini, lalu klik tombol Prediksi.**
""")

st.divider()

# Layout 3 kolom untuk input
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Data Pribadi & Akademik")
    
    # Pilihan status pernikahan dalam bentuk teks
    marital_status_map = {"Single": 1, "Menikah": 2, "Cerai": 3, "Janda/Duda": 4, "Lainnya 5": 5, "Lainnya 6": 6}
    marital_status_label = st.selectbox("Status Pernikahan", list(marital_status_map.keys()))
    marital_status = marital_status_map[marital_status_label]
    
    application_mode = st.number_input("Mode Pendaftaran", min_value=1, step=1, value=1)
    application_order = st.number_input("Urutan Pendaftaran", min_value=1, step=1, value=1)
    course = st.number_input("Kode Kursus", min_value=1, step=1, value=171)
    
    # Pilihan jadwal kuliah dalam bentuk teks
    daytime_map = {"Malam": 0, "Siang": 1}
    daytime_label = st.selectbox("Jadwal Kuliah", list(daytime_map.keys()))
    daytime_evening_attendance = daytime_map[daytime_label]
    
    previous_qualification = st.number_input("Kualifikasi Sebelumnya", min_value=1, step=1, value=1)
    previous_qualification_grade = st.number_input("Nilai Kualifikasi Sebelumnya (0-200)", 
                                                    min_value=0.0, max_value=200.0, value=120.0)

with col2:
    st.subheader("Data Demografis")
    nacionality = st.number_input("Kode Kewarganegaraan", min_value=1, step=1, value=1)
    mothers_qualification = st.number_input("Kualifikasi Ibu", min_value=1, step=1, value=1)
    fathers_qualification = st.number_input("Kualifikasi Ayah", min_value=1, step=1, value=1)
    admission_grade = st.number_input("Nilai Masuk (0-200)", min_value=0.0, max_value=200.0, value=120.0)
    
    # Pemetaan untuk pilihan biner (Ya/Tidak)
    bin_map = {"Tidak": 0, "Ya": 1}
    
    displaced_label = st.selectbox("Tempat Tinggal (Terlantar)", list(bin_map.keys()))
    displaced = bin_map[displaced_label]
    
    educational_special_needs_label = st.selectbox("Kebutuhan Khusus", list(bin_map.keys()))
    educational_special_needs = bin_map[educational_special_needs_label]
    
    debtor_label = st.selectbox("Status Hutang (Debtor)", list(bin_map.keys()))
    debtor = bin_map[debtor_label]
    
    tuition_fees_up_to_date_label = st.selectbox("SPP Lunas", list(bin_map.keys()))
    tuition_fees_up_to_date = bin_map[tuition_fees_up_to_date_label]
    
    # Pilihan jenis kelamin dalam bentuk teks
    gender_map = {"Laki-laki": 0, "Perempuan": 1}
    gender_label = st.selectbox("Jenis Kelamin", list(gender_map.keys()))
    gender = gender_map[gender_label]

with col3:
    st.subheader("Data Tambahan")
    
    scholarship_holder_label = st.selectbox("Penerima Beasiswa", list(bin_map.keys()))
    scholarship_holder = bin_map[scholarship_holder_label]
    
    age_at_enrollment = st.number_input("Usia Saat Mendaftar", min_value=17, max_value=70, value=20)
    
    international_label = st.selectbox("Mahasiswa Internasional", list(bin_map.keys()))
    international = bin_map[international_label]
    
    curricular_units_1st_sem_grade = st.number_input("Nilai Semester 1 (0-20)", 
                                                      min_value=0.0, max_value=20.0, value=10.0)
    curricular_units_2nd_sem_grade = st.number_input("Nilai Semester 2 (0-20)", 
                                                      min_value=0.0, max_value=20.0, value=10.0)
    unemployment_rate = st.number_input("Tingkat Pengangguran (%)", min_value=0.0, value=10.0)
    inflation_rate = st.number_input("Tingkat Inflasi (%)", min_value=0.0, value=2.0)
    gdp = st.number_input("GDP", min_value=-10.0, value=1.0)

st.divider()

# ============================================
# PREDIKSI
# ============================================
if st.button("Prediksi Status Mahasiswa", use_container_width=True, type="primary"):
    
    # Gabungkan semua input menjadi satu array (24 fitur)
    input_data = np.array([[
        marital_status, application_mode, application_order, course, daytime_evening_attendance,
        previous_qualification, previous_qualification_grade, nacionality, mothers_qualification,
        fathers_qualification, admission_grade, displaced, educational_special_needs, debtor,
        tuition_fees_up_to_date, gender, scholarship_holder, age_at_enrollment, international,
        curricular_units_1st_sem_grade, curricular_units_2nd_sem_grade, unemployment_rate,
        inflation_rate, gdp
    ]])
    
    # Scaling data input
    input_data_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)
    
    # Ambil classes dari model
    classes = model.classes_
    result = prediction[0]
    
    # Tampilkan hasil prediksi
    col_result, col_proba = st.columns(2)
    
    with col_result:
        if result == "Dropout":
            st.error(f"### Hasil Prediksi: {result}")
            st.warning("Mahasiswa ini berisiko tinggi dropout. Segera lakukan intervensi!")
        else:
            st.success(f"### Hasil Prediksi: {result}")
            st.balloons()
            st.info("Mahasiswa ini diprediksi akan lulus dengan baik.")
    
    with col_proba:
        st.subheader("Tingkat Keyakinan Model:")
        
        proba_values = prediction_proba[0]
        
        proba_df = pd.DataFrame({
            "Status": classes,
            "Probabilitas": proba_values
        })
        
        st.dataframe(proba_df, use_container_width=True, hide_index=True)
        st.bar_chart(proba_df.set_index("Status"))
    
    # ============================================
    # REKOMENDASI ACTION ITEMS
    # ============================================
    st.divider()
    st.subheader("Rekomendasi Tindakan untuk Jaya Jaya Institut")
    
    if result == "Dropout":
        st.markdown("""
        **Tindakan yang Disarankan:**
        
        1. Konseling Akademik - Jadwalkan pertemuan rutin dengan dosen pembimbing
        2. Bantuan Finansial - Evaluasi kelayakan beasiswa tambahan atau program cicilan
        3. Program Remedial - Tawarkan kelas tambahan untuk mata kuliah dengan nilai rendah
        4. Keterlibatan Orang Tua - Libatkan orang tua dalam proses monitoring
        5. Monitoring Berkala - Lakukan evaluasi setiap akhir semester
        """)
    else:
        st.markdown("""
        **Tindakan yang Disarankan:**
        
        1. Apresiasi Prestasi - Berikan penghargaan atas pencapaian akademik
        2. Persiapan Wisuda - Bantu persiapan administrasi dan syarat kelulusan
        3. Jaringan Alumni - Ajak bergabung ke komunitas alumni
        4. Dokumentasi - Catat faktor kesuksesan sebagai bahan evaluasi institusi
        5. Role Model - Jadikan sebagai mentor untuk adik tingkat
        """)

    # ============================================
    # INFORMASI MODEL
    # ============================================
    with st.expander("Informasi Model Prediksi"):
        st.markdown("""
        **Informasi Model:**
        - Algoritma: Random Forest Classifier
        - Akurasi: 88%
        - Kelas yang Diprediksi: Dropout dan Graduate
        
        **5 Faktor Paling Berpengaruh:**
        1. Nilai Semester 2 - 18.4%
        2. Nilai Semester 1 - 14.5%
        3. Nilai Ujian Masuk - 7.3%
        4. Nilai Kualifikasi Sebelumnya - 6.5%
        5. Usia Saat Mendaftar - 6.4%
        """)

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("2026 - Sistem Prediksi Status Mahasiswa | Jaya Jaya Institut")