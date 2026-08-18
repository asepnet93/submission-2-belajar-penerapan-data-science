# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Meskipun telah mencetak banyak lulusan dengan reputasi yang sangat baik, institusi ini masih menghadapi tantangan besar dalam hal tingkat dropout (mahasiswa yang tidak menyelesaikan pendidikannya) yang cukup tinggi. Tingkat dropout yang tinggi ini berdampak buruk pada reputasi institusi, pemborosan sumber daya, dan kehilangan potensi mahasiswa. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi sedini mungkin mahasiswa yang berpotensi melakukan dropout agar dapat diberikan bimbingan dan intervensi khusus.

### Permasalahan Bisnis
1. Tingginya angka dropout mahasiswa yang tidak terkendali dan merugikan institusi.
2. Belum teridentifikasinya faktor-faktor utama yang menyebabkan mahasiswa memutuskan untuk dropout.
3. Kurangnya sistem monitoring untuk memantau performa akademik dan faktor risiko dropout secara real-time.
4. Belum adanya model prediktif untuk mengidentifikasi mahasiswa yang berpotensi dropout sehingga institusi dapat melakukan intervensi dini.
5. Belum adanya rekomendasi strategi retensi mahasiswa yang berbasis data dan terukur.

### Cakupan Proyek
1. Analisis Data Eksploratif (EDA): Memahami dataset mahasiswa, mengidentifikasi pola, dan mencari korelasi awal antara berbagai faktor dengan status dropout.
2. Preprocessing Data: Menangani missing values, mengencode variabel kategorikal, dan melakukan scaling fitur.
3. Pembuatan Model Machine Learning: Menggunakan algoritma **Random Forest Classifier** untuk memprediksi status mahasiswa (**Dropout vs Graduate**). Data dengan status **Enrolled** tidak digunakan dalam proses training, namun digunakan untuk inferensi/prediksi.
4. Business Dashboard: Membuat dashboard interaktif untuk memonitor faktor-faktor risiko dropout dan performa mahasiswa.
5. Sistem Machine Learning (Prototype): Membuat aplikasi web sederhana menggunakan Streamlit yang dapat digunakan untuk memprediksi status mahasiswa baru secara real-time.
6. Rekomendasi Strategi Retensi: Memberikan action items berbasis data untuk menurunkan angka dropout.

### Persiapan

**Sumber Data:** 
Dataset diperoleh dari platform Dicoding dengan nama `students_performance.csv`. Dataset ini berisi data 4.424 mahasiswa dengan 36 kolom atribut. Dataset dapat diakses melalui tautan berikut:
[students_performance.csv](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

**Setup Environment:**
Python version yang digunakan: 3.10.11

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment (Windows)
venv\Scripts\activate

# Install library yang dibutuhkan
pip install -r requirements.txt
```

---

## Business Dashboard

Dashboard ini dibuat untuk membantu departemen akademik Jaya Jaya Institut dalam memonitor faktor-faktor yang mempengaruhi dropout rate menggunakan **Google Looker Studio**. Dashboard dapat diakses melalui link berikut:

**Akses Dashboard:**
- **Link:** [https://datastudio.google.com/reporting/bf8cdd66-7680-4083-ba24-ae48d162316e] 

### Komponen Dashboard
Dashboard ini menampilkan visualisasi untuk memantau performa mahasiswa dan faktor risiko dropout, antara lain:
1. **Overview:** Menampilkan total mahasiswa, jumlah dropout, enrolled, dan graduate.
2. **Distribusi Status:** Menampilkan proporsi status mahasiswa secara keseluruhan.
3. **Faktor Risiko:** Menampilkan hubungan antara nilai semester (Curricular Units Grade) dengan status mahasiswa.
4. **Demografi:** Menampilkan distribusi status berdasarkan usia, jenis kelamin, dan status beasiswa.

---

## Menjalankan Sistem Machine Learning

Prototype sistem machine learning dibuat menggunakan **Streamlit**. Aplikasi ini memungkinkan pengguna untuk memasukkan data mahasiswa dan mendapatkan prediksi status (**Dropout** atau **Graduate**) secara instan.

### Cara Menjalankan Secara Lokal (Local Development)
1. Install seluruh library yang dibutuhkan dengan menjalankan perintah `pip install -r requirements.txt`.
2. Pastikan file `model_rf.pkl` (model yang sudah dilatih) berada dalam folder yang sama dengan `app.py`.
3. Buka terminal atau command prompt pada folder proyek.
4. Jalankan perintah berikut:
   ```bash
   streamlit run app.py
   ```
5. Aplikasi akan terbuka secara otomatis di browser Anda pada alamat `http://localhost:8501`.

### Akses Prototype Online (Streamlit Community Cloud)
Prototype sistem machine learning ini telah di-deploy ke Streamlit Community Cloud dan dapat diakses secara online melalui link berikut:

**Akses Prototype:**
- **Link:** [https://asepnet93-submission-2-belajar-penerapan-data-science.streamlit.app/]

---

## Conclusion

### Kesimpulan Analisis Data (Faktor dan Karakteristik Dropout)
Berdasarkan analisis eksplorasi data yang telah dilakukan pada dataset mahasiswa Jaya Jaya Institut, ditemukan beberapa faktor dan karakteristik utama yang berkaitan dengan dropout:

1. **Distribusi Status:** Dari total 4.424 mahasiswa, terdapat 2.209 Graduate (49.9%), 1.421 Dropout (32.1%), dan 794 Enrolled (18.0%). Proporsi dropout yang signifikan menjadi perhatian utama institusi.

2. **Nilai Akademik:** Mahasiswa dengan nilai semester 1 dan semester 2 (Curricular Units Grade) yang rendah memiliki risiko dropout yang sangat tinggi. Nilai di bawah 10 (dari skala 0-20) mengindikasikan potensi dropout yang besar.

3. **Beasiswa:** Mahasiswa penerima beasiswa (Scholarship Holder) cenderung memiliki tingkat dropout lebih rendah dibandingkan mahasiswa non-beasiswa.

4. **Usia:** Mahasiswa dengan usia yang lebih muda saat mendaftar (di bawah 20 tahun) memiliki risiko dropout yang lebih tinggi dibandingkan mahasiswa yang lebih dewasa.

5. **Nilai Masuk:** Admission Grade (nilai ujian masuk) juga menjadi faktor penting, di mana mahasiswa dengan nilai masuk rendah cenderung memiliki risiko dropout lebih tinggi.

### Kesimpulan Performa Model Machine Learning

Berdasarkan model **Random Forest Classifier** yang telah dilatih **hanya menggunakan data Dropout dan Graduate** (data Enrolled digunakan untuk inferensi):

#### Performa Model
| Metrik | Nilai |
|--------|-------|
| **Akurasi** | **88%** |
| Precision (Dropout) | 89% |
| Recall (Dropout) | 79% |
| F1-Score (Dropout) | 84% |
| Precision (Graduate) | 87% |
| Recall (Graduate) | 94% |
| F1-Score (Graduate) | 90% |

#### Confusion Matrix
```
              Predicted
              Dropout  Graduate
Actual Dropout   224       60
Actual Graduate   28      414
```

#### Feature Importance (5 Faktor Paling Berpengaruh)
1. Curricular_units_2nd_sem_grade (Nilai Semester 2) - 33.49%
2. Curricular_units_1st_sem_grade (Nilai Semester 1) - 18.59%
3. Age_at_enrollment (Usia Saat Mendaftar) - 5.57%
4. Admission_grade (Nilai Masuk) - 3.50%
5. Previous_qualification_grade (Nilai Kualifikasi Sebelumnya) - 2.90%

#### Prediksi Data Enrolled (Inferensi)
Model juga digunakan untuk memprediksi **794 siswa yang saat ini berstatus Enrolled** (masih aktif kuliah):

| Prediksi | Jumlah | Persentase |
|----------|--------|------------|
| **Graduate** | 527 | 66.4% |
| **Dropout** | 267 | 33.6% |

Hasil ini memberikan **early warning** bahwa 267 siswa aktif berpotensi dropout dan memerlukan intervensi dini.

### Rekomendasi Action Items
Berdasarkan hasil analisis data dan model machine learning, berikut beberapa rekomendasi action items yang dapat dilakukan Jaya Jaya Institut untuk menurunkan angka dropout:

#### 1. Program Bimbingan Akademik Intensif untuk Mahasiswa Berisiko
Berdasarkan model, nilai semester 1 dan 2 adalah prediktor terkuat untuk dropout (total 32.82% feature importance). Institusi harus mengidentifikasi mahasiswa dengan nilai di bawah ambang batas (misal: kurang dari 10) pada akhir semester pertama. Mereka harus segera diberikan program bimbingan belajar, tutor sebaya, dan konseling akademik sebelum semester berikutnya dimulai.

#### 2. Pengembangan Sistem Peringatan Dini (Early Warning System)
Implementasikan dashboard monitoring (yang telah dibuat) secara real-time untuk melacak performa mahasiswa. Jika sistem mendeteksi mahasiswa dengan penurunan nilai atau SKS yang tidak diambil secara konsisten, sistem harus mengirimkan notifikasi otomatis kepada dosen pembimbing akademik.

#### 3. Optimalisasi Program Beasiswa
Analisis menunjukkan mahasiswa penerima beasiswa memiliki tingkat dropout lebih rendah. Institusi harus memperluas program beasiswa berbasis prestasi dan kebutuhan, serta memberikan bimbingan karir tambahan bagi penerima beasiswa agar mereka tetap termotivasi.

#### 4. Program Orientasi dan Mentoring untuk Mahasiswa Baru
Mahasiswa dengan usia muda (di bawah 20 tahun) memiliki risiko dropout lebih tinggi. Buatlah program orientasi yang lebih intensif, dan pasangkan setiap mahasiswa baru dengan mentor (dosen atau mahasiswa senior) untuk membantu mereka beradaptasi dengan lingkungan akademik selama tahun pertama.

#### 5. Intervensi Dini untuk 267 Siswa Enrolled Berisiko
Berdasarkan prediksi model, terdapat **267 siswa aktif yang berpotensi dropout**. Institusi harus segera:
- Mengidentifikasi daftar siswa tersebut
- Memberikan konseling akademik personal
- Memonitor perkembangan nilai mereka setiap bulan
- Menyediakan program remedial jika diperlukan

#### 6. Evaluasi Kurikulum untuk Mahasiswa dengan Nilai Awal Rendah
Beberapa mahasiswa mungkin mengalami kesulitan karena kurikulum yang terlalu berat di awal. Pertimbangkan untuk menyediakan kelas remedial atau foundation year bagi mahasiswa dengan nilai admission grade yang rendah agar mereka dapat mengejar ketertinggalan sebelum memasuki mata kuliah inti.

---

## Kesimpulan Akhir

Model Random Forest yang dikembangkan berhasil mencapai **akurasi 88%** dalam memprediksi status mahasiswa (Dropout vs Graduate). Dengan akurasi ini, model sudah **siap digunakan** untuk membantu Jaya Jaya Institut dalam mengidentifikasi mahasiswa yang berpotensi dropout dan melakukan intervensi dini. Dashboard interaktif dan aplikasi Streamlit yang telah dikembangkan memungkinkan institusi untuk memonitor faktor risiko secara real-time dan mengambil tindakan preventif.

**Rekomendasi utama** adalah segera melakukan intervensi terhadap **267 siswa aktif yang diprediksi berpotensi dropout**, dengan fokus pada peningkatan nilai akademik di semester 1 dan 2, serta pemberian program bimbingan dan beasiswa yang lebih optimal.
```