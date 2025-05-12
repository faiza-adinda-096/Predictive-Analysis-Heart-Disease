# Laporan Proyek Machine Learning - Faiza Adinda Fakhira Batubara

## Domain Proyek

Penyakit jantung merupakan salah satu penyebab kematian tertinggi secara global. Masalah pada jantung dapat diklasifikasikan menjadi dua jenis utama, yaitu penyakit jantung dan serangan jantung. Menurut WHO, pada tahun 2019 terdapat 17.9 juta orang meninggal akibat penyakit jantung, mencakup 32% dari seluruh kematian global. Dari jumlah angka kematian tersebut, 85% disebabkan oleh serangan jantung dan stroke.

Di era perkembangan teknologi yang sangat pesat saat ini, teknologi dapat membantu kita untuk mencegah terkena penyakit jantung serta mengurangi angka kematian akibat penyakit jantung. Dengan menggunakan Machine Learning kita dapat memproses analisa data medis dengan menggunakan berbagai metode yang tersedia untuk menemukan pola atau informasi yang penting. Dengan memanfaatkan data klinis pasien seperti tekanan darah, kadar kolesterol, usia, jenis kelamin, dan gaya hidup, kita dapat membangun model prediktif yang membantu tenaga medis dalam mengidentifikasi individu dengan risiko tinggi.

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

- Bagaimana memanfaatkan data klinis pasien untuk mengidentifikasi kemungkinan seseorang menderita penyakit jantung?
- Algoritma machine learning mana yang paling efektif untuk memprediksi risiko penyakit jantung berdasarkan dataset yang tersedia?
- Bagaimana meningkatkan performa model prediksi agar lebih akurat dan dapat diandalkan dalam pengambilan keputusan medis?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Menggunakan fitur-fitur seperti usia, tekanan darah, kadar kolesterol, dll, untuk membangun model prediktif yang dapat mengklasifikasikan pasien berisiko tinggi
- Menerapkan dan membandingkan beberapa algoritma klasifikasi seperti Logistic Regression, Random Forest, dan Support Vector Machine (SVM) untuk memilih model dengan performa terbaik.
- Melakukan evaluasi dan tuning hyperparameter untuk meningkatkan akurasi model dan menurunkan kesalahan prediksi.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Heart Disease Dataset yang tersedia secara publik di Kaggle. Dataset ini dapat diakses melalui tautan berikut: https://www.kaggle.com/datasets/mexwell/heart-disease-dataset 

### Variabel-variabel pada Heart Disease Dataset adalah sebagai berikut:
- age : usia pasien
- sex : jenis kelamin pasien (Binary. 0 = perempuan, 1 = laki-laki)
- chest pain type: jenis nyeri dada (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)
- resting bps: tekanan darah saat istirahat
- cholestrol: kadar kolestrol
- fasting blood sugar: gula darah saat puasa >120 mg/dl (Binary. 0 = tidak, 1 = iya )
- resting ecg: hasil elctrocardiogram saat istirahat (0 = normal, 1 = kelainan gelombang, 2 = hipertrofi ventrikel kiri)
- max heart rate : maksimum detak jantung
- exercise angina: angina akibat olahraga (0 = tidak, 1 = iya)
- oldpeak: tingkat ST yang diinduksi oleh olahraga dibanding istirahat
- ST slope: kemiringan segmen ST selama latihan (0 = downsloping, 1 = flat, 2 = upsloping)
- target: diagnosis penyakit jantung (0 = tidak memiliki penyakit jantung, 1 = memiliki penyakit jantung)

 ### Exploratory Data Analysis
Exploratory data analysis atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.

Berikut ini adalah EDA yang dilakukan:
```python df.info()```

Output dari kode diatas yaitu:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1190 entries, 0 to 1189
Data columns (total 12 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   age                    1190 non-null   int64  
 1   sex                    1190 non-null   int64  
 2   chest pain type        1190 non-null   int64  
 3   resting bp s           1190 non-null   int64  
 4   cholesterol            1190 non-null   int64  
 5   fasting blood sugar    1190 non-null   int64  
 6   resting ecg            1190 non-null   int64  
 7   max heart rate         1190 non-null   int64  
 8   exercise angina        1190 non-null   int64  
 9   oldpeak                1190 non-null   float64
10   ST slope               1190 non-null   int64  
11   target                 1190 non-null   int64  
dtypes: float64(1), int64(11)
memory usage: 111.7 KB
```

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

## Referensi
