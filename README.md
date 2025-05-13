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
- Menerapkan dan membandingkan beberapa algoritma klasifikasi seperti Logistic Regression, Random Forest, dan Support Vector Machine untuk memilih model dengan performa terbaik.
- Melakukan evaluasi dan tuning hyperparameter untuk meningkatkan akurasi model dan menurunkan kesalahan prediksi.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Heart Disease Dataset yang tersedia secara publik di Kaggle. Dataset ini dapat diakses melalui tautan berikut: https://www.kaggle.com/datasets/mexwell/heart-disease-dataset 

### Variabel-variabel pada Heart Disease Dataset adalah sebagai berikut:
- ```age``` : usia pasien
- ```sex``` : jenis kelamin pasien (Binary. 0 = perempuan, 1 = laki-laki)
- ```chest pain type```: jenis nyeri dada (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)
- ```resting bps```: tekanan darah saat istirahat
- ```cholestrol```: kadar kolestrol
- ```fasting blood sugar```: gula darah saat puasa >120 mg/dl (Binary. 0 = tidak, 1 = iya )
- ```resting ecg```: hasil elctrocardiogram saat istirahat (0 = normal, 1 = kelainan gelombang, 2 = hipertrofi ventrikel kiri)
- ```max heart rate```: maksimum detak jantung
- ```exercise angina```: angina akibat olahraga (0 = tidak, 1 = iya)
- ```oldpeak```: tingkat ST yang diinduksi oleh olahraga dibanding istirahat
- ```ST slope```: kemiringan segmen ST selama latihan (0 = downsloping, 1 = flat, 2 = upsloping)
- ```target```: diagnosis penyakit jantung (0 = tidak memiliki penyakit jantung, 1 = memiliki penyakit jantung)

 ### Exploratory Data Analysis
Exploratory data analysis atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.

Berikut ini adalah EDA yang dilakukan:
```python
  heart.info()
  ```

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
Dapat dilihat dari output kode diatas terdapat 1190 baris data dengan 12 kolom pada dataset. Selain itu, tidak terdapat nilai null pada dataset, yang ditunjukkan oleh informasi ```non-null``` pada setiap kolom.

Selanjutnya dengan kode dibawah ini, kita akan melihat statistik deskriptif dataset
```python
  heart.describe()
  ```
Output kode diatas yaitu:
```
|       | age   | sex   | chest pain type | resting bp s | cholesterol | fasting blood sugar | resting ecg | max heart rate | exercise angina | oldpeak | ST slope | target |
|-------|-------|-------|------------------|---------------|-------------|----------------------|--------------|----------------|------------------|---------|----------|--------|
| count | 1190  | 1190  | 1190             | 1190          | 1190        | 1190                 | 1190         | 1190           | 1190             | 1190    | 1190     | 1190   |
| mean  | 53.72 | 0.76  | 3.23             | 132.15        | 210.36      | 0.21                 | 0.70         | 139.73         | 0.39             | 0.92    | 1.62     | 0.53   |
| std   | 9.36  | 0.42  | 0.94             | 18.37         | 101.42      | 0.41                 | 0.87         | 25.52          | 0.49             | 1.09    | 0.61     | 0.50   |
| min   | 28    | 0     | 1                | 0             | 0           | 0                    | 0            | 60             | 0                | -2.6    | 0        | 0      |
| 25%   | 47    | 1     | 3                | 120           | 188         | 0                    | 0            | 121            | 0                | 0       | 1        | 0      |
| 50%   | 54    | 1     | 4                | 130           | 229         | 0                    | 0            | 140.5          | 0                | 0.6     | 2        | 1      |
| 75%   | 60    | 1     | 4                | 140           | 269.75      | 0                    | 2            | 160            | 1                | 1.6     | 2        | 1      |
| max   | 77    | 1     | 4                | 200           | 603         | 1                    | 2            | 202            | 1                | 6.2     | 3        | 1      |
```
Lalu, dengan kode dibawah ini, kita akan melihat apakah ada nilai duplikat
```python
  heart.duplicated().sum()
  ```
Output:
```
np.int64(272)
```
Terlihat dari output diatas bahwa terdapat 272 nilai duplikat pada dataset. Saya akan menanganinya pada tahap **Data Preparation**

Selanjutnya, kita akan cek apakah ada outlier dengan kode dibawah ini
```
# Memilih fitur-fitur numerik yang rawan outlier
features_to_check = ['resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']

plt.figure(figsize=(15, 10))

for i, feature in enumerate(features_to_check, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=heart, y=feature, color='skyblue')
    plt.title(f'Boxplot of {feature}')
    plt.tight_layout()

plt.show()
```
Output:
![image](https://github.com/user-attachments/assets/cd419317-7c10-47ac-892a-d6125c2536d0)
Terlihat bahwa pada keempat fitur tersebut terdeteksi ada outlier

Selanjutnya, saya akan melakukan **Visualisasi Data**
- _Univariate Analysis_
  ![image](https://github.com/user-attachments/assets/bcca5737-cdb0-495a-8df4-cc620c0438bb)
Histogram di atas menunjukkan distribusi masing-masing fitur dalam dataset. Sebagian besar fitur memiliki distribusi normal atau mendekati normal, seperti ```age```, ```resting bp s```, dan ```max heart rate```. Namun,  beberapa fitur seperti ```cholesterol``` dan ```oldpeak``` memiliki outlier dan distribusi yang skewed (tidak simetris). Distribusi ini memberikan gambaran awal mengenai pola data dan potensi perlunya penanganan seperti normalisasi atau transformasi.

- _Multivariate Analysis_
  
  **Scatterplot**
  ![image](https://github.com/user-attachments/assets/ad493e17-59bb-45f2-8a3e-fc977bdb3a97)
Visualisasi pertama menggunakan scatterplot matrix (pairplot) bertujuan untuk melihat hubungan antar seluruh kombinasi fitur numerik secara pairwise. Pada visualisasi ini, setiap kombinasi dua fitur ditampilkan dalam bentuk scatter plot, sementara diagonalnya menunjukkan distribusi masing-masing fitur melalui histogram. Dari scatterplot matrix tersebut, tampak bahwa beberapa fitur seperti chest pain type, max heart rate, oldpeak, dan ST slope menunjukkan pola sebaran yang cukup berbeda antar nilai target (pasien dengan atau tanpa penyakit jantung), sehingga berpotensi menjadi prediktor penting dalam pemodelan.

 **Heatmap Correlation**
 ![image](https://github.com/user-attachments/assets/ed4c66c7-2187-432c-a4ef-2bf4e521a4e0)
Visualisasi kedua berupa heatmap korelasi memperlihatkan hubungan korelasi linear antar fitur numerik menggunakan koefisien Pearson. Hasil heatmap menunjukkan bahwa fitur yang paling berkorelasi dengan variabel target (penyakit jantung) adalah chest pain type (0.46), ST slope (0.51), exercise angina (-0.48), max heart rate (-0.41), dan oldpeak (-0.40). Korelasi positif menunjukkan bahwa semakin tinggi nilai fitur, semakin besar kemungkinan menderita penyakit jantung, dan sebaliknya untuk korelasi negatif. Di sisi lain, beberapa fitur seperti cholesterol, resting blood pressure, dan fasting blood sugar memiliki korelasi rendah terhadap target, yang mengindikasikan kontribusinya dalam prediksi kemungkinan tidak terlalu signifikan.

## Data Preparation
- **Cleaning data**

  Pada tahap ini saya melalukan cleaning data, yaitu menghapus data duplikat dan menghapus outlier. Melakukan cleaning data sangat penting karena dapat meningkatkan kualitas dan keakuratan data.
 
Kode dibawah ini melakukan penghapusan nilai duplikat dan cek kembali apakah ada masih ada nilai duplikat
```python
  heart = heart.drop_duplicates()
  heart.duplicated().sum()
  ```
Output:
```python
 np.int64(0)
  ```
Dapat dilihat dari output diatas bahwa dataset sudah bersih dari nilai duplikat. Selanjutnya saya akan menangani kolom-kolom yang terdapat outlier (```resting bp s```, ```cholesterol```, ```max heart rate```, ```oldpeak```) dengan kode dibawah
```python
  # Menangani Outlier
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Kolom yang ingin dibersihkan dari outlier
outlier_cols = ['resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
heart = remove_outliers_iqr(heart, outlier_cols)
  ```
- **Split Dataset**

  Tahapan split dataset diperlukan untuk memisahkan data menjadi data latih (training) dan data uji (testing), agar model dapat dilatih pada satu bagian data dan dievaluasi pada bagian data yang tidak pernah dilihat sebelumnya. Dengan memisahkan dataset, kita dapat memastikan bahwa evaluasi model lebih objektif dan mencerminkan performa di dunia nyata.
  
  Melakukan split dataset dan cek jumlah data sampel dengan kode dibawah ini:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  print(f'Total # of sample in whole dataset: {len(X)}')
  print(f'Total # of sample in train dataset: {len(X_train)}')
  print(f'Total # of sample in test dataset: {len(X_test)}')
  ```
  Output:
  ```
  Total # of sample in whole dataset: 701
  Total # of sample in train dataset: 560
  Total # of sample in test dataset: 141
  ```
  Dapat dilihat pada output diatas, bahwa split dataset telah berhasil dan telah membagi data menjadi data train dan data test, dengan jumlah sampel **560 pada data train** dan **141 pada data test**.
  
- **Standarisasi**

  Standarisasi dilakukan untuk memastikan seluruh fitur numerik berada dalam skala yang sama, yaitu dengan rata-rata 0 dan standar deviasi 1. Hal ini penting agar algoritma machine learning yang sensitif terhadap skala fitur, seperti Support Vector Machine (SVM) dan Logistic Regression, dapat bekerja secara optimal.

  Melakukan standarisasi dengan kode dibawah:
  ```python
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train) 
  X_test_scaled = scaler.transform(X_test) 
  ```

## Modeling
Pada tahap ini, saya membangun 3 model machine learning yaitu:

### 1. Logistic Regression
Logistic Regression digunakan sebagai baseline model untuk klasifikasi biner karena kesederhanaannya dan interpretabilitasnya yang tinggi. Model ini tidak memerlukan banyak tuning dan bekerja dengan baik jika hubungan antar fitur bersifat linier terhadap logit dari target.

- Parameter yang digunakan: default (solver='lbfgs', max_iter=1000 jika perlu).
- Kelebihan: sederhana, cepat, dan hasilnya mudah diinterpretasikan.
- Kekurangan: performa menurun jika data tidak linier atau banyak outlier.

Membangun model Logistic Regression dengan kode dibawah:
```phyton
# Membangun Model Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
```

### 2. Random Forest

### 3. Support Vector Machine (SVM)
SVM bekerja dengan memaksimalkan margin antara kelas dalam ruang fitur, cocok digunakan untuk data dengan dimensi yang tidak terlalu tinggi namun tidak linier.

- Parameter yang digunakan: default (kernel='rbf').
- Kelebihan: bekerja dengan baik pada dataset kecil-menengah dan robust terhadap outlier.
- Kekurangan: waktu pelatihan lama jika data besar; sulit diinterpretasikan.

Membangun model Logistic Regression dengan kode dibawah:
```phyton
svm = SVC()
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
```
  
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
- [WHO] (https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))
- [Scholar] (https://ejournal.upnvj.ac.id/informatik/article/view/4694/1852)
