# Laporan Proyek Machine Learning - Faiza Adinda Fakhira Batubara

## Domain Proyek

Penyakit jantung merupakan salah satu penyebab kematian tertinggi secara global. Masalah pada jantung dapat diklasifikasikan menjadi dua jenis utama, yaitu penyakit jantung dan serangan jantung. Menurut WHO, pada tahun 2019 terdapat 17.9 juta orang meninggal akibat penyakit jantung, mencakup 32% dari seluruh kematian global. Dari jumlah angka kematian tersebut, 85% disebabkan oleh serangan jantung dan stroke [1].

Di era perkembangan teknologi yang sangat pesat saat ini, teknologi dapat membantu kita untuk mencegah terkena penyakit jantung serta mengurangi angka kematian akibat penyakit jantung. Dengan menggunakan Machine Learning kita dapat memproses analisa data medis dengan menggunakan berbagai metode yang tersedia untuk menemukan pola atau informasi yang penting [2]. Dengan memanfaatkan data klinis pasien seperti tekanan darah, kadar kolesterol, usia, jenis kelamin, dan gaya hidup, kita dapat membangun model prediktif yang membantu tenaga medis dalam mengidentifikasi individu dengan risiko tinggi.

## Business Understanding

### Problem Statements

- Bagaimana memanfaatkan data klinis pasien untuk mengidentifikasi kemungkinan seseorang menderita penyakit jantung?
- Algoritma machine learning mana yang paling efektif untuk memprediksi risiko penyakit jantung berdasarkan dataset yang tersedia?
- Bagaimana mengevaluasi performa model prediksi agar hasilnya akurat dan dapat diandalkan untuk mendukung pengambilan keputusan medis?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Menggunakan fitur-fitur seperti usia, tekanan darah, kadar kolesterol, dll, untuk membangun model prediktif yang dapat mengklasifikasikan pasien berisiko tinggi
- Menerapkan dan membandingkan beberapa algoritma klasifikasi seperti Logistic Regression, Random Forest, dan Support Vector Machine untuk memilih model dengan performa terbaik.
- Melakukan evaluasi terhadap masing-masing model menggunakan metrik akurasi, precision, recall, dan F1 score untuk menilai efektivitas model dalam mendeteksi risiko penyakit jantung.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah _Heart Disease Dataset_ yang tersedia secara publik di Kaggle. Dataset ini dapat diakses melalui tautan berikut: [Heart-Disease-Dataset](https://www.kaggle.com/datasets/mexwell/heart-disease-dataset)

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
``` phyton
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

Terlihat bahwa pada keempat fitur tersebut terdeteksi ada outlier. Outlier akan ditangani pada tahap **Data Preparation**

### Visualisasi Data
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

- Parameter yang digunakan: default (```solver='lbfgs'```, ```max_iter=1000```).
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
Random Forest Classifier digunakan karena cocok untuk kasus klasifikasi seperti prediksi penyakit jantung. Model ini membangun banyak decision tree dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting.
- Parameter yang digunakan: default (```n_estimators=100```, ```random_state=42```).
- Kelebihan: kuat terhadap overfitting, mampu menangani fitur non-linear, dan bekerja baik tanpa perlu banyak preprocessing.
- Kekurangan: interpretasi model lebih sulit dibanding logistic regression, dan membutuhkan lebih banyak sumber daya komputasi dibanding model yang lebih sederhana.

### 3. Support Vector Machine (SVM)
SVM bekerja dengan memaksimalkan margin antara kelas dalam ruang fitur, cocok digunakan untuk data dengan dimensi yang tidak terlalu tinggi namun tidak linier.

- Parameter yang digunakan: default (```kernel='rbf'```).
- Kelebihan: bekerja dengan baik pada dataset kecil-menengah dan robust terhadap outlier.
- Kekurangan: waktu pelatihan lama jika data besar; sulit diinterpretasikan.

  Membangun model Logistic Regression dengan kode dibawah:
```phyton
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

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
Pada tahap evaluasi, digunakan beberapa metrik evaluasi untuk mengukur performa model klasifikasi yang telah dibangun, yaitu **akurasi, precision, recall, dan F1-score**. Metrik-metrik ini dipilih karena sesuai dengan permasalahan klasifikasi yang sedang diselesaikan.

- Akurasi untuk mengukur proporsi prediksi yang benar terhadap seluruh data.

Formula:

<p align="center">
  <img src="https://github.com/user-attachments/assets/dc5f13d2-8f08-4671-8ba9-4e6e2b30aa69" alt="Formula Akurasi" width="400">
</p>

- Precision mengukur seberapa tepat model dalam memprediksi kelas positif (berapa banyak dari prediksi positif yang benar).

Formula:

<p align="center">
  <img src="https://github.com/user-attachments/assets/d16e2027-7722-4b4c-9d30-a9cbff2de6f6" alt="Formula Precision" width="400">
</p>

- Recall mengukur seberapa baik model dalam menemukan semua kasus positif yang sebenarnya.

Formula:

<p align="center">
  <img src="https://github.com/user-attachments/assets/80cc663d-b807-4cb7-a3d2-4271d854d8f8" alt="Formula Recall" width="400">
</p>

- F1 Score merupakan gabungan antara precision dan recall. Metrik ini berguna saat dibutuhkan keseimbangan antara keduanya.

Formula:

<p align="center">
  <img src="https://github.com/user-attachments/assets/007c3a65-1de9-4fd8-a672-35c3c90ae4ca" alt="Formula F1 Score" width="400">
</p>

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Hasil Evaluasi Model
Tiga model yang diuji adalah Logistic Regression, Support Vector Machine (SVM), dan Random Forest Classifier. Berikut hasil evaluasi berdasarkan metrik:
```
Logistic Regression:
Accuracy: 0.85
Precision: 0.86
Recall: 0.85
F1 Score: 0.86

SVM:
Accuracy: 0.71
Precision: 0.78
Recall: 0.62
F1 Score: 0.69

Random Forest:
Accuracy: 0.89
Precision: 0.89
Recall: 0.91
F1 Score: 0.9
```
Berdasarkan hasil evaluasi, model **Random Forest Classifier** memberikan performa terbaik di antara ketiga model yang diuji, dengan nilai akurasi, precision, recall, dan F1-score tertinggi. Oleh karena itu, model ini dipilih sebagai model terbaik untuk menyelesaikan permasalahan klasifikasi pada proyek ini.

## Referensi
[1] World Health Organization, “Cardiovascular diseases (CVDs),” World Health Organization, Jun. 11, 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
[2] F. M. Adibah, S. Afifah, and R. Rachmawati, “Penerapan Algoritma Klasifikasi Untuk Prediksi Penyakit Jantung Menggunakan Metode K-NN,” Jurnal Informatika, vol. 12, no. 1, pp. 52–58, Mar. 2021. [Online]. Available: https://ejournal.upnvj.ac.id/informatik/article/view/4694/1852
