# Breast Cancer Tumor Classification using PySpark
Project ini berisi implementasi pemodelan Machine Learning menggunakan **PySpark** untuk mengklasifikasikan diagnosis tumor payudara (Ganas / *Malignant* vs Jinak / *Benign*) berdasarkan data fitur fisik tumor.

## Ringkasan Proyek

Pemodelan regresi logistik ini dirancang untuk menangani dataset kesehatan menggunakan pemrosesan data terdistribusi via Apache Spark (PySpark). Variabel target `diagnosis` dikonversi menjadi data biner:
* **1**: Malignant (Ganas)
* **0**: Benign (Jinak)

### Fitur Prediktor yang Digunakan:
- **Mean Features:** `radius_mean`, `texture_mean`, `area_mean`, `perimeter_mean`, `compactness_mean`
- **Standard Error (SE) Features:** `radius_se`, `area_se`, `perimeter_se`
- **Worst Features:** `radius_worst`, `texture_worst`, `area_worst`, `concavity_worst`, `symmetry_worst`

---

## Langkah-Langkah Pengerjaan

1. **Inisialisasi SparkSession:** Membangun lingkungan PySpark lokal.
2. **Data Ingestion & Preprocessing:** Membaca data `cancer_tumor.csv` dan mengonversi variabel respon `diagnosis` dari format kategori (`M`/`B`) menjadi nilai numerik (`1`/`0`).
3. **Feature Engineering:** Membentuk vektor fitur tunggal `Predictors` menggunakan `VectorAssembler`.
4. **Data Splitting:** Membagi data secara acak menjadi:
   - **80% Data Training**
   - **20% Data Testing**
5. **Model Training:** Melakukan estimasi model regresi logistik menggunakan `LogisticRegression` dengan `maxIter=1000`.
6. **Evaluasi Performa:** Evaluasi model berbasis nilai **Area Under ROC (AUC)** untuk data *training* dan *testing*.

---

## Hasil Evaluasi Model

| Dataset | Evaluasi Metric (AUC) |
| :--- | :--- |
| **Data Training** | **~0.9960** |
| **Data Testing** | **~0.9915** |

> **Kesimpulan:** Nilai AUC yang mendekati 1.0 pada data testing menunjukkan bahwa model regresi logistik PySpark ini memiliki kemampuan pemisahan/klasifikasi yang sangat baik dan konsisten terhadap data baru (tidak mengalami *overfitting*).
