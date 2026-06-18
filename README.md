# Deteksi Kemiripan Wajah dengan PCA/SVD (Eigenfaces)

Aplikasi web untuk mendeteksi kemiripan dua wajah menggunakan pendekatan
**PCA (Principal Component Analysis)** yang dihitung melalui **SVD (Singular Value Decomposition)**,
atau yang dikenal sebagai metode **Eigenfaces**.

## 🔍 Alur Sistem

1. **Upload** dua gambar wajah bebas.
2. **Center Crop** — bagian tengah gambar di-crop (80%) untuk membuang background dan rambut pinggir sebelum deteksi.
3. **Face Detection** — wajah dideteksi & di-crop otomatis menggunakan Haar Cascade (OpenCV).
4. **Preprocessing** — gambar diubah ke grayscale, di-resize ke 100×100, histogram equalization, dinormalisasi (0–1), lalu di-flatten menjadi vektor 10.000 elemen.
5. **Training PCA dari Dataset LFW** — PCA dilatih dari dataset **Labeled Faces in the Wild (LFW)**, ribuan foto wajah nyata dari berbagai orang, sehingga eigenfaces yang dihasilkan merepresentasikan struktur wajah manusia secara umum.
6. **Centering Data** — data dikurangi dengan rata-ratanya (mean face), dikerjakan otomatis oleh PCA.
7. **PCA berbasis SVD** — `X_c = U Σ V^T`, lalu kedua wajah yang diupload diproyeksikan ke ruang PCA berdimensi rendah: `Z = X_c V_k`.
8. **Euclidean Distance** — kemiripan dihitung dari jarak Euclidean antara dua vektor hasil proyeksi (`z1`, `z2`) di ruang PCA, sesuai metode Eigenfaces klasik (Turk & Pentland).
9. **Keputusan** — jika `distance <= threshold` → **Mirip**, jika tidak → **Tidak Mirip**.

## 🛠️ Teknologi

- [Streamlit](https://streamlit.io/) — antarmuka web
- [OpenCV](https://opencv.org/) — face detection (Haar Cascade) & image processing
- [NumPy](https://numpy.org/) — operasi matriks/vektor
- [Scikit-Learn](https://scikit-learn.org/) — PCA, dataset LFW, & Euclidean distance

## 🚀 Cara Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

> ⚠️ Pertama kali dijalankan, aplikasi akan mengunduh dataset LFW (~200MB) secara otomatis.
> Proses ini hanya terjadi sekali; selanjutnya data di-cache oleh Streamlit.

## 🌐 Demo

Link Streamlit: *(isi setelah deploy)*

## 📄 Referensi Konsep

Implementasi ini mengikuti konsep matematis PCA/SVD untuk deteksi kemiripan wajah
(Eigenfaces — Turk & Pentland, 1991): data wajah direpresentasikan sebagai vektor
berdimensi tinggi, direduksi dimensinya melalui PCA (SVD) yang dilatih dari dataset
LFW, kemudian dibandingkan menggunakan jarak Euclidean di ruang berdimensi rendah tersebut.
