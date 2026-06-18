# Deteksi Kemiripan Wajah dengan PCA/SVD (Eigenfaces)

Aplikasi web sederhana untuk mendeteksi kemiripan dua wajah menggunakan pendekatan
**PCA (Principal Component Analysis)** yang dihitung melalui **SVD (Singular Value Decomposition)**,
atau yang dikenal sebagai metode **Eigenfaces**.

## 🔍 Alur Sistem

1. **Upload** dua gambar wajah bebas.
2. **Face Detection** — wajah dideteksi & di-crop otomatis menggunakan Haar Cascade (OpenCV).
3. **Preprocessing** — gambar diubah ke grayscale, di-resize ke 100×100, dinormalisasi (0–1), lalu di-flatten menjadi vektor 10.000 elemen.
4. **Pembentukan Matriks Data X** — kedua vektor wajah disusun menjadi matriks `X`.
5. **Centering Data** — data dikurangi dengan rata-ratanya (mean face).
6. **PCA berbasis SVD** — `X_c = U Σ V^T`, lalu data diproyeksikan ke ruang PCA berdimensi rendah: `Z = X_c V_k`.
7. **Cosine Similarity** — kemiripan dihitung dari sudut antara dua vektor hasil proyeksi (`z1`, `z2`).
8. **Keputusan** — jika `similarity >= threshold` → **Mirip**, jika tidak → **Tidak Mirip**.

## 🛠️ Teknologi

- [Streamlit](https://streamlit.io/) — antarmuka web
- [OpenCV](https://opencv.org/) — face detection (Haar Cascade) & image processing
- [NumPy](https://numpy.org/) — operasi matriks/vektor
- [Scikit-Learn](https://scikit-learn.org/) — PCA & cosine similarity

## 🚀 Cara Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Demo

Link Streamlit: *(isi setelah deploy)*

## 📄 Referensi Konsep

Implementasi ini mengikuti konsep matematis PCA/SVD untuk deteksi kemiripan wajah:
data wajah direpresentasikan sebagai vektor berdimensi tinggi, direduksi dimensinya
melalui PCA (SVD), kemudian dibandingkan menggunakan cosine similarity di ruang
berdimensi rendah tersebut.
