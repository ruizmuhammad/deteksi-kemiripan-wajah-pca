import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ukuran standar gambar wajah
IMG_SIZE = (100, 100)

st.set_page_config(
    page_title="Deteksi Kemiripan Wajah - PCA/SVD",
    page_icon="🧑‍🤝‍🧑",
    layout="wide",
)


@st.cache_resource
def load_face_cascade():
    # load haar cascade buat deteksi wajah
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    # konversi gambar dari uploader streamlit ke format opencv
    rgb_image = np.array(pil_image.convert("RGB"))
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def detect_and_crop_face(img_bgr: np.ndarray, face_cascade, margin: float = 0.3):
    """
    Deteksi wajah dengan Haar Cascade, lalu crop dengan margin proporsional.

    Catatan: Haar Cascade kadang menghasilkan beberapa kandidat bbox untuk satu
    wajah yang sama -- salah satunya bisa jadi area "kepala + leher/bahu" yang
    kebesaran (bukan wajah murni). Kalau langsung ambil bbox dengan area terbesar,
    itu bisa salah pilih kandidat yang kebesaran ini, sehingga rasio crop antar
    foto jadi tidak konsisten dan PCA jadi menangkap "seberapa ketat crop-nya"
    bukan fitur wajah itu sendiri. Untuk mengurangi risiko ini, kandidat bbox
    difilter dulu berdasarkan rasio ukuran yang wajar untuk wajah (20-45% dari
    lebar gambar); kalau tidak ada yang masuk rentang itu, fallback ke kandidat
    terkecil (lebih aman daripada kandidat yang kebesaran).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    min_dim = min(h_img, w_img)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(int(min_dim * 0.15), int(min_dim * 0.15)),
    )

    if len(faces) == 0:
        return None, gray, None

    target_ratio_range = (0.20, 0.45)
    candidates_in_range = [f for f in faces if target_ratio_range[0] <= f[2] / w_img <= target_ratio_range[1]]

    if candidates_in_range:
        x, y, w, h = max(candidates_in_range, key=lambda f: f[2] * f[3])
    else:
        # tidak ada kandidat dengan rasio wajar -> ambil yang terkecil sebagai fallback
        x, y, w, h = min(faces, key=lambda f: f[2] * f[3])

    mx, my = int(w * margin), int(h * margin)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(w_img, x + w + mx)
    y1 = min(h_img, y + h + my)

    face_crop = gray[y0:y1, x0:x1]

    return face_crop, gray, (x0, y0, x1 - x0, y1 - y0)


def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
    # resize, equalisasi histogram (normalisasi pencahayaan), normalisasi, lalu flatten
    resized = cv2.resize(face_gray, IMG_SIZE)
    equalized = cv2.equalizeHist(resized)
    normalized = equalized / 255.0
    vector = normalized.flatten()
    return vector, equalized


def process_uploaded_image(uploaded_file, face_cascade):
    pil_image = Image.open(uploaded_file)
    img_bgr = pil_to_cv2(pil_image)

    face_crop, full_gray, bbox = detect_and_crop_face(img_bgr, face_cascade)

    face_detected = face_crop is not None
    if not face_detected:
        # kalau wajah ga ketemu, pakai gambar penuh sebagai fallback
        face_crop = full_gray

    vector, resized_face = preprocess_face(face_crop)

    return {
        "original_pil": pil_image,
        "vector": vector,
        "resized_face": resized_face,
        "face_detected": face_detected,
        "bbox": bbox,
    }


def augment_face(face_img_100x100: np.ndarray, rng: np.random.Generator, n_variations: int = 50) -> list:
    """
    Bikin variasi (augmentasi) dari SATU gambar wajah yang sudah di-resize 100x100.
    Setiap variasi tetap wajah yang SAMA secara struktur (mata/hidung/mulut di posisi
    yang konsisten secara relatif), hanya beda pose ringan (rotasi, scaling, translasi
    kecil), pencahayaan, flip, dan noise halus.

    Ini menggantikan pendekatan sebelumnya yang memakai noise acak murni + blur sebagai
    dataset training. Noise acak tidak punya struktur wajah sama sekali, sehingga arah
    eigenvector yang dihasilkan PCA tidak ada hubungannya dengan fitur wajah manusia --
    similarity yang dihasilkan jadi tergantung seed acak (terbukti goyang antara positif
    dan negatif kalau seed diganti). Dengan augmentasi dari wajah ASLI yang diupload,
    struktur wajah (posisi mata, hidung, kontur) tetap konsisten di semua sampel training,
    sehingga PCA belajar arah-arah variasi yang relevan secara visual -- sesuai prinsip
    dasar Eigenfaces (Turk & Pentland).
    """
    variations = []
    h, w = face_img_100x100.shape
    center = (w // 2, h // 2)

    for _ in range(n_variations):
        angle = rng.uniform(-10, 10)
        scale = rng.uniform(0.95, 1.05)
        tx = rng.uniform(-3, 3)
        ty = rng.uniform(-3, 3)
        brightness = rng.uniform(-15, 15)
        flip = rng.random() < 0.5

        rot_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rot_matrix[0, 2] += tx
        rot_matrix[1, 2] += ty
        transformed = cv2.warpAffine(
            face_img_100x100, rot_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
        )

        bright = np.clip(transformed.astype(np.float64) + brightness, 0, 255).astype(np.uint8)
        img = cv2.flip(bright, 1) if flip else bright

        noise = rng.normal(0, 2, size=img.shape)
        img_noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        variations.append(img_noisy)

    return variations


def build_training_dataset(face1_equalized: np.ndarray, face2_equalized: np.ndarray, n_variations: int = 50):
    """
    Bangun dataset training untuk PCA dari AUGMENTASI dua wajah yang diupload
    (bukan dari noise acak). Karena tidak ada dataset wajah orang lain yang
    tersedia, training set diperluas dengan augmentasi (rotasi ringan, flip,
    perubahan brightness, translasi kecil, noise halus) dari kedua foto asli.
    Ini menjaga struktur wajah tetap konsisten sehingga PCA bisa belajar arah-arah
    variasi yang benar-benar relevan terhadap fitur wajah (eigenfaces yang valid),
    bukan arah dari noise acak yang tidak punya makna visual.
    """
    rng = np.random.default_rng(seed=42)

    aug_1 = augment_face(face1_equalized, rng, n_variations=n_variations)
    aug_2 = augment_face(face2_equalized, rng, n_variations=n_variations)

    all_faces = aug_1 + aug_2
    X = np.array([f.flatten() / 255.0 for f in all_faces])
    return X


def compute_pca_similarity(face1_equalized: np.ndarray, face2_equalized: np.ndarray,
                            vector_1: np.ndarray, vector_2: np.ndarray, n_components: int):
    # bangun dataset training (X) berisi augmentasi dari kedua wajah asli
    X_train = build_training_dataset(face1_equalized, face2_equalized)

    max_components = min(X_train.shape[0], X_train.shape[1])
    n_components = min(n_components, max_components)

    # PCA dilatih dari dataset training (X_c = X - mean, lalu SVD)
    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    # proyeksikan dua wajah ASLI (bukan dataset training) ke ruang PCA
    Z = pca.transform(np.array([vector_1, vector_2]))

    z1 = Z[0].reshape(1, -1)
    z2 = Z[1].reshape(1, -1)

    similarity = cosine_similarity(z1, z2)[0][0]
    euclidean_distance = float(np.linalg.norm(z1 - z2))
    explained_variance = float(np.sum(pca.explained_variance_ratio_))

    return {
        "similarity": float(similarity),
        "distance": euclidean_distance,
        "z1": z1.flatten(),
        "z2": z2.flatten(),
        "explained_variance": explained_variance,
        "n_components_used": n_components,
        "n_training_samples": X_train.shape[0],
    }


st.title("🧑‍🤝‍🧑 Deteksi Kemiripan Wajah dengan PCA/SVD (Eigenfaces)")

st.markdown(
    """
    Aplikasi ini membandingkan dua gambar wajah menggunakan pendekatan
    **PCA (Principal Component Analysis)** berbasis **SVD**.

    **Alur proses:** Upload gambar → Deteksi & crop wajah (Haar Cascade) →
    Preprocessing (grayscale, resize, histogram equalization, normalisasi, flatten) →
    Augmentasi kedua wajah (rotasi, flip, brightness, translasi) untuk membentuk dataset
    training → PCA dilatih dari dataset hasil augmentasi → Proyeksi kedua wajah asli ke
    ruang PCA → Hitung **jarak Euclidean** di ruang PCA (sesuai metode Eigenfaces klasik) →
    Keputusan mirip / tidak mirip.
    """
)

st.divider()

with st.sidebar:
    st.header("⚙️ Pengaturan")
    distance_threshold = st.slider(
        "Threshold Euclidean Distance (ruang PCA)",
        min_value=10.0,
        max_value=60.0,
        value=33.0,
        step=0.5,
        help="Jika distance <= threshold, wajah dianggap mirip. Eigenfaces klasik (Turk & Pentland) menggunakan jarak Euclidean di ruang PCA sebagai metrik kemiripan -- bukan cosine similarity -- karena makna 'jarak antar wajah' lebih langsung dibanding sudut antar vektor pada ruang ini.",
    )
    n_components = st.slider(
        "Jumlah Komponen Utama PCA (k)",
        min_value=2,
        max_value=30,
        value=15,
        help="Jumlah dimensi hasil reduksi PCA. PCA dilatih dari dataset hasil augmentasi, lalu kedua wajah asli diproyeksikan ke ruang ini.",
    )
    st.caption(
        "ℹ️ PCA dilatih dari dataset hasil **augmentasi** kedua wajah yang diupload "
        "(rotasi ringan, flip horizontal, perubahan brightness, translasi kecil, noise "
        "halus) -- karena tidak ada dataset wajah orang lain yang tersedia. Augmentasi "
        "menjaga struktur wajah tetap konsisten, sehingga PCA belajar arah variasi yang "
        "relevan secara visual (bukan dari noise acak yang tidak punya makna wajah)."
    )
    st.caption(
        "⚠️ Karena training set terbatas (hanya dari augmentasi 2 foto, tanpa wajah "
        "orang lain sebagai pembanding), pemisahan identitas belum sepenuhnya akurat -- "
        "terutama untuk pasangan dengan perbedaan usia, pencahayaan, atau kualitas foto "
        "yang besar. Ini adalah limitasi metodologis PCA/Eigenfaces dengan data terbatas, "
        "bukan bug pada implementasi."
    )

face_cascade = load_face_cascade()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Gambar Wajah 1")
    file_1 = st.file_uploader("Upload gambar wajah pertama", type=["jpg", "jpeg", "png"], key="file1")

with col2:
    st.subheader("📷 Gambar Wajah 2")
    file_2 = st.file_uploader("Upload gambar wajah kedua", type=["jpg", "jpeg", "png"], key="file2")

st.divider()

if file_1 and file_2:
    if st.button("🔍 Deteksi Kemiripan Wajah", type="primary", use_container_width=True):
        with st.spinner("Memproses gambar..."):
            result_1 = process_uploaded_image(file_1, face_cascade)
            result_2 = process_uploaded_image(file_2, face_cascade)

        st.subheader("🔬 Hasil Preprocessing")
        prep_col1, prep_col2 = st.columns(2)

        with prep_col1:
            st.image(result_1["original_pil"], caption="Gambar Asli 1", use_container_width=True)
            st.image(
                result_1["resized_face"],
                caption=f"Wajah Setelah Preprocessing (100x100){'  ✅ Wajah terdeteksi' if result_1['face_detected'] else '  ⚠️ Wajah tidak terdeteksi, memakai seluruh gambar'}",
                use_container_width=True,
            )

        with prep_col2:
            st.image(result_2["original_pil"], caption="Gambar Asli 2", use_container_width=True)
            st.image(
                result_2["resized_face"],
                caption=f"Wajah Setelah Preprocessing (100x100){'  ✅ Wajah terdeteksi' if result_2['face_detected'] else '  ⚠️ Wajah tidak terdeteksi, memakai seluruh gambar'}",
                use_container_width=True,
            )

        pca_result = compute_pca_similarity(
            result_1["resized_face"], result_2["resized_face"],
            result_1["vector"], result_2["vector"],
            n_components=n_components,
        )

        similarity = pca_result["similarity"]
        distance = pca_result["distance"]
        is_similar = distance <= distance_threshold

        st.divider()
        st.subheader("📊 Hasil Analisis PCA")

        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("Dimensi Asli (per gambar)", f"{result_1['vector'].shape[0]} fitur")
        with info_col2:
            st.metric("Dimensi Setelah PCA (k)", f"{pca_result['n_components_used']} fitur")
        with info_col3:
            st.metric("Explained Variance", f"{pca_result['explained_variance']*100:.2f}%")
        with info_col4:
            st.metric("Sampel Training (augmentasi)", f"{pca_result['n_training_samples']}")

        st.divider()
        st.subheader("🎯 Hasil Perbandingan")

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Euclidean Distance (ruang PCA)", f"{distance:.4f}")
            st.caption(f"Threshold: {distance_threshold} (semakin kecil = semakin mirip)")
        with result_col2:
            st.metric("Cosine Similarity (info tambahan)", f"{similarity:.4f}")

        if is_similar:
            st.success(
                f"✅ **WAJAH MIRIP**\n\n"
                f"Distance ({distance:.4f}) <= Threshold ({distance_threshold}) "
                f"→ kedua wajah dianggap **mirip**."
            )
        else:
            st.error(
                f"❌ **WAJAH TIDAK MIRIP**\n\n"
                f"Distance ({distance:.4f}) > Threshold ({distance_threshold}) "
                f"→ kedua wajah dianggap **tidak mirip**."
            )

        with st.expander("📐 Lihat detail representasi vektor wajah di ruang PCA (z1, z2)"):
            st.write("**z1 (representasi wajah 1):**")
            st.code(np.round(pca_result["z1"], 4))
            st.write("**z2 (representasi wajah 2):**")
            st.code(np.round(pca_result["z2"], 4))

else:
    st.info("⬆️ Silakan upload dua gambar wajah terlebih dahulu untuk memulai analisis.")

st.divider()
st.caption(
    "Dibangun dengan Streamlit, OpenCV, NumPy, dan Scikit-Learn — "
    "implementasi konsep PCA/SVD (Eigenfaces) untuk deteksi kemiripan wajah."
)
