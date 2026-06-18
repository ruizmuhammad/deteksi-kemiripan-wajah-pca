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


def detect_and_crop_face(img_bgr: np.ndarray, face_cascade):
    # deteksi wajah, kalau ketemu langsung crop area wajahnya
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
    )

    if len(faces) == 0:
        return None, gray, None

    # ambil wajah dengan area paling besar
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = gray[y:y + h, x:x + w]

    return face_crop, gray, (x, y, w, h)


def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
    # resize, normalisasi, lalu flatten jadi vektor
    resized = cv2.resize(face_gray, IMG_SIZE)
    normalized = resized / 255.0
    vector = normalized.flatten()
    return vector, resized


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


def compute_pca_similarity(vector_1: np.ndarray, vector_2: np.ndarray, n_components: int):
    # bentuk matriks X dari 2 vektor wajah
    X = np.array([vector_1, vector_2])

    # batasi n_components biar ga lebih dari jumlah sampel/fitur
    max_components = min(X.shape[0], X.shape[1])
    n_components = min(n_components, max_components)

    # PCA disini udah otomatis centering data + hitung lewat SVD
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)

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
    }


st.title("🧑‍🤝‍🧑 Deteksi Kemiripan Wajah dengan PCA/SVD (Eigenfaces)")

st.markdown(
    """
    Aplikasi ini membandingkan dua gambar wajah menggunakan pendekatan
    **PCA (Principal Component Analysis)** berbasis **SVD**.

    **Alur proses:** Upload gambar → Deteksi & crop wajah (Haar Cascade) →
    Preprocessing (grayscale, resize, normalisasi, flatten) → PCA →
    Proyeksi ke ruang PCA → Hitung **cosine similarity** → Keputusan mirip / tidak mirip.
    """
)

st.divider()

with st.sidebar:
    st.header("⚙️ Pengaturan")
    threshold = st.slider(
        "Threshold Cosine Similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.80,
        step=0.01,
        help="Jika similarity >= threshold, wajah dianggap mirip (sesuai contoh pada dokumen, default 0.80).",
    )
    n_components = st.slider(
        "Jumlah Komponen Utama PCA (k)",
        min_value=1,
        max_value=2,
        value=1,
        help="Pada kasus 2 gambar, jumlah komponen maksimum dibatasi oleh jumlah sampel (2).",
    )
    st.caption(
        "ℹ️ Karena hanya membandingkan 2 gambar, jumlah komponen PCA (k) maksimum "
        "yang bisa dihitung secara matematis adalah 2 (mengikuti batas min(jumlah sampel, jumlah fitur))."
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
            result_1["vector"], result_2["vector"], n_components=n_components
        )

        similarity = pca_result["similarity"]
        distance = pca_result["distance"]
        is_similar = similarity >= threshold

        st.divider()
        st.subheader("📊 Hasil Analisis PCA")

        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Dimensi Asli (per gambar)", f"{result_1['vector'].shape[0]} fitur")
        with info_col2:
            st.metric("Dimensi Setelah PCA (k)", f"{pca_result['n_components_used']} fitur")
        with info_col3:
            st.metric("Explained Variance", f"{pca_result['explained_variance']*100:.2f}%")

        st.divider()
        st.subheader("🎯 Hasil Perbandingan")

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Cosine Similarity", f"{similarity:.4f}")
            st.caption(f"Threshold: {threshold}")
        with result_col2:
            st.metric("Euclidean Distance (ruang PCA)", f"{distance:.4f}")

        if is_similar:
            st.success(
                f"✅ **WAJAH MIRIP**\n\n"
                f"Similarity ({similarity:.4f}) >= Threshold ({threshold}) "
                f"→ kedua wajah dianggap **mirip**."
            )
        else:
            st.error(
                f"❌ **WAJAH TIDAK MIRIP**\n\n"
                f"Similarity ({similarity:.4f}) < Threshold ({threshold}) "
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
