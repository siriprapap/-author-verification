import streamlit as st
from sentence_transformers import SentenceTransformer, util

# โหลดโมเดล
model = SentenceTransformer("all-MiniLM-L6-v2")

# เพิ่มฟอนต์สวยๆ ด้วย Google Fonts
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ส่วนของแอป
st.title("ระบบตรวจสอบผู้เขียนอัตโนมัติ (Author Verification)")

doc1 = st.text_area("เอกสารที่ 1", height=150)
doc2 = st.text_area("เอกสารที่ 2", height=150)

if st.button("ตรวจสอบ"):
    with st.spinner("กำลังประมวลผล..."):
        emb1 = model.encode(doc1, convert_to_tensor=True)
        emb2 = model.encode(doc2, convert_to_tensor=True)

        similarity = util.cos_sim(emb1, emb2).item()
        result = "น่าจะเป็นผู้เขียนคนเดียวกัน" if similarity > 0.7 else "อาจเป็นคนละคนเขียน"
        st.success(f"{result} (ค่าความคล้าย: {similarity:.2f})")
