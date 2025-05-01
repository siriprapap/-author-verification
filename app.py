import streamlit as st
from sentence_transformers import SentenceTransformer, util

# โหลดโมเดล
model = SentenceTransformer("all-MiniLM-L6-v2")

# ปรับแต่งฟอนต์และรูปแบบให้สวยงาม
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600&family=Kanit:wght@500;600&display=swap');
    
    /* ฟอนต์หลัก */
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
        font-size: 16px;
        color: #333333;
    }
    
    /* หัวข้อหลัก */
    h1 {
        font-family: 'Kanit', sans-serif;
        font-weight: 600;
        font-size: 28px;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    
    /* หัวข้อย่อย */
    h2 {
        font-family: 'Kanit', sans-serif;
        font-weight: 500;
        font-size: 22px;
        color: #2c3e50;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    /* ช่องป้อนข้อความ */
    .stTextArea>textarea {
        font-family: 'Sarabun', sans-serif !important;
        font-size: 16px !important;
        border: 1px solid #dfe6e9 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        min-height: 200px;
    }
    
    /* ปุ่ม */
    .stButton>button {
        font-family: 'Kanit', sans-serif !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        background-color: #3498db !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ส่วนหัวแอปพลิเคชัน
st.title("ระบบตรวจสอบผู้เขียนอัตโนมัติ")
st.markdown("วิเคราะห์ความคล้ายคลึงของลายลักษณ์อักษรด้วยเทคโนโลยี AI")

# ส่วนป้อนข้อมูล
st.markdown("## ป้อนข้อความเพื่อวิเคราะห์")

doc1 = st.text_area(
    "**เอกสารที่ 1**",
    height=200,
    placeholder="เช่น: บทความ งานเขียน หรือข้อความตัวอย่าง..."
)

doc2 = st.text_area(
    "**เอกสารที่ 2**",
    height=200,
    placeholder="เช่น: บทความ งานเขียน หรือข้อความตัวอย่าง..."
)

# ปุ่มตรวจสอบ
if st.button("ตรวจสอบความคล้ายคลึง"):
    if doc1 and doc2:
        with st.spinner("กำลังวิเคราะห์ความคล้ายคลึง..."):
            # สร้าง Embedding
            emb1 = model.encode(doc1, convert_to_tensor=True)
            emb2 = model.encode(doc2, convert_to_tensor=True)
            
            # คำนวณความคล้ายคลึง
            similarity = util.cos_sim(emb1, emb2).item()
            
            # แสดงผลลัพธ์
            st.markdown("---")
            st.markdown("## ผลการวิเคราะห์")
            
            if similarity > 0.75:
                st.success(f"**ผลลัพธ์:** น่าจะเป็นผู้เขียนคนเดียวกัน (ความคล้ายคลึง: {similarity:.2%})")
            elif similarity > 0.5:
                st.warning(f"**ผลลัพธ์:** อาจมีลักษณะการเขียนคล้ายกัน (ความคล้ายคลึง: {similarity:.2%})")
            else:
                st.error(f"**ผลลัพธ์:** น่าจะเป็นคนละคนเขียน (ความคล้ายคลึง: {similarity:.2%})")
    else:
        st.warning("กรุณาป้อนข้อความทั้งสองช่องก่อนทำการตรวจสอบ")
