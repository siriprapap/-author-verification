import streamlit as st
from sentence_transformers import SentenceTransformer, util

# โหลดโมเดล
model = SentenceTransformer("all-MiniLM-L6-v2")

# ปรับแต่งฟอนต์ภาษาไทยให้สวยงาม
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&family=Kanit:wght@400;600&display=swap');
    
    /* ฟอนต์หลักสำหรับเนื้อหา */
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: #333333;
    }
    
    /* หัวข้อหลัก */
    h1 {
        font-family: 'Kanit', sans-serif;
        font-weight: 600;
        font-size: 28px;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    
    /* หัวข้อย่อย */
    h2, h3, h4, h5, h6 {
        font-family: 'Kanit', sans-serif;
        font-weight: 500;
        color: #2c3e50;
    }
    
    /* ปุ่ม */
    .stButton>button {
        font-family: 'Kanit', sans-serif !important;
        font-weight: 500 !important;
        background-color: #3498db !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-size: 16px !important;
        transition: all 0.3s !important;
    }
    
    .stButton>button:hover {
        background-color: #2980b9 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* ช่องป้อนข้อความ */
    .stTextArea>textarea {
        font-family: 'Sarabun', sans-serif !important;
        font-size: 16px !important;
        border: 1px solid #dfe6e9 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* การ์ดผลลัพธ์ */
    .stAlert {
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    
    /* ส่วนหัว */
    .css-1v0mbdj {
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ส่วนหัวแอปพลิเคชัน
st.title("🖋️ ระบบตรวจสอบผู้เขียนอัตโนมัติ")
st.markdown("วิเคราะห์ความคล้ายคลึงของลายลักษณ์อักษรด้วยเทคโนโลยี AI")

# ส่วนป้อนข้อมูล
st.subheader("📝 ป้อนข้อความเพื่อวิเคราะห์")
col1, col2 = st.columns(2)
with col1:
    doc1 = st.text_area("เอกสารที่ 1", height=200, placeholder="เช่น: บทความ งานเขียน หรือข้อความตัวอย่าง...")

with col2:
    doc2 = st.text_area("เอกสารที่ 2", height=200, placeholder="เช่น: บทความ งานเขียน หรือข้อความตัวอย่าง...")

# ปุ่มตรวจสอบ
if st.button("🔍 ตรวจสอบความคล้ายคลึง", type="primary"):
    if doc1 and doc2:
        with st.spinner("🔄 กำลังวิเคราะห์ความคล้ายคลึง..."):
            # สร้าง Embedding
            emb1 = model.encode(doc1, convert_to_tensor=True)
            emb2 = model.encode(doc2, convert_to_tensor=True)
            
            # คำนวณความคล้ายคลึง
            similarity = util.cos_sim(emb1, emb2).item()
            
            # แสดงผลลัพธ์
            st.markdown("---")
            st.subheader("📊 ผลการวิเคราะห์")
            
            if similarity > 0.75:
                st.success(f"""
                ### ผลลัพธ์: ✅ น่าจะเป็นผู้เขียนคนเดียวกัน
                **ระดับความคล้ายคลึง:** {similarity:.2%}  
                **การตีความ:** ข้อความทั้งสองมีลักษณะการเขียนที่คล้ายกันมาก
                """)
            elif similarity > 0.5:
                st.warning(f"""
                ### ผลลัพธ์: ⚠️ อาจมีลักษณะการเขียนคล้ายกัน
                **ระดับความคล้ายคลึง:** {similarity:.2%}  
                **การตีความ:** ข้อความทั้งสองมีบางส่วนที่คล้ายกัน
                """)
            else:
                st.error(f"""
                ### ผลลัพธ์: ❌ น่าจะเป็นคนละคนเขียน
                **ระดับความคล้ายคลึง:** {similarity:.2%}  
                **การตีความ:** ข้อความทั้งสองมีลักษณะการเขียนแตกต่างกัน
                """)
            
            # แสดงความช่วยเหลือในการตีความ
            with st.expander("ℹ️ คำแนะนำในการตีความผลลัพธ์"):
                st.markdown("""
                - **> 75%**: ความคล้ายคลึงสูงมาก (มีแนวโน้มว่าเป็นผู้เขียนคนเดียวกัน)
                - **50-75%**: ความคล้ายคลึงปานกลาง (อาจมีลักษณะการเขียนคล้ายกันบางส่วน)
                - **< 50%**: ความคล้ายคลึงต่ำ (น่าจะเป็นคนละคนเขียน)
                """)
    else:
        st.warning("⚠️ กรุณาป้อนข้อความทั้งสองช่องก่อนทำการตรวจสอบ")

# ส่วนท้าย
st.markdown("---")
st.markdown("""
<style>
footer {
    font-size: 14px !important;
    color: #777777 !important;
    text-align: center !important;
}
</style>
<div class='footer'>
<p>ระบบนี้ใช้โมเดล all-MiniLM-L6-v2 สำหรับวิเคราะห์ความคล้ายคลึงของข้อความ</p>
</div>
""", unsafe_allow_html=True)
