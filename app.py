import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# โหลด tokenizer และโมเดล
tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("all-MiniLM-L6-v2")

st.title("ระบบตรวจสอบผู้เขียนอัตโนมัติ (Author Verification)")

doc1 = st.text_area("เอกสารที่ 1", height=150)
doc2 = st.text_area("เอกสารที่ 2", height=150)

if st.button("ตรวจสอบ"):
    with st.spinner("กำลังประมวลผล..."):
        inputs = tokenizer(doc1 + " [SEP] " + doc2, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        score = probs[0][1].item()
        st.success(f"ความน่าจะเป็นว่าเป็นผู้เขียนคนเดียวกัน: {score:.2f}")
