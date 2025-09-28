import streamlit as st
import pandas as pd

# Header
st.title("📊 수업 잔존기간 통합 분석 도구")

st.subheader("1️⃣ 데이터 업로드 및 현재 생존분석")

# 파일 업로드
uploaded_file = st.file_uploader(
    "CSV 파일을 선택하세요",
    type=['csv']
)

if uploaded_file is not None:
    # 파일 읽기
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ 파일 업로드 완료! ({len(df):,}행)")
    st.dataframe(df.head())