import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from lifelines import KaplanMeierFitter
import plotly.graph_objects as go

from utils.data_processing import load_data, process_data
from utils.modeling import (
    perform_kaplan_meier_analysis,
    create_monthly_distribution_chart,
    display_auc_metrics,
    create_survival_curve_chart,
    create_grouped_survival_curves,
    create_auc_analysis_table
)

st.set_page_config(
    page_title="📊 수업 잔존기간 통합 분석 도구",
    page_icon="🏠",
    layout="wide"
)

st.subheader("1️⃣ 데이터 업로드 및 현재 생존분석")

# 파일 업로드
uploaded_file = st.file_uploader(
    "CSV 파일을 선택하세요",
    type=['csv']
)

if uploaded_file is not None:
    df, CURRNET_DATE, CUTOFF_DATE = load_data(uploaded_file)
    st.info(f"현재 시점 {CURRNET_DATE.strftime('%Y-%m-%d')}")

    # 날짜 범위 선택
    st.write("")
    st.subheader("📅 분석 대상 기간 설정")
    col1, col2 = st.columns(2)
    with col1:
        START_DATE = st.date_input(
            "시작 날짜",
            value=datetime(2023, 5, 1),
            format="YYYY-MM-DD"
        )
    with col2:
        END_DATE = st.date_input(
            "종료 날짜",
            value=datetime(2024, 4, 30),
            format="YYYY-MM-DD"
        )

    # 데이터 처리 실행
    processed_df = process_data(df, START_DATE, END_DATE, CUTOFF_DATE)
    st.write("")
 
    # 생존 분석 수행
    survival_df, auc_value = perform_kaplan_meier_analysis(processed_df)

    # 월별 분포 차트
    create_monthly_distribution_chart(processed_df)

    # AUC 메트릭 표시
    display_auc_metrics(survival_df, auc_value)

    # 생존곡선 차트
    create_survival_curve_chart(survival_df)

    # 결제기간별 생존곡선 비교
    create_grouped_survival_curves(processed_df)

    # AUC 분석 결과 표
    create_auc_analysis_table(processed_df)
    
    st.subheader("2️⃣ AUC 개선 목표 설정")