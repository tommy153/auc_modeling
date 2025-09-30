import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from lifelines import KaplanMeierFitter, CoxPHFitter
from utils.load_googlesheet import *
from utils.visualization import (
    create_monthly_bar_chart,
    create_weekly_bar_chart,
    create_survival_curve,
    create_grouped_survival_curves,
    create_churn_rate_timeline,
    create_survival_comparison_chart,
    calculate_auc,
    calculate_survival_rate_at_time,
    display_auc_improvement_results
)

st.set_page_config(
    page_title="📊 수업 잔존기간 통합 분석 도구",
    page_icon="📊",
    layout="wide"
)
st.subheader("1️⃣ 데이터 업로드 및 현재 생존분석")

df = load_google_sheets_data("이탈_RAW")

with st.status("구글시트 데이터 처리 중..."):
    df_processed = processing_google_sheet(df)
    st.success("처리가 완료되었습니다 ✅")

start_date = df_processed['결제등록일'].min().strftime("%Y-%m-%d")
end_date   = df_processed['결제등록일'].max().strftime("%Y-%m-%d")

st.markdown(f"""
### 📆 분석 기간: **{start_date}** ~ **{end_date}** """)

total_cnt = len(df_processed)  # 전체 수업 수
stop_cnt = (df_processed['이탈여부'] == 1).sum()  # 중단 수업 수
active_cnt = (df_processed['이탈여부'] == 0).sum()  # 활성 수업 수
# dm_cnt = (df_processed['DM보정여부'] == True).sum()  # DM 보정 건수 (컬럼명 확인 필요)

st.write("")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"📄 **총 수업 수**<br><span style='font-size:24px; font-weight:bold;'>{total_cnt:,}개</span>", unsafe_allow_html=True)
with col2:
    st.markdown(f"❌ **중단 수업**<br><span style='font-size:24px; font-weight:bold;'>{stop_cnt:,}개</span>", unsafe_allow_html=True)
with col3:
    st.markdown(f"✅ **활성 수업**<br><span style='font-size:24px; font-weight:bold;'>{active_cnt:,}개</span>", unsafe_allow_html=True)
with col4:
    st.markdown(f"🛠️ **DM 보정**<br><span style='font-size:24px; font-weight:bold;'>43개</span>", unsafe_allow_html=True)

st.write("")

# 시각화: 월별/주별 신규 수업 시작 수
unit = st.radio("단위 선택", ["월별", "주별"], horizontal=True)

if unit == "월별":
    fig_month = create_monthly_bar_chart(df_processed)
    st.plotly_chart(fig_month, use_container_width=True)
elif unit == "주별":
    fig_week = create_weekly_bar_chart(df_processed)
    st.plotly_chart(fig_week, use_container_width=True)
    
st.write("")

# Kaplan-Meier 생존 분석
kmf = KaplanMeierFitter()
kmf.fit(durations=df_processed['duration_days'], event_observed=df_processed['이탈여부'], label="KM Estimate")

# 단위 선택
unit = st.radio("단위 선택", ["개월", "주"], horizontal=True)

# AUC 및 36개월 생존율 계산
auc_months = calculate_auc(kmf, max_months=36)
survival_36m = calculate_survival_rate_at_time(kmf, months=36)

# KPI 카드
col1, col2 = st.columns(2)

with col1:
    st.markdown("📊 **AUC (평균 생존기간)**")
    st.markdown(f"<span style='font-size:24px; font-weight:bold;'>{auc_months:.2f}개월</span>", unsafe_allow_html=True)

with col2:
    st.markdown("☑️ **36개월 생존율**")
    st.markdown(f"<span style='font-size:24px; font-weight:bold;'>{survival_36m*100:.1f}%</span>", unsafe_allow_html=True)

# Kaplan-Meier 생존 곡선
fig = create_survival_curve(df_processed, unit=unit)
st.plotly_chart(fig, use_container_width=True)

st.write("")

# 결제개월수별 Kaplan-Meier 생존 곡선
fig_grouped = create_grouped_survival_curves(df_processed)
st.plotly_chart(fig_grouped, use_container_width=True)

# -----------------------------
# 2️⃣ 그룹별 요약 통계 추출
# -----------------------------
groups = [
    ("전체", None),
    ("1개월 구매", "1"),
    ("3개월 구매", "3"),
    ("6개월 구매", "6"),
    ("12개월 구매", "12")
]

results = []

for group_name, pay_month in groups:
    if pay_month is None:
        data = df_processed
    else:
        data = df_processed[df_processed['결제개월수'] == pay_month]

    if len(data) > 0:
        sample_size = len(data)
        churn_count = data['이탈여부'].sum()   # 1=이탈, 0=생존
        churn_rate = churn_count / sample_size * 100

        kmf = KaplanMeierFitter()
        kmf.fit(data["donemonth"], event_observed=data["이탈여부"].astype(int))

        # 36개월까지만 survival 데이터
        survival_df = kmf.survival_function_.reset_index()
        survival_df.columns = ["개월", "생존확률"]
        survival_df = survival_df[(survival_df["개월"] <= 36) & (survival_df["개월"] >= 0)]

        # AUC 계산
        auc_value = np.trapz(survival_df["생존확률"], survival_df["개월"])

        # 중위 생존기간
        median_survival = kmf.median_survival_time_
        if pd.isna(median_survival):
            median_disp = "도달 안함"
        else:
            median_disp = f"{median_survival:.1f}개월"

        results.append({
            "구분": group_name,
            "샘플 수": f"{sample_size:,}개",
            "중단율": f"{churn_rate:.1f}%",
            "AUC (36개월)": f"{auc_value:.2f}개월",
            "중위 생존기간": median_disp
        })

# -----------------------------
# 3️⃣ Streamlit 표 출력
# -----------------------------
results_df = pd.DataFrame(results)
st.table(results_df)   # 고정형 표

st.write("")

# =============================================================================
# 2️⃣ AUC 개선 목표 설정
# =============================================================================
st.subheader("2️⃣ AUC 개선 목표 설정")


