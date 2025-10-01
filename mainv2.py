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
    create_survival_duration_boxplot,
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

st.write("")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"📄 **총 수업 수**<br><span style='font-size:24px; font-weight:bold;'>{total_cnt:,}개</span>", unsafe_allow_html=True)
with col2:
    st.markdown(f"❌ **중단 수업**<br><span style='font-size:24px; font-weight:bold;'>{stop_cnt:,}개</span>", unsafe_allow_html=True)
with col3:
    st.markdown(f"✅ **활성 수업**<br><span style='font-size:24px; font-weight:bold;'>{active_cnt:,}개</span>", unsafe_allow_html=True)

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

# 분석 단위 선택
st.markdown("#### 📊 생존분석 설정")
analysis_unit = st.radio("분석 단위 선택", ["주", "개월"], horizontal=True, help="생존분석과 시각화에 사용할 시간 단위를 선택하세요")

# 단위에 따른 duration 계산
if analysis_unit == "주":
    duration = df_processed['duration_days'] / 7  # 일을 주로 변환
else:  # 개월
    duration = df_processed['duration_days'] / 30.44  # 일을 개월로 변환 (평균 월 일수)

# Kaplan-Meier 생존 분석
kmf = KaplanMeierFitter()
kmf.fit(durations=duration, event_observed=df_processed['이탈여부'], label="KM Estimate")

# AUC 및 생존율 계산 (단위에 따라 조정)
if analysis_unit == "주":
    max_time = 36 * 4.35  # 36개월을 주로 변환 (약 156주)
    time_point = 36 * 4.35  # 36개월 시점
    time_label = "36개월"
else:  # 개월
    max_time = 36
    time_point = 36
    time_label = "36개월"

auc_value = calculate_auc(kmf, max_time=max_time, unit=analysis_unit)
survival_rate = calculate_survival_rate_at_time(kmf, time_point, unit=analysis_unit)

# KPI 카드
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"📊 **AUC (평균 생존기간, {analysis_unit})**")
    st.markdown(f"<span style='font-size:24px; font-weight:bold;'>{auc_value:.2f}{analysis_unit}</span>", unsafe_allow_html=True)

with col2:
    st.markdown(f"☑️ **{time_label} 생존율**")
    st.markdown(f"<span style='font-size:24px; font-weight:bold;'>{survival_rate*100:.1f}%</span>", unsafe_allow_html=True)

# Kaplan-Meier 생존 곡선
fig = create_survival_curve(df_processed, unit=analysis_unit)
st.plotly_chart(fig, use_container_width=True)

st.write("")

# 결제개월수별 Kaplan-Meier 생존 곡선
fig_grouped = create_grouped_survival_curves(df_processed, unit=analysis_unit)
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

        # 선택한 단위에 따라 duration 계산
        if analysis_unit == "주":
            group_duration = data['duration_days'] / 7
            max_time = 36 * 4.35  # 36개월을 주로 변환
            time_label_unit = "주"
        else:  # 개월
            group_duration = data['duration_days'] / 30.44
            max_time = 36
            time_label_unit = "개월"

        kmf = KaplanMeierFitter()
        kmf.fit(group_duration, event_observed=data["이탈여부"].astype(int))

        # AUC 계산 (선택한 단위 기준)
        auc_value = calculate_auc(kmf, max_time=max_time, unit=analysis_unit)

        # 중위 생존기간
        median_survival = kmf.median_survival_time_
        if pd.isna(median_survival):
            median_disp = "도달 안함"
        else:
            median_disp = f"{median_survival:.1f}{time_label_unit}"

        # 실제 관찰된 duration의 중앙값 (박스플롯과 비교용)
        observed_median = group_duration.median()

        results.append({
            "구분": group_name,
            "샘플 수": f"{sample_size:,}개",
            "중단율": f"{churn_rate:.1f}%",
            f"AUC (36개월, {time_label_unit})": f"{auc_value:.2f}{time_label_unit}",
            f"KM 중위생존기간": median_disp,
            f"관찰 중앙값": f"{observed_median:.1f}{time_label_unit}"
        })

# -----------------------------
# 3️⃣ Streamlit 표 출력
# -----------------------------
results_df = pd.DataFrame(results)
st.table(results_df)   # 고정형 표

st.write("")

# -----------------------------
# 4️⃣ 결제개월수별 생존 기간 분포 박스 플롯
# -----------------------------
st.subheader("📦 결제개월수별 생존 기간 분포")
fig_boxplot = create_survival_duration_boxplot(df_processed, unit=analysis_unit)
st.plotly_chart(fig_boxplot, use_container_width=True)

st.write("")

# =============================================================================
# 2️⃣ AUC 개선 목표 설정
# =============================================================================
st.subheader("2️⃣ AUC 개선 목표 설정")


