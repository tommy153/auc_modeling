import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import plotly.graph_objects as go

def perform_kaplan_meier_analysis(processed_df):
    """Kaplan-Meier 생존 분석 수행"""
    # Kaplan-Meier 피팅
    durations = processed_df["done_month_corrected"]
    events = processed_df["churn"]

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, label="전체")

    # 생존곡선 데이터프레임 만들기
    survival_df = kmf.survival_function_.reset_index()
    survival_df.columns = ["개월", "생존확률"]

    # 36개월까지만 필터
    survival_df = survival_df[(survival_df["개월"] <= 36)&(survival_df["개월"]>=0)]

    # AUC 계산
    auc_value = np.trapz(survival_df["생존확률"], survival_df["개월"])

    return survival_df, auc_value

def create_monthly_distribution_chart(processed_df):
    """월별 수업 시작 분포 차트 생성"""
    st.subheader("📅 월별 수업 시작 분포")

    # 월별 그룹화
    processed_df['년월'] = processed_df['crda'].dt.to_period('M').astype(str)
    monthly_counts = processed_df['년월'].value_counts().sort_index()

    # 막대 그래프
    fig_monthly = go.Figure(data=[
        go.Bar(x=monthly_counts.index, y=monthly_counts.values,
               marker_color='steelblue',
               text=monthly_counts.values,
               textposition='auto')
    ])

    fig_monthly.update_layout(
        title="월별 신규 수업 시작 수",
        xaxis_title="년월",
        yaxis_title="수업 수",
        template="plotly_white",
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig_monthly)

def display_auc_metrics(survival_df, auc_value):
    """AUC 및 생존율 메트릭 표시"""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 AUC", f"{auc_value:.2f}개월")
    with col2:
        st.metric("📈 36개월 생존율", f"{survival_df['생존확률'].iloc[-1]:.1%}")

def create_survival_curve_chart(survival_df):
    """Kaplan-Meier 생존곡선 차트 생성"""
    fig = go.Figure()

    # 선 + 면적 채우기
    fig.add_trace(go.Scatter(
        x=survival_df["개월"],
        y=survival_df["생존확률"],
        mode="lines",
        name="생존곡선",
        fill="tozeroy",         # 선 밑을 면적으로 채움
        fillcolor="rgba(0, 150, 255, 0.2)"
    ))

    fig.update_layout(
        title="📊 Kaplan-Meier 생존곡선 + AUC 면적",
        xaxis_title="개월",
        yaxis_title="생존확률",
        yaxis=dict(
            range=[0,1],
            dtick=0.1,
            showgrid=False
        ),
        template="plotly_white"
    )

    st.plotly_chart(fig)

def create_grouped_survival_curves(processed_df):
    """fst_months별로 그룹화된 생존곡선 생성"""
    st.subheader("📊 결제기간별 생존곡선 비교")

    # 특정 결제기간만 선택
    fst_months_groups = [1, 3, 6, 12]

    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, group in enumerate(fst_months_groups):
        # 그룹별 데이터 필터링
        group_data = processed_df[processed_df['fst_months'] == group]

        if len(group_data) > 0:
            # Kaplan-Meier 피팅
            durations = group_data["done_month_corrected"]
            events = group_data["churn"]

            kmf = KaplanMeierFitter()
            kmf.fit(durations, event_observed=events, label=f"{group}개월")

            # 생존곡선 데이터 생성
            survival_df = kmf.survival_function_.reset_index()
            survival_df.columns = ["개월", "생존확률"]
            survival_df = survival_df[(survival_df["개월"] <= 36) & (survival_df["개월"] >= 0)]

            # 라인 추가
            fig.add_trace(go.Scatter(
                x=survival_df["개월"],
                y=survival_df["생존확률"],
                mode="lines",
                name=f"{group}개월 결제",
                line=dict(color=colors[i % len(colors)])
            ))

    fig.update_layout(
        title="결제기간별 생존곡선 비교",
        xaxis_title="개월",
        yaxis_title="생존확률",
        yaxis=dict(
            range=[0,1],
            dtick=0.1,
            showgrid=False
        ),
        template="plotly_white",
        legend=dict(x=0.7, y=0.9)
    )

    st.plotly_chart(fig)

def create_auc_analysis_table(processed_df):
    """AUC 분석 결과 표 생성"""
    st.subheader("📊 AUC 분석 결과")

    results = []
    groups = [
        ("전체", None),
        ("1개월 구매", 1),
        ("3개월 구매", 3),
        ("6개월 구매", 6),
        ("12개월 구매", 12)
    ]

    for group_name, fst_month in groups:
        # 데이터 필터링
        if fst_month is None:
            data = processed_df
        else:
            data = processed_df[processed_df['fst_months'] == fst_month]

        if len(data) > 0:
            # 기본 통계
            sample_size = len(data)
            churn_count = data['churn'].sum()
            churn_rate = churn_count / sample_size * 100

            # Kaplan-Meier 피팅
            durations = data["done_month_corrected"]
            events = data["churn"]

            kmf = KaplanMeierFitter()
            kmf.fit(durations, event_observed=events)

            # 생존곡선 데이터 생성 (36개월까지)
            survival_df = kmf.survival_function_.reset_index()
            survival_df.columns = ["개월", "생존확률"]
            survival_df = survival_df[(survival_df["개월"] <= 36) & (survival_df["개월"] >= 0)]

            # AUC 계산
            auc_value = np.trapz(survival_df["생존확률"], survival_df["개월"])

            # 중위 생존기간 계산
            median_survival = kmf.median_survival_time_
            median_survival = median_survival if not pd.isna(median_survival) else "도달 안함"

            results.append({
                "구분": group_name,
                "샘플 수": f"{sample_size:,}개",
                "중단율": f"{churn_rate:.1f}%",
                "AUC (36개월)": f"{auc_value:.2f}개월",
                "중위 생존기간": f"{median_survival:.1f}개월" if median_survival != "도달 안함" else median_survival
            })

    # 데이터프레임으로 변환하여 표시
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, width='stretch')