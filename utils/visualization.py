import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter


def create_monthly_bar_chart(df_processed):
    """월별 신규 수업 시작 수 차트 생성"""
    df_monthly = (
        df_processed
        .groupby(pd.Grouper(key="결제등록일", freq="M"))
        .size()
        .reset_index(name="수업 수")
    )

    fig_month = px.bar(
        df_monthly,
        x="결제등록일",
        y="수업 수",
        text="수업 수",
        title="월별 신규 수업 시작 수"
    )

    fig_month.update_traces(textposition="outside")
    fig_month.update_layout(
        xaxis_title="연월",
        yaxis_title="수업 수",
        template="plotly_white",
        xaxis=dict(
            tickmode="linear",
            dtick="M1",
            tickformat="%Y-%m"
        )
    )

    return fig_month


def create_weekly_bar_chart(df_processed):
    """주별 신규 수업 시작 수 차트 생성"""
    df_temp = df_processed.copy()
    df_temp['연도'] = df_temp['결제등록일'].dt.isocalendar().year
    df_temp['주차'] = df_temp['결제등록일'].dt.isocalendar().week
    df_temp['연주차코드'] = df_temp['연도'] * 100 + df_temp['주차']

    df_weekly = (
        df_temp
        .groupby(['연도','주차','연주차코드'])
        .size()
        .reset_index(name="수업 수")
    )

    df_weekly['연도-주차'] = df_weekly['연도'].astype(str) + "-W" + df_weekly['주차'].astype(str)

    fig_week = px.bar(
        df_weekly,
        x="연도-주차",
        y="수업 수",
        text="수업 수",
        title="주별 신규 수업 시작 수",
        category_orders={"연도-주차": df_weekly.sort_values("연주차코드")["연도-주차"].tolist()}
    )

    return fig_week


def create_survival_curve(df_processed, unit="주"):
    """Kaplan-Meier 생존 곡선 생성"""
    df_km = pd.DataFrame({
        "duration": df_processed['duration_days'],
        "event": df_processed['이탈여부']
    })

    kmf = KaplanMeierFitter()
    kmf.fit(durations=df_km['duration'], event_observed=df_km['event'], label="KM Estimate")

    timeline = kmf.survival_function_.index
    survival = kmf.survival_function_["KM Estimate"]

    # 단위 변환
    if unit == "주":
        x = timeline / 7
        xlabel = "주"
    elif unit == "개월":
        x = timeline / 30
        xlabel = "개월"
    else:
        x = timeline
        xlabel = "일"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=survival,
        mode='lines',
        line=dict(color='blue', width=2),
        name='생존 확률',
        line_shape='hv',
        fill='tozeroy',
        fillcolor='rgba(0, 123, 255, 0.2)'
    ))

    fig.update_layout(
        title="Kaplan–Meier 생존 곡선",
        xaxis_title=xlabel,
        yaxis_title="생존 확률",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="LightGray", borderwidth=1
        )
    )

    fig.update_yaxes(tick0=0.0, dtick=0.1, range=[0, 1], showgrid=False)
    fig.update_xaxes(showgrid=False)

    return fig


def create_grouped_survival_curves(df_processed):
    """결제개월수별 Kaplan-Meier 생존 곡선 생성"""
    groups = [
        ("전체", None),
        ("1개월 구매", "1"),
        ("3개월 구매", "3"),
        ("6개월 구매", "6"),
        ("12개월 구매", "12")
    ]

    fig = go.Figure()
    kmf = KaplanMeierFitter()

    for group_name, pay_month in groups:
        if pay_month is None:
            data = df_processed
        else:
            data = df_processed[df_processed['결제개월수'] == pay_month]

        if len(data) > 0:
            durations = data["donemonth"]
            events = data["이탈여부"].astype(int)

            kmf.fit(durations, event_observed=events, label=group_name)

            survival = kmf.survival_function_[group_name]
            timeline = kmf.survival_function_.index

            fig.add_trace(go.Scatter(
                x=timeline,
                y=survival,
                mode='lines',
                line_shape='hv',
                name=group_name
            ))

    fig.update_layout(
        title="Kaplan–Meier 생존 곡선 (결제개월수별)",
        xaxis_title="개월",
        yaxis_title="생존 확률",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="LightGray", borderwidth=1
        )
    )
    fig.update_yaxes(tick0=0.0, dtick=0.1, range=[0,1], showgrid=False)
    fig.update_xaxes(showgrid=False)

    return fig


def create_churn_rate_timeline(time_churn_detail, time_unit_option):
    """시간대별 이탈률 추이 그래프 생성"""
    if len(time_churn_detail) == 0:
        return None

    fig_time_churn = px.line(
        time_churn_detail,
        x='시간그룹',
        y='이탈률(%)',
        title=f'{time_unit_option} 이탈률 추이',
        markers=True
    )
    fig_time_churn.update_layout(
        template="plotly_white",
        xaxis_title="기간",
        yaxis_title="이탈률(%)"
    )

    return fig_time_churn


def create_survival_comparison_chart(kmf_current, kmf_improved):
    """현재 vs 개선 후 생존 곡선 비교 그래프"""
    fig_comparison = go.Figure()

    # 현재 곡선
    fig_comparison.add_trace(go.Scatter(
        x=kmf_current.survival_function_.index,
        y=kmf_current.survival_function_['Current'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='현재',
        line_shape='hv'
    ))

    # 개선 후 곡선
    fig_comparison.add_trace(go.Scatter(
        x=kmf_improved.survival_function_.index,
        y=kmf_improved.survival_function_['Improved'],
        mode='lines',
        line=dict(color='green', width=2),
        name='개선 후',
        line_shape='hv'
    ))

    fig_comparison.update_layout(
        title="생존 곡선 비교: 현재 vs 개선 후",
        xaxis_title="개월",
        yaxis_title="생존 확률",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="LightGray", borderwidth=1
        )
    )
    fig_comparison.update_yaxes(tick0=0.0, dtick=0.1, range=[0,1], showgrid=False)
    fig_comparison.update_xaxes(showgrid=False)

    return fig_comparison


def calculate_auc(kmf, max_months=36):
    """AUC 계산 (36개월 기준)"""
    survival_df = kmf.survival_function_.reset_index()
    survival_df.columns = ["개월", "생존확률"]
    survival_df = survival_df[(survival_df["개월"] <= max_months) & (survival_df["개월"] >= 0)]
    auc_value = np.trapz(survival_df["생존확률"], survival_df["개월"])
    return auc_value


def calculate_survival_rate_at_time(kmf, months):
    """특정 시점의 생존율 계산"""
    return kmf.predict(months * 30)  # 개월을 일로 변환


def display_auc_improvement_results(auc_current, auc_improved):
    """AUC 개선 결과 표시"""
    import streamlit as st

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**현재 AUC**")
        st.markdown(f"<span style='font-size:20px; font-weight:bold;'>{auc_current:.2f}개월</span>", unsafe_allow_html=True)

    with col2:
        st.markdown("**개선 후 예상 AUC**")
        st.markdown(f"<span style='font-size:20px; font-weight:bold; color:green;'>{auc_improved:.2f}개월</span>", unsafe_allow_html=True)

    with col3:
        improvement = auc_improved - auc_current
        improvement_pct = (improvement / auc_current) * 100
        st.markdown("**개선 효과**")
        st.markdown(f"<span style='font-size:20px; font-weight:bold; color:blue;'>+{improvement:.2f}개월 ({improvement_pct:+.1f}%)</span>", unsafe_allow_html=True)