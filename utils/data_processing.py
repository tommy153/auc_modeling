import streamlit as st
import pandas as pd

def load_data(uploaded_file=None):
    """데이터를 로드하고 전처리하는 함수"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("data/AUC기본소스2508_lvt_done_month_수정.csv")

    if 'student_name' in df.columns:
        df = df.drop(columns='student_name')

    df['crda'] = pd.to_datetime(df['crda'])
    df['reactive_datetime'] = pd.to_datetime(df['reactive_datetime'])
    df['fst_pay_date'] = pd.to_datetime(df['fst_pay_date'])
    df['lst_done_at'] = pd.to_datetime(df['lst_done_at'])
    df['lst_tutoring_datetime'] = pd.to_datetime(
        df['lst_tutoring_datetime'],
        errors='coerce'
    )
    
    CURRNET_DATE = df['crda'].max()
    CUTOFF_DATE = CURRNET_DATE - pd.Timedelta(days=30)
    return df, CURRNET_DATE, CUTOFF_DATE

def process_data(df, START_DATE, END_DATE, CUTOFF_DATE):
    """
    원본 데이터를 분석용으로 전처리
    """
    # 완료 상태 정의
    FINISHED_STATES = ['FINISH', 'AUTO_FINISH', 'DONE', 'NOCARD', 'NOPAY']

    # 1. 기간 필터링
    st.write(f"📆 기간 필터링: {START_DATE.strftime('%Y-%m-%d')} ~ {END_DATE.strftime('%Y-%m-%d')}")

    # 유효한 결제일이 있는 데이터만 필터링
    valid_pay_date = df['fst_pay_date'].notna()
    in_period = (
        (df['crda'] >= pd.to_datetime(START_DATE)) &
        (df['crda'] <= pd.to_datetime(END_DATE))
    )

    filtered_data = df[valid_pay_date & in_period].copy()

    # 고유한 수업 식별자 생성 (lecture_vt_No + p_rn)
    filtered_data['lesson_id'] = (filtered_data['lecture_vt_No'].astype(str) + '_' +
                                filtered_data['p_rn'].astype(str))

    st.success(f"✅ 기간 필터링 결과: {len(filtered_data):,}개 행 (원본의 {len(filtered_data)/len(df)*100:.1f}%)")

    # 2. 수업 완료 상태 및 done_month 보정

    def determine_status_and_correct_month(row):
        """각 행의 완료 상태 판정 및 done_month 보정"""
        churn = False
        corrected_done_month = row['done_month'] if pd.notna(row['done_month']) else 0

        # 1) 명시적 완료 상태 확인
        if row['tutoring_state'] in FINISHED_STATES:
            churn = True

        # 2) 암시적 완료 상태 확인 (ACTIVE이지만 마지막 수업일이 기준일보다 이전)
        elif (row['tutoring_state'] == 'ACTIVE' and
              pd.notna(row['lst_tutoring_datetime']) and
              row['lst_tutoring_datetime'] < CUTOFF_DATE):
            churn = True

            # done_month 보정 로직
            if (pd.notna(row['crda']) and pd.notna(row['lst_tutoring_datetime'])):
                # 실제 수업 기간 계산 (28일 = 1개월로 가정)
                actual_days = (row['lst_tutoring_datetime'] - row['crda']).days
                actual_months = actual_days / 28

                # done_month가 실제 기간보다 크면 보정
                if row['done_month'] > actual_months:
                    corrected_done_month = actual_months * 0.8  # 80%로 보정

        return pd.Series({
            'churn': churn,
            'done_month_corrected': corrected_done_month
        })

    # 상태 및 보정 적용
    status_correction = filtered_data.apply(determine_status_and_correct_month, axis=1)
    processed_data = pd.concat([filtered_data, status_correction], axis=1)

    # 결과 요약
    total_count = len(processed_data)
    finished_count = processed_data['churn'].sum()
    active_count = total_count - finished_count
    corrected_rows = (
        processed_data['done_month_corrected'] !=
        processed_data['done_month'].fillna(0)
    ).sum()


    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 총 수업 수", f"{total_count:,}개")
    with col2:
        st.metric("❌ 중단 수업", f"{finished_count:,}개")
    with col3:
        st.metric("✅ 활성 수업", f"{active_count:,}개")
    with col4:
        st.metric("🔧 DM 보정", f"{corrected_rows:,}개")

    return processed_data