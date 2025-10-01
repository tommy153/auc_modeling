import numpy as np
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import json
import streamlit as st

@st.cache_data
def load_google_sheets_data(worksheet_name: str):
    print(f"[{datetime.now()}] Google Sheet 데이터 로드")
    """Google Sheets에서 데이터 로드 - 중복 헤더 오류 해결"""
    
    with open('pj_appscript.json', 'r') as f:
        credentials_info = json.load(f)

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
    client = gspread.authorize(creds)

    # 스프레드시트 열기
    spreadsheet = client.open("🔥🔥🔥 경험그룹_KPI (수업 기준!!!!!) 🔥🔥🔥")
    worksheet = spreadsheet.worksheet(worksheet_name)

    try:
        # 모든 데이터 가져오기
        all_values = worksheet.get_all_values()
        
        if not all_values:
            print("워크시트가 비어있습니다.")
            return pd.DataFrame()
        
        # 두 번째 행을 헤더로 사용 (첫 번째 행은 스킵)
        headers = all_values[1]
        data_rows = all_values[2:]
        
        # 중복된 빈 헤더 처리
        processed_headers = []
        for i, header in enumerate(headers):
            if header == '' or header in processed_headers:
                # 빈 헤더나 중복 헤더에 고유한 이름 부여
                processed_headers.append(f'unnamed_column_{i}')
            else:
                processed_headers.append(header)
        
        # DataFrame 생성
        df = pd.DataFrame(data_rows, columns=processed_headers)
        
        # 빈 행 제거
        df = df.dropna(how='all')
        
        # 불필요한 unnamed 컬럼들 제거 (모든 값이 비어있는 경우)
        cols_to_drop = []
        for col in df.columns:
            if col.startswith('unnamed_column_') and (df[col].isna().all() or (df[col] == '').all()):
                cols_to_drop.append(col)
        
        df = df.drop(columns=cols_to_drop)
        
        print(f"[{datetime.now()}] 데이터 로드 성공: {len(df)}행, {len(df.columns)}열")
        return df
        
    except Exception as e:
        print(f"데이터 로드 중 오류: {str(e)}")
        return pd.DataFrame()


def processing_google_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Google Sheet → Pandas 데이터 전처리 함수
    1. 각 컬럼별 '' → np.nan 변환
    2. dtype 보정 (infer_objects)
    3. 주요 컬럼 타입 변환 (날짜, 숫자, 카테고리 등)
    4. 최종 분석용 컬럼만 선택
    """

    def donemonth_to_days_bucketed(series: pd.Series) -> pd.Series:
        """
        done_month 값을 7일 단위 버킷으로 변환
        - 정수부: floor(done_month) × 28일
        - 소수부: 0→0일, (0~0.25]→7일, (0.25~0.5]→14일, (0.5~0.75]→21일, (0.75~1)→28일(이월)
        - 28일은 다음 달로 이월 처리
        """
        int_part = np.floor(series).astype(int)  # numpy int 사용
        frac = (series - np.floor(series)).fillna(0)

        # 소수부 → 일수 매핑
        conditions = [
            (frac == 0),
            (frac > 0) & (frac <= 0.25),
            (frac > 0.25) & (frac <= 0.5),
            (frac > 0.5) & (frac <= 0.75),
            (frac > 0.75) & (frac < 1),
        ]
        choices = [0, 7, 14, 21, 28]
        add_days = np.select(conditions, choices, default=0)

        # 28일은 이월 처리
        carry_mask = add_days == 28
        int_part = int_part + carry_mask.astype(int)  # numpy int로만 처리
        add_days = np.where(carry_mask, 0, add_days)

        # 최종적으로 pandas Series로 리턴할 때만 Int64 변환
        return pd.Series(int_part * 28 + add_days, index=series.index).astype("Int64")
    
    weight_map = {'W1': 0.25, 'W2': 0.125, 'W3': 0.0833}

    def correct_done_month(df):
        df['opt_prefix'] = df['option'].str.extract(r'^(W\d)')
        df['opt_weight'] = df['opt_prefix'].map(weight_map)

        cond = (df['donemonth'] == 0) & df['opt_weight'].notna()
        df.loc[cond, 'donemonth'] = df.loc[cond, 'cycle_count'] * df.loc[cond, 'opt_weight']

        return df

    print(f"[{datetime.now()}] 전처리 시작")

    # 1. 데이터 복사
    df = df.copy()

    # 2. 컬럼별로 빈 문자열 → NaN 치환
    df = df.mask(df == '', np.nan)

    # 3. dtype 보정 (FutureWarning 방지)
    df = df.infer_objects(copy=False)

    # 4. 'T' 값 제거 (테스트 데이터 제외)
    df = df[df['이탈여부'] != 'T']

    # 5. 타입 변환
    # 날짜형
    df['결제등록일'] = pd.to_datetime(df['payment_regdate'], errors='coerce')
    if '중단예정일' in df.columns:
        df['중단예정일'] = pd.to_datetime(df['중단예정일'], errors='coerce')

    # 숫자형 (nullable Int64 사용)
    df['lvt'] = pd.to_numeric(df['lvt'], errors='coerce').astype('Int64')
    df['user_No'] = pd.to_numeric(df['user_No'], errors='coerce').astype('Int64')
    df['단계'] = pd.to_numeric(df['단계'], errors='coerce').fillna(0).astype(int)
    df['donemonth'] = pd.to_numeric(df['done_month'], errors='coerce')
    df['stage_count'] = pd.to_numeric(df['stage_count'], errors='coerce').fillna(0).astype(int)
    df['cycle_count'] = pd.to_numeric(df['cycle_count'], errors='coerce').fillna(0).astype(int)
    if '중단 예정 DONEMONTH' in df.columns:
        df['중단 예정 DONEMONTH'] = pd.to_numeric(df['중단 예정 DONEMONTH'], errors='coerce')

    # 카테고리형 (이탈여부: A=0, P=1)
    df['이탈여부'] = df['이탈여부'].map({"A": 0, "P": 1})
    df = correct_done_month(df)
    df['duration_days'] = donemonth_to_days_bucketed(df['donemonth'])
    df['결제개월수'] = df['최초 개월 수']

    # 6. 최종 사용할 컬럼만 선택 (존재하는 것만 유지)
    keep_cols = [
        '결제등록일', 'lvt', 'user_No', 'option', '단계', '이탈여부', 'donemonth', 'duration_days',
        '학년', '교과/탐구', '결제개월수', 'stage_count', 'cycle_count',
        '과외상태', '수업상태', '중단예정일', '중단 예정 DONEMONTH'
    ]
    df = df[keep_cols]

    print(f"[{datetime.now()}] 전처리 완료: shape={df.shape}")
    return df