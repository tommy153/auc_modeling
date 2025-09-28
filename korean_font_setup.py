# korean_font_setup.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import warnings
warnings.filterwarnings('ignore')

def setup_korean_font():
    """운영체제별 한글 폰트 설정"""
    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            font_candidates = ['AppleGothic', 'Apple SD Gothic Neo', 'Nanum Gothic', 'Malgun Gothic']
        elif system == "Windows":
            font_candidates = ['Malgun Gothic', 'NanumGothic', 'Dotum', 'Gulim']
        else:  # Linux
            font_candidates = ['Nanum Gothic', 'NanumGothic', 'DejaVu Sans', 'Liberation Sans']

        available_fonts = [f.name for f in fm.fontManager.ttflist]

        for font in font_candidates:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                plt.rcParams['axes.unicode_minus'] = False
                print(f"한글 폰트 설정 완료: {font}")
                return True

        # fallback
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        print("한글 폰트를 찾지 못했습니다. 기본 폰트 사용")
        return False

    except Exception as e:
        print(f"폰트 설정 오류: {e}")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        return False

def show_simple_plot():
    """폰트 확인용 예제 플롯"""
    plt.figure(figsize=(4, 3))
    plt.title("예제")
    plt.plot([0, 1], [0, 1], 'r-')
    plt.show()

# import 시 자동 실행
if setup_korean_font():
    show_simple_plot()
