import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
import os

# ------------------------
# 0. 한글 폰트 설정
# ------------------------
def set_korean_font():
    font_path = "NanumGothic-Regular.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
    else:
        print("❗ NanumGothic-Regular.ttf 파일을 찾을 수 없습니다.")

set_korean_font()

# ------------------------
# 1. 페이지 설정
# ------------------------
st.set_page_config(page_title="월별 기준금리 기반 아파트 가격 예측기", layout="centered")
st.title("📆 월별 기준금리 기반 아파트 평균가격 예측기")

# ------------------------
# 2. 데이터 로딩
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("월별_아파트_기준금리_통합.csv")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.dropna(subset=["기준금리", "평균가격"])
    df["년월"] = df["날짜"].dt.strftime("%Y년 %m월")
    return df

data = load_data()

# ------------------------
# 3. 사용자 입력
# ------------------------
st.sidebar.header("사용자 설정")
regions = sorted(data["지역"].unique())
selected_region = st.sidebar.selectbox("📍 지역 선택", regions)

# 연월 슬라이더 설정
ym_options = data["년월"].unique().tolist()
ym_options.sort()
def ym_to_date(ym_str):
    return pd.to_datetime(ym_str.replace("년 ", "-").replace("월", "-01"))

start_ym, end_ym = st.sidebar.select_slider("📅 분석 기간 설정 (연월)",
    options=ym_options,
    value=(ym_options[0], ym_options[-1]))

start_date = ym_to_date(start_ym)
end_date = ym_to_date(end_ym)

input_rate = st.sidebar.slider("📉 기준금리 입력 (%)", 0.0, 10.0, 3.5, step=0.1)

# ------------------------
# 4. 데이터 필터링
# ------------------------
region_data = data[(data["지역"] == selected_region) &
                   (data["날짜"] >= start_date) & (data["날짜"] <= end_date)]

if not region_data.empty and len(region_data) >= 3:
    region_data = region_data.copy()

    # ------------------------
    # 5. 회귀 모델 학습 및 예측 (가중치 제거됨)
    # ------------------------
    X = region_data[["기준금리"]]
    y = region_data["평균가격"]
    model = LinearRegression()
    model.fit(X, y)  # ✅ 가중치 없음
    predicted_price = model.predict(np.array([[input_rate]]))[0]

    # ------------------------
    # 6. 출력
    # ------------------------
    corr = region_data["기준금리"].corr(region_data["평균가격"])
    st.subheader(f"🔍 {selected_region} 지역 기준금리 {input_rate:.1f}%에 대한 예측")
    st.metric("📊 예상 평균 아파트 가격", f"{predicted_price:,.0f} 백만원")
    st.write(f"📈 기준금리와 아파트 평균가격 간 상관계수: **{corr:.3f}**")
    st.caption(f"※ 선택된 기간: {start_ym} ~ {end_ym}, 총 {len(region_data)}개월")

    # ------------------------
    # 7. 산점도 + 회귀선
    # ------------------------
    fig, ax = plt.subplots()
    sns.regplot(data=region_data, x="기준금리", y="평균가격", ax=ax, scatter_kws={"s": 40})
    ax.scatter(input_rate, predicted_price, color="red", s=100, label="예측값")
    ax.set_title(f"[ {selected_region} ] 기준금리와 아파트 평균가격 관계")
    ax.set_xlabel("기준금리 (%)")
    ax.set_ylabel("평균 아파트 가격 (백만원)")
    ax.legend()
    st.pyplot(fig)

    # ------------------------
    # 8. 시간 흐름에 따른 추이
    # ------------------------
    fig2, ax1 = plt.subplots(figsize=(8, 4))
    color1 = "tab:blue"
    ax1.set_xlabel("날짜")
    ax1.set_ylabel("평균 아파트 가격", color=color1)
    ax1.plot(region_data["날짜"], region_data["평균가격"], marker='o', color=color1, label="평균가격")
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("기준금리 (%)", color=color2)
    ax2.plot(region_data["날짜"], region_data["기준금리"], marker='s', linestyle='--', color=color2, label="기준금리")
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(f"[ {selected_region} ] 월별 평균 아파트 가격 및 기준금리 추이")
    fig2.tight_layout()
    st.pyplot(fig2)

else:
    st.warning("해당 지역의 데이터가 부족하거나 선택한 기간 내 정보가 충분하지 않습니다.")
