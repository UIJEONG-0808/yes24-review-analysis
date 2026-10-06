import streamlit as st

st.set_page_config(
    page_title="YES24 리뷰 데이터 분석",
    page_icon="📚",
    layout="wide"
)

st.title("YES24 리뷰 데이터 분석")

st.write(
    """
    YES24 도서 리뷰 데이터를 직접 수집하고 전처리하여
    별점과 리뷰 텍스트의 감성 관계를 분석한 프로젝트입니다.
    """
)

st.divider()

st.header("프로젝트 소개")

st.write(
    """
    이 프로젝트에서는 YES24의 도서 리뷰 및 한줄평 데이터를 수집하여
    리뷰 유형과 별점, 텍스트 감성 사이의 관계를 분석했습니다.
    """
)

st.header("분석 과정")

st.markdown(
    """
    1. YES24 리뷰 데이터 수집
    2. 텍스트 데이터 전처리
    3. 리뷰 감성 분석
    4. 별점과 텍스트 감성 비교
    5. 장문 리뷰와 단문 리뷰 비교
    """
)

st.header("주요 분석 결과")

col1, col2 = st.columns(2)

with col1:
    st.metric("장문 리뷰 긍정 비율", "62%")

with col2:
    st.metric("단문 리뷰 긍정 비율", "50%")

st.write(
    """
    분석 결과, 높은 별점을 부여한 리뷰에서도 부정적인 표현이 함께 나타나는
    사례를 확인했습니다. 이를 통해 별점만으로는 독자의 실제 반응을
    충분히 설명하기 어렵다는 점을 확인했습니다.
    """
)
