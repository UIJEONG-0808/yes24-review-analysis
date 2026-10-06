import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="YES24 Review Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 스타일
# --------------------------------------------------

st.markdown("""
<style>

.block-container {
    max-width: 1200px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

h1 {
    font-size: 2.7rem !important;
    font-weight: 750 !important;
}

h2 {
    margin-top: 2.5rem !important;
    font-weight: 700 !important;
}

h3 {
    font-weight: 650 !important;
}

[data-testid="stMetric"] {
    background: #fafafa;
    border: 1px solid #e8e8e8;
    padding: 18px;
    border-radius: 12px;
}

.insight-box {
    padding: 18px 22px;
    border-radius: 12px;
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    margin-top: 10px;
    margin-bottom: 20px;
}

.small-text {
    color: #777;
    font-size: 0.93rem;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 사이드바
# --------------------------------------------------

with st.sidebar:

    st.title("YES24 Review Analysis")

    page = st.radio(
        "페이지",
        [
            "Overview",
            "Analysis",
            "Polarizing Books",
            "Conclusion"
        ]
    )

    st.divider()

    st.caption("Text Analysis Portfolio")

    st.markdown("""
    **Tools**

    Python  
    Playwright  
    Pandas  
    KoNLPy  
    TF-IDF  
    Sentiment Analysis
    """)


# --------------------------------------------------
# Overview
# --------------------------------------------------

if page == "Overview":

    st.title("YES24 도서 리뷰 텍스트 분석")

    st.markdown(
        """
        ### 별점만으로는 독자의 실제 감정을 설명할 수 있을까?

        YES24 도서 리뷰와 한줄평을 직접 수집하여  
        **키워드 · 감성 · 별점 · 리뷰 유형의 관계**를 분석한 프로젝트입니다.
        """
    )

    st.divider()

    # 핵심 숫자
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("분석 도서", "15권")

    with c2:
        st.metric("수집 리뷰", "18,163건")

    with c3:
        st.metric("장르", "5개")

    with c4:
        st.metric("리뷰 유형", "2종")

    st.caption(
        "로맨스 · 판타지 · 문학 · 추리스릴러 · 호불호 / comment · review"
    )

    st.header("프로젝트 배경")

    left, right = st.columns([1, 1])

    with left:

        st.subheader("Why")

        st.write(
            """
            텍스트 데이터를 분석할 때 가장 쉽게 접할 수 있는 데이터 중 하나가
            온라인 리뷰입니다.

            특히 소설 리뷰에는 독자가 작품을 읽으면서 느낀 감정과 의견이
            비교적 직접적으로 표현됩니다.

            평소 독서를 좋아한다는 개인적인 관심사를 데이터 분석과 연결하여
            **'독자의 평가와 실제 감정은 얼마나 일치하는가'**라는 질문에서
            프로젝트를 시작했습니다.
            """
        )

    with right:

        st.subheader("Research Question")

        st.markdown(
            """
            - 별점과 리뷰 텍스트의 감성은 일치하는가?
            - 장르별로 자주 등장하는 키워드는 다른가?
            - 한줄평과 장문 리뷰의 감성 표현에는 차이가 있는가?
            - 호불호가 강한 작품은 어떤 특징을 갖는가?
            """
        )

    st.header("데이터 수집")

    st.markdown(
        """
        Python **Playwright**를 이용해 YES24의 동적 웹페이지에서
        리뷰 데이터를 직접 수집했습니다.
        """
    )

    pipeline = pd.DataFrame(
        {
            "단계": [
                "01 수집",
                "02 전처리",
                "03 키워드 분석",
                "04 감성 분석",
                "05 비교·해석"
            ],
            "내용": [
                "YES24 리뷰 · 한줄평 크롤링",
                "텍스트 정제 및 장르 분류",
                "명사 추출 · TF-IDF",
                "긍정·부정 키워드 기반 점수화",
                "별점 · 장르 · 리뷰 유형 비교"
            ]
        }
    )

    st.dataframe(
        pipeline,
        hide_index=True,
        use_container_width=True
    )

    st.header("수집 데이터")

    st.code(
        """
rating_5
content
helpful
review_type
date
is_purchase
        """,
        language="text"
    )

    st.markdown(
        """
        <div class="insight-box">
        <b>포인트</b><br>
        주어진 데이터셋을 분석한 것이 아니라,
        분석 목적에 필요한 데이터를 직접 수집하고
        분석 가능한 형태로 가공하는 과정부터 수행했습니다.
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Analysis
# --------------------------------------------------

elif page == "Analysis":

    st.title("전체 분석")

    tab1, tab2, tab3 = st.tabs(
        [
            "별점 × 감성",
            "Comment vs Review",
            "장르별 키워드"
        ]
    )

    # --------------------------------------------------
    # 별점 vs 감성
    # --------------------------------------------------

    with tab1:

        st.subheader("별점과 텍스트 감성은 얼마나 일치할까?")

        col1, col2 = st.columns([1, 2])

        with col1:

            st.metric(
                "별점-감성 상관계수",
                "r = 0.207"
            )

            st.write("p < 0.001")

        with col2:

            st.markdown(
                """
                <div class="insight-box">
                별점이 높을수록 감성 점수도 높아지는 경향은 있었지만,
                상관계수는 <b>0.207</b>로 낮았습니다.<br><br>

                즉, 독자는 높은 별점을 주면서도 리뷰에는
                부정적인 표현을 함께 사용할 수 있었습니다.
                </div>
                """,
                unsafe_allow_html=True
            )

        # 개념적 시각화
        fig = go.Figure()

        fig.add_indicator(
            mode="gauge+number",
            value=0.207,
            number={"valueformat": ".3f"},
            title={"text": "Rating ↔ Sentiment Correlation"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"thickness": 0.35}
            }
        )

        fig.update_layout(height=330)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("핵심 발견")

        st.markdown(
            """
            - 별점과 텍스트 감성은 **완전히 동일한 정보가 아님**
            - 높은 별점 + 부정적 표현이 동시에 존재
            - 낮은 별점 + 긍정적 표현 역시 일부 존재
            - 숫자 하나로는 독자의 복합적인 감정을 설명하기 어려움
            """
        )

    # --------------------------------------------------
    # comment vs review
    # --------------------------------------------------

    with tab2:

        st.subheader("단문과 장문 리뷰의 차이")

        review_df = pd.DataFrame(
            {
                "리뷰 유형": [
                    "comment (단문)",
                    "review (장문)"
                ],
                "긍정 비율": [
                    50,
                    62
                ]
            }
        )

        fig = px.bar(
            review_df,
            x="리뷰 유형",
            y="긍정 비율",
            text="긍정 비율"
        )

        fig.update_traces(
            texttemplate="%{text}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="긍정 리뷰 비율 (%)",
            xaxis_title="",
            yaxis_range=[0, 75],
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Comment 긍정 비율",
                "50%"
            )

            st.write(
                """
                감성 표현이 없는 단순 기록성 리뷰가
                상대적으로 많이 나타났습니다.
                """
            )

        with c2:

            st.metric(
                "Review 긍정 비율",
                "62%"
            )

            st.write(
                """
                장문 리뷰에서는 독자의 감정과
                추천·공감 의도가 더 풍부하게 나타났습니다.
                """
            )

        st.markdown(
            """
            <div class="insight-box">
            <b>Insight</b><br><br>
            장문 리뷰는 단문 한줄평보다 더 풍부한 감성 정보를 담고 있었습니다.
            리뷰 길이와 독자의 몰입 정도가 텍스트 감성 표현과
            관련될 가능성을 확인했습니다.
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------
    # 장르별 분석
    # --------------------------------------------------

    with tab3:

        st.subheader("장르별 독자의 관심 포인트")

        genre = st.selectbox(
            "장르를 선택하세요",
            [
                "로맨스",
                "판타지",
                "문학",
                "추리스릴러",
                "호불호"
            ]
        )

        genre_info = {

            "로맨스": {
                "keyword": "제목 · 영화",
                "insight":
                    "원작과 영화의 비교가 자주 등장해 "
                    "미디어 믹스에 민감한 장르로 나타났습니다."
            },

            "판타지": {
                "keyword": "백화점 · 달러 등 고유 소재",
                "insight":
                    "작품의 고유 세계관과 소재가 주요 키워드로 등장해 "
                    "세계관 몰입도가 높은 특징을 보였습니다."
            },

            "문학": {
                "keyword": "작가 · 한강",
                "insight":
                    "작품 자체뿐 아니라 작가를 중심으로 한 "
                    "담론이 상대적으로 강하게 나타났습니다."
            },

            "추리스릴러": {
                "keyword": "살인 · 사건 · 좀비",
                "insight":
                    "사건과 장르적 소재가 핵심 키워드에 "
                    "명확하게 반영되었습니다."
            },

            "호불호": {
                "keyword": "사랑 · 반전",
                "insight":
                    "특정 감정 요소와 반전이 "
                    "평가가 갈리는 주요 요소로 나타났습니다."
            }
        }

        info = genre_info[genre]

        st.metric(
            "주요 키워드",
            info["keyword"]
        )

        st.markdown(
            f"""
            <div class="insight-box">
            <b>{genre} 분석</b><br><br>
            {info["insight"]}
            </div>
            """,
            unsafe_allow_html=True
        )


# --------------------------------------------------
# 호불호 도서 분석
# --------------------------------------------------

elif page == "Polarizing Books":

    st.title("호불호 도서 집중 분석")

    st.write(
        """
        전체 15권 중 독자 평가가 뚜렷하게 갈리는 작품을 중심으로
        별점 분포와 텍스트 감성을 추가 분석했습니다.
        """
    )

    book_stats = pd.DataFrame(
        {
            "도서": [
                "급류",
                "홍학의 자리",
                "구의 증명"
            ],
            "별점 표준편차": [
                1.00,
                0.78,
                0.87
            ]
        }
    )

    fig = px.bar(
        book_stats,
        x="도서",
        y="별점 표준편차",
        text="별점 표준편차"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="별점 표준편차",
        xaxis_title="",
        height=450,
        yaxis_range=[0, 1.15]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "표준편차가 클수록 독자별 평가 차이가 크다는 의미로 해석했습니다."
    )

    selected_book = st.selectbox(
        "작품을 선택하세요",
        [
            "급류",
            "홍학의 자리",
            "구의 증명"
        ]
    )

    book_info = {

        "급류": {
            "std": "1.00",
            "headline": "3권 중 가장 강한 호불호",
            "text":
                """
                1~2점 저평가 리뷰가 세 작품 중 가장 많았습니다.

                긍정·부정 리뷰 모두에서 '사랑'이 주요 키워드로 나타났으며,
                낮은 별점 리뷰에서는 '베스트셀러'라는 표현도 등장했습니다.

                발표 분석에서는 이를
                **베스트셀러에 대한 높은 기대와 실제 독서 경험 사이의 괴리**
                로 해석했습니다.
                """,
            "extra": "저별점-긍정 2.4% / 고별점-부정 1.9%"
        },

        "홍학의 자리": {
            "std": "0.78",
            "headline": "반전이 평가를 가르는 핵심 요소",
            "text":
                """
                세 작품 중 5점 집중도가 가장 높고
                상대적으로 평가가 일관되었습니다.

                높은 평가와 낮은 평가 모두에서
                **'반전'**이 주요 키워드로 등장해,
                동일한 요소에 대한 독자 반응이 크게 갈렸습니다.

                긍정 리뷰 비율은 **71.3%**로
                세 작품 중 가장 높았습니다.
                """,
            "extra": "긍정 비율 71.3%"
        },

        "구의 증명": {
            "std": "0.87",
            "headline": "높은 평가 속에 부정적 감정이 공존",
            "text":
                """
                높은 별점을 주면서도 부정적 감정 표현을 사용하는
                리뷰가 상대적으로 많이 나타났습니다.

                발표 분석에서는 이를
                작품을 긍정적으로 평가하면서도
                슬픔과 같은 강한 감정을 동시에 표현하는
                **복합 감정 리뷰**로 해석했습니다.

                고별점-부정 감성 리뷰 비율은 **2.6%**로
                세 작품 중 가장 높았습니다.
                """,
            "extra": "고별점-부정 감성 2.6%"
        }
    }

    info = book_info[selected_book]

    c1, c2 = st.columns([1, 2])

    with c1:

        st.metric(
            "별점 표준편차",
            info["std"]
        )

        st.metric(
            "추가 지표",
            info["extra"]
        )

    with c2:

        st.subheader(info["headline"])

        st.markdown(info["text"])


# --------------------------------------------------
# Conclusion
# --------------------------------------------------

elif page == "Conclusion":

    st.title("Conclusion")

    st.subheader("프로젝트에서 발견한 4가지")

    st.markdown(
        """
        ### 01. 별점과 텍스트 감성은 완전히 일치하지 않는다

        전체 별점-감성 상관계수는 **r = 0.207**로,
        양의 관계는 존재하지만 강하지 않았습니다.

        독자는 높은 별점을 주면서도
        부정적인 감정을 텍스트로 표현할 수 있었습니다.

        ---

        ### 02. 장르마다 독자가 주목하는 요소가 다르다

        - 로맨스 → 미디어 믹스
        - 판타지 → 세계관
        - 문학 → 작가
        - 추리스릴러 → 사건과 소재
        - 호불호 → 반전

        ---

        ### 03. 장문 리뷰가 더 풍부한 감성 신호를 담는다

        review의 긍정 비율은 **62%**,
        comment는 **50%**였습니다.

        장문의 리뷰에서 독자의 감정과
        적극적인 평가가 더 많이 나타났습니다.

        ---

        ### 04. 같은 '호불호'라도 원인은 서로 다르다

        **급류**  
        기대와 실제 독서 경험 사이의 괴리

        **홍학의 자리**  
        반전 요소에 대한 극단적인 평가 차이

        **구의 증명**  
        강한 감정 소비와 취향 의존성
        """
    )

    st.divider()

    st.subheader("What I Learned")

    st.write(
        """
        이 프로젝트를 통해 주어진 데이터만 분석하는 것이 아니라,
        분석 목적에 맞는 데이터를 직접 수집하고
        전처리 기준과 분석 기준을 설정하는 경험을 쌓았습니다.

        또한 별점과 같은 정량 데이터만으로는 설명하기 어려운
        사용자의 반응을 텍스트 데이터를 함께 활용해
        해석하는 과정의 중요성을 확인했습니다.
        """
    )

    st.success(
        "Data Collection → Preprocessing → Analysis → Interpretation"
    )
