import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 0. 기본 설정
# =============================================

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 결과 이미지 저장 폴더
OUTPUT_DIR = 'analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 장르 매핑 (파일명 속 책 제목 기준)
GENRE_MAP = {
    '오늘 밤, 세계에서 이 사랑이 사라진다 해도': '로맨스',
    '오만과 편견':                               '로맨스',
    '너의 췌장을 먹고 싶어 (일반판)':            '로맨스',
    '지구 끝의 온실':                            '판타지',
    '나미야 잡화점의 기적':                      '판타지',
    '달러구트 꿈 백화점':                        '판타지',
    '모순':                                      '문학',
    '인간 실격':                                 '문학',
    '채식주의자':                                '문학',
    '용의자 X의 헌신':                           '추리스릴러',
    '칵테일, 러브, 좀비':                        '추리스릴러',
    '봉제인형 살인사건':                         '추리스릴러',
    '급류':                                      '호불호',
    '홍학의 자리 (금기 에디션)':                 '호불호',
    '구의 증명':                                 '호불호',
}

GENRE_COLORS = {
    '로맨스':    '#e91e8c',
    '판타지':    '#7c4dff',
    '문학':      '#00897b',
    '추리스릴러':'#f4511e',
    '호불호':    '#546e7a',
}

# =============================================
# 1. 데이터 로드 (ALL 파일 사용)
# =============================================

DATA_DIR = 'yes24_reviews_output'
all_file = os.path.join(DATA_DIR, 'yes24_ALL_15books_20260610_184051.csv')

print("데이터 로드 중...")
df = pd.read_csv(all_file, encoding='utf-8-sig')
print(f"전체 리뷰 수: {len(df):,}개")

# 책 제목 컬럼 정리 (공백 제거)
df['book_title'] = df['book_title'].str.strip()

# 장르 컬럼 추가
df['genre'] = df['book_title'].map(GENRE_MAP)

# 매핑 안 된 책 확인
unmapped = df[df['genre'].isna()]['book_title'].unique()
if len(unmapped) > 0:
    print(f"장르 미매핑 책: {unmapped}")

df = df.dropna(subset=['genre'])
print(f"장르 매핑 후 리뷰 수: {len(df):,}개")
print(df.groupby('genre').size())


# =============================================
# 공통 함수 정의
# =============================================

okt = Okt()

STOPWORDS = {
    '이', '가', '을', '를', '은', '는', '에', '의', '도', '으로', '로', '와', '과',
    '하다', '있다', '되다', '이다', '하고', '해서', '에서', '지만', '그리고', '하는',
    '것', '수', '그', '이런', '저런', '더', '좀', '잘', '너무', '정말', '진짜',
    '많이', '같이', '같다', '않다', '없다', '않는', '입니다', '합니다', '했습니다',
    '해요', '이에요', '예요', '네요', '어요', '아요', '구요', '거요', '죠',
    '책', '소설', '작품', '리뷰', '읽다', '읽었', '구매', '구입', '이거', '저거',
    '때문', '생각', '내용', '부분', '느낌', '이번', '정도', '경우', '사람', '이후'
}

def extract_nouns(text):
    if pd.isna(text) or len(str(text).strip()) < 2:
        return []
    nouns = okt.nouns(str(text))
    return [n for n in nouns if len(n) >= 2 and n not in STOPWORDS]

POSITIVE_WORDS = [
    '재미', '재밌', '좋아', '좋다', '좋은', '훌륭', '명작', '최고', '강추', '추천',
    '설레', '감동', '즐거', '행복', '만족', '완벽', '멋지', '아름다', '흥미', '흥분',
    '술술', '빠져', '몰입', '두근', '기대', '사랑', '소장', '필독', '감사',
    '신선', '탁월', '우아', '매력', '공감', '재독', '뛰어', '인상', '깊이'
]
NEGATIVE_WORDS = [
    '지루', '실망', '별로', '아쉽', '힘들', '어렵', '불만', '싫', '나쁘', '최악',
    '후회', '낡은', '지겹', '억지', '어색', '평범', '노잼', '짜증', '진부',
    '부족', '불편', '불쾌', '이해안', '퇴색'
]

def sentiment_score(text):
    if pd.isna(text) or len(str(text).strip()) < 2:
        return 0.0
    text = str(text)
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)

def sentiment_label(score):
    if score > 0:   return '긍정'
    elif score < 0: return '부정'
    else:           return '중립'

def save(fname):
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"저장 완료: {path}")


# =============================================
# 1단계: 키워드 분석
# =============================================
print("\n========== 1단계: 키워드 분석 ==========")
print("명사 추출 중... (수 분 소요될 수 있습니다)")

df['nouns'] = df['content'].apply(extract_nouns)
df['nouns_str'] = df['nouns'].apply(lambda x: ' '.join(x))

# --- 1-1. 장르별 워드클라우드 (2x3 그리드) ---
genres = list(GENRE_MAP.values())
genres = list(dict.fromkeys(genres))  # 순서 유지 중복 제거

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, genre in enumerate(genres):
    genre_nouns = [
        noun
        for nouns in df[df['genre'] == genre]['nouns']
        for noun in nouns
    ]
    freq = Counter(genre_nouns)
    if len(freq) == 0:
        continue
    wc = WordCloud(
        font_path='C:/Windows/Fonts/malgun.ttf',
        width=600, height=350,
        background_color='white',
        max_words=80,
        colormap='tab10'
    )
    wc.generate_from_frequencies(freq)
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].axis('off')
    axes[i].set_title(f'{genre}', fontsize=14, fontweight='bold',
                      color=GENRE_COLORS[genre], pad=10)

plt.suptitle('장르별 키워드 워드클라우드', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
save('1_wordcloud_장르별.png')

# --- 1-2. 장르별 상위 10개 키워드 비교 바차트 ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, genre in enumerate(genres):
    genre_nouns = [
        noun
        for nouns in df[df['genre'] == genre]['nouns']
        for noun in nouns
    ]
    freq = Counter(genre_nouns)
    top10 = freq.most_common(10)
    if not top10:
        continue
    words_list, counts_list = zip(*top10)
    axes[i].barh(list(words_list)[::-1], list(counts_list)[::-1],
                 color=GENRE_COLORS[genre], alpha=0.85)
    axes[i].set_title(f'{genre} 상위 10 키워드', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('빈도')

plt.suptitle('장르별 상위 10개 키워드', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
save('1_top10_keywords_장르별.png')

# --- 1-3. TF-IDF 장르별 핵심 키워드 ---
print("\n[TF-IDF 장르별 핵심 키워드]")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, genre in enumerate(genres):
    g_df = df[df['genre'] == genre]
    g_df = g_df[g_df['nouns_str'].str.strip() != '']
    if len(g_df) < 5:
        continue
    tfidf = TfidfVectorizer(max_features=20)
    matrix = tfidf.fit_transform(g_df['nouns_str'])
    scores = dict(zip(tfidf.get_feature_names_out(),
                      matrix.toarray().mean(axis=0)))
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    w, s = zip(*top)
    axes[i].barh(list(w)[::-1], list(s)[::-1],
                 color=GENRE_COLORS[genre], alpha=0.85)
    axes[i].set_title(f'{genre} TF-IDF 핵심 키워드', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('TF-IDF 점수')

plt.suptitle('장르별 TF-IDF 핵심 키워드', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
save('1_tfidf_장르별.png')


# =============================================
# 2단계: 감성 분석 + 별점 비교
# =============================================
print("\n========== 2단계: 감성 분석 ==========")

df['sentiment'] = df['content'].apply(sentiment_score)
df['sentiment_label'] = df['sentiment'].apply(sentiment_label)

print("\n전체 감성 분포:")
print(df['sentiment_label'].value_counts())

# --- 2-1. 장르별 감성 점수 분포 박스플롯 ---
plt.figure(figsize=(12, 6))
order = genres
palette = {g: GENRE_COLORS[g] for g in genres}
sns.boxplot(data=df, x='genre', y='sentiment', order=order,
            palette=palette, width=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
plt.xlabel('장르', fontsize=12)
plt.ylabel('감성 점수', fontsize=12)
plt.title('장르별 감성 점수 분포', fontsize=16, fontweight='bold')
plt.tight_layout()
save('2_sentiment_장르별_boxplot.png')

# --- 2-2. 별점 대비 감성 점수 비교 (장르별) ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, genre in enumerate(genres):
    g_df = df[df['genre'] == genre]
    rating_sent = g_df.groupby('rating_5')['sentiment'].mean().reset_index()

    axes[i].bar(rating_sent['rating_5'].astype(str),
                rating_sent['sentiment'],
                color=GENRE_COLORS[genre], alpha=0.85, edgecolor='white')
    axes[i].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[i].set_title(f'{genre}', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('별점')
    axes[i].set_ylabel('평균 감성 점수')

    # 상관계수
    valid = g_df[g_df['sentiment'] != 0]
    if len(valid) > 5:
        corr, pval = pearsonr(valid['rating_5'], valid['sentiment'])
        axes[i].text(0.05, 0.95, f'r={corr:.3f} (p={pval:.3f})',
                     transform=axes[i].transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('장르별 별점 대비 평균 감성 점수', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
save('2_rating_vs_sentiment_장르별.png')

# --- 2-3. 전체 별점 vs 감성 점수 산점도 ---
plt.figure(figsize=(10, 6))
for genre in genres:
    g_df = df[df['genre'] == genre]
    plt.scatter(g_df['rating_5'], g_df['sentiment'],
                alpha=0.25, s=15,
                color=GENRE_COLORS[genre], label=genre)

plt.axhline(y=0, color='red', linestyle='--', linewidth=1.2)
plt.xlabel('별점', fontsize=12)
plt.ylabel('감성 점수', fontsize=12)
plt.title('별점 vs 감성 점수 산점도 (장르별)', fontsize=16, fontweight='bold')
plt.legend(title='장르', fontsize=10)

valid_all = df[df['sentiment'] != 0]
if len(valid_all) > 5:
    corr, pval = pearsonr(valid_all['rating_5'], valid_all['sentiment'])
    plt.text(0.05, 0.95, f'전체 상관계수: r={corr:.3f} (p={pval:.4f})',
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
plt.tight_layout()
save('2_scatter_별점vs감성.png')

# --- 2-4. 불일치 리뷰 탐지 ---
print("\n[별점 높은데(4.0↑) 부정적인 리뷰 상위 5개]")
mismatch_neg = df[(df['rating_5'] >= 4.0) & (df['sentiment'] < 0)].sort_values('sentiment')
print(mismatch_neg[['book_title', 'genre', 'rating_5', 'sentiment', 'content']].head(5).to_string())

print("\n[별점 낮은데(3.0↓) 긍정적인 리뷰 상위 5개]")
mismatch_pos = df[(df['rating_5'] <= 3.0) & (df['sentiment'] > 0)].sort_values('sentiment', ascending=False)
print(mismatch_pos[['book_title', 'genre', 'rating_5', 'sentiment', 'content']].head(5).to_string())


# =============================================
# 3단계: review vs comment 감성 분포 비교
# =============================================
print("\n========== 3단계: review vs comment 비교 ==========")

print("\n타입별 감성 점수 기술통계:")
print(df.groupby('review_type')['sentiment'].describe().round(4))

# --- 3-1. 전체 박스플롯 ---
plt.figure(figsize=(9, 6))
sns.boxplot(data=df, x='review_type', y='sentiment',
            palette={'comment': 'steelblue', 'review': 'coral'},
            width=0.45)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
plt.xlabel('리뷰 타입', fontsize=12)
plt.ylabel('감성 점수', fontsize=12)
plt.title('comment vs review 감성 점수 분포 (전체)', fontsize=15, fontweight='bold')
plt.xticks([0, 1], ['comment (단문)', 'review (장문)'], fontsize=12)
plt.tight_layout()
save('3_comment_vs_review_boxplot.png')

# --- 3-2. 장르별 comment vs review 감성 점수 ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, genre in enumerate(genres):
    g_df = df[df['genre'] == genre]
    sns.boxplot(data=g_df, x='review_type', y='sentiment',
                palette={'comment': 'steelblue', 'review': 'coral'},
                width=0.45, ax=axes[i])
    axes[i].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[i].set_title(f'{genre}', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('리뷰 타입')
    axes[i].set_ylabel('감성 점수')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['comment', 'review'])

plt.suptitle('장르별 comment vs review 감성 점수 분포', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
save('3_comment_vs_review_장르별.png')

# --- 3-3. 감성 레이블 비율 비교 ---
type_label = df.groupby(['review_type', 'sentiment_label']).size().unstack(fill_value=0)
type_label_pct = type_label.div(type_label.sum(axis=1), axis=0) * 100

type_label_pct.plot(kind='bar', figsize=(10, 6),
                    color=['#e74c3c', '#95a5a6', '#3498db'],
                    edgecolor='white', width=0.5)
plt.xlabel('리뷰 타입', fontsize=12)
plt.ylabel('비율 (%)', fontsize=12)
plt.title('comment vs review 감성 레이블 비율', fontsize=15, fontweight='bold')
plt.xticks([0, 1], ['comment (단문)', 'review (장문)'], rotation=0, fontsize=12)
plt.legend(title='감성', fontsize=11)
plt.tight_layout()
save('3_comment_vs_review_ratio.png')

# =============================================
# 완료
# =============================================
print("\n========== 전체 분석 완료 ==========")
print(f"결과 이미지가 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")
