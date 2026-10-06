import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from konlpy.tag import Okt
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 0. 기본 설정
# =============================================

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'analysis_results_hobulho'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOOKS = {
    '급류':         'yes24_reviews_output/yes24_116586303_급류_20260610_184051.csv',
    '홍학의 자리':  'yes24_reviews_output/yes24_102791425_홍학의 자리 (금기 에디션)_20260610_184051.csv',
    '구의 증명':    'yes24_reviews_output/yes24_118578901_구의 증명_20260610_184051.csv',
}

COLORS = {
    '급류':        '#e74c3c',
    '홍학의 자리': '#3498db',
    '구의 증명':   '#2ecc71',
}

FONT_PATH = 'C:/Windows/Fonts/malgun.ttf'

# =============================================
# 1. 데이터 로드
# =============================================

print("데이터 로드 중...")
dfs = {}
for title, path in BOOKS.items():
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['book_title'] = title
    dfs[title] = df
    print(f"  {title}: {len(df):,}개 리뷰")

df_all = pd.concat(dfs.values(), ignore_index=True)
print(f"전체: {len(df_all):,}개\n")

# =============================================
# 2. 공통 함수
# =============================================

okt = Okt()

STOPWORDS = {
    '이', '가', '을', '를', '은', '는', '에', '의', '도', '으로', '로', '와', '과',
    '하다', '있다', '되다', '이다', '하고', '해서', '에서', '지만', '그리고', '하는',
    '것', '수', '그', '이런', '저런', '더', '좀', '잘', '너무', '정말', '진짜',
    '많이', '같이', '같다', '않다', '없다', '않는', '입니다', '합니다', '했습니다',
    '해요', '이에요', '예요', '네요', '어요', '아요', '구요', '거요', '죠',
    '책', '소설', '작품', '리뷰', '읽다', '읽었', '구매', '구입', '이거', '저거',
    '때문', '생각', '내용', '부분', '느낌', '이번', '정도', '경우', '사람', '이후',
    '급류', '홍학', '자리', '구의', '증명'
}

POSITIVE_WORDS = [
    '재미', '재밌', '좋아', '좋다', '좋은', '훌륭', '명작', '최고', '강추', '추천',
    '설레', '감동', '즐거', '행복', '만족', '완벽', '멋지', '아름다', '흥미', '흥분',
    '술술', '빠져', '몰입', '두근', '기대', '사랑', '소장', '필독', '감사',
    '신선', '탁월', '우아', '매력', '공감', '재독', '뛰어', '인상', '깊이', '반전'
]
NEGATIVE_WORDS = [
    '지루', '실망', '별로', '아쉽', '힘들', '어렵', '불만', '싫', '나쁘', '최악',
    '후회', '낡은', '지겹', '억지', '어색', '평범', '노잼', '짜증', '진부',
    '부족', '불편', '불쾌', '이해안', '퇴색', '속았', '후회', '혹평'
]

def extract_nouns(text):
    if pd.isna(text) or len(str(text).strip()) < 2:
        return []
    nouns = okt.nouns(str(text))
    return [n for n in nouns if len(n) >= 2 and n not in STOPWORDS]

def sentiment_score(text):
    if pd.isna(text) or len(str(text).strip()) < 2:
        return 0.0
    text = str(text)
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    return float(pos - neg)

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
# 3. 감성 점수 & 명사 추출
# =============================================

print("감성 점수 계산 중...")
df_all['sentiment'] = df_all['content'].apply(sentiment_score)
df_all['sentiment_label'] = df_all['sentiment'].apply(sentiment_label)

print("명사 추출 중... (수 분 소요될 수 있습니다)")
df_all['nouns'] = df_all['content'].apply(extract_nouns)


# =============================================
# 분석 1 — 별점 분포 비교
# =============================================
print("\n[분석 1] 별점 분포 비교")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, (title, color) in enumerate(COLORS.items()):
    g_df = dfs[title]
    rating_counts = g_df['rating_5'].value_counts().sort_index()
    std = g_df['rating_5'].std()
    mean = g_df['rating_5'].mean()

    axes[i].bar(rating_counts.index.astype(str), rating_counts.values,
                color=color, alpha=0.85, edgecolor='white')
    axes[i].set_title(f'{title}\n평균 {mean:.2f}점 / 표준편차 {std:.2f}',
                      fontsize=13, fontweight='bold')
    axes[i].set_xlabel('별점')
    axes[i].set_ylabel('리뷰 수')

plt.suptitle('호불호 도서 3권 별점 분포 비교', fontsize=16, fontweight='bold')
plt.tight_layout()
save('1_별점_분포_비교.png')


# =============================================
# 분석 2 — 평균 별점 & 표준편차 비교
# =============================================
print("\n[분석 2] 평균 별점 & 표준편차")

stats = []
for title in BOOKS:
    g = dfs[title]['rating_5']
    stats.append({'책': title, '평균 별점': g.mean(), '표준편차': g.std()})
stats_df = pd.DataFrame(stats)
print(stats_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(stats_df['책'], stats_df['평균 별점'],
            color=list(COLORS.values()), alpha=0.85, edgecolor='white')
axes[0].set_ylim(0, 5)
axes[0].set_ylabel('평균 별점')
axes[0].set_title('평균 별점 비교', fontsize=13, fontweight='bold')
for i, v in enumerate(stats_df['평균 별점']):
    axes[0].text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=11)

axes[1].bar(stats_df['책'], stats_df['표준편차'],
            color=list(COLORS.values()), alpha=0.85, edgecolor='white')
axes[1].set_ylabel('표준편차')
axes[1].set_title('별점 표준편차 비교\n(클수록 호불호 심함)', fontsize=13, fontweight='bold')
for i, v in enumerate(stats_df['표준편차']):
    axes[1].text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=11)

plt.suptitle('호불호 도서 3권 별점 통계 비교', fontsize=16, fontweight='bold')
plt.tight_layout()
save('2_별점_통계_비교.png')


# =============================================
# 분석 3 — 긍정 vs 부정 리뷰 워드클라우드
# =============================================
print("\n[분석 3] 긍정 vs 부정 워드클라우드")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

for i, title in enumerate(BOOKS):
    g_df = df_all[df_all['book_title'] == title]

    pos_nouns = [n for nouns in g_df[g_df['sentiment'] > 0]['nouns'] for n in nouns]
    neg_nouns = [n for nouns in g_df[g_df['sentiment'] < 0]['nouns'] for n in nouns]

    for j, (nouns, label, cmap) in enumerate([
        (pos_nouns, '긍정', 'Blues'),
        (neg_nouns, '부정', 'Reds')
    ]):
        freq = Counter(nouns)
        if len(freq) == 0:
            axes[i][j].axis('off')
            continue
        wc = WordCloud(
            font_path=FONT_PATH,
            width=600, height=350,
            background_color='white',
            max_words=60,
            colormap=cmap
        )
        wc.generate_from_frequencies(freq)
        axes[i][j].imshow(wc, interpolation='bilinear')
        axes[i][j].axis('off')
        axes[i][j].set_title(f'{title} — {label} 리뷰 키워드',
                              fontsize=13, fontweight='bold')

plt.suptitle('긍정 vs 부정 리뷰 키워드 비교', fontsize=16, fontweight='bold')
plt.tight_layout()
save('3_긍부정_워드클라우드.png')


# =============================================
# 분석 4 — 별점 1~2점 vs 4~5점 키워드 비교
# =============================================
print("\n[분석 4] 극단 별점 키워드 비교")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

for i, title in enumerate(BOOKS):
    g_df = df_all[df_all['book_title'] == title]

    low_nouns  = [n for nouns in g_df[g_df['rating_5'] <= 2]['nouns'] for n in nouns]
    high_nouns = [n for nouns in g_df[g_df['rating_5'] >= 4]['nouns'] for n in nouns]

    for j, (nouns, label, color) in enumerate([
        (high_nouns, '별점 4~5점 (긍정)', '#3498db'),
        (low_nouns,  '별점 1~2점 (부정)', '#e74c3c'),
    ]):
        freq = Counter(nouns)
        top10 = freq.most_common(10)
        if not top10:
            axes[i][j].axis('off')
            continue
        words_list, counts_list = zip(*top10)
        axes[i][j].barh(list(words_list)[::-1], list(counts_list)[::-1],
                        color=color, alpha=0.85)
        axes[i][j].set_title(f'{title} — {label}',
                              fontsize=12, fontweight='bold')
        axes[i][j].set_xlabel('빈도')

plt.suptitle('극단 별점(1~2점 vs 4~5점) 키워드 비교', fontsize=16, fontweight='bold')
plt.tight_layout()
save('4_극단별점_키워드.png')


# =============================================
# 분석 5 — 불일치 리뷰 탐지
# =============================================
print("\n[분석 5] 불일치 리뷰 탐지")

for title in BOOKS:
    g_df = df_all[df_all['book_title'] == title]
    print(f"\n{'='*50}")
    print(f"[{title}] 별점 높은데(4↑) 부정적인 리뷰")
    mismatch_neg = g_df[(g_df['rating_5'] >= 4) & (g_df['sentiment'] < 0)].sort_values('sentiment')
    print(mismatch_neg[['rating_5', 'sentiment', 'content']].head(3).to_string())

    print(f"\n[{title}] 별점 낮은데(2↓) 긍정적인 리뷰")
    mismatch_pos = g_df[(g_df['rating_5'] <= 2) & (g_df['sentiment'] > 0)].sort_values('sentiment', ascending=False)
    print(mismatch_pos[['rating_5', 'sentiment', 'content']].head(3).to_string())

# 불일치 비율 시각화
mismatch_stats = []
for title in BOOKS:
    g_df = df_all[df_all['book_title'] == title]
    total = len(g_df)
    neg_mismatch = len(g_df[(g_df['rating_5'] >= 4) & (g_df['sentiment'] < 0)])
    pos_mismatch = len(g_df[(g_df['rating_5'] <= 2) & (g_df['sentiment'] > 0)])
    mismatch_stats.append({
        '책': title,
        '고별점-부정감성 (%)': round(neg_mismatch / total * 100, 1),
        '저별점-긍정감성 (%)': round(pos_mismatch / total * 100, 1),
    })

ms_df = pd.DataFrame(mismatch_stats)
x = range(len(BOOKS))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars1 = ax.bar([i - width/2 for i in x], ms_df['고별점-부정감성 (%)'],
               width, label='고별점-부정감성', color='#e74c3c', alpha=0.85)
bars2 = ax.bar([i + width/2 for i in x], ms_df['저별점-긍정감성 (%)'],
               width, label='저별점-긍정감성', color='#3498db', alpha=0.85)

ax.set_xticks(list(x))
ax.set_xticklabels(ms_df['책'], fontsize=12)
ax.set_ylabel('전체 리뷰 중 비율 (%)')
ax.set_title('불일치 리뷰 비율 (별점 ↔ 감성 불일치)', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.1f}%', ha='center', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.1f}%', ha='center', fontsize=10)

plt.tight_layout()
save('5_불일치_리뷰_비율.png')


# =============================================
# 분석 6 — comment vs review 감성 비교
# =============================================
print("\n[분석 6] comment vs review 감성 비교")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for i, title in enumerate(BOOKS):
    g_df = df_all[df_all['book_title'] == title]
    sns.boxplot(data=g_df, x='review_type', y='sentiment',
                palette={'comment': 'steelblue', 'review': 'coral'},
                width=0.45, ax=axes[i])
    axes[i].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[i].set_title(title, fontsize=13, fontweight='bold')
    axes[i].set_xlabel('리뷰 타입')
    axes[i].set_ylabel('감성 점수')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['comment', 'review'])

plt.suptitle('호불호 도서 comment vs review 감성 점수 분포', fontsize=15, fontweight='bold')
plt.tight_layout()
save('6_comment_vs_review.png')


# =============================================
# 분석 7 — 감성 레이블 비율 비교
# =============================================
print("\n[분석 7] 감성 레이블 비율")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, title in enumerate(BOOKS):
    g_df = df_all[df_all['book_title'] == title]
    label_counts = g_df['sentiment_label'].value_counts()
    label_pct = label_counts / label_counts.sum() * 100

    color_map = {'긍정': '#e74c3c', '중립': '#95a5a6', '부정': '#3498db'}
    colors = [color_map.get(l, 'gray') for l in label_pct.index]

    axes[i].pie(label_pct.values, labels=label_pct.index,
                autopct='%1.1f%%', colors=colors,
                startangle=90, textprops={'fontsize': 11})
    axes[i].set_title(title, fontsize=13, fontweight='bold')

plt.suptitle('호불호 도서 감성 레이블 비율', fontsize=15, fontweight='bold')
plt.tight_layout()
save('7_감성_레이블_비율.png')


# =============================================
# 완료
# =============================================
print(f"\n========== 분석 완료 ==========")
print(f"결과 이미지가 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")
print("생성된 파일:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
