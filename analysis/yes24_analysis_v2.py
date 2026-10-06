import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 0. 기본 설정
# =============================================

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'analysis_results_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 장르 매핑 ──────────────────────────────────────────────
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

GENRE_ORDER  = ['로맨스', '판타지', '문학', '추리스릴러', '호불호']

# 로즈쿼츠·세레니티 기반 팔레트
GENRE_COLORS = {
    '로맨스':    '#E8829A',
    '판타지':    '#7B9ED4',
    '문학':      '#6BAE95',
    '추리스릴러':'#E0956A',
    '호불호':    '#9E8FC4',
}

C_COMMENT = '#92A8D1'   # 세레니티
C_REVIEW  = '#F7CAC9'   # 로즈쿼츠
C_COMMENT_DK = '#5C7EAF'
C_REVIEW_DK  = '#D4908E'

# ── 감성 사전 ──────────────────────────────────────────────
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
    t = str(text)
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    return float(pos - neg)

def save(fname):
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f'저장: {path}')

# =============================================
# 1. 데이터 로드
# =============================================
print('데이터 로드 중...')
df = pd.read_csv(
    'yes24_reviews_output/yes24_ALL_15books_20260610_184051.csv',
    encoding='utf-8-sig'
)
df['book_title'] = df['book_title'].str.strip()
df['genre']      = df['book_title'].map(GENRE_MAP)
df               = df.dropna(subset=['genre'])
df['sentiment']  = df['content'].apply(sentiment_score)

def label(s):
    if s > 0:  return '긍정'
    elif s < 0: return '부정'
    return '중립'

df['sent_label'] = df['sentiment'].apply(label)
print(f'총 {len(df):,}건  |  장르별:')
print(df.groupby('genre').size().to_string())

# 감성 표현 있는 리뷰 (0 제외)
df_nz = df[df['sentiment'] != 0].copy()

# =============================================
# 분석 1 — 장르별 감성 점수 박스플롯
# =============================================
print('\n[1] 장르별 감성 점수 박스플롯')

fig, ax = plt.subplots(figsize=(13, 6))
ax.set_facecolor('#FAFAFA')

# boxplot
for i, genre in enumerate(GENRE_ORDER):
    data = df_nz[df_nz['genre'] == genre]['sentiment']
    bp = ax.boxplot(
        data, positions=[i], widths=0.52,
        patch_artist=True, showfliers=True, notch=False,
        flierprops=dict(marker='o', markersize=3,
                        markerfacecolor=GENRE_COLORS[genre],
                        markeredgecolor=GENRE_COLORS[genre], alpha=0.25),
        medianprops=dict(color='white', linewidth=2.5),
        boxprops=dict(facecolor=GENRE_COLORS[genre],
                      edgecolor=GENRE_COLORS[genre], linewidth=1.5),
        whiskerprops=dict(color=GENRE_COLORS[genre], linewidth=1.5),
        capprops=dict(color=GENRE_COLORS[genre], linewidth=2),
    )
    median_val = data.median()
    mean_val   = data.mean()
    ax.text(i, median_val + 0.1, f'중앙값\n{median_val:.1f}',
            ha='center', fontsize=9, color='#333333', fontweight='bold')
    ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] > -5 else -4.5,
            f'n={len(data):,}', ha='center', fontsize=8.5, color='#888888')

ax.axhline(0, color='#D4908E', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_xticks(range(len(GENRE_ORDER)))
ax.set_xticklabels(GENRE_ORDER, fontsize=12)
ax.set_ylabel('감성 점수 (긍정 단어 수 − 부정 단어 수)', fontsize=11)
ax.set_title('장르별 감성 점수 분포 (감성 표현 리뷰 기준)', fontsize=15, fontweight='bold', pad=14)
ax.yaxis.grid(True, color='#E8E8E8', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
save('1_장르별_감성_박스플롯.png')

# =============================================
# 분석 2 — 장르별 별점 대비 감성 점수
# =============================================
print('\n[2] 장르별 별점 대비 감성 점수')

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()

for i, genre in enumerate(GENRE_ORDER):
    ax = axes[i]
    ax.set_facecolor('#FAFAFA')
    g  = df[df['genre'] == genre]

    # 별점별 평균 감성
    mean_s = g.groupby('rating_5')['sentiment'].mean().reset_index()
    ax.bar(mean_s['rating_5'].astype(str), mean_s['sentiment'],
           color=GENRE_COLORS[genre], edgecolor='white', alpha=0.85, width=0.6)
    ax.axhline(0, color='#D4908E', linestyle='--', linewidth=1.2, alpha=0.7)

    # 상관계수
    valid = g[g['sentiment'] != 0]
    if len(valid) > 5:
        r, p = pearsonr(valid['rating_5'], valid['sentiment'])
        ax.text(0.05, 0.95, f'r = {r:.3f}  (p {"< 0.001" if p < 0.001 else f"= {p:.3f}"})',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F7CAC9',
                          edgecolor='#D4908E', alpha=0.85))

    ax.set_title(genre, fontsize=13, fontweight='bold', color=GENRE_COLORS[genre])
    ax.set_xlabel('별점', fontsize=10)
    ax.set_ylabel('평균 감성 점수', fontsize=10)
    ax.yaxis.grid(True, color='#E8E8E8', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[5].set_visible(False)
plt.suptitle('장르별 별점 대비 평균 감성 점수', fontsize=16, fontweight='bold')
plt.tight_layout()
save('2_장르별_별점대비_감성.png')

# =============================================
# 분석 3 — 전체 별점 vs 감성 산점도 (jitter)
# =============================================
print('\n[3] 별점 vs 감성 산점도')

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_facecolor('#FAFAFA')

np.random.seed(42)
for genre in GENRE_ORDER:
    g = df[(df['genre'] == genre) & df['rating_5'].between(1, 5)]
    xj = g['rating_5'] + np.random.uniform(-0.18, 0.18, len(g))
    yj = g['sentiment'] + np.random.uniform(-0.06, 0.06, len(g))
    ax.scatter(xj, yj, color=GENRE_COLORS[genre],
               alpha=0.18, s=16, linewidths=0, label=genre, rasterized=True)

# 전체 별점별 평균 추이
mean_all = df[df['rating_5'].between(1, 5)].groupby('rating_5')['sentiment'].mean()
ax.plot(mean_all.index, mean_all.values,
        color='#3D3D3D', linewidth=2.2, marker='o',
        markersize=7, markerfacecolor='white', markeredgewidth=2,
        zorder=5, label='전체 평균 추이')

ax.axhline(0, color='#D4908E', linestyle='--', linewidth=1.5, alpha=0.7)

valid_all = df[(df['sentiment'] != 0) & df['rating_5'].between(1, 5)]
r, p = pearsonr(valid_all['rating_5'], valid_all['sentiment'])
ax.text(0.03, 0.96,
        f'전체 상관계수: r = {r:.3f}  (p {"< 0.001" if p < 0.001 else f"= {p:.3f}"})',
        transform=ax.transAxes, fontsize=12, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F7CAC9',
                  edgecolor='#D4908E', alpha=0.85))

ax.set_xlim(0.5, 5.5)
ax.set_xticks([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
ax.set_xticklabels(['1점','1.5','2점','2.5','3점','3.5','4점','4.5','5점'], fontsize=11)
ax.set_xlabel('별점', fontsize=12)
ax.set_ylabel('감성 점수', fontsize=12)
ax.set_title('별점 vs 감성 점수 산점도 (장르별)', fontsize=15, fontweight='bold', pad=14)
ax.yaxis.grid(True, color='#E8E8E8', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

patches = [mpatches.Patch(color=GENRE_COLORS[g], label=g, alpha=0.75)
           for g in GENRE_ORDER]
mean_line = plt.Line2D([0],[0], color='#3D3D3D', linewidth=2,
                        marker='o', markersize=6,
                        markerfacecolor='white', markeredgewidth=2,
                        label='전체 평균 추이')
patches.append(mean_line)
ax.legend(handles=patches, loc='lower right', fontsize=11,
          framealpha=0.9, edgecolor='#E0E0E0')

plt.tight_layout()
save('3_별점vs감성_산점도.png')

# =============================================
# 분석 4 — comment vs review 감성 비교
# =============================================
print('\n[4] comment vs review 비교')

fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                         gridspec_kw={'width_ratios': [1.2, 1]})
fig.patch.set_facecolor('#FFFFFF')

# ── 왼쪽: 박스플롯 ────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor('#FAFAFA')

PAL  = {'comment': C_COMMENT, 'review': C_REVIEW}
EDGE = {'comment': C_COMMENT_DK, 'review': C_REVIEW_DK}

for i, rt in enumerate(['comment', 'review']):
    data = df_nz[df_nz['review_type'] == rt]['sentiment']
    ax1.boxplot(
        data, positions=[i], widths=0.46,
        patch_artist=True, showfliers=True,
        flierprops=dict(marker='o', markersize=3,
                        markerfacecolor=EDGE[rt],
                        markeredgecolor=EDGE[rt], alpha=0.25),
        medianprops=dict(color='white', linewidth=2.5),
        boxprops=dict(facecolor=PAL[rt], edgecolor=EDGE[rt], linewidth=1.5),
        whiskerprops=dict(color=EDGE[rt], linewidth=1.5),
        capprops=dict(color=EDGE[rt], linewidth=2),
    )
    med = data.median()
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    ax1.text(i, med + 0.1,  f'중앙값 {med:.1f}',  ha='center', fontsize=9.5, fontweight='bold', color='#333333')
    ax1.text(i, q3  + 0.22, f'Q3={q3:.1f}', ha='center', fontsize=8.5, color=EDGE[rt])
    ax1.text(i, q1  - 0.38, f'Q1={q1:.1f}', ha='center', fontsize=8.5, color=EDGE[rt])
    n = len(data)
    ax1.text(i, -3.8, f'n={n:,}', ha='center', fontsize=8.5, color='#999999')

ax1.axhline(0, color='#D4908E', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['comment\n(단문)', 'review\n(장문)'], fontsize=12)
ax1.set_ylabel('감성 점수', fontsize=11)
ax1.set_title('감성 점수 분포\n(감성 표현 리뷰 기준)', fontsize=13, fontweight='bold', pad=12)
ax1.yaxis.grid(True, color='#E8E8E8', linewidth=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── 오른쪽: 평균 감성 + 긍정 비율 ────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#FAFAFA')

rtypes = ['comment', 'review']
means  = [df_nz[df_nz['review_type']==t]['sentiment'].mean() for t in rtypes]
pos_r  = [len(df[(df['review_type']==t) & (df['sentiment']>0)]) /
          len(df[df['review_type']==t]) * 100 for t in rtypes]

x, w = np.array([0, 1]), 0.3
b1 = ax2.bar(x - w/2, means,  width=w,
             color=[PAL[t] for t in rtypes],
             edgecolor=[EDGE[t] for t in rtypes], linewidth=1.5)
b2 = ax2.bar(x + w/2, pos_r, width=w,
             color=[EDGE[t] for t in rtypes],
             edgecolor=[EDGE[t] for t in rtypes], linewidth=1.5, alpha=0.65)

for bar, v in zip(b1, means):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{v:.2f}', ha='center', fontsize=10, fontweight='bold', color='#3D3D3D')
for bar, v in zip(b2, pos_r):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#3D3D3D')

ax2.set_xticks(x)
ax2.set_xticklabels(['comment\n(단문)', 'review\n(장문)'], fontsize=12)
ax2.set_title('평균 감성 점수 &\n긍정 리뷰 비율', fontsize=13, fontweight='bold', pad=12)
ax2.yaxis.grid(True, color='#E8E8E8', linewidth=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

p1 = mpatches.Patch(color=C_COMMENT,    label='평균 감성 점수')
p2 = mpatches.Patch(color=C_COMMENT_DK, alpha=0.65, label='긍정 리뷰 비율 (%)')
ax2.legend(handles=[p1, p2], fontsize=9, loc='upper left',
           framealpha=0.9, edgecolor='#E0E0E0')

plt.suptitle('comment vs review 감성 점수 비교', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
save('4_comment_vs_review.png')

# =============================================
# 분석 5 — 장르별 comment vs review 박스플롯
# =============================================
print('\n[5] 장르별 comment vs review 비교')

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()

for i, genre in enumerate(GENRE_ORDER):
    ax = axes[i]
    ax.set_facecolor('#FAFAFA')
    g  = df_nz[df_nz['genre'] == genre]

    for j, rt in enumerate(['comment', 'review']):
        data = g[g['review_type'] == rt]['sentiment']
        if len(data) == 0:
            continue
        ax.boxplot(
            data, positions=[j], widths=0.44,
            patch_artist=True, showfliers=True,
            flierprops=dict(marker='o', markersize=2.5,
                            markerfacecolor=EDGE[rt],
                            markeredgecolor=EDGE[rt], alpha=0.2),
            medianprops=dict(color='white', linewidth=2),
            boxprops=dict(facecolor=PAL[rt], edgecolor=EDGE[rt], linewidth=1.5),
            whiskerprops=dict(color=EDGE[rt], linewidth=1.3),
            capprops=dict(color=EDGE[rt], linewidth=1.8),
        )
        med = data.median()
        ax.text(j, med + 0.12, f'{med:.1f}',
                ha='center', fontsize=9, fontweight='bold', color='#333333')

    ax.axhline(0, color='#D4908E', linestyle='--', linewidth=1.2, alpha=0.65)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['comment', 'review'], fontsize=11)
    ax.set_title(genre, fontsize=13, fontweight='bold',
                 color=GENRE_COLORS[genre])
    ax.set_ylabel('감성 점수', fontsize=10)
    ax.yaxis.grid(True, color='#E8E8E8', linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[5].set_visible(False)
plt.suptitle('장르별 comment vs review 감성 점수 분포', fontsize=16, fontweight='bold')
plt.tight_layout()
save('5_장르별_comment_vs_review.png')

print('\n========== 완료 ==========')
print(f'결과 저장 위치: {OUTPUT_DIR}/')
