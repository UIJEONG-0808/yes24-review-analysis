#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YES24 모바일 한줄평 + 리뷰 + 별점 크롤러 (다중 도서)
======================================================
GOODS_LIST 에 도서번호를 넣으면 순서대로 모두 수집한다.
- 책마다 개별 CSV 즉시 저장(중간에 끊겨도 그때까지 안전)
- 마지막에 전체를 합친 CSV 도 생성
- goods_no / book_title 열 포함

수집 항목
    review_id, goods_no, book_title, review_type(comment/review),
    rating_10(10점), rating_5(5점환산), content, author(마스킹),
    grade(YES마니아 등급), date, helpful(공감), format(종이책/eBook),
    is_purchase, is_spoiler

구매/전체 필터
    한줄평: COMMENT_SCOPE  'all'(전체) | 'purchase'(구매한줄평)
    리뷰  : REVIEW_SCOPE   'all'(전체) | 'purchase'(구매리뷰)
    전체가 구매의 상위집합이고 is_purchase 열로 나중에 구매분만 분리 가능.
"""



import re
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ── 설정 ─────────────────────────────────────────────────
# 수집할 도서번호 목록 (URL 끝 숫자). 한 권만 할 거면 원소 하나만.
GOODS_LIST = [
    '116586303', '102791425', '118578901', # 급류, 홍학의 자리, 구의 증명
    '102360203', '402246', '37922672',  # 오늘 밤, 세계에서 이 사랑이 사라진다 해도, 오만과 편견, 너의 췌장을 먹고싶어
    '103026125', '116586056', '91065309', # 지구 끝의 온실, 나미야 잡화점의 기적, 달러구트 꿈 백화점
    '8759796', '1387488', '108422348',  # 모순, 인간실격, 채식주의자
    '45353675', '89913383', '53802600' # 용의자 X의 헌신, 칵테일러브좀비, 봉제인형 살인사건
    
    # "여기에", "도서번호", "추가",
]

HEADLESS      = True
COMMENT_SCOPE = "all"        # 'all' = 전체 한줄평 / 'purchase' = 구매한줄평
REVIEW_SCOPE  = "all"        # 'all' = 리뷰전체   / 'purchase' = 구매리뷰
MAX_CLICKS    = 600          # 더보기 최대 클릭(안전장치)
GROW_WAIT_MS  = 8000         # 클릭 후 항목 증가 대기 한도(ms)
STALL_LIMIT   = 4            # 증가 없음이 연속 N회면 종료
SLEEP_BETWEEN = 2.0          # 책 사이 대기(초) - 과도한 요청 방지
OUTPUT_DIR    = Path("yes24_reviews_output")
OUTPUT_DIR.mkdir(exist_ok=True)

MOBILE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) "
             "AppleWebKit/605.1.15 (KHTML, like Gecko) "
             "Version/16.5 Mobile/15E148 Safari/604.1")

RATING_RE = re.compile(r"total_rating_(\d+)")


def make_pages(no):
    return [
        {"type": "comment", "label": "한줄평",
         "url": f"https://m.yes24.com/Goods/OneComment/{no}",
         "container": "div.revwGrp#oneCommentList", "more": "#oneCommentMoreBtn",
         "more_fn": "oneComment.getMore()"},
        {"type": "review", "label": "리뷰",
         "url": f"https://m.yes24.com/Goods/Review/{no}",
         "container": "div.revwGrp#reviewList", "more": "#reviewMoreBtn",
         "more_fn": "review.getMore()"},
    ]


# ── Jupyter 호환 실행 헬퍼 ───────────────────────────────
def run_in_thread(fn):
    box = {}
    def _run():
        if sys.platform == "win32":
            import asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            box["v"] = fn()
        except Exception as e:
            box["e"] = e
    t = threading.Thread(target=_run); t.start(); t.join()
    if "e" in box:
        raise box["e"]
    return box.get("v")


# ── 파서 ─────────────────────────────────────────────────
def parse_set(s, review_type):
    cont = s.select_one("div.revw_cont")
    content = cont.get_text("\n", strip=True) if cont else ""
    if not content:
        return None  # display:none 더미 등 제외

    r10 = None
    tr = s.select_one("span.total_rating")
    if tr:
        for c in tr.get("class", []):
            m = RATING_RE.match(c)
            if m:
                r10 = int(m.group(1)); break
    r5 = round(r10 / 2, 1) if r10 is not None else None

    a = s.select_one(".revw_id .id_txt")
    author = a.get_text(strip=True) if a else ""
    d = s.select_one(".revw_date .date")
    date = d.get_text(strip=True).rstrip(".") if d else ""

    helpful = 0
    h = s.select_one("em.num[name=recommend_count]")
    if h:
        digits = re.sub(r"\D", "", h.get_text())
        helpful = int(digits) if digits else 0

    fmt = ""
    is_purchase = False
    for ic in s.select("span.iconC"):
        t = ic.get_text(strip=True)
        if t in ("종이책", "eBook"):
            fmt = t
        if t == "구매":
            is_purchase = True

    grade = ""
    g = s.select_one("strong.id_ico")
    if g:
        gm = re.search(r"YES마니아\s*:\s*(\S+)", g.get_text())
        grade = gm.group(1) if gm else ""

    is_spoiler = bool(s.find_parent("div", class_="spoiler")) \
        or ("spoiler" in (s.get("class") or []))

    rid = s.get("data-comment-seq") or s.get("data-review-seq") or ""

    return {
        "review_id": rid, "review_type": review_type,
        "rating_10": r10, "rating_5": r5,
        "content": content, "author": author, "grade": grade,
        "date": date, "helpful": helpful,
        "format": fmt, "is_purchase": is_purchase, "is_spoiler": is_spoiler,
    }


def parse_container(html, container_sel, review_type):
    grp = BeautifulSoup(html, "lxml").select_one(container_sel)
    if not grp:
        return []
    rows = [parse_set(s, review_type) for s in grp.select("div.revwSet")]
    return [r for r in rows if r]


# ── 더보기 끝까지 (안쪽 a 클릭 + getMore 폴백) ───────────
def load_all(page, cfg):
    container = cfg["container"]
    count_js = ("() => document.querySelectorAll('"
                + container + " div.revwSet div.revw_cont').length")
    visible_js = ("() => { const b = document.querySelector('" + cfg["more"]
                  + "'); if (!b) return false; const s = getComputedStyle(b);"
                  " return s.display !== 'none' && s.visibility !== 'hidden'; }")

    def wait_more_visible(ms=5000):
        w = 0
        while w < ms:
            if page.evaluate(visible_js):
                return True
            page.wait_for_timeout(300); w += 300
        return False

    stall = 0
    for i in range(MAX_CLICKS):
        if not wait_more_visible(5000):
            break
        before = page.evaluate(count_js)

        clicked = False
        try:
            a = page.locator(f"{cfg['more']} a").first
            if a.count() > 0:
                a.scroll_into_view_if_needed(timeout=2000)
                a.click(timeout=3000)
                clicked = True
        except Exception:
            pass
        if not clicked:
            try:
                page.evaluate(cfg["more_fn"])
            except Exception:
                pass

        grew = False; waited = 0
        while waited < GROW_WAIT_MS:
            page.wait_for_timeout(300); waited += 300
            if page.evaluate(count_js) > before:
                grew = True; break
        if grew:
            stall = 0
        else:
            stall += 1
            if stall >= STALL_LIMIT:
                break

        if (i + 1) % 10 == 0:
            print(f"      더보기 {i+1}회 / 누적 {page.evaluate(count_js)}개")

    return page.evaluate(count_js)


def get_title(page):
    try:
        t = page.title()                       # 예: "[한줄평] 급류 - 예스24"
        t = re.sub(r"^\[[^\]]*\]\s*", "", t)
        t = re.sub(r"\s*-\s*예스24.*$", "", t)
        return t.strip()
    except Exception:
        return ""


def collect_page(page, cfg):
    page.goto(cfg["url"], wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(1500)

    if cfg["type"] == "comment" and COMMENT_SCOPE == "all":
        try:
            page.evaluate(
                "var c=document.getElementById('chkOrderOneComment');"
                "if(c&&c.checked){c.checked=false;oneComment.changePurchase(c);}")
            page.wait_for_timeout(1800)
        except Exception:
            pass
    elif cfg["type"] == "review":
        try:
            page.evaluate("review.changeType(1)" if REVIEW_SCOPE == "all"
                          else "review.changeType(1, 1)")
            page.wait_for_timeout(1800)
        except Exception:
            pass

    load_all(page, cfg)
    return parse_container(page.content(), cfg["container"], cfg["type"])


def safe_name(s):
    return re.sub(r'[\\/:*?"<>|]', "_", s)[:40].strip() or "untitled"


def crawl(goods_list):
    all_rows = []
    summary = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(
            user_agent=MOBILE_UA, viewport={"width": 390, "height": 844},
            locale="ko-KR", timezone_id="Asia/Seoul",
            is_mobile=True, has_touch=True)
        ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
        page = ctx.new_page()

        for idx, no in enumerate(goods_list, 1):
            print(f"\n{'#'*54}\n[{idx}/{len(goods_list)}] 도서 {no}")
            book_title, rows = "", []
            try:
                for j, cfg in enumerate(make_pages(no)):
                    print(f"  - {cfg['label']}")
                    if cfg["type"] == "comment":
                        # 제목은 첫 페이지에서 한 번
                        page.goto(cfg["url"], wait_until="domcontentloaded", timeout=30000)
                        page.wait_for_timeout(800)
                        book_title = get_title(page)
                        print(f"    도서명: {book_title}")
                    part = collect_page(page, cfg)
                    for r in part:
                        r["goods_no"] = no
                        r["book_title"] = book_title
                    rows.extend(part)
                    print(f"    {cfg['label']} {len(part)}개")
            except Exception as e:
                print(f"  ! 도서 {no} 수집 중 오류: {e} (다음 도서로 진행)")

            # 책별 CSV 즉시 저장
            if rows:
                dfb = pd.DataFrame(rows).drop_duplicates(
                    subset=["review_id", "review_type"]).reset_index(drop=True)
                fn = OUTPUT_DIR / f"yes24_{no}_{safe_name(book_title)}_{ts}.csv"
                dfb.to_csv(fn, index=False, encoding="utf-8-sig")
                print(f"  → 저장: {fn.name}  ({len(dfb)}건)")
                all_rows.extend(rows)
                summary.append((no, book_title, len(dfb)))
            else:
                summary.append((no, book_title, 0))

            if idx < len(goods_list):
                time.sleep(SLEEP_BETWEEN)

        browser.close()
    return all_rows, summary, ts


def main():
    all_rows, summary, ts = run_in_thread(lambda: crawl(GOODS_LIST))

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("\n수집 0건.")
        return df

    df["rating_10"] = pd.to_numeric(df["rating_10"], errors="coerce")
    df["rating_5"] = pd.to_numeric(df["rating_5"], errors="coerce")
    df = df.drop_duplicates(
        subset=["goods_no", "review_id", "review_type"]).reset_index(drop=True)

    cols = ["goods_no", "book_title", "review_id", "review_type",
            "rating_10", "rating_5", "content", "author", "grade",
            "date", "helpful", "format", "is_purchase", "is_spoiler"]
    df = df[cols]

    combined = OUTPUT_DIR / f"yes24_ALL_{len(GOODS_LIST)}books_{ts}.csv"
    df.to_csv(combined, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 54)
    print(f"전체 합본 저장 → {combined}  (총 {len(df)}건)")
    print(f"(한줄평 scope={COMMENT_SCOPE}, 리뷰 scope={REVIEW_SCOPE})")
    print("\n도서별 수집:")
    for no, title, n in summary:
        print(f"  {no}  {title[:24]:<24}  {n}건")
    print(f"\n종류별:\n{df['review_type'].value_counts().to_string()}")
    print(f"\n별점(10점) 분포:\n{df['rating_10'].value_counts(dropna=False).sort_index().to_string()}")
    print(f"\n별점 결측: {df['rating_10'].isna().sum()}건 / 스포일러: {df['is_spoiler'].sum()}건")
    return df


if __name__ == "__main__":
    main()
