import time
from pathlib import Path
import warnings
import requests
from bs4 import BeautifulSoup

import pandas as pd
import yfinance as yf

# ---- 静音所有 warning ----
warnings.filterwarnings("ignore")

# ---- 基本配置 ----
INPUT_FILE  = "stocks.csv"
OUTPUT_FILE = "stock_with_industry.csv"
SYMBOL_COL = "Symbol"
NAME_COL = "Security Name"
SLEEP_BASE = 0.6
MAX_RETRY = 3

def get_sector_industry_yf(symbol: str):
    """
    用 yfinance 通过 symbol 抓取 sector / industry
    返回 (sector, industry) 或 (None, None)
    """
    t = (symbol or "").strip()
    if not t:
        return None, None
    t = t.replace(" ", "").replace(".", "-")

    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            tk = yf.Ticker(t)
            info = tk.get_info()
            sector = info.get("sector")
            industry = info.get("industry")
            if sector or industry:
                return sector, industry
        except Exception as e:
            last_err = e
            time.sleep(SLEEP_BASE * attempt)
    
    return None, None

def get_sector_industry_stockanalysis(symbol: str):
    """
    从 stockanalysis.com 抓取 sector / industry
    返回 (sector, industry) 或 (None, None)
    """
    try:
        url = f"https://stockanalysis.com/stocks/{symbol.lower()}/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 尝试找到包含Sector和Industry的表格或div
            sector = None
            industry = None
            
            # 查找包含"Sector"和"Industry"的文本
            for tag in soup.find_all(['td', 'div', 'span']):
                text = tag.get_text(strip=True)
                if 'Sector' in text:
                    next_sibling = tag.find_next_sibling()
                    if next_sibling:
                        sector = next_sibling.get_text(strip=True)
                if 'Industry' in text:
                    next_sibling = tag.find_next_sibling()
                    if next_sibling:
                        industry = next_sibling.get_text(strip=True)
            
            if sector or industry:
                print(f"    ✓ StockAnalysis找到: Sector={sector}, Industry={industry}")
                return sector, industry
                
    except Exception as e:
        print(f"    StockAnalysis抓取失败: {e}")
    
    return None, None

def main():
    in_path = Path(INPUT_FILE)
    if not in_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_FILE}")

    df = pd.read_csv(in_path)
    
    if SYMBOL_COL not in df.columns:
        raise ValueError(f"缺少必需的列：{SYMBOL_COL}")

    if "Industry" not in df.columns:
        df["Industry"] = pd.NA
    if "Source" not in df.columns:
        df["Source"] = pd.NA

    cache = {}
    ok = fail = from_stockanalysis = 0
    n = len(df)

    for idx, row in df.iterrows():
        sym = str(row[SYMBOL_COL]).strip()
        
        print(f"[{idx+1}/{n}] 处理 {sym}...", flush=True)
        
        if sym in cache:
            sector, industry, source = cache[sym]
        else:
            # 方法1: 直接用Symbol在Yahoo Finance查询
            sector, industry = get_sector_industry_yf(sym)
            source = "Yahoo Finance"
            
            # # 方法2: 如果Yahoo找不到，试试StockAnalysis
            # if not sector and not industry:
            #     print(f"  Yahoo未找到，尝试StockAnalysis...")
            #     sector, industry = get_sector_industry_stockanalysis(sym)
            #     if sector or industry:
            #         source = "StockAnalysis"
            #         from_stockanalysis += 1
            #     else:
            #         source = "N/A"
            
            cache[sym] = (sector, industry, source)
            time.sleep(SLEEP_BASE)

        # 写入结果
        df.loc[idx, "Industry"] = sector if sector else "N/A"
        df.loc[idx, "Source"] = source
        
        if sector:
            ok += 1
        else:
            fail += 1

    # 保存结果
    out_path = Path(OUTPUT_FILE)
    df.to_csv(out_path, index=False)
    print(f"\n完成！")
    print(f"  成功: {ok} 条")
    print(f"  失败: {fail} 条")
    print(f"  通过StockAnalysis找到: {from_stockanalysis} 条")
    print(f"  已输出: {out_path.resolve()}")

if __name__ == "__main__":
    main()