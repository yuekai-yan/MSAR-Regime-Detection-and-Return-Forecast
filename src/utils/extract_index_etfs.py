"""
Extract index ETFs from ETFs.xlsx and export results (CSV + XLSX).

Added:
  - Exclude rows where 'Test Issue' == 'Y' (default on; toggle with CLI).

Requirements: pandas, openpyxl
Usage:
  python extract_index_etfs.py --src ETFs.xlsx --outdir ./output
"""

import argparse
import re
from pathlib import Path
import pandas as pd

def load_etf_sheet(src: Path) -> pd.DataFrame:
    """Read ETFs.xlsx whose first row contains header names."""
    raw = pd.read_excel(src, dtype=str).fillna("")
    header = list(raw.iloc[0])
    df = raw.iloc[1:].copy()
    df.columns = header

    # Ensure columns exist
    for col in ["Nasdaq Traded","Symbol","Security Name","Listing Exchange",
                "Market Category","ETF","Round Lot Size","Test Issue",
                "Financial Status","CQS Symbol","NASDAQ Symbol","NextShares"]:
        if col not in df.columns:
            df[col] = ""

    # Normalize key fields
    df["Symbol"] = df["Symbol"].str.strip().str.upper()
    df["Security Name"] = df["Security Name"].str.strip()
    df["ETF"] = df["ETF"].str.strip().str.upper()
    df["Test Issue"] = df["Test Issue"].str.strip().str.upper()
    df["NextShares"] = df["NextShares"].str.strip().str.upper()
    return df

def classify_index_etfs(df: pd.DataFrame, exclude_test: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (index_etfs, maybe_needs_review, non_index)."""

    # Base set: ETF=Y; Exclude Test Issues as needed; Exclude NextShares
    base = df[df["ETF"] == "Y"].copy()
    if exclude_test:
        base = base[base["Test Issue"] != "Y"]
    base = base[base["NextShares"] != "Y"]

    # Index ETFs identification rules (including + excluding keywords)
    include_pat = re.compile(
        r"(Index|Indices|Benchmark|Composite|S&P|Dow Jones|NASDAQ-100|Russell|MSCI|FTSE|CRSP|Bloomberg|Barclays|ICE|iBoxx|Solactive|Morningstar|CSI|CBOE|NYSE)",
        re.IGNORECASE
    )
    exclude_pat = re.compile(
        r"(Leveraged|Ultra|\b[23]x\b|-2x|-3x|Inverse|Short|Bear|Futures|ETN|Commodity Trust|Bitcoin|Crypto|Oil|Gold Trust|Silver Trust|Active|Managed|AdvisorShares|ARK|ETMF|NextShares|Covered Call|BuyWrite|Option Income|Buffer|Defined Outcome)",
        re.IGNORECASE
    )

    name = base["Security Name"]
    base["__hit_include"] = name.str.contains(include_pat, na=False)
    base["__hit_exclude"] = name.str.contains(exclude_pat, na=False)
    base["is_index_etf"] = base["__hit_include"] & ~base["__hit_exclude"]

    index_etfs = base[base["is_index_etf"]].copy()
    maybe_idx = base[(~base["is_index_etf"]) & base["__hit_include"]].copy()
    non_index = base[(~base["is_index_etf"]) & (~base["__hit_include"])].copy()
    return index_etfs, maybe_idx, non_index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Path to ETFs.xlsx")
    ap.add_argument("--outdir", type=Path, default=Path("./output"), help="Directory to save results")
    ap.add_argument("--include-test", action="store_true",
                    help="Do NOT exclude rows where 'Test Issue'=='Y' (default is to exclude).")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_etf_sheet(args.src)
    index_etfs, maybe_idx, non_index = classify_index_etfs(df, exclude_test=not args.include_test)

    cols = ["Symbol","Security Name","Listing Exchange","Market Category","ETF",
            "Test Issue","Financial Status","CQS Symbol","NASDAQ Symbol"]

    # Export
    index_etfs[cols].to_csv(args.outdir / "index_etfs.csv", index=False)
    index_etfs[cols].to_excel(args.outdir / "index_etfs.xlsx", index=False)
    maybe_idx[cols].to_csv(args.outdir / "index_etfs_needs_review.csv", index=False)

    # Summary
    total = len(df)
    etf_base = len(df[df["ETF"] == "Y"])
    if args.include_test:
        base_after = len(df[(df["ETF"] == "Y") & (df["NextShares"] != "Y")])
    else:
        base_after = len(df[(df["ETF"] == "Y") & (df["Test Issue"] != "Y") & (df["NextShares"] != "Y")])

    print(f"Total rows: {total}")
    print(f"ETF base (ETF=Y): {etf_base}")
    print(f"After exclusions (TestIssue={'excluded' if not args.include_test else 'kept'}, NextShares=excluded): {base_after}")
    print(f"Index ETFs: {len(index_etfs)}")
    print(f"Needs review: {len(maybe_idx)}")
    print(f"Non-index ETFs: {len(non_index)}")
    print(f"Saved to: {args.outdir.resolve()}")

if __name__ == "__main__":
    main()