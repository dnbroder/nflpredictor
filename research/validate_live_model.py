"""
Live Model Validation Against Historic 2H Lines
================================================
Validates the Q1/H1 pace signal against actual second half betting lines
from SBR historical data (2013-2019).

This answers the key question:
  "If I bet the 2H over/under based on Q1 pace, what's my edge?"

The live model thesis:
  - Pre-game model fires (cycle state, PIC features)
  - Q1 plays out — read Q1 combined score / closing_total
  - Compare Q1 pace to 2H line to find market inefficiency
  - Bet the 2H over/under accordingly

Usage:
    python research/validate_live_model.py

Input:
    data/odds/sbr_closing_lines_2013_2019.parquet
    (must be re-generated with updated sbr_ingest.py that captures 2H lines)
"""

import pandas as pd
import numpy as np


def validate_q1_pace_vs_h2_line(df):
    """
    Core validation: does Q1 pace predict 2H over/under outcomes?

    Q1 pace = q1_combined / closing_total
    2H result = h2_combined vs h2_line

    Expected finding:
      - High Q1 pace (>25%) → game is scoring at above-expected rate
        → 2H over hits at elevated rate (market underpriced the 2H)
      - Low Q1 pace (<10%) → game is scoring below expected
        → 2H under hits at elevated rate

    This is the live betting edge: Q1 tells you the market
    mispriced the 2H total.
    """
    # Filter to games with complete data
    valid = df.dropna(subset=['q1_pct_of_total', 'h2_over', 'h2_line', 'closing_total'])
    print(f"\nGames with complete Q1 + 2H line data: {len(valid)}")

    if len(valid) < 100:
        print("⚠  Insufficient data — re-run sbr_ingest.py with updated script first")
        return

    baseline_h2_over = valid['h2_over'].mean()
    print(f"Baseline 2H over rate: {baseline_h2_over:.3f}")
    print(f"Average 2H line: {valid['h2_line'].mean():.1f}")
    print(f"Average closing total: {valid['closing_total'].mean():.1f}")
    print(f"Average h2/total ratio: {(valid['h2_line'] / valid['closing_total']).mean():.3f}")

    print("\n" + "="*65)
    print("Q1 PACE → 2H OVER RATE")
    print("="*65)
    print(f"{'Q1 Bucket':<15} {'Games':>6} {'2H Over%':>10} {'vs Baseline':>12} {'Edge':>8}")
    print("-"*65)

    buckets = [
        ('<10%',   valid['q1_pct_of_total'] < 0.10),
        ('10-15%', (valid['q1_pct_of_total'] >= 0.10) & (valid['q1_pct_of_total'] < 0.15)),
        ('15-20%', (valid['q1_pct_of_total'] >= 0.15) & (valid['q1_pct_of_total'] < 0.20)),
        ('20-25%', (valid['q1_pct_of_total'] >= 0.20) & (valid['q1_pct_of_total'] < 0.25)),
        ('25-30%', (valid['q1_pct_of_total'] >= 0.25) & (valid['q1_pct_of_total'] < 0.30)),
        ('>30%',   valid['q1_pct_of_total'] >= 0.30),
    ]

    for label, mask in buckets:
        group = valid[mask]
        if len(group) < 10:
            continue
        rate = group['h2_over'].mean()
        edge = rate - baseline_h2_over
        print(f"{label:<15} {len(group):>6} {rate*100:>9.1f}% {edge*100:>+11.1f}% {'★' if abs(edge) > 0.08 else ''}")

    print("\n" + "="*65)
    print("H1 PACE → 2H OVER RATE (halftime signal)")
    print("="*65)
    print(f"{'H1 Bucket':<15} {'Games':>6} {'2H Over%':>10} {'vs Baseline':>12} {'Edge':>8}")
    print("-"*65)

    h1_buckets = [
        ('<30%',   valid['h1_pct_of_total'] < 0.30),
        ('30-40%', (valid['h1_pct_of_total'] >= 0.30) & (valid['h1_pct_of_total'] < 0.40)),
        ('40-50%', (valid['h1_pct_of_total'] >= 0.40) & (valid['h1_pct_of_total'] < 0.50)),
        ('50-60%', (valid['h1_pct_of_total'] >= 0.50) & (valid['h1_pct_of_total'] < 0.60)),
        ('60-70%', (valid['h1_pct_of_total'] >= 0.60) & (valid['h1_pct_of_total'] < 0.70)),
        ('>70%',   valid['h1_pct_of_total'] >= 0.70),
    ]

    for label, mask in h1_buckets:
        group = valid[mask]
        if len(group) < 10:
            continue
        rate = group['h2_over'].mean()
        edge = rate - baseline_h2_over
        print(f"{label:<15} {len(group):>6} {rate*100:>9.1f}% {edge*100:>+11.1f}% {'★' if abs(edge) > 0.08 else ''}")

    print("\n" + "="*65)
    print("LIVE MODEL BETTING SIMULATION")
    print("Bet 2H over when Q1 pace >= 25% | Bet 2H under when Q1 pace <= 10%")
    print("="*65)

    # Simulate flat betting at -110 (need 52.4% to break even)
    bet_over = valid[valid['q1_pct_of_total'] >= 0.25]
    bet_under = valid[valid['q1_pct_of_total'] <= 0.10]

    if len(bet_over) > 0:
        over_win_rate = bet_over['h2_over'].mean()
        over_roi = (over_win_rate * 100 - (1 - over_win_rate) * 110) / 110 * 100
        print(f"\nBet 2H OVER (Q1 >= 25%): {len(bet_over)} games")
        print(f"  Win rate:  {over_win_rate*100:.1f}%")
        print(f"  ROI @-110: {over_roi:+.1f}%")

    if len(bet_under) > 0:
        under_win_rate = 1 - bet_under['h2_over'].mean()
        under_roi = (under_win_rate * 100 - (1 - under_win_rate) * 110) / 110 * 100
        print(f"\nBet 2H UNDER (Q1 <= 10%): {len(bet_under)} games")
        print(f"  Win rate:  {under_win_rate*100:.1f}%")
        print(f"  ROI @-110: {under_roi:+.1f}%")

    # Combined strategy
    total_bets = len(bet_over) + len(bet_under)
    if total_bets > 0:
        over_profit = bet_over['h2_over'].sum() * 100 - (1 - bet_over['h2_over']).sum() * 110
        under_profit = (1 - bet_under['h2_over']).sum() * 100 - bet_under['h2_over'].sum() * 110
        total_profit = over_profit + under_profit
        total_wagered = total_bets * 110
        combined_roi = total_profit / total_wagered * 100
        print(f"\nCombined strategy: {total_bets} total bets")
        print(f"  Total profit: ${total_profit:,.0f} (per $110 unit)")
        print(f"  Combined ROI: {combined_roi:+.1f}%")

    print("\n" + "="*65)
    print("REGRESSION TO MEAN: Does Q1 pace predict remaining scoring?")
    print("="*65)
    buckets_simple = [
        ('Low Q1 (<=10%)',   valid['q1_pct_of_total'] <= 0.10),
        ('Mid Q1 (10-25%)',  (valid['q1_pct_of_total'] > 0.10) & (valid['q1_pct_of_total'] <= 0.25)),
        ('High Q1 (>25%)',   valid['q1_pct_of_total'] > 0.25),
    ]
    for label, mask in buckets_simple:
        group = valid[mask]
        if len(group) < 10:
            continue
        avg_q1 = group['q1_combined'].mean()
        avg_h2 = group['h2_combined'].mean()
        avg_total = group['combined_score'].mean()
        print(f"\n{label} (n={len(group)}):")
        print(f"  Avg Q1 score:      {avg_q1:.1f}")
        print(f"  Avg H2 score:      {avg_h2:.1f}")
        print(f"  Avg total:         {avg_total:.1f}")
        print(f"  H2/Total ratio:    {avg_h2/avg_total:.3f}")


def main():
    import os
    path = 'data/odds/sbr_closing_lines_2013_2019.parquet'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        print("Run sbr_ingest.py first (updated version with 2H line capture)")
        return

    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} games from {path}")
    print(f"Columns: {list(df.columns)}")

    if 'h2_line' not in df.columns:
        print("\n⚠  h2_line not in parquet — re-run sbr_ingest.py with updated script")
        print("The updated script is in research/sbr_ingest.py")
        return

    h2_coverage = df['h2_line'].notna().sum()
    print(f"Games with 2H line: {h2_coverage}/{len(df)} ({h2_coverage/len(df)*100:.0f}%)")

    validate_q1_pace_vs_h2_line(df)


if __name__ == '__main__':
    main()