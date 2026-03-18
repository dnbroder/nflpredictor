"""
SBR Historical Closing Lines Ingestion
=======================================
Scrapes closing lines from sportsbookreviewsonline.com for NFL seasons 2013-2019.

Usage:
  1. Save each season's raw SBR text to sbr_raw/sbr_{year}.txt
     (copy-paste table from sportsbookreviewsonline.com/scoresoddsarchives/nfl/)
  2. Run: python research/sbr_ingest.py
  3. Output: data/odds/sbr_closing_lines_2013_2019.parquet
  4. Upload to S3: nfl-predictor-bucket/data/odds/sbr_closing_lines_2013_2019.parquet

Seasons covered: 2013-2019 (7 seasons, ~1,870 games)
This expands training data from ~3,520 rows to ~5,100+ rows.

pic_history_start should be set to 2012 when training with these seasons.
"""
import pandas as pd
import os
import boto3

# SBR team name → nfl_data_py abbreviation
# Covers all name variations 2013-2019 including franchise moves
SBR_TO_NFL = {
    # AFC East
    'Buffalo': 'BUF', 'Miami': 'MIA', 'NewEngland': 'NE', 'NYJets': 'NYJ',
    # AFC North
    'Baltimore': 'BAL', 'Cincinnati': 'CIN', 'Cleveland': 'CLE', 'Pittsburgh': 'PIT',
    # AFC South
    'Houston': 'HOU', 'Indianapolis': 'IND', 'Jacksonville': 'JAX', 'Tennessee': 'TEN',
    # AFC West
    'Denver': 'DEN', 'KansasCity': 'KC', 'Oakland': 'LV', 'SanDiego': 'LAC',
    # NFC East
    'Dallas': 'DAL', 'NYGiants': 'NYG', 'Philadelphia': 'PHI', 'Washington': 'WAS',
    # NFC North
    'Chicago': 'CHI', 'Detroit': 'DET', 'GreenBay': 'GB', 'Minnesota': 'MIN',
    # NFC South
    'Atlanta': 'ATL', 'Carolina': 'CAR', 'NewOrleans': 'NO', 'TampaBay': 'TB',
    # NFC West
    'Arizona': 'ARI', 'LosAngeles': 'LAR', 'SanFrancisco': 'SF', 'Seattle': 'SEA',
    # Post-move names
    'LARams': 'LAR', 'LAChargers': 'LAC',
    # Alternate spellings SBR uses
    'NewYorkJets': 'NYJ', 'NewYorkGiants': 'NYG', 'KansasCity': 'KC',
    'GreenBay': 'GB', 'NewEngland': 'NE', 'NewOrleans': 'NO', 'TampaBay': 'TB',
    'SanFrancisco': 'SF', 'SanDiego': 'LAC', 'LasVegas': 'LV',
}


def parse_sbr_season(text, season_year):
    """
    Parse raw SBR paste for one NFL season.

    Column layout (tab-separated):
      Date(0) Rot(1) VH(2) Team(3) Q1(4) Q2(5) Q3(6) Q4(7) Final(8) Open(9) Close(10) ML(11) 2H(12)

    Games come in consecutive pairs: Visitor row then Home row.
    Closing total = whichever numeric value in Open or Close is between 30 and 80.
    Values of 'pk' mean pick-em (spread = 0), not a total — skip those for total detection.
    """
    rows = []
    lines = text.strip().split('\n')

    # Clean and filter to data rows only
    clean = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip header lines
        if any(line.startswith(h) for h in ['NFL', 'Date', 'Dat', 'Rot', 'Team']):
            continue
        parts = line.split('\t')
        if len(parts) >= 9:
            clean.append([p.strip() for p in parts])

    j = 0
    skipped = 0
    while j < len(clean) - 1:
        v, h = clean[j], clean[j + 1]
        try:
            away_sbr = v[3]
            home_sbr = h[3]

            # Validate row pair — both should have team names (no digits-only team)
            if away_sbr.isdigit() or home_sbr.isdigit():
                j += 1
                continue

            away_score = float(v[8])
            home_score = float(h[8])
            combined = away_score + home_score

            # Find closing total: numeric value 30 < x < 80 in Close(10) or Open(9)
            closing_total = None
            for row in [v, h]:
                for idx in [10, 9]:
                    try:
                        raw = row[idx].replace('pk', '').strip()
                        if not raw:
                            continue
                        val = float(raw)
                        if 30 < val < 80:
                            closing_total = val
                            break
                    except (ValueError, IndexError):
                        pass
                if closing_total is not None:
                    break

            if closing_total is not None:
                away_nfl = SBR_TO_NFL.get(away_sbr)
                home_nfl = SBR_TO_NFL.get(home_sbr)
                rows.append({
                    'season':         season_year,
                    'away_team':      away_nfl,
                    'home_team':      home_nfl,
                    'away_team_sbr':  away_sbr,
                    'home_team_sbr':  home_sbr,
                    'away_score':     int(away_score),
                    'home_score':     int(home_score),
                    'combined_score': combined,
                    'closing_total':  closing_total,
                    'over':           int(combined > closing_total),
                })
            else:
                skipped += 1

        except Exception:
            skipped += 1

        j += 2

    df = pd.DataFrame(rows)
    matched = len(df.dropna(subset=['away_team', 'home_team']))
    print(f"  {season_year}: {matched} games matched, {len(df) - matched} unmapped teams, {skipped} skipped")

    # Report unmapped team names so they can be added to SBR_TO_NFL
    if len(df) > 0:
        unmapped_df = df[df['away_team'].isna() | df['home_team'].isna()]
        if len(unmapped_df) > 0:
            names = set(unmapped_df['away_team_sbr'].tolist() + unmapped_df['home_team_sbr'].tolist())
            names = {n for n in names if n and not n.isdigit()}
            if names:
                print(f"    ⚠  Unmapped SBR names: {sorted(names)}")
                print(f"    Add to SBR_TO_NFL and re-run")

    return df.dropna(subset=['away_team', 'home_team'])


def main():
    raw_dir = 'sbr_raw'
    all_dfs = []

    # 2013-2019: 7 seasons, ~1,870 games
    # With 2020-2022 Odds API data: ~2,700 total games → ~5,400 training rows
    seasons = [2013, 2014, 2015, 2016, 2017, 2018, 2019]

    print(f"Processing {len(seasons)} seasons: {seasons}")
    for season in seasons:
        path = os.path.join(raw_dir, f'sbr_{season}.txt')
        if not os.path.exists(path):
            print(f"  {season}: file missing ({path}) — skipping")
            continue
        with open(path) as f:
            text = f.read()
        df = parse_sbr_season(text, season)
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        print("\nNo data found. Add season files to sbr_raw/ and re-run.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} games across {combined['season'].nunique()} seasons")
    print(combined['season'].value_counts().sort_index().to_string())

    # Quick validation — over rate should be ~47-50%
    over_rate = combined['over'].mean()
    print(f"\nOver rate: {over_rate:.3f} (expect 0.47-0.50 vs closing line)")
    if not 0.40 < over_rate < 0.60:
        print("  ⚠  Over rate out of range — check data quality")

    # Determine output filename based on season range
    min_season = combined['season'].min()
    max_season = combined['season'].max()
    out_name = f'sbr_closing_lines_{min_season}_{max_season}.parquet'

    os.makedirs('data/odds', exist_ok=True)
    out_path = f'data/odds/{out_name}'
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Upload to S3
    bucket = 'nfl-predictor-bucket'
    s3_key = f'data/odds/{out_name}'
    try:
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.upload_file(out_path, bucket, s3_key)
        print(f"Uploaded: s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"S3 upload failed: {e}")
        print(f"  Run manually: aws s3 cp {out_path} s3://{bucket}/{s3_key}")

    print(f"""
Next steps:
  1. Upload {out_path} to S3 via AWS Console if auto-upload failed
  2. Update nfl_data.py load_sbr_closing_lines() to use {out_name}
  3. Retrain with: train_start=2013&train_end=2022&pic_history_start=2012
""")


if __name__ == '__main__':
    main()