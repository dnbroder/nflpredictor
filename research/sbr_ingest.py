"""
SBR Historical Closing Lines Ingestion
Usage: 
  1. Save each season's raw SBR text (copy-paste from website) to:
       sbr_raw/sbr_2016.txt, sbr_raw/sbr_2017.txt, sbr_raw/sbr_2018.txt, sbr_raw/sbr_2019.txt
  2. Run: python sbr_ingest.py
  3. Output: data/odds/sbr_closing_lines_2016_2019.parquet
     Upload this to S3: nfl-predictor-bucket/data/odds/sbr_closing_lines_2016_2019.parquet
"""
import pandas as pd
import os
import boto3

SBR_TO_NFL = {
    'Carolina': 'CAR', 'Denver': 'DEN', 'TampaBay': 'TB', 'Atlanta': 'ATL',
    'Minnesota': 'MIN', 'Tennessee': 'TEN', 'Cleveland': 'CLE', 'Philadelphia': 'PHI',
    'Cincinnati': 'CIN', 'NYJets': 'NYJ', 'Oakland': 'LV', 'NewOrleans': 'NO',
    'SanDiego': 'LAC', 'KansasCity': 'KC', 'Buffalo': 'BUF', 'Baltimore': 'BAL',
    'Chicago': 'CHI', 'Houston': 'HOU', 'GreenBay': 'GB', 'Jacksonville': 'JAX',
    'Miami': 'MIA', 'Seattle': 'SEA', 'NYGiants': 'NYG', 'Dallas': 'DAL',
    'Detroit': 'DET', 'Indianapolis': 'IND', 'NewEngland': 'NE', 'Arizona': 'ARI',
    'Pittsburgh': 'PIT', 'Washington': 'WAS', 'LosAngeles': 'LAR', 'SanFrancisco': 'SF',
    'LARams': 'LAR', 'LAChargers': 'LAC',
}

def parse_sbr_season(text, season_year):
    """
    Parse raw SBR paste. Games come in pairs: Visitor row then Home row.
    Columns: Date Rot VH Team Q1 Q2 Q3 Q4 Final Open Close ML 2H
    Indices: 0    1   2  3    4  5  6  7  8     9    10   11 12
    The closing total is whichever of Open/Close is >30 and <80.
    """
    rows = []
    lines = text.strip().split('\n')
    clean = []
    for l in lines:
        l = l.strip()
        if not l or l.startswith('NFL') or l.startswith('Date') or l.startswith('Dat'):
            continue
        parts = l.split('\t')
        if len(parts) >= 11:
            clean.append([p.strip() for p in parts])

    j = 0
    skipped = 0
    while j < len(clean) - 1:
        v, h = clean[j], clean[j+1]
        try:
            away_sbr = v[3]
            home_sbr = h[3]
            away_score = float(v[8])
            home_score = float(h[8])
            combined = away_score + home_score

            # Total is the value >30 <80 in Open(9) or Close(10) from either row
            closing_total = None
            for row in [v, h]:
                for idx in [10, 9]:
                    try:
                        val = float(row[idx].replace('pk', '0'))
                        if 30 < val < 80:
                            closing_total = val
                            break
                    except:
                        pass
                if closing_total:
                    break

            if closing_total:
                rows.append({
                    'season': season_year,
                    'away_team': SBR_TO_NFL.get(away_sbr),
                    'home_team': SBR_TO_NFL.get(home_sbr),
                    'away_team_sbr': away_sbr,
                    'home_team_sbr': home_sbr,
                    'away_score': int(away_score),
                    'home_score': int(home_score),
                    'combined_score': combined,
                    'closing_total': closing_total,
                })
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
        j += 2

    df = pd.DataFrame(rows)
    print(f"  {season_year}: {len(df)} games, {skipped} skipped")

    # Report unmapped teams
    unmapped = df[df['away_team'].isna() | df['home_team'].isna()]
    if len(unmapped):
        sbr_names = set(unmapped['away_team_sbr'].tolist() + unmapped['home_team_sbr'].tolist())
        print(f"    ⚠ Unmapped SBR names: {sbr_names}")
        print(f"    Add them to SBR_TO_NFL dict and re-run")

    return df.dropna(subset=['away_team', 'home_team'])


def main():
    raw_dir = 'sbr_raw'
    all_dfs = []

    for season in [2016, 2017, 2018, 2019]:
        path = os.path.join(raw_dir, f'sbr_{season}.txt')
        if not os.path.exists(path):
            print(f"Missing: {path} — skipping season {season}")
            continue
        with open(path) as f:
            text = f.read()
        df = parse_sbr_season(text, season)
        all_dfs.append(df)

    if not all_dfs:
        print("No data found. Create sbr_raw/ and add season text files.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} games across {combined['season'].nunique()} seasons")
    print(combined['season'].value_counts().sort_index().to_string())

    os.makedirs('data/odds', exist_ok=True)
    out_path = 'data/odds/sbr_closing_lines_2016_2019.parquet'
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Upload to S3
    bucket = 'nfl-predictor-bucket'
    s3_key = 'data/odds/sbr_closing_lines_2016_2019.parquet'
    try:
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.upload_file(out_path, bucket, s3_key)
        print(f"Uploaded to s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"S3 upload failed (run manually): {e}")
        print(f"  aws s3 cp {out_path} s3://{bucket}/{s3_key}")


if __name__ == '__main__':
    main()