import pandas as pd

def move_duplicates(input_file: str, date_col: str = "date") -> None:
    # Load CSV
    df = pd.read_csv(input_file)
    df.columns = [c.strip().lower() for c in df.columns]  # normalize names

    # Auto-detect date column if needed
    if date_col not in df.columns:
        date_col = [c for c in df.columns if "date" in c][0]

    # Parse dates safely
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Find duplicates
    duplicates = df[df[date_col].duplicated(keep=False)].sort_values(by=date_col)
    cleaned = df.drop_duplicates(subset=[date_col], keep="first")

    # Save outputs
    duplicates.to_csv("duplicates.csv", index=False)
    cleaned.to_csv("cleaned.csv", index=False)

    print(f"✅ Found {len(duplicates)} duplicate rows.")
    print("📁 Saved duplicates → duplicates.csv")
    print("📁 Saved cleaned data → cleaned.csv")

if __name__ == "__main__":
    move_duplicates("PCCE_EQUITIES_CBOE.csv")
