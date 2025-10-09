"""CLI entry point for earnings analyzer"""
from analysis import batch_analyze, analyze_ticker
from output import export_to_csv, export_to_json


def main():
    """Main entry point"""
    
    tickers = [
        "DAL", "PEP", "FAST", "BLK", "C", "DPZ", "GS", "JNJ", "JPM", "WFC",
        "OMC", "ABT", "BAC", "CFG", "MS", "PGR", "PLD", "PNC", "SYF", "JBHT",
        "UAL", "BK", "KEY", "MMC", "MTB", "SCHW", "SNA", "TRV", "USB", "CSX",
        "AXP", "FITB", "HBAN", "RF", "SLB", "STT", "TFC", "STLD"
    ]
    
    results = batch_analyze(
        tickers,
        lookback_quarters=24,
        debug=False,
        fetch_iv=True,
        parallel=False,
        max_workers=4
    )
    
    if results is not None:
        export_to_csv(results)


if __name__ == "__main__":
    main()