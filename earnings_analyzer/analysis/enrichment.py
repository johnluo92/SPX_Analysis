"""IV data enrichment for analysis results

Separated from batch processing to make it a reusable step
"""
from typing import List, Dict
from ..core.models import AnalysisResult
from ..data_sources import YahooFinanceClient


def enrich_with_iv(results: List[AnalysisResult], 
                   debug: bool = False) -> tuple[List[AnalysisResult], Dict]:
    """
    Enrich analysis results with current IV data
    
    Args:
        results: List of AnalysisResult objects
        debug: Print debug information
    
    Returns:
        (enriched_results, iv_summary_dict)
    """
    yf_client = YahooFinanceClient()
    iv_summary = {'success': [], 'failed': []}
    
    for result in results:
        try:
            iv_data = yf_client.get_current_iv(result.ticker)
            
            if iv_data and 'iv' in iv_data and iv_data['iv'] is not None:
                # Update the result object
                result.current_iv = iv_data['iv']
                result.iv_dte = iv_data.get('dte', None)
                
                # Calculate IV elevation vs RVol45d
                if result.rvol_45d is not None:
                    iv_elevation = ((iv_data['iv'] - result.rvol_45d) / result.rvol_45d) * 100
                    result.iv_elevation = round(iv_elevation, 1)
                else:
                    result.iv_elevation = None
                
                iv_summary['success'].append(result.ticker)
            else:
                # Explicit None assignment ensures ticker still appears
                result.current_iv = None
                result.iv_dte = None
                result.iv_elevation = None
                iv_summary['failed'].append(result.ticker)
                
                if debug:
                    print(f"\n⚠️  IV fetch returned no data for {result.ticker}")
                    
        except Exception as e:
            # Catch any unexpected errors
            result.current_iv = None
            result.iv_dte = None
            result.iv_elevation = None
            iv_summary['failed'].append(result.ticker)
            
            if debug:
                print(f"\n⚠️  IV fetch error for {result.ticker}: {e}")
    
    return results, iv_summary


def enrich_with_iv_dict(results: List[Dict], 
                        debug: bool = False) -> tuple[List[Dict], Dict]:
    """
    Legacy version that works with dictionaries
    
    This is for backward compatibility with old batch.py
    """
    yf_client = YahooFinanceClient()
    iv_summary = {'success': [], 'failed': []}
    
    for result in results:
        ticker = result['ticker']
        
        try:
            iv_data = yf_client.get_current_iv(ticker)
            
            if iv_data and 'iv' in iv_data and iv_data['iv'] is not None:
                result['current_iv'] = iv_data['iv']
                result['iv_dte'] = iv_data.get('dte', None)
                
                # IV elevation vs RVol45d
                if 'rvol_45d' in result and result['rvol_45d'] is not None:
                    iv_elevation = ((iv_data['iv'] - result['rvol_45d']) / result['rvol_45d']) * 100
                    result['iv_elevation'] = round(iv_elevation, 1)
                else:
                    result['iv_elevation'] = None
                
                iv_summary['success'].append(ticker)
            else:
                result['current_iv'] = None
                result['iv_dte'] = None
                result['iv_elevation'] = None
                iv_summary['failed'].append(ticker)
                
                if debug:
                    print(f"\n⚠️  IV fetch returned no data for {ticker}")
                    
        except Exception as e:
            result['current_iv'] = None
            result['iv_dte'] = None
            result['iv_elevation'] = None
            iv_summary['failed'].append(ticker)
            
            if debug:
                print(f"\n⚠️  IV fetch error for {ticker}: {e}")
    
    return results, iv_summary