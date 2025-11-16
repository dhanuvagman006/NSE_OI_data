import pandas as pd
from nsepython import oi_chain_builder
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import threading
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
class Config:
    INDEX_SYMBOL = os.getenv("INDEX_SYMBOL", "NIFTY")
    TOP_N = int(os.getenv("TOP_N", "10"))
    CACHE_DURATION = int(os.getenv("CACHE_DURATION", "60"))  # seconds
    PORT = int(os.getenv("PORT", "5000"))
    HOST = os.getenv("HOST", "127.0.0.1")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Simple in-memory cache
class DataCache:
    def __init__(self):
        self._cache = {}
        self._cache_lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Dict]:
        with self._cache_lock:
            if key in self._cache:
                data, timestamp = self._cache[key]
                if datetime.now() - timestamp < timedelta(seconds=Config.CACHE_DURATION):
                    return data
                else:
                    del self._cache[key]
        return None
    
    def set(self, key: str, data: Dict) -> None:
        with self._cache_lock:
            self._cache[key] = (data, datetime.now())
    
    def clear(self) -> None:
        with self._cache_lock:
            self._cache.clear()

cache = DataCache()

def rate_limit(max_calls: int = 10, window: int = 60):
    """Simple rate limiting decorator"""
    calls = {}
    lock = threading.Lock()
    
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            client_id = request.remote_addr if request else 'local'
            now = time.time()
            
            with lock:
                if client_id not in calls:
                    calls[client_id] = []
                
                # Remove old calls outside the window
                calls[client_id] = [call_time for call_time in calls[client_id] 
                                  if now - call_time < window]
                
                if len(calls[client_id]) >= max_calls:
                    return jsonify({"error": "Rate limit exceeded"}), 429
                
                calls[client_id].append(now)
            
            return f(*args, **kwargs)
        return wrapper
    return decorator

def validate_symbol(symbol: str) -> bool:
    """Validate if symbol is supported"""
    supported_symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
    return symbol.upper() in supported_symbols

def get_oi_data(symbol: str, top_n: int) -> Optional[Dict[str, Any]]:
    """Fetches and processes OI data, returning it as a dictionary.

    Returns OI for both calls and puts for strikes near the spot price
    (LTP). The `top_n` parameter is interpreted as the number of strikes
    above and below LTP to include, where data is available.
    """
    cache_key = f"{symbol}_{top_n}"
    
    # Check cache first
    cached_data = cache.get(cache_key)
    if cached_data:
        logger.info(f"Returning cached data for {symbol}")
        return cached_data
    
    try:
        logger.info(f"Fetching fresh data for {symbol}")
        df_full, ltp, crontime = oi_chain_builder(symbol)
        
        if df_full is None or df_full.empty:
            logger.error(f"Empty data received for {symbol}")
            return None
        
        # Ensure required columns exist and handle different column names
        calls_oi_col = None
        puts_oi_col = None
        strike_col = None
        
        # Find the correct column names (case-insensitive)
        for col in df_full.columns:
            col_lower = col.lower()
            if 'strike' in col_lower and 'price' in col_lower:
                strike_col = col
            elif 'call' in col_lower and 'oi' in col_lower:
                calls_oi_col = col
            elif 'put' in col_lower and 'oi' in col_lower:
                puts_oi_col = col
        
        # Fallback column names
        if not strike_col:
            strike_col = 'Strike Price'
        if not calls_oi_col:
            calls_oi_col = 'CALLS_OI'
        if not puts_oi_col:
            puts_oi_col = 'PUTS_OI'
        
        required_cols = [strike_col, calls_oi_col, puts_oi_col]
        missing_cols = [col for col in required_cols if col not in df_full.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None

        # Clean and validate data
        df_clean = df_full.dropna(subset=required_cols).copy()
        
        # Ensure numeric types
        for col in [calls_oi_col, puts_oi_col]:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[strike_col] = pd.to_numeric(df_clean[strike_col], errors='coerce')
        
        # Remove rows with NaN values after conversion
        df_clean = df_clean.dropna(subset=required_cols)
        
        if df_clean.empty:
            logger.error("No valid data after cleaning")
            return None

        # Select strikes near the LTP (spot price)
        df_clean["distance_from_ltp"] = (df_clean[strike_col] - float(ltp)).abs()
        df_sorted = df_clean.sort_values("distance_from_ltp")

        # Take strikes nearest to LTP. We keep up to (2 * top_n + 1)
        # rows so that you get a band around the spot price.
        near_strikes = df_sorted.head(2 * top_n + 1).copy()

        # Sort selected strikes in ascending order for nicer plotting
        near_strikes = near_strikes.sort_values(strike_col)

        # For backward compatibility, also compute "top" by OI if needed
        top_ce = df_clean.nlargest(top_n, calls_oi_col)
        top_pe = df_clean.nlargest(top_n, puts_oi_col)

        # Calculate additional metrics
        total_ce_oi = df_clean[calls_oi_col].sum()
        total_pe_oi = df_clean[puts_oi_col].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

        # Find support and resistance levels
        max_ce_strike = top_ce.iloc[0][strike_col] if not top_ce.empty else None
        max_pe_strike = top_pe.iloc[0][strike_col] if not top_pe.empty else None

        # Prepare data for JSON/Chart
        chart_data = {
            'ltp': float(ltp),
            'crontime': str(crontime),
            # Strikes near spot price
            'near_strikes': near_strikes[strike_col].astype(float).tolist(),
            'near_ce_oi': near_strikes[calls_oi_col].astype(int).tolist(),
            'near_pe_oi': near_strikes[puts_oi_col].astype(int).tolist(),

            # Legacy top-by-OI data (if you still want it elsewhere)
            'ce_strikes': top_ce[strike_col].astype(float).tolist(),
            'ce_oi': top_ce[calls_oi_col].astype(int).tolist(),
            'pe_strikes': top_pe[strike_col].astype(float).tolist(),
            'pe_oi': top_pe[puts_oi_col].astype(int).tolist(),
            'total_ce_oi': int(total_ce_oi),
            'total_pe_oi': int(total_pe_oi),
            'pcr': round(pcr, 3),
            'max_ce_strike': float(max_ce_strike) if max_ce_strike else None,
            'max_pe_strike': float(max_pe_strike) if max_pe_strike else None,
            'symbol': symbol.upper(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache the data
        cache.set(cache_key, chart_data)
        
        return chart_data

    except Exception as e:
        logger.error(f"Runtime error occurred for {symbol}: {e}", exc_info=True)
        return None

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html', symbol=Config.INDEX_SYMBOL, top_n=Config.TOP_N)

@app.route('/data')
def get_data():
    """API endpoint to serve the OI data as JSON."""
    data = get_oi_data(Config.INDEX_SYMBOL, Config.TOP_N)
    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "Failed to fetch OI data"}), 500

if __name__ == '__main__':
    # Use threaded=True for development or run with gunicorn/waitress for production
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, threaded=True)