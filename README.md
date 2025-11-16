# NSE OI Data Dashboard

Simple Flask-based dashboard to visualise NSE index option open interest around the spot price.

## Setup

```powershell
cd D:\NSE
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Development

```powershell
cd D:\NSE
.\.venv\Scripts\Activate.ps1
$env:DEBUG="True"
python .\core.py
```

Visit http://127.0.0.1:5000 in your browser.

## Production (Waitress)

```powershell
cd D:\NSE
.\.venv\Scripts\Activate.ps1
$env:DEBUG="False"
$env:HOST="0.0.0.0"
$env:PORT="5000"
python .\wsgi.py
```

Behind a reverse proxy (e.g. Nginx), point traffic to `HOST:PORT`.

### Environment variables

- `INDEX_SYMBOL` (default `NIFTY`)
- `TOP_N` (default `10`)
- `CACHE_DURATION` in seconds (default `60`)
- `HOST` (default `0.0.0.0`)
- `PORT` (default `5000`)
- `DEBUG` (`True`/`False`, default `False`)
