FROM python:3.12-slim

WORKDIR /app

# System deps: TA-Lib C library + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    libpq-dev \
    && wget -q https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr && make && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && apt-get remove -y build-essential wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in layers (cache-friendly: slow deps first)
COPY requirements.txt .

# autogluon + ray are large — install first so Docker caches them
RUN pip install --no-cache-dir "autogluon.tabular[lightgbm,xgboost]==1.5.0"
RUN pip install --no-cache-dir "ray>=2.43.0,<2.53.0"

# TA-Lib Python binding (needs C library installed above)
RUN pip install --no-cache-dir ta-lib

# spaCy model (required at runtime)
RUN pip install --no-cache-dir spacy && python -m spacy download en_core_web_sm

# Remaining requirements (excludes autogluon/ray/ta-lib — already installed)
RUN pip install --no-cache-dir \
    fastapi==0.115.5 \
    "uvicorn[standard]==0.32.1" \
    pydantic==2.10.3 \
    pydantic-settings==2.6.1 \
    python-dotenv==1.0.1 \
    SQLAlchemy==2.0.36 \
    psycopg2-binary==2.9.10 \
    asyncpg==0.30.0 \
    alembic==1.14.0 \
    APScheduler==3.11.0 \
    "yfinance>=1.4.1" \
    pandas-datareader==0.10.0 \
    sec-edgar-api==1.1.0 \
    "tradingview-screener==3.2.1" \
    beautifulsoup4==4.12.3 \
    httpx==0.28.1 \
    lxml==5.3.0 \
    transformers==4.47.1 \
    "torch==2.5.1" \
    lightgbm==4.5.0 \
    scikit-learn==1.6.0 \
    "pandas==2.3.2" \
    "numpy==2.2.6" \
    "pyarrow>=24.0.0" \
    cachetools==5.5.0 \
    python-dateutil==2.9.0 \
    websockets==14.1 \
    Pillow==11.0.0 \
    "networkx>=3.3" \
    "finvizfinance>=0.14" \
    "sentence-transformers>=2.7.0" \
    "groq>=0.4.0" \
    "agno>=1.0.0" \
    "chromadb>=0.5.0" \
    "optuna>=3.6" \
    pytz \
    python-telegram-bot==21.8

# Git-based deps
RUN pip install --no-cache-dir \
    "alpaca-py @ git+https://github.com/bangalore-killua/trading-system-v2-alpaca-py.git@da55156db81af18cd6c171ccddf1921017d98d3d" \
    "feedparser @ git+https://github.com/bangalore-killua/trading-system-v2-feedparser.git@6.0.11" \
    "quantstats @ git+https://github.com/bangalore-killua/trading-system-v2-quantstats.git@0.0.62" \
    "ultralytics @ git+https://github.com/bangalore-killua/trading-system-v2-ultralytics.git@v8.3.40" \
    "mplfinance @ git+https://github.com/bangalore-killua/trading-system-v2-mplfinance.git@v0.12.10b0"

# Copy application code
COPY . .

# Pre-trained models are part of the repo — already copied above

EXPOSE 8000

# Healthcheck: hit the API every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "run_server.py"]
