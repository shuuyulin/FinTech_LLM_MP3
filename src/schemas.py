def _s(name, desc, props, req):
    return {
        "type": "function",
        "function": {
            "name":        name,
            "description": desc,
            "parameters":  {"type": "object", "properties": props, "required": req},
        },
    }


SCHEMA_TICKERS = _s(
    "get_tickers_by_sector",
    "Return all stocks in a sector or industry from the local database. "
    "Use broad sector names ('Information Technology', 'Energy') or sub-sectors ('semiconductor', 'insurance').",
    {"sector": {"type": "string", "description": "Sector or industry name"}},
    ["sector"],
)

SCHEMA_PRICE = _s(
    "get_price_performance",
    "Get % price change for a list of tickers over a time period. "
    "Use 'period' for relative ranges ('1mo','3mo','6mo','ytd','1y'), "
    "or 'start'/'end' (YYYY-MM-DD) for a specific date range. "
    "If both are provided, start/end takes precedence.",
    {
        "tickers": {"type": "array", "items": {"type": "string"}},
        "period":  {"type": "string", "default": "1y"},
        "start":   {"type": "string", "description": "Start date in YYYY-MM-DD format"},
        "end":     {"type": "string", "description": "End date in YYYY-MM-DD format"},
    },
    ["tickers"],
)

SCHEMA_OVERVIEW = _s(
    "get_company_overview",
    "Get fundamentals for one stock: P/E ratio, EPS, market cap, 52-week high and low.",
    {"ticker": {"type": "string", "description": "Ticker symbol e.g. 'AAPL'"}},
    ["ticker"],
)

SCHEMA_STATUS = _s(
    "get_market_status",
    "Check whether global stock exchanges are currently open or closed.",
    {}, [],
)

SCHEMA_MOVERS = _s(
    "get_top_gainers_losers",
    "Get today's top gaining, top losing, and most actively traded stocks.",
    {}, [],
)

SCHEMA_NEWS = _s(
    "get_news_sentiment",
    "Get latest news headlines and Bullish/Bearish/Neutral sentiment scores for a stock.",
    {"ticker": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
    ["ticker"],
)

SCHEMA_SQL = _s(
    "query_local_db",
    "Run a SQL SELECT on stocks.db. "
    "Table 'stocks': ticker, company, sector, industry, market_cap (Large/Mid/Small), exchange.",
    {"sql": {"type": "string", "description": "A valid SQL SELECT statement"}},
    ["sql"],
)

# Full set — used by single agent
ALL_SCHEMAS = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_OVERVIEW, SCHEMA_STATUS, SCHEMA_MOVERS, SCHEMA_NEWS, SCHEMA_SQL]

# Specialist subsets — used by multi-agent
MARKET_TOOLS      = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_STATUS, SCHEMA_MOVERS]
FUNDAMENTAL_TOOLS = [SCHEMA_OVERVIEW, SCHEMA_SQL, SCHEMA_TICKERS]
SENTIMENT_TOOLS   = [SCHEMA_NEWS, SCHEMA_SQL]
