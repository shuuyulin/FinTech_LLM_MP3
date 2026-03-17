import sqlite3
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, time
from zoneinfo import ZoneInfo

from src.config import ALPHAVANTAGE_API_KEY, DB_PATH


def _av_rate_limited(data: dict) -> bool:
    """Return True if AlphaVantage responded with a rate-limit or access message."""
    return "Note" in data or "Information" in data


def get_price_performance(tickers: list, period: str = "1y", start: str = None, end: str = None) -> dict:
    results = {}
    for ticker in tickers:
        try:
            if start or end:
                data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                label = f"{start} to {end}"
            else:
                data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                label = period
            if data.empty:
                results[ticker] = {"error": "No data — possibly delisted"}
                continue
            open_price  = float(data["Close"].iloc[0].item())
            close_price = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(open_price, 2),
                "end_price":   round(close_price, 2),
                "pct_change":  round((close_price - open_price) / open_price * 100, 2),
                "period":      label,
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return results


def get_market_status() -> dict:
    try:
        data = requests.get(
            f"https://www.alphavantage.co/query?function=MARKET_STATUS&apikey={ALPHAVANTAGE_API_KEY}",
            timeout=10,
        ).json()
        if not _av_rate_limited(data):
            return data
    except Exception:
        pass
    # Fallback: infer NYSE open/closed from current time
    now = datetime.now(ZoneInfo("America/New_York"))
    is_weekday = now.weekday() < 5
    status = (
        "open"
        if is_weekday and time(9, 30) <= now.time() <= time(16, 0)
        else "closed"
    )
    return {
        "markets": [{"market_type": "Equity", "primary_exchanges": "NYSE, NASDAQ", "current_status": status}],
        "note": "Estimated from NYSE trading hours — AlphaVantage limit reached",
    }


def get_top_gainers_losers() -> dict:
    try:
        data = requests.get(
            f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={ALPHAVANTAGE_API_KEY}",
            timeout=10,
        ).json()
        if not _av_rate_limited(data):
            return data
    except Exception:
        pass
    return {"error": "Top gainers/losers unavailable — AlphaVantage limit reached and no free alternative exists"}


def get_news_sentiment(ticker: str, limit: int = 5) -> dict:
    try:
        data = requests.get(
            f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
            f"&tickers={ticker}&limit={limit}&apikey={ALPHAVANTAGE_API_KEY}",
            timeout=10,
        ).json()
        if not _av_rate_limited(data):
            articles = data.get("feed", [])
            if articles:
                return {
                    "ticker": ticker,
                    "articles": [
                        {
                            "title":     a.get("title"),
                            "source":    a.get("source"),
                            "sentiment": a.get("overall_sentiment_label"),
                            "score":     a.get("overall_sentiment_score"),
                        }
                        for a in articles[:limit]
                    ],
                }
            return {"ticker": ticker, "articles": [], "note": f"No recent news data available for {ticker}"}
    except Exception:
        pass
    # Fallback: yfinance news (no sentiment scores)
    try:
        raw = yf.Ticker(ticker).news or []
        articles = [
            {
                "title":     n["content"]["title"],
                "source":    n["content"]["provider"]["displayName"],
                "sentiment": "N/A",
                "score":     None,
            }
            for n in raw[:limit]
            if n.get("content")
        ]
        return {
            "ticker": ticker,
            "articles": articles,
            "note": "Sentiment unavailable — AlphaVantage limit reached, using yfinance news",
        }
    except Exception as e:
        return {"ticker": ticker, "articles": [], "error": str(e)}


def query_local_db(sql: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(sql, conn)
        conn.close()
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}


def get_company_overview(ticker: str) -> dict:
    """Try Alpha Vantage first, fall back to yfinance on rate-limit or error."""
    url = (
        f"https://www.alphavantage.co/query?function=OVERVIEW"
        f"&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    )
    try:
        data = requests.get(url, timeout=10).json()
        if data and "Symbol" in data:
            return {
                "ticker":      ticker,
                "name":        data.get("Name", ""),
                "pe_ratio":    data.get("PERatio", "N/A"),
                "eps":         data.get("EPS", "N/A"),
                "market_cap":  data.get("MarketCapitalization", "N/A"),
                "week52_high": data.get("52WeekHigh", "N/A"),
                "week52_low":  data.get("52WeekLow", "N/A"),
            }
        # Fallback to yfinance
        info = yf.Ticker(ticker).info
        return {
            "ticker":      ticker,
            "name":        info.get("longName", ""),
            "pe_ratio":    str(info.get("trailingPE", "N/A")),
            "eps":         str(info.get("trailingEps", "N/A")),
            "market_cap":  str(info.get("marketCap", "N/A")),
            "week52_high": str(info.get("fiftyTwoWeekHigh", "N/A")),
            "week52_low":  str(info.get("fiftyTwoWeekLow", "N/A")),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def get_tickers_by_sector(sector: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT ticker, company, sector, industry, market_cap FROM stocks WHERE sector = ?",
            conn, params=(sector,),
        )
        if df.empty:
            df = pd.read_sql_query(
                "SELECT ticker, company, sector, industry, market_cap FROM stocks WHERE industry LIKE ?",
                conn, params=(f"%{sector}%",),
            )
        conn.close()
        return {"sector": sector, "stocks": df.to_dict(orient="records")}
    except Exception as e:
        return {"sector": sector, "stocks": [], "error": str(e)}


# Dispatch table used by the agent loop
ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector":  get_tickers_by_sector,
    "get_price_performance":  get_price_performance,
    "get_company_overview":   get_company_overview,
    "get_market_status":      get_market_status,
    "get_top_gainers_losers": get_top_gainers_losers,
    "get_news_sentiment":     get_news_sentiment,
    "query_local_db":         query_local_db,
}
