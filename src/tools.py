import sqlite3
import requests
import yfinance as yf
import pandas as pd

from src.config import ALPHAVANTAGE_API_KEY, DB_PATH


def get_price_performance(tickers: list, period: str = "1y") -> dict:
    results = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                results[ticker] = {"error": "No data — possibly delisted"}
                continue
            start = float(data["Close"].iloc[0].item())
            end   = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price":   round(end, 2),
                "pct_change":  round((end - start) / start * 100, 2),
                "period":      period,
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return results


def get_market_status() -> dict:
    return requests.get(
        f"https://www.alphavantage.co/query?function=MARKET_STATUS&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=10,
    ).json()


def get_top_gainers_losers() -> dict:
    return requests.get(
        f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=10,
    ).json()


def get_news_sentiment(ticker: str, limit: int = 5) -> dict:
    try:
        data = requests.get(
            f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
            f"&tickers={ticker}&limit={limit}&apikey={ALPHAVANTAGE_API_KEY}",
            timeout=10,
        ).json()
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
