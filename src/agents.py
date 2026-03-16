import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import client
from src.models import AgentResult
from src.tools import ALL_TOOL_FUNCTIONS
from src.schemas import (
    ALL_SCHEMAS,
    MARKET_TOOLS, FUNDAMENTAL_TOOLS, SENTIMENT_TOOLS,
)

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────

SINGLE_AGENT_PROMPT = """You are a precise financial data analyst with access to tools.

INSTRUCTIONS:
1. Answer questions using ONLY real data retrieved from tools
2. ALWAYS call the relevant tools to answer the question
3. If a tool returns an error or empty data, report it clearly (e.g., "No news articles available")
4. Present the data you retrieve with specific numbers
5. Never fabricate numbers or make up data
6. If you cannot retrieve a data point, say so explicitly with reason (rate limit, no data, API error)

IMPORTANT: Some tools may return empty results (e.g., no news articles) — this is normal, not an error.
Report the fact that data is unavailable rather than making up information."""

ORCHESTRATOR_PROMPT = """You are a financial analysis orchestrator.
Given a user question, decompose it into three focused sub-tasks for specialist agents.

1. market_task       — aspects about stock price performance, market open/closed status, or top movers
2. fundamentals_task — aspects about P/E ratios, EPS, market cap, 52-week high/low, or stock filtering
3. sentiment_task    — aspects about news sentiment

For each sub-task, be specific about which tickers or sectors to analyze.
If a domain is NOT relevant to the question, output exactly "NOT APPLICABLE".

Respond ONLY in valid JSON (no markdown):
{
  "market_task": "<specific sub-task or NOT APPLICABLE>",
  "fundamentals_task": "<specific sub-task or NOT APPLICABLE>",
  "sentiment_task": "<specific sub-task or NOT APPLICABLE>"
}"""

MARKET_PROMPT = """You are a market data specialist. \
Answer questions about stock price performance, market status, and market movers using ONLY real tool data.
For sector questions, ALWAYS call get_tickers_by_sector first to get actual tickers, \
then call get_price_performance on those tickers.
Never fabricate prices or % changes. If data is unavailable, say so explicitly."""

FUNDAMENTALS_PROMPT = """You are a fundamentals analyst. \
Answer questions about P/E ratios, EPS, market cap, and 52-week ranges \
using ONLY real data from get_company_overview.
For filtering stocks (large-cap, NASDAQ, by sector), use query_local_db or get_tickers_by_sector first.
Never fabricate financial metrics. If a ticker returns an error, report it."""

SENTIMENT_PROMPT = """You are a news sentiment analyst. \
Answer questions about recent news headlines and sentiment scores \
using ONLY real data from get_news_sentiment.
For sector-level sentiment, use query_local_db to get tickers first.
Report sentiment as Bullish / Bearish / Neutral with article titles."""

CRITIC_PROMPT = """You are a fact-checker. Review a specialist's answer against their raw tool data.
Check whether every specific claim (numbers, tickers, sentiments) is directly supported by the tool output.
Flag any claim that appears fabricated or contradicts the tool data.
Respond ONLY in valid JSON (no markdown):
{
  "verified": true,
  "issues": []
}"""

SYNTHESIZER_PROMPT = """You are a financial analysis synthesizer.
You receive outputs from market, fundamentals, and sentiment specialists, plus any critic flags.
Combine the relevant information into a single, coherent, factual answer to the original question.
Prioritize verified data. Omit specialists that returned NOT APPLICABLE or had no relevant data.
If the critic flagged issues, note them as caveats rather than presenting them as facts.
Lead with a direct answer, then provide supporting detail."""


# ─────────────────────────────────────────────────────────────
# CORE AGENT LOOP
# ─────────────────────────────────────────────────────────────

def run_specialist_agent(
    agent_name: str,
    system_prompt: str,
    task: str,
    tool_schemas: list,
    model: str = "gpt-4o-mini",
    max_iters: int = 8,
    verbose: bool = False,
) -> AgentResult:
    messages     = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
    tools_called = []
    raw_data     = {}

    for _ in range(max_iters):
        kwargs = {"model": model, "messages": messages}
        if tool_schemas:
            kwargs["tools"]       = tool_schemas
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        msg      = response.choices[0].message

        if not msg.tool_calls:
            return AgentResult(
                agent_name=agent_name, answer=msg.content or "",
                tools_called=tools_called, raw_data=raw_data,
            )

        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            fn      = ALL_TOOL_FUNCTIONS.get(fn_name)
            result  = fn(**fn_args) if fn else {"error": f"Unknown tool: {fn_name}"}
            tools_called.append(fn_name)
            raw_data[fn_name] = result
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

        if verbose:
            print(f"[{agent_name}] called {fn_name}")

    return AgentResult(
        agent_name=agent_name, answer="Max iterations reached",
        tools_called=tools_called, raw_data=raw_data,
    )


# ─────────────────────────────────────────────────────────────
# SINGLE AGENT
# ─────────────────────────────────────────────────────────────

def run_single_agent(question: str, model: str = "gpt-4o-mini", verbose: bool = False) -> AgentResult:
    return run_specialist_agent(
        agent_name="SingleAgent",
        system_prompt=SINGLE_AGENT_PROMPT,
        task=question,
        tool_schemas=ALL_SCHEMAS,
        model=model,
        verbose=verbose,
    )


# ─────────────────────────────────────────────────────────────
# MULTI-AGENT  (Orchestrator → Parallel Specialists → Critic → Synthesizer)
# ─────────────────────────────────────────────────────────────

def _run_specialist(name, prompt, task, schemas, model, verbose) -> AgentResult:
    """Skip NOT APPLICABLE tasks without making any API calls."""
    if task.strip().upper() == "NOT APPLICABLE":
        return AgentResult(agent_name=name, answer="NOT APPLICABLE", tools_called=[], raw_data={})
    return run_specialist_agent(
        agent_name=name, system_prompt=prompt, task=task,
        tool_schemas=schemas, model=model, verbose=verbose,
    )


def run_multi_agent(question: str, model: str = "gpt-4o-mini", verbose: bool = False) -> dict:
    t0 = time.time()

    # ── Step 1: Orchestrator decomposes the question ──────────
    orch_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user",   "content": question},
        ],
        response_format={"type": "json_object"},
    )
    try:
        tasks = json.loads(orch_resp.choices[0].message.content)
    except Exception:
        tasks = {}

    market_task       = tasks.get("market_task",       question)
    fundamentals_task = tasks.get("fundamentals_task", question)
    sentiment_task    = tasks.get("sentiment_task",    question)

    if verbose:
        print(f"[Orchestrator] market:       {market_task}")
        print(f"[Orchestrator] fundamentals: {fundamentals_task}")
        print(f"[Orchestrator] sentiment:    {sentiment_task}")

    # ── Step 2: Parallel specialist execution ─────────────────
    specialist_configs = [
        ("MarketSpecialist",       MARKET_PROMPT,       market_task,       MARKET_TOOLS),
        ("FundamentalsSpecialist", FUNDAMENTALS_PROMPT, fundamentals_task, FUNDAMENTAL_TOOLS),
        ("SentimentSpecialist",    SENTIMENT_PROMPT,    sentiment_task,    SENTIMENT_TOOLS),
    ]

    agent_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_specialist, name, prompt, task, schemas, model, verbose): name
            for name, prompt, task, schemas in specialist_configs
        }
        for future in as_completed(futures):
            agent_results.append(future.result())

    # ── Step 3: Critic reviews each specialist ────────────────
    all_critic_issues = []
    for result in agent_results:
        if result.answer == "NOT APPLICABLE" or not result.tools_called:
            continue
        critic_input = (
            f"Specialist: {result.agent_name}\n"
            f"Answer: {result.answer}\n"
            f"Raw tool data: {json.dumps(result.raw_data, default=str)[:2000]}"
        )
        try:
            critic_resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CRITIC_PROMPT},
                    {"role": "user",   "content": critic_input},
                ],
                response_format={"type": "json_object"},
            )
            critic_out = json.loads(critic_resp.choices[0].message.content)
            issues = critic_out.get("issues", [])
            if issues:
                all_critic_issues.extend(issues)
                result.issues_found = issues
        except Exception:
            pass

    # ── Step 4: Synthesizer produces final answer ─────────────
    synthesis_input = f"Original question: {question}\n\n"
    for r in agent_results:
        synthesis_input += f"--- {r.agent_name} ---\n{r.answer}\n\n"
    if all_critic_issues:
        synthesis_input += "--- Critic flags ---\n" + "\n".join(str(i) for i in all_critic_issues) + "\n"

    synth_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYNTHESIZER_PROMPT},
            {"role": "user",   "content": synthesis_input},
        ],
    )

    return {
        "final_answer":  synth_resp.choices[0].message.content,
        "agent_results": agent_results,
        "elapsed_sec":   round(time.time() - t0, 2),
        "architecture":  "orchestrator-parallel-critic",
    }
