import os, re, json, logging, time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import snowflake.connector
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

LOG_FILE = Path(__file__).parent / "query_log.jsonl"

def log_query(entry: dict):
    """Append one JSON record per query to query_log.jsonl."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.warning(f"Failed to write query log: {e}")

app = FastAPI(title="Travel Insights Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Client and model setup
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

DB  = "BOOKINGS"
SCH = "PUBLIC"
TBL = "INFO"

def get_sf_connection():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=DB,
        schema=SCH,
        session_parameters={"QUERY_TAG": "travel-agent"},
    )

# Schema
# Column names verified directly from Snowflake information_schema.
# All column names are uppercase as stored in Snowflake.

TABLE_CATALOG = f"""
DATABASE: {DB}  SCHEMA: {SCH}  TABLE: {TBL}
Fully qualified table name: "{DB}"."{SCH}"."{TBL}"

This table contains hotel booking records from two Portuguese hotels (2015-2017).
Each row is one booking. Total rows: ~119,000.

Hotel booking dataset schema:

HOTEL               TEXT    — Hotel type: 'Resort Hotel' or 'City Hotel'
IS_CANCELED         NUMBER  — 1 = booking was canceled, 0 = not canceled
RESERVATION_STATUS  TEXT    — Final status: 'Check-Out', 'Canceled', 'No-Show'
RESERVATION_STATUS_DATE DATE — Date of last status change

Arrival date is split into 4 columns

ARRIVAL_DATE_YEAR         NUMBER  — Year of arrival: 2015, 2016, or 2017
ARRIVAL_DATE_MONTH        TEXT    — Month name: 'January' through 'December'
ARRIVAL_DATE_WEEK_NUMBER  NUMBER  — ISO week number (1-53)
ARRIVAL_DATE_DAY_OF_MONTH NUMBER  — Day of month (1-31)

Stay duration is split into weekday vs weekend nights

STAYS_IN_WEEKEND_NIGHTS   NUMBER  — Nights stayed on Saturday/Sunday
STAYS_IN_WEEK_NIGHTS      NUMBER  — Nights stayed Monday-Friday
-- Total stay = STAYS_IN_WEEKEND_NIGHTS + STAYS_IN_WEEK_NIGHTS

Guests

ADULTS    NUMBER  — Number of adults
CHILDREN  NUMBER  — Number of children
BABIES    NUMBER  — Number of babies

Booking details

LEAD_TIME        NUMBER  — Days between booking date and arrival date
                          (higher = booked further in advance)
MEAL             TEXT    — Meal plan: 'BB' (Bed & Breakfast), 'HB' (Half Board),
                           'FB' (Full Board), 'SC' (Self Catering), 'Undefined'
MARKET_SEGMENT   TEXT    — How the booking was sourced:
                           'Online TA' (online travel agent e.g. Expedia),
                           'Offline TA/TO' (offline travel agent/tour operator),
                           'Direct', 'Corporate', 'Groups', 'Complementary', 'Aviation'
DISTRIBUTION_CHANNEL TEXT — Booking channel: 'TA/TO', 'Direct', 'Corporate',
                            'GDS' (global distribution system), 'Undefined'
DEPOSIT_TYPE     TEXT    — 'No Deposit', 'Non Refund', 'Refundable'
CUSTOMER_TYPE    TEXT    — 'Transient', 'Transient-Party', 'Contract', 'Group'
BOOKING_CHANGES  NUMBER  — Number of changes made to the booking
DAYS_IN_WAITING_LIST NUMBER — Days the booking was on the waiting list

Guest history

IS_REPEATED_GUEST              NUMBER — 1 = returning guest, 0 = first time
PREVIOUS_CANCELLATIONS         NUMBER — Number of previous cancellations by this guest
PREVIOUS_BOOKINGS_NOT_CANCELED NUMBER — Number of previous non-canceled bookings

Room

RESERVED_ROOM_TYPE TEXT — Room type originally reserved (A-H, L, P)
ASSIGNED_ROOM_TYPE TEXT — Room type actually assigned (may differ from reserved)
-- Room type mismatch: RESERVED_ROOM_TYPE != ASSIGNED_ROOM_TYPE

Pricing

ADR NUMBER — Average Daily Rate in EUR (revenue / total nights stayed)
            — Use AVG(ADR) for average, filter WHERE ADR > 0 to exclude free stays
REQUIRED_CAR_PARKING_SPACES NUMBER — Number of parking spaces requested
TOTAL_OF_SPECIAL_REQUESTS   NUMBER — Number of special requests made

Agent and company info

AGENT   TEXT — ID of travel agent (numeric string or 'NULL')
COMPANY TEXT — ID of company that made booking (numeric string or 'NULL')
COUNTRY TEXT — Country of origin of guest (ISO 3166-1 alpha-3 code, e.g. 'PRT', 'GBR', 'USA')
"""

# guardrails - deciding what is vs what is not allowed to be answered 

ALLOWED_TOPICS = """
ALLOWED topics (answer these):
- Hotel booking patterns, trends, and volumes
- Cancellation rates, reasons, and patterns
- Revenue and pricing (ADR — average daily rate)
- Market segments (Online TA, Direct, Corporate, etc.)
- Distribution channels
- Lead time analysis (how far in advance bookings are made)
- Guest demographics (country of origin, adults/children, repeat guests)
- Stay duration (weekend vs weekday nights)
- Room type analysis (reserved vs assigned, upgrades/downgrades)
- Seasonality (monthly, weekly, yearly trends)
- Special requests and parking
- Deposit type analysis
- Waiting list analysis
- Comparisons between Resort Hotel and City Hotel
- Comparisons between market segments, channels, customer types

REJECT these (do not answer):
- Current events, news, politics
- Stock market, cryptocurrency, financial advice
- Questions about specific real people or companies by name
- Medical, legal, or personal advice
- Anything unrelated to hotel bookings or travel analytics
- Predictions beyond what the data supports
- Offensive, or harmful content
- Questions about data not in this dataset (flights, restaurants, car rentals, etc.)
"""

# Prompts

SYSTEM_SQL = f"""You are a Snowflake SQL expert for a hotel booking analytics dataset.

{ALLOWED_TOPICS}

{TABLE_CATALOG}

Respond ONLY with a raw JSON object — no markdown, no code fences, no explanation outside JSON.

If the question is on an ALLOWED topic:
{{"type":"sql","sql":"SELECT ...","explanation":"one line: what this measures","chart_type":"bar"|"line"|"none"}}

Set chart_type to:
- "line" if the question is about trends over time (monthly, yearly, weekly patterns)
- "bar" if the question is about rankings or comparisons across categories
- "none" for all other questions (single values, ratios, etc.)

If the question should be REJECTED:
{{"type":"reject","message":"I can only answer questions about hotel booking analytics. [brief friendly reason]"}}

SQL RULES — follow exactly:
1. Table: always use fully qualified "{DB}"."{SCH}"."{TBL}"
2. Column names: UPPERCASE exactly as listed in schema above
3. Cancellation rate: ROUND(100.0 * SUM(IS_CANCELED) / COUNT(*), 1) AS cancellation_rate
4. Average ADR: AVG(CASE WHEN ADR > 0 THEN ADR END) AS avg_adr — exclude zero-rate bookings
5. Total stay nights: STAYS_IN_WEEKEND_NIGHTS + STAYS_IN_WEEK_NIGHTS
6. Monthly ordering: use ARRIVAL_DATE_MONTH — order results by month chronologically using a CASE statement
7. Filter by hotel: WHERE HOTEL = 'Resort Hotel' or WHERE HOTEL = 'City Hotel'
8. Filter by channel: WHERE MARKET_SEGMENT = 'Online TA' (use exact values from schema)
9. LIMIT 20 for multi-row results; omit for single-row aggregates
10. Always use descriptive English aliases (cancellation_rate, avg_daily_rate, total_bookings)
11. For month ordering use:
    CASE ARRIVAL_DATE_MONTH
      WHEN 'January' THEN 1 WHEN 'February' THEN 2 WHEN 'March' THEN 3
      WHEN 'April' THEN 4 WHEN 'May' THEN 5 WHEN 'June' THEN 6
      WHEN 'July' THEN 7 WHEN 'August' THEN 8 WHEN 'September' THEN 9
      WHEN 'October' THEN 10 WHEN 'November' THEN 11 WHEN 'December' THEN 12
    END AS month_num

FEW-SHOT EXAMPLES — use these as a guide for SQL style:

Q: What is the overall cancellation rate?
A: {{"type":"sql","sql":"SELECT ROUND(100.0 * SUM(IS_CANCELED) / COUNT(*), 1) AS cancellation_rate FROM \\"{DB}\\".\\"{SCH}\\".\\"{TBL}\\"","explanation":"Overall cancellation rate across all bookings","chart_type":"none"}}

Q: Which market segment has the highest cancellation rate?
A: {{"type":"sql","sql":"SELECT MARKET_SEGMENT, COUNT(*) AS total_bookings, ROUND(100.0 * SUM(IS_CANCELED) / COUNT(*), 1) AS cancellation_rate FROM \\"{DB}\\".\\"{SCH}\\".\\"{TBL}\\" GROUP BY MARKET_SEGMENT ORDER BY cancellation_rate DESC LIMIT 20","explanation":"Cancellation rate by market segment","chart_type":"bar"}}

Q: How do bookings trend by month?
A: {{"type":"sql","sql":"SELECT ARRIVAL_DATE_MONTH, COUNT(*) AS total_bookings, CASE ARRIVAL_DATE_MONTH WHEN 'January' THEN 1 WHEN 'February' THEN 2 WHEN 'March' THEN 3 WHEN 'April' THEN 4 WHEN 'May' THEN 5 WHEN 'June' THEN 6 WHEN 'July' THEN 7 WHEN 'August' THEN 8 WHEN 'September' THEN 9 WHEN 'October' THEN 10 WHEN 'November' THEN 11 WHEN 'December' THEN 12 END AS month_num FROM \\"{DB}\\".\\"{SCH}\\".\\"{TBL}\\" GROUP BY ARRIVAL_DATE_MONTH ORDER BY month_num","explanation":"Monthly booking volume trend","chart_type":"line"}}

Q: What is the average daily rate by hotel type?
A: {{"type":"sql","sql":"SELECT HOTEL, ROUND(AVG(CASE WHEN ADR > 0 THEN ADR END), 2) AS avg_daily_rate, COUNT(*) AS total_bookings FROM \\"{DB}\\".\\"{SCH}\\".\\"{TBL}\\" GROUP BY HOTEL ORDER BY avg_daily_rate DESC","explanation":"Average daily rate by hotel type","chart_type":"bar"}}

Q: Does lead time correlate with cancellations?
A: {{"type":"sql","sql":"SELECT CASE WHEN LEAD_TIME < 7 THEN '0-7 days' WHEN LEAD_TIME < 30 THEN '8-30 days' WHEN LEAD_TIME < 90 THEN '31-90 days' WHEN LEAD_TIME < 180 THEN '91-180 days' ELSE '180+ days' END AS lead_time_bucket, COUNT(*) AS total_bookings, ROUND(100.0 * SUM(IS_CANCELED) / COUNT(*), 1) AS cancellation_rate FROM \\"{DB}\\".\\"{SCH}\\".\\"{TBL}\\" GROUP BY lead_time_bucket ORDER BY MIN(LEAD_TIME)","explanation":"Cancellation rate by lead time bucket","chart_type":"bar"}}
"""

SYSTEM_RESPOND = """You are a sharp, insightful travel analytics analyst — like a data scientist at Expedia.

A user asked a question, SQL was run against a hotel bookings dataset (2015-2017, two Portuguese hotels), and results came back.
Write a clear, insightful 2-4 sentence answer in plain English.

Rules:
- Round numbers to readable form: "about 37%" not "37.2341%"
- For percentages, use 1 decimal place
- Highlight the single most interesting or surprising finding
- For multi-row results, describe the pattern and name the standout values
- Never mention SQL, databases, column names, or anything technical
- Add one sentence of business context or implication where relevant (e.g. "This suggests Online TA bookings carry higher risk for revenue leakage")
- Be concise — 2-4 sentences max
- Data covers 2015-2017 from two hotels in Portugal — mention this only if directly relevant"""


# Request model 
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


# Agent logic
def ask_llm(messages: list, system: str, max_tokens: int = 800) -> str:
    all_messages = [{"role": "system", "content": system}] + messages
    resp = groq_client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=all_messages,
    )
    return resp.choices[0].message.content.strip()


def fix_column_case(sql: str) -> str:
    """Snowflake uppercases unquoted identifiers.
    For this dataset columns are already uppercase so this is a safety net."""
    return sql


def run_snowflake_query(sql: str) -> tuple[list, list]:
    conn = get_sf_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(50)
        cols = [d[0] for d in cur.description]
        return rows, cols
    finally:
        conn.close()


def parse_json_response(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def agent(user_message: str, history: list) -> dict:
    """Two-step agent: generate SQL → run → explain."""
    _start = time.time()

    # Build message history for context (last 8 turns)
    messages = history[-8:] + [{"role": "user", "content": user_message}]

    # Step 1: Generate SQL (or reject)
    raw = ask_llm(messages, SYSTEM_SQL, max_tokens=800)
    log.info(f"LLM SQL response: {raw[:200]}")

    try:
        parsed = parse_json_response(raw)
    except json.JSONDecodeError:
        return {"type": "error", "message": "I had trouble understanding that question. Could you rephrase it?"}

    if parsed["type"] == "reject":
        log_query({"ts": datetime.now(timezone.utc).isoformat(), "question": user_message,
                   "outcome": "rejected", "response_s": round(time.time() - _start, 2)})
        return {"type": "reject", "message": parsed["message"]}

    sql = parsed["sql"]
    chart_type = parsed.get("chart_type", "none")
    log.info(f"Running SQL: {sql[:300]}")

    # Step 2: Run query
    try:
        rows, cols = run_snowflake_query(sql)
    except Exception as e:
        log.error(f"Snowflake error: {e}")
        fix_msg = [{"role": "user", "content": f"The SQL failed with: {str(e)}\nOriginal question: {user_message}\nBroken SQL: {sql}\nPlease fix it."}]
        raw2 = ask_llm(fix_msg, SYSTEM_SQL, max_tokens=800)
        try:
            parsed2 = parse_json_response(raw2)
            rows, cols = run_snowflake_query(parsed2["sql"])
            sql = parsed2["sql"]
            chart_type = parsed2.get("chart_type", "none")
        except Exception as e2:
            log_query({"ts": datetime.now(timezone.utc).isoformat(), "question": user_message,
                       "outcome": "error", "error": str(e2),
                       "response_s": round(time.time() - _start, 2)})
            return {"type": "error", "message": "I couldn't find data for that question. Try rephrasing or asking about a different topic."}

    # Step 3: Format response
    result_summary = f"Columns: {cols}\nData (first 20 rows): {rows[:20]}"
    explain_messages = [{"role": "user", "content": f"User asked: {user_message}\n\nQuery result:\n{result_summary}"}]
    answer = ask_llm(explain_messages, SYSTEM_RESPOND, max_tokens=400)

    # Serialize rows
    serializable_rows = []
    for row in rows:
        serializable_rows.append([
            float(v) if hasattr(v, '__float__') and not isinstance(v, (int, str, bool)) else v
            for v in row
        ])

    log_query({
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": user_message,
        "outcome": "success",
        "sql": sql,
        "rows_returned": len(rows),
        "chart_type": chart_type,
        "response_s": round(time.time() - _start, 2),
    })

    return {
        "type": "answer",
        "message": answer,
        "sql": sql,
        "columns": cols,
        "rows": serializable_rows,
        "chart_type": chart_type,
    }


# Routes (API endpoints)
@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        result = agent(req.message, req.history)
        return JSONResponse(content=result)
    except Exception as e:
        log.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/logs")
def get_logs(n: int = 50):
    """Return the last n query log entries."""
    if not LOG_FILE.exists():
        return JSONResponse(content={"logs": [], "total": 0})
    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return JSONResponse(content={"logs": entries, "total": len(lines)})
