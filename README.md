# 🗺️ Travel Insights Agent

An interactive chat agent for exploring hotel booking analytics. Ask questions in plain English - the agent converts them to SQL, queries Snowflake, and returns a conversational answer. Trend questions automatically render a Chart.js line chart.

---
## Demo

> URL: [Chat with my agent here](https://travel-insights-agent.onrender.com)
- Note : No login required, may take a minute to wake up (Render free tier limits on inactivity)

> A quick demo video - [Demo Link](https://github.com/EESHAK02/travel-insights-agent/blob/main/demo_video.mp4)

---
## Stack

| Layer | Technology |
|---|---|
| LLM | Groq - Llama 3.3 70B Versatile (free tier) |
| Data | Snowflake - Hotel Booking Demand dataset (2015–2017, ~119k bookings) |
| Backend | Python 3.11 + FastAPI |
| Frontend | Single-page HTML/CSS/JS + Chart.js |
| Hosting | Render.com (free tier) |

---

## Dataset

[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) - Jesse Mostipak, Kaggle.

Two Portuguese hotels (Resort Hotel and City Hotel), 119,390 bookings, 32 features including cancellations, lead time, market segment, ADR, country of origin, and more.

Loaded into Snowflake as: `"BOOKINGS"."PUBLIC"."INFO"`

---

## What you can ask

- **Cancellations:** "Which market segment cancels the most?" / "Does lead time affect cancellations?"
- **Pricing:** "What is the average daily rate by hotel type?" / "How has ADR changed by year?"
- **Segments:** "How does Direct compare to Online TA in cancellation rate?"
- **Seasonality:** "How do bookings trend by month?" ← auto-renders a line chart
- **Geography:** "Which countries book the most?"
- **Guest behaviour:** "What percentage of guests are repeat visitors?"

---

## How it works

```
User asks question
  - Guardrail check (allowed topics only)
  - LLM generates Snowflake SQL + chart_type signal
  - Snowflake executes query against booking data
  - LLM formats result in plain English
  - Response shown with optional chart (trend questions) or data table
```

**Charts:** The LLM returns a `chart_type` field (`"line"`, `"bar"`, or `"none"`). Time-series and trend questions get a `"line"` chart rendered via Chart.js. The raw data table is always accessible via a toggle below the chart.

**Conversation context:** Last 8 turns sent for follow-up question support.

**Guardrails:** Off-topic questions (politics, finance, predictions, etc.) are rejected before SQL generation. 

**SQL transparency:** Every answer includes a collapsible SQL inspector.

---

## Key Insights the Agent Can Surface

Insights that internal teams can check for:

1. **OTA vs Direct cancellation gap** - Core revenue leakage problem for OTAs.
2. **Lead time risk** - Bookings made 180+ days in advance cancel at ~50%+. Short lead time bookings are far more reliable.
3. **Non-refundable deposit paradox** - Non-refundable bookings have high cancellation rates despite the penalty, suggesting guests accept the loss.
4. **Repeat guest loyalty** - Repeat guests cancel at significantly lower rates.

---

## Evaluation

```bash
python eval.py                                                     # local
python eval.py --url https://travel-insights-agent.onrender.com/   # deployed
python eval.py --skip-judge                                        # rule checks only
```

14 test cases: 10 valid analytics questions + 4 guardrail tests - all checks passed. 
Extra check for visualizations: time-series questions verified to trigger `chart_type=line`. 

---

## Monitoring

Every query logged to `query_log.jsonl`. For the deployed version, queries can be monitored at `/logs` on the live URL.

---

## Future Improvements

- **Cross-model eval judge** - use Gemini/GPT-4 instead of same model to avoid self-evaluation bias
- **Date range filter** - UI toggle to filter to 2015 / 2016 / 2017 separately
- **Streaming responses** - token-by-token streaming instead of waiting for full response
- **Session persistence** - save conversation to localStorage so refresh doesn't lose context

--- 

## Local Development

```bash
git clone https://github.com/EESHAK02/travel-insights-agent
cd travel-insights-agent
python -m venv venv && venv\Scripts\activate   # Mac users: source venv/bin/activate  
pip install -r requirements.txt
cp .env.example .env   # fill in your credentials
uvicorn main:app --reload --port 8000
# Open locally accessible link that pops up
```

---

## Deploy to Render

1. Push this repo to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [render.com](https://render.com) - **New Web Service**
3. Connect GitHub repo
4. Settings:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add **Environment Variables** (from `.env` file and Python version 3.11.0) in Render's dashboard
6. Deploy to get a public url

---
