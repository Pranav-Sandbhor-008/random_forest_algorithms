"""
Workforce Risk Predictor  -  Flask app for Vercel
-------------------------------------------------
Model : RandomForestClassifier (random.pkl), 21 input features, 3 classes (0, 1, 2)

Project layout (all in the repo root):
    app.py
    requirements.txt
    random.pkl
    encoders.pkl   <- OPTIONAL (see ENCODING NOTE below)

ENCODING NOTE (important)
-------------------------
Your model was trained on numbers, but several inputs (industry, country, hiring_trend ...)
are text categories. The pickle does NOT contain the encoders used during training, so:

  * Best option : save your training encoders as `encoders.pkl`, a dict like
                  {"industry": LabelEncoder, "country": LabelEncoder, ...}
                  (a plain dict {"Technology": 3, ...} per column also works).
  * Fallback    : this app sorts the option lists below alphabetically and uses the index
                  (exactly what sklearn's LabelEncoder does). For correct results the
                  OPTIONS lists must match the unique values in your training data.

CLASS_INFO below maps the predicted class (0/1/2) to a human label. Edit it to match
what 0, 1 and 2 mean in your dataset.
"""

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
from flask import Flask, Response, jsonify, request

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "random.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

app = Flask(__name__)

# --------------------------------------------------------------------------------------
# 1. CONFIG  (edit these to match your dataset)
# --------------------------------------------------------------------------------------

LEVEL3 = ["Low", "Medium", "High"]

# The order of this list MUST match the model's training column order.
FEATURES = [
    # ---- Company -------------------------------------------------------------
    {"name": "company_name", "label": "Company name", "group": "Company", "type": "text",
     "options": ["Amazon", "Flipkart", "Google", "Infosys", "Meta", "Microsoft", "Oracle",
                 "Salesforce", "TCS", "Wipro"],
     "help": "Start typing or pick a suggestion."},
    {"name": "industry", "label": "Industry", "group": "Company", "type": "select",
     "options": ["Education", "Energy", "Finance", "Healthcare", "Logistics", "Manufacturing",
                 "Media", "Retail", "Technology", "Telecom"]},
    {"name": "country", "label": "Country", "group": "Company", "type": "select",
     "options": ["Australia", "Canada", "Germany", "India", "Singapore", "UK", "USA"]},
    {"name": "company_size", "label": "Company size", "group": "Company", "type": "select",
     "options": ["Startup", "Small", "Medium", "Large", "Enterprise"]},
    {"name": "month", "label": "Month", "group": "Company", "type": "number",
     "min": 1, "max": 12, "step": 1, "default": 6, "help": "1 = January, 12 = December."},
    {"name": "year", "label": "Year", "group": "Company", "type": "number",
     "min": 2018, "max": 2030, "step": 1, "default": 2026},

    # ---- Layoffs -------------------------------------------------------------
    {"name": "layoffs_count", "label": "People laid off", "group": "Layoffs", "type": "number",
     "min": 0, "max": 50000, "step": 10, "default": 500, "unit": "people"},
    {"name": "layoff_percentage", "label": "Share of workforce laid off", "group": "Layoffs",
     "type": "number", "min": 0, "max": 100, "step": 0.5, "default": 5, "unit": "%"},
    {"name": "reason_for_layoffs", "label": "Main reason for layoffs", "group": "Layoffs",
     "type": "select",
     "options": ["AI Automation", "Budget Cuts", "Cost Cutting", "Market Slowdown",
                 "Over-hiring", "Restructuring"]},

    # ---- Hiring & market -------------------------------------------------------
    {"name": "open_roles", "label": "Open roles", "group": "Hiring & market", "type": "number",
     "min": 0, "max": 20000, "step": 10, "default": 300, "unit": "roles"},
    {"name": "hiring_trend", "label": "Hiring trend", "group": "Hiring & market",
     "type": "select", "options": ["Decreasing", "Stable", "Increasing"]},
    {"name": "top_hiring_role", "label": "Most hired role", "group": "Hiring & market",
     "type": "select",
     "options": ["AI/ML Engineer", "Customer Support", "Data Scientist", "Designer",
                 "Operations", "Sales", "Software Engineer"]},
    {"name": "remote_jobs_percentage", "label": "Remote jobs", "group": "Hiring & market",
     "type": "number", "min": 0, "max": 100, "step": 1, "default": 30, "unit": "%"},
    {"name": "stock_growth_percent", "label": "Stock growth", "group": "Hiring & market",
     "type": "number", "min": -100, "max": 200, "step": 0.5, "default": 4, "unit": "%"},
    {"name": "revenue_growth_percent", "label": "Revenue growth", "group": "Hiring & market",
     "type": "number", "min": -100, "max": 200, "step": 0.5, "default": 6, "unit": "%"},
    {"name": "salary_budget_change", "label": "Salary budget change", "group": "Hiring & market",
     "type": "number", "min": -50, "max": 50, "step": 0.5, "default": 2, "unit": "%"},

    # ---- AI & people ---------------------------------------------------------
    {"name": "ai_automation_impact", "label": "AI automation impact", "group": "AI & people",
     "type": "select", "options": LEVEL3},
    {"name": "ai_replacement_risk", "label": "AI replacement risk", "group": "AI & people",
     "type": "select", "options": LEVEL3},
    {"name": "ai_adoption_level", "label": "AI adoption level", "group": "AI & people",
     "type": "select", "options": LEVEL3},
    {"name": "employee_sentiment", "label": "Employee sentiment", "group": "AI & people",
     "type": "select", "options": ["Negative", "Neutral", "Positive"]},
    {"name": "job_security_score", "label": "Job security score", "group": "AI & people",
     "type": "number", "min": 0, "max": 100, "step": 1, "default": 60, "unit": "/ 100",
     "help": "Higher means people feel safer in their jobs."},
]

# What each predicted class means. EDIT THESE to match your dataset.
CLASS_INFO = {
    0: {
        "label": "Low risk", "tone": "good",
        "headline": "The outlook looks stable",
        "meaning": "Based on these numbers, the model sees few warning signs. Conditions "
                   "look similar to companies that stayed steady.",
        "advice": ["Keep an eye on hiring and revenue trends each quarter.",
                   "Keep building skills that stay valuable as AI tools spread.",
                   "No urgent action needed, but stay informed."],
    },
    1: {
        "label": "Moderate risk", "tone": "warn",
        "headline": "Some signs of pressure",
        "meaning": "The numbers show a mix of healthy and worrying signals. This company "
                   "resembles ones that faced some disruption but were not in crisis.",
        "advice": ["Update your resume and portfolio so you are ready if things change.",
                   "Build skills in areas the company is actively hiring for.",
                   "Watch for further layoff news, hiring freezes or falling revenue."],
    },
    2: {
        "label": "High risk", "tone": "bad",
        "headline": "Strong warning signs",
        "meaning": "The numbers resemble companies that went through serious workforce "
                   "disruption. Conditions here deserve close attention.",
        "advice": ["Prepare now: refresh your resume and start networking.",
                   "Build an emergency fund if you can.",
                   "Look at roles the market is hiring for and upskill toward them."],
    },
}

PRESETS = {
    "Stable company": {
        "company_name": "Microsoft", "industry": "Technology", "country": "USA",
        "company_size": "Enterprise", "month": 6, "year": 2026, "layoffs_count": 50,
        "layoff_percentage": 0.5, "reason_for_layoffs": "Restructuring", "open_roles": 2500,
        "hiring_trend": "Increasing", "top_hiring_role": "AI/ML Engineer",
        "remote_jobs_percentage": 40, "stock_growth_percent": 18, "revenue_growth_percent": 14,
        "salary_budget_change": 5, "ai_automation_impact": "Low", "ai_replacement_risk": "Low",
        "ai_adoption_level": "Medium", "employee_sentiment": "Positive",
        "job_security_score": 85,
    },
    "Under pressure": {
        "company_name": "Wipro", "industry": "Technology", "country": "India",
        "company_size": "Large", "month": 3, "year": 2026, "layoffs_count": 4000,
        "layoff_percentage": 14, "reason_for_layoffs": "AI Automation", "open_roles": 120,
        "hiring_trend": "Decreasing", "top_hiring_role": "Customer Support",
        "remote_jobs_percentage": 15, "stock_growth_percent": -22, "revenue_growth_percent": -9,
        "salary_budget_change": -6, "ai_automation_impact": "High", "ai_replacement_risk": "High",
        "ai_adoption_level": "High", "employee_sentiment": "Negative",
        "job_security_score": 28,
    },
}

# --------------------------------------------------------------------------------------
# 2. LOAD MODEL (+ optional encoders)
# --------------------------------------------------------------------------------------

MODEL = None
MODEL_ERROR = None
try:
    with open(MODEL_PATH, "rb") as fh:
        MODEL = pickle.load(fh)
except Exception as exc:  # keep the app alive so the UI can show a helpful message
    MODEL_ERROR = f"{type(exc).__name__}: {exc}"

ENCODERS = {}
if os.path.exists(ENCODERS_PATH):
    try:
        with open(ENCODERS_PATH, "rb") as fh:
            ENCODERS = pickle.load(fh)
    except Exception:
        ENCODERS = {}

FEATURE_BY_NAME = {f["name"]: f for f in FEATURES}
FEATURE_ORDER = [f["name"] for f in FEATURES]
if MODEL is not None and hasattr(MODEL, "feature_names_in_"):
    FEATURE_ORDER = list(MODEL.feature_names_in_)  # trust the model's own column order

IMPORTANCE = {}
if MODEL is not None and hasattr(MODEL, "feature_importances_"):
    IMPORTANCE = dict(zip(FEATURE_ORDER, [float(v) for v in MODEL.feature_importances_]))


# --------------------------------------------------------------------------------------
# 3. HELPERS
# --------------------------------------------------------------------------------------

def encode_category(name, value):
    """Return (encoded_number, seen_in_training)."""
    enc = ENCODERS.get(name)
    if enc is not None:
        if isinstance(enc, dict):
            if value in enc:
                return float(enc[value]), True
        elif hasattr(enc, "classes_"):
            classes = [str(c) for c in enc.classes_]
            if str(value) in classes:
                return float(classes.index(str(value))), True
        return 0.0, False

    classes = sorted(FEATURE_BY_NAME[name].get("options", []))
    lowered = [c.lower() for c in classes]
    if str(value).strip().lower() in lowered:
        return float(lowered.index(str(value).strip().lower())), True
    return 0.0, False


def build_row(values):
    """Validate + encode user input. Returns (row, errors, notes, display_values)."""
    row, errors, notes, display = [], {}, [], {}
    for name in FEATURE_ORDER:
        spec = FEATURE_BY_NAME.get(name)
        raw = values.get(name)
        if spec is None:
            errors[name] = "Unknown feature."
            continue
        if raw is None or str(raw).strip() == "":
            errors[name] = "This field is required."
            continue

        if spec["type"] == "number":
            try:
                num = float(raw)
                if np.isnan(num) or np.isinf(num):
                    raise ValueError
            except (TypeError, ValueError):
                errors[name] = "Enter a valid number."
                continue
            if num < spec["min"] or num > spec["max"]:
                errors[name] = f"Use a value between {spec['min']} and {spec['max']}."
                continue
            row.append(num)
            display[name] = num
        else:
            code, seen = encode_category(name, raw)
            if not seen:
                notes.append(f"\u201c{raw}\u201d for {spec['label'].lower()} was not recognised, "
                             "so it was treated as a default value.")
            row.append(code)
            display[name] = str(raw)
    return row, errors, notes, display


def fmt_num(v):
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"


def build_insights(v):
    """Plain-language observations about the inputs. Each: {tone, text}."""
    out = []

    def add(tone, text):
        out.append({"tone": tone, "text": text})

    pct = v.get("layoff_percentage")
    if pct is not None:
        if pct >= 15:
            add("bad", f"{fmt_num(pct)}% of the workforce was laid off, which is a very large cut.")
        elif pct >= 5:
            add("warn", f"{fmt_num(pct)}% of staff were laid off, a noticeable reduction.")
        else:
            add("good", f"Only {fmt_num(pct)}% of staff were laid off, which is small.")

    cnt = v.get("layoffs_count")
    if cnt is not None and cnt >= 1000:
        add("warn", f"About {fmt_num(cnt)} people lost their jobs, a high absolute number.")

    rev = v.get("revenue_growth_percent")
    if rev is not None:
        if rev < 0:
            add("bad", f"Revenue is shrinking ({fmt_num(rev)}%), which often leads to more cost cutting.")
        elif rev >= 10:
            add("good", f"Revenue is growing well (+{fmt_num(rev)}%), which supports job stability.")

    stock = v.get("stock_growth_percent")
    if stock is not None and stock <= -10:
        add("warn", f"The stock fell {fmt_num(abs(stock))}%, which adds pressure on management to cut costs.")

    trend = v.get("hiring_trend")
    if trend == "Decreasing":
        add("bad", "Hiring is slowing down, so fewer new roles are opening up.")
    elif trend == "Increasing":
        add("good", "Hiring is picking up, a sign the company is investing in people.")

    if v.get("open_roles") is not None and v["open_roles"] >= 1000:
        add("good", f"There are {fmt_num(v['open_roles'])} open roles, so demand for talent is high.")

    if v.get("ai_replacement_risk") == "High":
        add("bad", "AI replacement risk is high, so some jobs here could be automated.")
    elif v.get("ai_replacement_risk") == "Low":
        add("good", "AI replacement risk is low, so most roles look safe from automation.")

    if v.get("employee_sentiment") == "Negative":
        add("warn", "Employee sentiment is negative, which often signals unrest or low morale.")
    elif v.get("employee_sentiment") == "Positive":
        add("good", "Employee sentiment is positive, a healthy sign.")

    jss = v.get("job_security_score")
    if jss is not None:
        if jss < 40:
            add("bad", f"The job security score is low ({fmt_num(jss)}/100).")
        elif jss >= 70:
            add("good", f"The job security score is strong ({fmt_num(jss)}/100).")

    sal = v.get("salary_budget_change")
    if sal is not None and sal < 0:
        add("warn", f"The salary budget is being cut ({fmt_num(sal)}%).")

    order = {"bad": 0, "warn": 1, "good": 2}
    out.sort(key=lambda x: order[x["tone"]])
    return out[:7]


def confidence_text(p):
    if p >= 0.75:
        return "The model is very sure about this result."
    if p >= 0.55:
        return "The model leans this way, but it is not a certain call."
    return "This is a close call. The model sees mixed signals, so treat it with caution."


def describe_drivers(display):
    drivers = sorted(IMPORTANCE.items(), key=lambda kv: kv[1], reverse=True)[:5]
    out = []
    for name, imp in drivers:
        spec = FEATURE_BY_NAME.get(name, {"label": name, "unit": ""})
        val = display.get(name, "")
        if isinstance(val, float):
            val = fmt_num(val)
        unit = spec.get("unit", "")
        out.append({"label": spec["label"], "value": f"{val} {unit}".strip(),
                    "weight": round(imp * 100, 1)})
    return out


# --------------------------------------------------------------------------------------
# 4. ROUTES
# --------------------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify(status="ok" if MODEL is not None else "model_not_loaded", error=MODEL_ERROR)


@app.get("/api/schema")
def schema():
    return jsonify(features=FEATURES, presets=PRESETS,
                   model_ready=MODEL is not None, model_error=MODEL_ERROR)


@app.post("/api/predict")
def predict():
    if MODEL is None:
        return jsonify(ok=False, error="The model file could not be loaded. "
                                       "Make sure random.pkl is in the project root."), 500

    payload = request.get_json(silent=True) or {}
    values = payload.get("values", payload)
    row, errors, notes, display = build_row(values)
    if errors:
        return jsonify(ok=False, error="Please fix the highlighted fields.", fields=errors), 400

    try:
        X = np.array([row], dtype=float)
        probs = MODEL.predict_proba(X)[0]
        classes = [int(c) for c in MODEL.classes_]
        best = int(np.argmax(probs))
        pred = classes[best]
    except Exception as exc:
        return jsonify(ok=False, error=f"Prediction failed: {exc}"), 500

    info = CLASS_INFO.get(pred, {"label": f"Class {pred}", "tone": "warn",
                                 "headline": f"Predicted class {pred}",
                                 "meaning": "No description is configured for this class.",
                                 "advice": []})
    conf = float(probs[best])
    return jsonify(
        ok=True,
        prediction=pred,
        label=info["label"],
        tone=info["tone"],
        headline=info["headline"],
        meaning=info["meaning"],
        advice=info["advice"],
        confidence=round(conf * 100, 1),
        confidence_text=confidence_text(conf),
        probabilities=[
            {"label": CLASS_INFO.get(c, {"label": f"Class {c}"})["label"],
             "tone": CLASS_INFO.get(c, {"tone": "warn"})["tone"],
             "value": round(float(p) * 100, 1)}
            for c, p in zip(classes, probs)
        ],
        insights=build_insights(display),
        drivers=describe_drivers(display),
        notes=notes,
        time=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


# --------------------------------------------------------------------------------------
# 5. FRONT-END (single page, light + dark theme)
# --------------------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Workforce Risk Predictor</title>
<meta name="color-scheme" content="light dark">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script>
  (function () {
    var saved = null;
    try { saved = localStorage.getItem("theme"); } catch (e) {}
    var dark = saved ? saved === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  })();
</script>
<style>
:root{
  --bg:#F3F5F8; --surface:#FFFFFF; --surface2:#F7F9FB; --ink:#142030; --muted:#5A6B7D;
  --line:#DCE3EA; --accent:#1F5FD1; --accent-ink:#FFFFFF; --accent-soft:#E6EEFB;
  --good:#16885E; --good-soft:#E1F4EC; --warn:#B87A06; --warn-soft:#FBF0D8;
  --bad:#CC3B33; --bad-soft:#FBE6E4; --shadow:0 1px 2px rgba(20,32,48,.06),0 8px 24px rgba(20,32,48,.06);
  --radius:14px; color-scheme:light;
}
:root[data-theme="dark"]{
  --bg:#0C1118; --surface:#141B24; --surface2:#19222D; --ink:#E7EDF4; --muted:#93A2B2;
  --line:#25303D; --accent:#6C9DFF; --accent-ink:#0A1220; --accent-soft:#1B2C4A;
  --good:#3FD6A0; --good-soft:#12302A; --warn:#F2B84B; --warn-soft:#352A12;
  --bad:#FF7B72; --bad-soft:#3A1C1B; --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
  color-scheme:dark;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Manrope",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased;transition:background .25s,color .25s}
button,input,select{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}

/* header */
header.top{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.top .wrap{display:flex;align-items:center;justify-content:space-between;height:62px;gap:12px}
.brand{display:flex;align-items:center;gap:10px;font-weight:800;letter-spacing:-.01em;font-size:17px}
.mark{width:30px;height:30px;border-radius:9px;background:var(--accent);display:grid;place-items:center}
.mark svg{width:18px;height:18px;stroke:var(--accent-ink)}
.tools{display:flex;gap:8px;align-items:center}
.btn{border:1px solid var(--line);background:var(--surface);border-radius:10px;padding:8px 13px;cursor:pointer;
  font-weight:600;font-size:13.5px;display:inline-flex;align-items:center;gap:7px;transition:background .15s,border-color .15s,transform .1s}
.btn:hover{background:var(--surface2);border-color:color-mix(in srgb,var(--accent) 40%,var(--line))}
.btn:active{transform:translateY(1px)}
.btn svg{width:16px;height:16px}
.btn.primary{background:var(--accent);color:var(--accent-ink);border-color:transparent}
.btn.primary:hover{background:color-mix(in srgb,var(--accent) 88%,#000)}
.btn.ghost{background:transparent}
.btn[disabled]{opacity:.55;cursor:not-allowed}
.theme .sun{display:none}
:root[data-theme="dark"] .theme .sun{display:block}
:root[data-theme="dark"] .theme .moon{display:none}

/* intro */
.intro{padding:34px 0 20px;max-width:720px}
.intro h1{font-size:clamp(28px,4.2vw,40px);line-height:1.12;letter-spacing:-.025em;margin:0 0 10px;font-weight:800}
.intro p{margin:0;color:var(--muted);font-size:16.5px}

/* layout */
.grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:22px;align-items:start;padding-bottom:60px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
.card-h{padding:18px 22px 0;display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.card-h h2{margin:0;font-size:17px;font-weight:700;letter-spacing:-.01em}
.card-b{padding:18px 22px 22px}
.result-card{position:sticky;top:78px}

/* progress + tabs */
.progress{height:6px;background:var(--surface2);border-radius:99px;overflow:hidden;margin:14px 22px 0;border:1px solid var(--line)}
.progress i{display:block;height:100%;width:0;background:var(--accent);border-radius:99px;transition:width .35s ease}
.progress-label{padding:6px 22px 0;color:var(--muted);font-size:12.5px;font-weight:600}
.tabs{display:flex;gap:6px;padding:14px 22px 0;overflow-x:auto;scrollbar-width:none}
.tabs::-webkit-scrollbar{display:none}
.tab{border:1px solid var(--line);background:var(--surface);color:var(--muted);padding:8px 14px;border-radius:99px;cursor:pointer;
  font-weight:600;font-size:13.5px;white-space:nowrap;display:inline-flex;gap:7px;align-items:center;transition:all .15s}
.tab:hover{color:var(--ink)}
.tab[aria-selected="true"]{background:var(--accent-soft);color:var(--accent);border-color:color-mix(in srgb,var(--accent) 45%,var(--line))}
.tab .dot{width:8px;height:8px;border-radius:50%;background:var(--line)}
.tab.done .dot{background:var(--good)}

/* fields */
.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px 18px}
.field{display:flex;flex-direction:column;gap:6px;min-width:0}
.field.wide{grid-column:1/-1}
.field label{font-weight:600;font-size:13.5px;display:flex;justify-content:space-between;gap:8px}
.field .hint{font-size:12px;color:var(--muted);font-weight:500}
.ctl{width:100%;border:1px solid var(--line);background:var(--surface2);border-radius:10px;padding:10px 12px;transition:border-color .15s,box-shadow .15s,background .15s}
.ctl:hover{border-color:color-mix(in srgb,var(--accent) 35%,var(--line))}
.ctl:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 22%,transparent);background:var(--surface)}
.num{display:flex;align-items:center;gap:8px}
.num .ctl{width:116px;flex:none;text-align:right;font-variant-numeric:tabular-nums}
.unit{font-size:12.5px;color:var(--muted);min-width:36px;font-weight:600}
input[type=range]{flex:1;min-width:0;accent-color:var(--accent);height:22px;cursor:pointer}
.field.invalid .ctl{border-color:var(--bad)}
.err{color:var(--bad);font-size:12.5px;font-weight:600;min-height:0}
.nav{display:flex;justify-content:space-between;gap:10px;margin-top:22px;flex-wrap:wrap}
.nav .right{display:flex;gap:8px;margin-left:auto}
.presets{display:flex;gap:6px;flex-wrap:wrap}

/* result */
.empty{padding:34px 10px;text-align:center;color:var(--muted)}
.empty svg{width:44px;height:44px;stroke:var(--muted);margin-bottom:8px}
.empty h3{margin:0 0 4px;color:var(--ink);font-size:16px}
.verdict{display:flex;flex-direction:column;align-items:center;text-align:center;padding-top:4px}
.gauge{width:230px;max-width:100%;position:relative}
.gauge svg{width:100%;display:block;overflow:visible}
.gauge .track{stroke:var(--line)}
.gauge .fill{transition:stroke-dashoffset .9s cubic-bezier(.2,.8,.2,1)}
.gauge .pct{position:absolute;inset:auto 0 4px 0;text-align:center;font-size:34px;font-weight:800;letter-spacing:-.03em;line-height:1}
.gauge .pct small{display:block;font-size:12px;font-weight:600;color:var(--muted);letter-spacing:0;margin-top:3px}
.pill{display:inline-block;padding:5px 14px;border-radius:99px;font-weight:800;font-size:14px;margin-top:12px}
.tone-good{--t:var(--good);--ts:var(--good-soft)}
.tone-warn{--t:var(--warn);--ts:var(--warn-soft)}
.tone-bad{--t:var(--bad);--ts:var(--bad-soft)}
.pill{background:var(--ts);color:var(--t)}
.verdict h3{margin:10px 0 6px;font-size:21px;letter-spacing:-.015em}
.verdict p{margin:0;color:var(--muted);max-width:44ch}
.conf-note{margin-top:8px;font-size:13px;color:var(--muted)}
.sec{margin-top:22px;padding-top:18px;border-top:1px solid var(--line)}
.sec h4{margin:0 0 10px;font-size:14px;font-weight:700}
.bars{display:flex;flex-direction:column;gap:9px}
.bar-row{display:grid;grid-template-columns:104px 1fr 48px;gap:10px;align-items:center;font-size:13px;font-weight:600}
.bar-row .b{height:9px;border-radius:99px;background:var(--surface2);border:1px solid var(--line);overflow:hidden}
.bar-row .b i{display:block;height:100%;background:var(--t);width:0;transition:width .8s cubic-bezier(.2,.8,.2,1)}
.bar-row span:last-child{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.list li{display:flex;gap:10px;align-items:flex-start;font-size:14px}
.list .ic{flex:none;width:20px;height:20px;border-radius:50%;background:var(--ts);color:var(--t);display:grid;place-items:center;font-size:12px;font-weight:800;margin-top:1px}
.drv{display:grid;grid-template-columns:1fr auto;gap:2px 10px;font-size:13.5px}
.drv .n{font-weight:600}.drv .v{color:var(--muted);text-align:right}
.drv .w{grid-column:1/-1;height:4px;background:var(--surface2);border-radius:9px;overflow:hidden;margin-bottom:8px}
.drv .w i{display:block;height:100%;background:var(--accent)}
.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:20px}
.note{margin-top:14px;font-size:12.5px;color:var(--muted);background:var(--surface2);border:1px solid var(--line);border-radius:10px;padding:9px 12px}
.disclaimer{margin-top:16px;font-size:12px;color:var(--muted)}

/* history */
.hist{margin-top:22px}
.hist-item{width:100%;text-align:left;display:flex;align-items:center;justify-content:space-between;gap:10px;border:1px solid var(--line);
  background:var(--surface);border-radius:10px;padding:9px 12px;margin-top:8px;cursor:pointer;font-size:13.5px;transition:background .15s}
.hist-item:hover{background:var(--surface2)}
.hist-item .tag{padding:2px 10px;border-radius:99px;font-weight:700;font-size:12.5px;background:var(--ts);color:var(--t)}
.hist-item small{color:var(--muted)}

.spin{width:16px;height:16px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;animation:sp .7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.toast{position:fixed;left:50%;bottom:24px;transform:translate(-50%,20px);background:var(--ink);color:var(--bg);padding:10px 16px;border-radius:10px;
  font-weight:600;font-size:13.5px;opacity:0;pointer-events:none;transition:all .25s;z-index:50}
.toast.show{opacity:1;transform:translate(-50%,0)}
.banner{background:var(--bad-soft);color:var(--bad);border:1px solid var(--bad);border-radius:12px;padding:12px 16px;margin-bottom:16px;font-weight:600}
.hidden{display:none!important}

@media (max-width:900px){
  .grid{grid-template-columns:1fr}
  .result-card{position:static}
}
@media (max-width:560px){
  .fields{grid-template-columns:1fr}
  .btn .lbl{display:none}
  .btn.keep .lbl{display:inline}
  .bar-row{grid-template-columns:88px 1fr 44px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
</head>
<body>
<header class="top">
  <div class="wrap">
    <div class="brand">
      <span class="mark"><svg viewBox="0 0 24 24" fill="none" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/></svg></span>
      Workforce Risk Predictor
    </div>
    <div class="tools">
      <button class="btn theme" id="themeBtn" aria-label="Switch between light and dark theme" title="Switch theme">
        <svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/></svg>
        <svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
        <span class="lbl" id="themeLbl">Dark</span>
      </button>
    </div>
  </div>
</header>

<main class="wrap">
  <section class="intro">
    <h1>See how risky a company's workforce situation looks</h1>
    <p>Fill in what you know about layoffs, hiring and AI. You get a clear result in plain language, with the reasons behind it.</p>
  </section>

  <div id="banner" class="banner hidden" role="alert"></div>

  <div class="grid">
    <!-- FORM -->
    <section class="card" aria-labelledby="formTitle">
      <div class="card-h">
        <h2 id="formTitle">Company details</h2>
        <div class="presets" id="presets" aria-label="Load an example"></div>
      </div>
      <div class="progress-label" id="progLabel">0 of 0 fields filled</div>
      <div class="progress" aria-hidden="true"><i id="progBar"></i></div>
      <div class="tabs" id="tabs" role="tablist"></div>
      <div class="card-b">
        <form id="form" novalidate>
          <div id="panels"></div>
          <div class="nav">
            <button type="button" class="btn ghost" id="resetBtn">Reset</button>
            <div class="right">
              <button type="button" class="btn" id="backBtn">Back</button>
              <button type="button" class="btn" id="nextBtn">Next</button>
              <button type="submit" class="btn primary keep" id="goBtn"><span class="lbl">Predict</span></button>
            </div>
          </div>
        </form>
      </div>
    </section>

    <!-- RESULT -->
    <aside class="card result-card" aria-live="polite" aria-labelledby="resTitle">
      <div class="card-h"><h2 id="resTitle">Result</h2></div>
      <div class="card-b" id="result">
        <div class="empty">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19V5M4 19h16M8 15v-4M12 15V8M16 15v-6"/></svg>
          <h3>No result yet</h3>
          <p>Complete the form and choose Predict, or load an example above. Shortcut: Ctrl + Enter.</p>
        </div>
      </div>
    </aside>
  </div>

  <section class="hist wrap" style="padding:0 0 60px;max-width:720px;margin:-30px 0 0;display:none" id="histBox">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <h2 style="font-size:16px;margin:0">Recent predictions</h2>
      <button class="btn ghost" id="clearHist" type="button">Clear</button>
    </div>
    <div id="histList"></div>
  </section>
</main>
<div class="toast" id="toast" role="status"></div>

<script>
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const safeStore = {
  get(k){ try { return localStorage.getItem(k); } catch(e){ return null; } },
  set(k,v){ try { localStorage.setItem(k,v); } catch(e){} }
};

let SCHEMA = null, GROUPS = [], active = 0, LAST = null;

/* ---------- theme ---------- */
function applyThemeLabel(){
  const dark = document.documentElement.getAttribute("data-theme") === "dark";
  $("#themeLbl").textContent = dark ? "Light" : "Dark";
}
$("#themeBtn").addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  safeStore.set("theme", next); applyThemeLabel();
});
applyThemeLabel();

function toast(msg){
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2200);
}

/* ---------- build form ---------- */
async function init(){
  try{
    const r = await fetch("/api/schema"); SCHEMA = await r.json();
  }catch(e){
    showBanner("Could not load the form. Please refresh the page."); return;
  }
  if(!SCHEMA.model_ready) showBanner("The model file could not be loaded. Make sure random.pkl is in the project root. " + (SCHEMA.model_error || ""));
  GROUPS = [...new Set(SCHEMA.features.map(f => f.group))];
  buildTabs(); buildPanels(); buildPresets(); showTab(0); updateProgress(); renderHistory();
}
function showBanner(t){ const b = $("#banner"); b.textContent = t; b.classList.remove("hidden"); }

function buildTabs(){
  const wrap = $("#tabs"); wrap.innerHTML = "";
  GROUPS.forEach((g, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "tab"; b.id = "tab" + i; b.dataset.i = i;
    b.setAttribute("role", "tab");
    b.innerHTML = `<span class="dot"></span>${esc(g)}`;
    b.addEventListener("click", () => showTab(i));
    wrap.appendChild(b);
  });
}

function fieldHTML(f){
  const id = "f_" + f.name;
  const hint = f.help ? `<span class="hint">${esc(f.help)}</span>` : "";
  let ctl = "";
  if(f.type === "number"){
    ctl = `<div class="num">
      <input type="range" id="${id}_r" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.default}" aria-label="${esc(f.label)} slider">
      <input class="ctl" type="number" inputmode="decimal" id="${id}" name="${f.name}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.default}">
      <span class="unit">${esc(f.unit || "")}</span></div>`;
  } else if(f.type === "select"){
    ctl = `<select class="ctl" id="${id}" name="${f.name}"><option value="">Choose...</option>` +
      f.options.map(o => `<option value="${esc(o)}">${esc(o)}</option>`).join("") + `</select>`;
  } else {
    ctl = `<input class="ctl" id="${id}" name="${f.name}" list="${id}_l" autocomplete="off" placeholder="e.g. ${esc(f.options[0] || "")}">
      <datalist id="${id}_l">${f.options.map(o => `<option value="${esc(o)}">`).join("")}</datalist>`;
  }
  return `<div class="field ${f.type === "number" ? "wide" : ""}" data-name="${f.name}">
    <label for="${id}"><span>${esc(f.label)}</span>${hint}</label>${ctl}<div class="err" id="${id}_e"></div></div>`;
}

function buildPanels(){
  const wrap = $("#panels"); wrap.innerHTML = "";
  GROUPS.forEach((g, i) => {
    const p = document.createElement("div");
    p.className = "fields hidden"; p.id = "panel" + i; p.setAttribute("role", "tabpanel");
    p.innerHTML = SCHEMA.features.filter(f => f.group === g).map(fieldHTML).join("");
    wrap.appendChild(p);
  });
  SCHEMA.features.forEach(f => {
    const el = $("#f_" + f.name);
    el.addEventListener("input", () => {
      if(f.type === "number"){ const r = $("#f_" + f.name + "_r"); if(r) r.value = el.value; }
      clearError(f.name); updateProgress();
    });
    if(f.type === "number"){
      $("#f_" + f.name + "_r").addEventListener("input", e => {
        el.value = e.target.value; clearError(f.name); updateProgress();
      });
    }
  });
}

function buildPresets(){
  const w = $("#presets"); w.innerHTML = "";
  Object.keys(SCHEMA.presets).forEach(name => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "btn"; b.textContent = name;
    b.addEventListener("click", () => { setValues(SCHEMA.presets[name]); toast("Loaded: " + name); });
    w.appendChild(b);
  });
}

function showTab(i){
  active = Math.max(0, Math.min(GROUPS.length - 1, i));
  GROUPS.forEach((_, k) => {
    $("#panel" + k).classList.toggle("hidden", k !== active);
    $("#tab" + k).setAttribute("aria-selected", k === active);
  });
  $("#backBtn").classList.toggle("hidden", active === 0);
  $("#nextBtn").classList.toggle("hidden", active === GROUPS.length - 1);
}
$("#backBtn").addEventListener("click", () => showTab(active - 1));
$("#nextBtn").addEventListener("click", () => showTab(active + 1));

/* ---------- values ---------- */
function getValues(){
  const v = {};
  SCHEMA.features.forEach(f => v[f.name] = $("#f_" + f.name).value.trim());
  return v;
}
function setValues(vals){
  SCHEMA.features.forEach(f => {
    if(vals[f.name] === undefined) return;
    const el = $("#f_" + f.name); el.value = vals[f.name];
    if(f.type === "number") $("#f_" + f.name + "_r").value = vals[f.name];
    clearError(f.name);
  });
  updateProgress();
}
function isFilled(f){ return $("#f_" + f.name).value.trim() !== ""; }
function updateProgress(){
  const total = SCHEMA.features.length;
  const done = SCHEMA.features.filter(isFilled).length;
  $("#progBar").style.width = (done / total * 100) + "%";
  $("#progLabel").textContent = `${done} of ${total} fields filled`;
  GROUPS.forEach((g, i) => {
    const fs = SCHEMA.features.filter(f => f.group === g);
    $("#tab" + i).classList.toggle("done", fs.every(isFilled));
  });
}
function clearError(name){
  const f = $(`.field[data-name="${name}"]`); if(!f) return;
  f.classList.remove("invalid"); $("#f_" + name + "_e").textContent = "";
}
function setError(name, msg){
  const f = $(`.field[data-name="${name}"]`); if(!f) return;
  f.classList.add("invalid"); $("#f_" + name + "_e").textContent = msg;
}
$("#resetBtn").addEventListener("click", () => {
  SCHEMA.features.forEach(f => {
    const el = $("#f_" + f.name);
    el.value = f.type === "number" ? f.default : "";
    if(f.type === "number") $("#f_" + f.name + "_r").value = f.default;
    clearError(f.name);
  });
  updateProgress(); showTab(0); toast("Form reset");
});

/* ---------- predict ---------- */
$("#form").addEventListener("submit", async e => { e.preventDefault(); await predict(); });
document.addEventListener("keydown", e => {
  if((e.ctrlKey || e.metaKey) && e.key === "Enter"){ e.preventDefault(); predict(); }
});

async function predict(){
  if(!SCHEMA) return;
  const values = getValues();
  let firstBad = -1;
  SCHEMA.features.forEach(f => {
    const val = values[f.name];
    if(val === ""){ setError(f.name, "This field is required."); }
    else if(f.type === "number" && (isNaN(+val) || +val < f.min || +val > f.max)){
      setError(f.name, `Use a value between ${f.min} and ${f.max}.`);
    } else { clearError(f.name); return; }
    if(firstBad < 0) firstBad = GROUPS.indexOf(f.group);
  });
  if(firstBad >= 0){ showTab(firstBad); toast("Please fix the highlighted fields"); return; }

  const btn = $("#goBtn"); const old = btn.innerHTML;
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span><span class="lbl">Analysing</span>';
  try{
    const r = await fetch("/api/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({values})});
    const d = await r.json();
    if(!d.ok){
      if(d.fields){ Object.entries(d.fields).forEach(([k, m]) => setError(k, m)); }
      toast(d.error || "Something went wrong"); return;
    }
    LAST = {data: d, values};
    renderResult(d); saveHistory(d, values);
    if(window.matchMedia("(max-width:900px)").matches) $("#result").scrollIntoView({behavior:"smooth", block:"start"});
  }catch(err){
    toast("Network error. Please try again.");
  }finally{
    btn.disabled = false; btn.innerHTML = old;
  }
}

function renderResult(d){
  const tone = "tone-" + d.tone;
  const insights = d.insights.length ? `<div class="sec"><h4>What stands out</h4><ul class="list">` +
    d.insights.map(i => `<li class="tone-${i.tone}"><span class="ic">${i.tone === "good" ? "+" : i.tone === "bad" ? "!" : "~"}</span><span>${esc(i.text)}</span></li>`).join("") + `</ul></div>` : "";
  const advice = d.advice.length ? `<div class="sec"><h4>What you can do</h4><ul class="list ${tone}">` +
    d.advice.map(a => `<li><span class="ic">&#10003;</span><span>${esc(a)}</span></li>`).join("") + `</ul></div>` : "";
  const bars = d.probabilities.map(p => `<div class="bar-row tone-${p.tone}"><span>${esc(p.label)}</span><div class="b"><i data-w="${p.value}"></i></div><span>${p.value}%</span></div>`).join("");
  const drivers = d.drivers.map(x => `<div class="n">${esc(x.label)}</div><div class="v">${esc(x.value)}</div><div class="w"><i style="width:${Math.min(100, x.weight * 2.4)}%"></i></div>`).join("");
  const notes = d.notes.length ? d.notes.map(n => `<div class="note">${esc(n)}</div>`).join("") : "";

  $("#result").innerHTML = `
    <div class="verdict ${tone}">
      <div class="gauge">
        <svg viewBox="0 0 200 110" aria-hidden="true">
          <path class="track" d="M10 100 A90 90 0 0 1 190 100" fill="none" stroke-width="14" stroke-linecap="round" pathLength="100"/>
          <path class="fill" id="gFill" d="M10 100 A90 90 0 0 1 190 100" fill="none" stroke="var(--t)" stroke-width="14" stroke-linecap="round" pathLength="100" stroke-dasharray="100" stroke-dashoffset="100"/>
        </svg>
        <div class="pct">${d.confidence}%<small>model confidence</small></div>
      </div>
      <span class="pill">${esc(d.label)}</span>
      <h3>${esc(d.headline)}</h3>
      <p>${esc(d.meaning)}</p>
      <div class="conf-note">${esc(d.confidence_text)}</div>
    </div>
    ${notes}
    <div class="sec"><h4>How the model splits its vote</h4><div class="bars">${bars}</div></div>
    ${insights}
    <div class="sec"><h4>Inputs the model weighs most</h4><div class="drv">${drivers}</div></div>
    ${advice}
    <div class="actions">
      <button class="btn" id="copyBtn" type="button">Copy summary</button>
      <button class="btn" id="dlBtn" type="button">Download JSON</button>
    </div>
    <p class="disclaimer">This is a statistical estimate from a machine learning model. Use it as one input, not as a final decision.</p>`;

  requestAnimationFrame(() => requestAnimationFrame(() => {
    const g = $("#gFill"); if(g) g.style.strokeDashoffset = 100 - d.confidence;
    document.querySelectorAll(".bar-row .b i").forEach(i => i.style.width = i.dataset.w + "%");
  }));
  $("#copyBtn").addEventListener("click", copySummary);
  $("#dlBtn").addEventListener("click", downloadJSON);
}

function copySummary(){
  if(!LAST) return;
  const d = LAST.data;
  const text = `Workforce risk result: ${d.label} (${d.confidence}% confidence)\n${d.headline}. ${d.meaning}\n\nKey points:\n` +
    d.insights.map(i => "- " + i.text).join("\n");
  (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject())
    .then(() => toast("Summary copied"), () => toast("Copy is not available here"));
}
function downloadJSON(){
  if(!LAST) return;
  const blob = new Blob([JSON.stringify(LAST, null, 2)], {type:"application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "prediction.json"; a.click(); URL.revokeObjectURL(a.href);
}

/* ---------- history ---------- */
function loadHist(){ try{ return JSON.parse(safeStore.get("history") || "[]"); }catch(e){ return []; } }
function saveHistory(d, values){
  const h = loadHist();
  h.unshift({t: Date.now(), label: d.label, tone: d.tone, conf: d.confidence, values});
  safeStore.set("history", JSON.stringify(h.slice(0, 8))); renderHistory();
}
function renderHistory(){
  const h = loadHist(); const box = $("#histBox");
  box.style.display = h.length ? "block" : "none";
  $("#histList").innerHTML = h.map((x, i) => `<button type="button" class="hist-item tone-${x.tone}" data-i="${i}">
    <span><span class="tag">${esc(x.label)}</span> &nbsp;${x.conf}%</span>
    <small>${esc(x.values.company_name || "Company")} &middot; ${new Date(x.t).toLocaleString([], {dateStyle:"medium", timeStyle:"short"})}</small></button>`).join("");
  document.querySelectorAll(".hist-item").forEach(b => b.addEventListener("click", () => {
    setValues(loadHist()[b.dataset.i].values); showTab(0); window.scrollTo({top:0, behavior:"smooth"}); toast("Inputs restored");
  }));
}
$("#clearHist").addEventListener("click", () => { safeStore.set("history", "[]"); renderHistory(); });

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)