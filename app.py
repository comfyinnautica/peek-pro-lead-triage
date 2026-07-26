import os
import json
import requests
import pandas as pd
import random
import re
import csv
import io
from datetime import datetime
import streamlit as st
from openai import OpenAI

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Peek Pro | Multi-Channel Lead Triage Engine",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Hide "Press Enter to submit form" Input Instructions */
    div[data-testid="InputInstructions"] {
        display: none !important;
    }

    /* Make ALL Primary Buttons Green (Run, Approve, Download) */
    button[kind="primary"] {
        background-color: #28a745 !important;
        color: white !important;
        border-color: #28a745 !important;
    }
    button[kind="primary"]:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State Initialization ─────────────────────────────────────────────
if "lead_queue" not in st.session_state:
    st.session_state["lead_queue"] = []

if "form_key_idx" not in st.session_state:
    st.session_state["form_key_idx"] = 0

if "raw_form_key" not in st.session_state:
    st.session_state["raw_form_key"] = 0

if "csv_uploader_key" not in st.session_state:
    st.session_state["csv_uploader_key"] = 0

if "flash_success" not in st.session_state:
    st.session_state["flash_success"] = None

CORRECTIONS_FILE = "corrections.jsonl"

# ── Comprehensive Global Countries List ───────────────────────────────────────
ALL_COUNTRIES = [
    "United States", "Canada", "United Kingdom", "Australia", "New Zealand", "Mexico", 
    "Argentina", "Austria", "Bahamas", "Belgium", "Belize", "Brazil", "Chile", "Colombia", 
    "Costa Rica", "Czech Republic", "Denmark", "Dominican Republic", "Ecuador", "Egypt", 
    "Finland", "France", "Germany", "Greece", "Guatemala", "Honduras", "Hungary", "Iceland", 
    "India", "Indonesia", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Kenya", 
    "Malaysia", "Maldives", "Morocco", "Netherlands", "Norway", "Panama", "Peru", 
    "Philippines", "Poland", "Portugal", "Puerto Rico", "Saudi Arabia", "Singapore", 
    "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Thailand", "Turkey", 
    "United Arab Emirates", "Uruguay", "Vietnam", "Other"
]

# ── Layer 1: Deterministic Normalizer ─────────────────────────────────────────
COLUMN_ALIASES = {
    "first_name": ["first_name", "firstname", "fname", "given_name"],
    "last_name": ["last_name", "lastname", "lname", "surname"],
    "name": ["full_name", "contact_name", "lead_name", "name", "contact"],
    "email": ["email_address", "contact_email", "mail", "email"],
    "website": ["site_url", "domain", "url", "web", "website"],
    "phone": ["phone_number", "mobile", "telephone", "phone"],
    "country": ["country_of_operations", "country", "location", "geo"],
    "annual_sales_volume": ["annual_sales_volume", "annual_sales", "revenue", "annual_rev", "booking_volume", "sales_tier", "rev"],
    "company": ["company_name", "co_name", "organization", "account", "business", "company"],
    "title": ["job_title", "position", "role", "title"],
    "channel": ["lead_source", "channel", "source", "medium", "origin"],
    "competitor_software": ["current_software", "pos", "ticketing_system", "competitor", "current_pos", "competitor_pos"],
    "timeline": ["urgency", "timeframe", "timeline", "implementation_timeline"],
    "notes": ["message", "comments", "description", "transcript", "raw_notes", "notes", "raw_message"]
}

def normalize_csv_row(raw_row: dict) -> dict:
    normalized = {}
    for standard_key, aliases in COLUMN_ALIASES.items():
        for col_name, val in raw_row.items():
            col_clean = str(col_name).lower().strip().replace(" ", "_")
            if col_clean in aliases or col_clean == standard_key:
                if pd.notna(val) and str(val).strip():
                    normalized[standard_key] = str(val).strip()
                break

    if "name" not in normalized and ("first_name" in normalized or "last_name" in normalized):
        fn = normalized.get("first_name", "")
        ln = normalized.get("last_name", "")
        normalized["name"] = f"{fn} {ln}".strip()

    return normalized

def sanitize_lead_payload(lead: dict) -> dict:
    sanitized = {}
    for k, v in lead.items():
        if v is not None and str(v).strip() != "" and str(v) != "nan":
            val_str = str(v).strip()
            if k in ["notes", "raw_transcript"]:
                val_str = val_str[:1000]
            sanitized[k] = val_str
    return sanitized

# ── Intelligence-First Enrichment Guardrails ───────────────────────────────
def should_enrich_lead(lead: dict) -> bool:
    if not lead.get("company") and not lead.get("email"):
        return False

    high_volume_tiers = ["$150,000 to $1,500,000", "$1,500,000 to $4m", "More than $4m"]
    if lead.get("annual_sales_volume") in high_volume_tiers:
        return True
    if lead.get("competitor_software") in ["FareHarbor", "Xola", "Anchor", "Resmark"]:
        return True

    if not lead.get("annual_sales_volume") or not lead.get("competitor_software") or lead.get("competitor_software") == "Other / Unknown":
        return True

    low_volume_tiers = ["New business / less than 1 year of sales", "less than $50,000"]
    if lead.get("annual_sales_volume") in low_volume_tiers:
        return False

    return True

def clean_apollo_payload(enrichment_data: dict) -> dict:
    clean = {}
    for k, v in enrichment_data.items():
        if v and str(v).strip().lower() not in ["unknown", "none", "null", ""]:
            clean[k] = str(v).strip()
    return clean

def enrich_with_apollo(email: str, company: str, api_key: str) -> dict:
    if not api_key:
        return {}

    domain = email.split("@")[-1] if email and "@" in email else ""
    generic_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"]
    if domain in generic_domains:
        domain = ""

    url = "https://api.apollo.io/v1/organizations/enrich"
    headers = {"Cache-Control": "no-cache", "Content-Type": "application/json"}
    payload = {"api_key": api_key}

    if domain:
        payload["domain"] = domain
    elif company:
        payload["organization_name"] = company
    else:
        return {}

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=5)
        if res.status_code == 200:
            data = res.json().get("organization", {})
            return clean_apollo_payload({
                "annual_revenue": data.get("annual_revenue_printed"),
                "employees": data.get("estimated_num_employees"),
                "industry": data.get("industry"),
                "location": f"{data.get('city', '')}, {data.get('state', '')}, {data.get('country', '')}".strip(", "),
            })
    except Exception:
        pass
    return {}

# ── Sidebar Settings & Reference Guides ───────────────────────────────────────
with st.sidebar:
    st.title("⚙️ System Settings")
    st.divider()

    with st.expander("🔑 API Keys & Integrations", expanded=False):
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            help="Paste your OpenAI API key here. Required for AI triage.",
        )
        apollo_key = st.text_input(
            "Apollo API Key (Optional)",
            type="password",
            value=os.environ.get("APOLLO_API_KEY", ""),
            help="Enables smart tie-breaker enrichment for leads.",
        )

    st.divider()
    st.subheader("📥 Queue Manager")

    if not st.session_state["lead_queue"]:
        st.info("Queue is empty. Add leads from the main panel.")
    else:
        st.success(f"{len(st.session_state['lead_queue'])} lead(s) currently queued.")

        with st.expander("👀 View Queued Leads", expanded=False):
            for i, queued_lead in enumerate(st.session_state["lead_queue"]):
                label = queued_lead.get("name") or queued_lead.get("company") or queued_lead.get("channel", "Unknown Lead")
                st.caption(f"**{i+1}.** {label}")

        if st.button("🗑️ Clear Queue", use_container_width=True):
            st.session_state["lead_queue"] = []
            st.rerun()

    st.divider()
    st.subheader("🎭 Session Bias")
    session_bias = st.text_area(
        "Session Context / SDR Mood",
        placeholder="e.g., 'It's Friday, wish them a great weekend.' or 'The Mets won last night.'",
        help="Injects real-time human context into the email drafts. Limited to 120 characters to preserve A/B test integrity.",
        max_chars=120
    )

    st.divider()

    with st.expander("🎯 Target ICP Criteria", expanded=False):
        st.markdown("""
        **Peek Pro Target ICP:**
        * **Industry:** Tour, Activity, Rental, Museum, & Experience Operators.
        * **Sales Volume Tiers:**
          * **High ICP:** `150,000 USD to 1.5M USD`, `1.5M USD to 4M USD`, `More than 4M USD`.
          * **PLG / Nurture:** `Less than 50,000 USD`, `50,000 USD to 150,000 USD`.
          * **Self-Serve:** `New business / less than 1 year`.
        * **High-Intent Signals:**
          * Current software: *FareHarbor, Xola, Anchor, Resmark*.
          * Multi-location setup or fast launch timeline.
        """)

    with st.expander("👤 Target Buyer Personas", expanded=False):
        st.markdown("""
1. **Owner-Operator**
   * *Titles:* Owner, Founder, CEO, President, Captain/Owner, Co-Founder.
   * *Focus:* Direct ROI, quick setup, fee transparency, saving time.

2. **Ops Manager**
   * *Titles:* GM, General Manager, Operations Director, Fleet Manager, Guest Experience Lead.
   * *Focus:* Staff scheduling, daily check-ins, waiver automation, POS speed.

3. **Multi-Location Director**
   * *Titles:* Regional VP, Managing Director, VP Operations, Franchise Owner.
   * *Focus:* Centralized reporting, franchise scale, custom API/CRM sync.

4. **Marketing / Revenue Lead**
   * *Titles:* Head of Growth, Marketing Director, Revenue Manager, E-Commerce Lead.
   * *Focus:* Online checkout conversion rates, dynamic pricing, OTA channel manager.
        """)

# ── Header & Flash Messaging ──────────────────────────────────────────────────
st.title("🏔️ Peek Pro — Multi-Channel Lead Triage Engine")
st.caption("AI-powered GTM lead scoring, A/B Testing templates, conditional enrichment & human review gate.")

# Safely display and clear the persistent success message post-rerun
if st.session_state["flash_success"]:
    st.success(st.session_state["flash_success"])
    st.session_state["flash_success"] = None

st.divider()

# ── Learning Loop Helpers ─────────────────────────────────────────────────────
def get_recent_corrections(persona: str, limit: int = 2) -> list:
    if not os.path.exists(CORRECTIONS_FILE):
        return []
    corrections = []
    with open(CORRECTIONS_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("persona") == persona:
                    corrections.append(data)
            except Exception:
                pass
    return corrections[-limit:]

def save_correction(persona: str, next_step: str, lead_context: str, reviewer_notes: str, orig_draft: str, corr_draft: str):
    record = {
        "persona": persona,
        "next_step": next_step,
        "context": lead_context,
        "reviewer_notes": reviewer_notes,
        "original_draft": orig_draft,
        "corrected_draft": corr_draft
    }
    with open(CORRECTIONS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

# ── AI Triage Pipelines ───────────────────────────────────────────────────────
def run_stage_1_scoring(client: OpenAI, lead: dict) -> dict:
    system_prompt = """
You are a Senior Revenue Operations Analyst for Peek Pro (experiences booking platform).
Evaluate the lead against Peek Pro's Ideal Customer Profile (ICP).

STRICT GROUND TRUTH & GENERAL RULES:
- GENERAL RULE: Err on the safe side. It is preferable to assign "Unknown" to missing information. Never assume or infer; deduce only from facts.
- INTENT OVERRIDE RULE (CRITICAL): If the `notes` clearly indicate they are a reporter, student, or vendor NOT looking to buy, route to "Disqualify". HOWEVER, do NOT disqualify based solely on an unusual job title (e.g., "Mayor", "Illusionist") if they express clear intent for booking software, mention conversion issues, or name a competitor. Intent trumps unusual titles.
- URGENCY & COMPETITOR RULE (CRITICAL): If the lead explicitly expresses HIGH URGENCY (e.g., "need to switch this week") OR mentions using a direct competitor (FareHarbor, Xola, Anchor, Resmark), they MUST be routed to "Sales" and given "High" priority, completely overriding low sales volume or unusual titles.
- FORMATTING RULE: NEVER use the '$' symbol for currency (e.g., write "150,000 USD" instead of "$150,000") to prevent UI rendering errors. 
- Form fields directly submitted by the user (annual_sales_volume, notes, competitor) are GROUND TRUTH.
- DO NOT assume or hallucinate competitors. ONLY list competitor displacement signals if explicitly stated.

PERSONA RULE & JOB TITLE GUIDELINES:
- Owner-Operator: Owner, Founder, CEO, President, Captain/Owner, Co-Founder.
- Ops Manager: GM, General Manager, Operations Director/Manager, Fleet Manager, Guest Experience Lead.
- Multi-Location Director: Regional VP, Managing Director, VP Operations, Franchise Owner.
- Marketing / Revenue Lead: Head of Growth, Marketing Director, Revenue Manager, E-Commerce Lead.
- CRITICAL: If the title is missing or nonsensical, it MUST default to "Unknown".

ICP Scoring Rules:
- Sales Volume 150k+ = Strong score (7-10). Target for Sales routing.
- Sales Volume < 50k or New Business = Lower score (1-5). Route to Self-serve / PLG or Nurture.

Return strict JSON with these exact keys:
- "fit_score": integer 1-10
- "priority": "High", "Medium", or "Low"
- "persona": One of ["Owner-Operator", "Ops Manager", "Multi-Location Director", "Marketing / Revenue Lead", "Unknown"]
- "persona_rationale": 1 sentence explaining why this persona was selected
- "icp_signals": JSON object with ratings ("Strong", "Moderate", "Weak", "Unknown") for:
    "industry_fit", "business_model", "company_size", "pain_indicators", "booking_method", "growth_signals", "geo_market", "multi_channel", "tech_readiness"
- "buying_signals": list of strings
- "missing_info": list of strings
- "recommended_next_step": One of ["Sales", "Self-serve / PLG", "Nurture", "Customer Support", "Disqualify", "Needs More Information"]
- "explanation": 2 sentence summary justification for the score and route
"""
    clean_payload = sanitize_lead_payload(lead)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Lead Data:\n{json.dumps(clean_payload, indent=2)}"}
        ]
    )
    return json.loads(response.choices[0].message.content)

def run_stage_2_drafting(client: OpenAI, lead: dict, stage1: dict, model: str, variation: str, session_bias: str) -> dict:
    persona = stage1.get("persona", "Unknown")
    corrections = get_recent_corrections(persona)

    few_shot_str = ""
    if corrections:
        few_shot_str = "\n\nPast Approved User Corrections for this Persona:\n"
        for idx, c in enumerate(corrections, 1):
            notes_str = f" (Reason: {c['reviewer_notes']})" if c.get('reviewer_notes') else ""
            few_shot_str += f"Example {idx}:\nOriginal: {c['original_draft']}\nApproved Edit{notes_str}: {c['corrected_draft']}\n"

    if variation == "A":
        ab_test_prompt = "A/B TEST STRATEGY A: Focus heavily on direct ROI, fee transparency, and keeping the opening punchy and short."
    else:
        ab_test_prompt = "A/B TEST STRATEGY B: Focus on social proof, avoiding operational headaches, and ending with a process-oriented question."

    bias_prompt = f"\nSESSION CONTEXT / SDR MOOD: Include this context naturally in the email if appropriate: '{session_bias}'" if session_bias.strip() else ""

    system_prompt = f"""
You are an expert SDR writing a 1-to-1 outreach email for Peek Pro.
Tailor the value proposition to the buyer persona:
- Owner-Operator: Focus on simplicity, quick setup, fee transparency, direct ROI.
- Ops Manager: Focus on operational efficiency, staff scheduling, automation.
- Multi-Location Director: Focus on centralized reporting, scale, API/CRM integrations.
- Marketing/Revenue Lead: Focus on conversion rates, dynamic pricing, OTA sync.

{ab_test_prompt}{bias_prompt}

CRITICAL GUARDRAILS:
- GENERAL RULE: Err on the safe side. Never assume or infer; deduce only from verified facts.
- FORMATTING RULE: NEVER use the '$' symbol for currency (use "USD" instead) to prevent UI errors.
- TONALITY RULE: You are STRICTLY FORBIDDEN from using em dashes (—) or en dashes (–). Use commas, colons, or separate sentences instead.
- QUESTION RULE: Limit yourself to EXACTLY ONE call-to-action question at the very end of the email. Do not ask about pain points they have already explained. Propose a specific time to connect.
- SIGN-OFF RULE: STRICTLY FORBIDDEN from including sign-offs like "Best regards,", "Best,", or placeholders like "[Your Name]". End the email immediately after the final question mark.
- COMPETITOR RULE: STRICTLY FORBIDDEN from assuming or mentioning competitors UNLESS explicitly provided.
- PERSONA RULE: Address the lead based strictly on verified facts. If the title is missing, it must default to Unknown.

Keep the email concise (under 120 words).{few_shot_str}
"""
    clean_lead = sanitize_lead_payload(lead)
    user_payload = {
        "lead": clean_lead,
        "persona": persona,
        "next_step": stage1.get("recommended_next_step"),
        "buying_signals": stage1.get("buying_signals"),
        "explanation": stage1.get("explanation")
    }

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Draft email for:\n{json.dumps(user_payload, indent=2)}"}
        ]
    )

    # --- Python Post-Processor (Bulletproof Sanitization) ---
    raw_text = response.choices[0].message.content.strip()

    # 1. Scrub em/en dashes and replace with commas or standard hyphens
    clean_text = raw_text.replace("—", ", ").replace("–", "-")

    # 2. Aggressively scrub common AI signature placeholders via Regex
    clean_text = re.sub(r"(?i)\n*(Best regards|Best|Sincerely|Cheers|Looking forward to hearing from you)[,\s]*\n*.*", "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(r"\[Your Name\]|\[Name\]|\[Your Title\]|Peek Pro", "", clean_text, flags=re.IGNORECASE)

    return {
        "text": clean_text.strip(),
        "tokens": response.usage.total_tokens
    }

# ── Multi-Channel Input Architecture (Stateful Tabs via Radio) ────────────────
input_method = st.radio(
    "Select Input Method",
    ["📋 Peek Web Form", "💬 Raw Text", "📁 Batch Upload"],
    horizontal=True,
    label_visibility="collapsed"
)

# --- TAB 1: Structured Web Form ---
if input_method == "📋 Peek Web Form":
    with st.form(f"peek_web_lead_form_{st.session_state['form_key_idx']}"):
        st.caption("Matches Peek Pro's official web form intake layout.")

        c1, c2 = st.columns(2)
        f_fname = c1.text_input("First Name*", placeholder="Sarah")
        f_lname = c2.text_input("Last Name*", placeholder="Jenkins")

        c3, c4, c5 = st.columns(3)
        f_email = c3.text_input("Email", placeholder="sarah@coastalkayaks.com")
        f_phone = c4.text_input("Phone", placeholder="+1 (555) 234-5678")
        f_title = c5.text_input("Job Title", placeholder="Owner & General Manager")

        c6, c7 = st.columns(2)
        f_company = c6.text_input("Company Name*", placeholder="Coastal Kayak Tours")
        f_website = c7.text_input("Website", placeholder="https://coastalkayaks.com")

        c8, c9 = st.columns(2)
        f_country = c8.selectbox(
            "Country of Operations*", 
            options=ALL_COUNTRIES,
            index=None,
            placeholder="Type to search or select..."
        )
        f_sales_volume = c9.selectbox(
            "Annual sales volume",
            [
                "New business / less than 1 year of sales",
                "less than 50,000 USD",
                "50,000 USD to 150,000 USD",
                "150,000 USD to 1,500,000 USD",
                "1,500,000 USD to 4m USD",
                "More than 4m USD"
            ],
            index=None,
            placeholder="Select sales volume..."
        )

        c10, c11, c12 = st.columns(3)
        f_channel = c10.selectbox("Lead channel/ origin", ["Web Form", "Live Chat", "Outbound Reply", "Conference"], index=None, placeholder="Select channel...")
        f_timeline = c11.selectbox("Launch Urgency", ["Immediately", "Next Season (1-3 mos)", "Just Browsing"], index=None, placeholder="Select urgency...")
        f_competitor = c12.selectbox("Current POS Software", ["FareHarbor", "Xola", "Anchor", "Resmark", "Paper / Pen", "Other / Unknown"], index=None, placeholder="Select software...")

        f_notes = st.text_area(
            "Lead Notes/ Context",
            placeholder="We operate 2 locations. Tired of FareHarbor's high fee structure and need quick onboarding before summer.",
            height=80
        )

        if st.form_submit_button("Queue Web Form Lead"):
            if not all([f_fname.strip(), f_lname.strip(), f_company.strip(), f_country]):
                st.error("⚠️ Please fill out all required fields: First Name, Last Name, Company Name, and Country of Operations.")
            else:
                st.session_state["lead_queue"].append({
                    "first_name": f_fname.strip(),
                    "last_name": f_lname.strip(),
                    "name": f"{f_fname.strip()} {f_lname.strip()}",
                    "email": f_email.strip(),
                    "website": f_website.strip(),
                    "phone": f_phone.strip(),
                    "country_of_operations": f_country,
                    "annual_sales_volume": f_sales_volume,
                    "company": f_company.strip(),
                    "title": f_title.strip(),
                    "channel": f_channel or "Web Form",
                    "competitor_software": f_competitor,
                    "timeline": f_timeline,
                    "notes": f_notes.strip()
                })
                st.session_state["form_key_idx"] += 1
                st.session_state["flash_success"] = "Web Form Lead added successfully to the queue!"
                st.rerun()

# --- TAB 2: Unstructured Text / Email Paste ---
elif input_method == "💬 Raw Text":
    st.markdown("Paste an inbound email reply, SDR call note, or chat transcript. The AI parses and triages in one pass.")

    with st.form(f"raw_text_form_{st.session_state['raw_form_key']}"):
        raw_transcript_input = st.text_area(
            "Raw Message / Transcript",
            value="",
            placeholder="Example:\nFrom: Mike Ramirez <mike@miamijetski.com>\nSubject: Switching off FareHarbor\n\nHey team, we run Miami JetSki Rentals across 5 locations in South Florida doing ~$2.5M annually. Currently on FareHarbor but our local reps complain about mobile checkout slowness. Need to switch over before Memorial Day. Can someone call me tomorrow?",
            height=180
        )
        if st.form_submit_button("Queue Raw Transcript Lead"):
            if raw_transcript_input.strip():
                st.session_state["lead_queue"].append({
                    "channel": "Unstructured Email / Chat Paste",
                    "raw_transcript": raw_transcript_input
                })
                st.session_state["raw_form_key"] += 1
                st.session_state["flash_success"] = "Unstructured transcript lead added successfully to the queue!"
                st.rerun()
            else:
                st.warning("Please paste some text first!")

# --- TAB 3: Batch CSV & Excel Upload ---
elif input_method == "📁 Batch Upload":
    st.markdown("Upload any campaign CSV or Excel file. The **Layer 1 Normalizer** maps headers (e.g. `fname`, `co_name`, `annual_sales`) to Peek standards without burning extra AI tokens.")

    uploaded_file = st.file_uploader("Drop leads file here", type=["csv", "xlsx", "xls"], key=f"csv_uploader_{st.session_state['csv_uploader_key']}")

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                # Bulletproof CSV parser for dirty lists
                content = uploaded_file.read().decode('utf-8', errors='ignore')
                reader = csv.reader(io.StringIO(content))
                data = list(reader)

                if data:
                    header = data[0]
                    # Find the maximum number of columns in any row
                    max_cols = max(len(row) for row in data)

                    # If some rows have more columns than the header, pad the header dynamically
                    if max_cols > len(header):
                        header.extend([f"Unnamed_{i}" for i in range(len(header), max_cols)])

                    raw_df = pd.DataFrame(data[1:], columns=header)
                else:
                    raw_df = pd.DataFrame()
            else:
                raw_df = pd.read_excel(uploaded_file)

            st.info(f"Loaded {len(raw_df)} raw rows. Normalizing headers...")
            cleaned_leads = [normalize_csv_row(row.to_dict()) for _, row in raw_df.iterrows()]

            st.dataframe(pd.DataFrame(cleaned_leads), use_container_width=True)

            if st.button(f"➕ Add {len(cleaned_leads)} Leads to Queue"):
                st.session_state["lead_queue"].extend(cleaned_leads)
                st.session_state["csv_uploader_key"] += 1
                st.session_state["flash_success"] = f"{len(cleaned_leads)} leads added successfully to the queue!"
                st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")

st.divider()

# ── Execution Trigger ─────────────────────────────────────────────────────────
if st.button("🚀 Run 2-Stage AI Triage Pipeline", type="primary", disabled=len(st.session_state["lead_queue"]) == 0):

    if not openai_key:
        st.error("⚠️ OpenAI API Key is missing! Please enter it in the sidebar.")
        st.stop()

    client = OpenAI(api_key=openai_key)
    results = []

    progress_bar = st.progress(0)
    queue_length = len(st.session_state["lead_queue"])

    for idx, raw_lead in enumerate(st.session_state["lead_queue"]):
        lead_payload = dict(raw_lead)

        apollo_status_msg = "Skipped (Auto-Rules)"
        if apollo_key and should_enrich_lead(lead_payload):
            enrichment = enrich_with_apollo(lead_payload.get("email", ""), lead_payload.get("company", ""), apollo_key)
            if enrichment:
                lead_payload["apollo_enrichment"] = enrichment
                apollo_status_msg = "Enriched Successfully"
            else:
                apollo_status_msg = "Attempted (No Org Match Found)"
        elif not apollo_key:
            apollo_status_msg = "No API Key Provided"

        stage1_res = run_stage_1_scoring(client, lead_payload)

        priority = stage1_res.get("priority", "Low")
        rec_step = stage1_res.get("recommended_next_step", "")

        variation = random.choice(["A", "B"])

        # Hardcoded Routing Engine based on Priority
        if rec_step == "Disqualify":
            draft = "Thank you for reaching out to Peek Pro. Based on your current setup, our platform is not a fit at this time."
            draft_source = "Hardcoded Disqualification"
            draft_tokens = 0

        elif priority == "High" and rec_step in ["Sales", "Self-serve / PLG", "Nurture", "Needs More Information"]:
            draft_response = run_stage_2_drafting(client, lead_payload, stage1_res, "gpt-4o", variation, session_bias)
            draft = draft_response["text"]
            draft_tokens = draft_response["tokens"]
            draft_source = "AI Drafted (gpt-4o)"

        elif priority == "Medium" and rec_step in ["Sales", "Self-serve / PLG", "Nurture", "Needs More Information"]:
            draft_response = run_stage_2_drafting(client, lead_payload, stage1_res, "gpt-4o-mini", variation, session_bias)
            draft = draft_response["text"]
            draft_tokens = draft_response["tokens"]
            draft_source = "AI Drafted (gpt-4o-mini)"

        else:
            fname = lead_payload.get('first_name') or lead_payload.get('name', 'there')
            if variation == "A":
                draft = f"Hi {fname},\n\nThanks for reaching out! To help route you to the right team, could you share a bit more about your typical monthly booking volume?"
            else:
                draft = f"Hi {fname},\n\nThanks for your interest in Peek Pro! We'd love to learn more about your setup. Are you currently using a booking system?"
            draft_source = "Hardcoded Template ($0)"
            draft_tokens = 0

        results.append({
            "lead": lead_payload,
            "stage1": stage1_res,
            "draft": draft,
            "draft_tokens": draft_tokens,
            "variation": variation,
            "draft_source": draft_source,
            "apollo_status": apollo_status_msg,
            "human_edited": False,
            "review_status": "PENDING",
            "final_next_step": rec_step,
            "final_draft": draft,
            "reviewer_notes": ""
        })

        progress_bar.progress((idx + 1) / queue_length)

    st.session_state["triage_results"] = results
    st.session_state["lead_queue"] = [] 

    st.session_state["flash_success"] = f"Successfully triaged {len(results)} lead(s)!"
    st.rerun()

# ── Step 3: Human Review Gate (HITL) ──────────────────────────────────────────
if "triage_results" in st.session_state:
    st.header("✍️ Step 3: Human Review Gate")
    st.caption("Review AI classifications, override action routes, and refine outreach drafts.")

    results_list = st.session_state["triage_results"]

    for idx, item in enumerate(results_list):
        s1 = item["stage1"]
        lead = item["lead"]
        priority = s1.get("priority", "Low")
        persona = s1.get("persona", "Unknown")
        status = item.get("review_status", "PENDING")

        display_name = lead.get("name") or lead.get("company") or f"Lead #{idx+1}"

        if status == "APPROVED":
            status_prefix = "🟢 [APPROVED]"
        elif status == "APPROVED_WITH_EDITS":
            status_prefix = "🟡 [APPROVED W/ EDITS]"
        elif status == "DELETED":
            status_prefix = "🔴 [DELETED]"
        else:
            status_prefix = "⏳ [PENDING]"

        priority_emoji = "🟢" if priority == "High" else "🟠" if priority == "Medium" else "🔴"

        expander_title = f"{status_prefix} Lead #{idx+1}: {display_name} | Priority: {priority_emoji} {priority} | Persona: {persona}"
        is_expanded = (status == "PENDING")

        with st.expander(expander_title, expanded=is_expanded):

            priority_color = "green" if priority == "High" else "orange" if priority == "Medium" else "red"

            st.markdown(f"### {status_prefix} Lead #{idx+1}: {display_name} | Priority: :{priority_color}[{priority}] | Persona: {persona}")
            st.markdown(f"**Suggested next step:** {s1.get('recommended_next_step', 'N/A')}")

            # Removed st.form to allow dynamic UI updates when routing is overridden
            col_info, col_grid = st.columns([3, 1])

            with col_info:
                st.markdown(f"**Persona Rationale:** {s1.get('persona_rationale', '')}")
                st.markdown(f"**AI Explanation:** {s1.get('explanation', '')}")
                st.markdown(f"**Buying Signals:** {', '.join(s1.get('buying_signals', []))}")
                if s1.get("missing_info"):
                    st.warning(f"Missing Info: {', '.join(s1.get('missing_info'))}")

                if lead.get("apollo_enrichment"):
                    st.info(f"ℹ️ **Apollo Verified:** {lead.get('apollo_enrichment')}")
                else:
                    st.caption(f"🔍 **Apollo Status:** {item.get('apollo_status', 'N/A')}")

                st.divider()

                c_step, c_notes = st.columns([1, 2])
                current_route = item["final_next_step"]

                new_step = c_step.selectbox(
                    "Override Next Step:",
                    ["Sales", "Self-serve / PLG", "Nurture", "Customer Support", "Disqualify", "Needs More Information"],
                    index=["Sales", "Self-serve / PLG", "Nurture", "Customer Support", "Disqualify", "Needs More Information"].index(current_route),
                    key=f"route_override_{idx}"
                )

                # Instantly overwrite the text area with a hardcoded template if the SDR changes the route
                if new_step != current_route:
                    item["final_next_step"] = new_step
                    fname = lead.get('first_name', lead.get('name', 'there'))

                    if new_step == "Disqualify":
                        new_template = "Thank you for reaching out to Peek Pro. Based on your current setup, our platform is not a fit at this time."
                    elif new_step == "Self-serve / PLG":
                        new_template = f"Hi {fname},\n\nThanks for your interest! The best way to explore Peek Pro is to sign up for our self-serve platform directly on our website."
                    elif new_step == "Nurture":
                        new_template = f"Hi {fname},\n\nThanks for reaching out! We'd love to stay in touch and keep you updated with the latest news and features from Peek Pro as you grow."
                    elif new_step == "Customer Support":
                        new_template = f"Hi {fname},\n\nI've routed your message to our technical support team. They will be in touch shortly to assist you."
                    elif new_step == "Needs More Information":
                        new_template = f"Hi {fname},\n\nThanks for reaching out! To help route you to the right team, could you share a bit more about your typical monthly booking volume?"
                    elif new_step == "Sales":
                        new_template = f"Hi {fname},\n\nThanks for your interest in Peek Pro! We'd love to learn more about your setup. When is a good time to connect next week?"

                    item["final_draft"] = new_template
                    st.session_state[f"draft_edit_{idx}"] = new_template
                    st.rerun()

                # Text Area logic
                # Ensure the text area respects the session state if we programmatically updated it above
                if f"draft_edit_{idx}" not in st.session_state:
                    st.session_state[f"draft_edit_{idx}"] = item["final_draft"]

                new_draft = st.text_area("Outreach Email Draft (Editable):", key=f"draft_edit_{idx}", height=220)

                # Update reviewer notes
                if f"notes_{idx}" not in st.session_state:
                    st.session_state[f"notes_{idx}"] = item["reviewer_notes"]
                rev_notes = c_notes.text_input("Reviewer Notes:", key=f"notes_{idx}", placeholder="e.g. 'Changed tone to be more casual'")

            with col_grid:
                st.markdown("**ICP Signals Grid:**")
                signals = s1.get("icp_signals", {})

                color_map = {"Strong": "green", "Moderate": "orange", "Weak": "red", "Unknown": "gray"}
                for k, v in signals.items():
                    c = color_map.get(v, "gray")
                    st.markdown(f"- {k.replace('_', ' ').title()}: :{c}[**{v}**]")

                st.divider()
                st.caption(f"**Draft Variation Tested:** {item['variation']}")
                st.caption(f"**Draft Source:** {item['draft_source']}")

                if item['draft_tokens'] > 0:
                    st.caption(f"**Draft Token Usage:** {item['draft_tokens']} tokens")
                else:
                    st.caption("**Draft Token Usage:** 0 tokens")

            st.divider()

            # Use columns to make the buttons massive (use_container_width=True spans the whole column)
            btn_app, btn_rej = st.columns(2)

            # Using type="primary" makes it inherit the green CSS we added at the top
            if btn_app.button("✅ Approve", type="primary", use_container_width=True, key=f"btn_approve_{idx}"):
                item["final_next_step"] = new_step
                item["final_draft"] = new_draft
                item["reviewer_notes"] = rev_notes

                # Auto-detect human edits and save to learning loop
                if new_draft.strip() != item["draft"].strip() or new_step != item["stage1"].get("recommended_next_step"):
                    item["human_edited"] = True
                    item["review_status"] = "APPROVED_WITH_EDITS" # 🟡 This brings back the yellow title!
                    save_correction(persona, new_step, str(lead), rev_notes, item["draft"], new_draft)
                else:
                    item["human_edited"] = False
                    item["review_status"] = "APPROVED"

                st.rerun()

            # Leaving this as default styling keeps it neutral/gray, balancing the layout visually
            if btn_rej.button("🗑️ Delete (Spam / Test / Error)", use_container_width=True, key=f"btn_reject_{idx}"):
                item["review_status"] = "DELETED"
                st.rerun()

    st.divider()

    # ── Step 4: Export Section ────────────────────────────────────────────────
    st.header("📊 Step 4: Export & Sync")

    all_processed = all(r["review_status"] != "PENDING" for r in results_list)

    export_rows = []
    for r in results_list:
        export_rows.append({
            "First Name": r["lead"].get("first_name", "N/A"),
            "Last Name": r["lead"].get("last_name", "N/A"),
            "Email": r["lead"].get("email", "N/A"),
            "Company": r["lead"].get("company", "N/A"),
            "Sales Volume Tier": r["lead"].get("annual_sales_volume", "N/A"),
            "Priority": r["stage1"].get("priority"),
            "Persona": r["stage1"].get("persona"),
            "A/B Variation Tested": r["variation"],
            "Draft Gen Source": r["draft_source"],
            "Draft Tokens Used": r.get("draft_tokens", 0),
            "Apollo Status": r["apollo_status"],
            "Human Edits Required": r["human_edited"],
            "Review Status": r["review_status"],
            "Final Action Route": r["final_next_step"],
            "Final Email Draft": r["final_draft"],
            "Reviewer Notes": r["reviewer_notes"],
        })

    export_df = pd.DataFrame(export_rows)
    approved_df = export_df[export_df["Review Status"].str.startswith("APPROVED")]

    # Append datetime to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button(
        "✅ Download Approved Leads (CSV)",
        data=approved_df.to_csv(index=False).encode("utf-8"),
        file_name=f"approved_leads_{timestamp}.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True
    )

    col_dl2.download_button(
        "⬇️ Download Full Audit Log (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"all_triage_results_{timestamp}.csv",
        mime="text/csv",
        type="secondary",
        use_container_width=True,
        disabled=not all_processed
    )

    if not all_processed:
        st.info("💡 The Full Audit Log will unlock once all leads have been reviewed.")