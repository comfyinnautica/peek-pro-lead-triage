import os
import json
import requests
import pandas as pd
import random
import re
from datetime import datetime
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# ── Load Environment Variables ────────────────────────────────────────────────
load_dotenv()

# ── Key Sanitizer Guardrail ───────────────────────────────────────────────────
def sanitize_api_key(key: str) -> str:
    if not key:
        return ""
    clean_key = re.sub(r'[^\x00-\x7F]+', '', key)
    return clean_key.strip().strip("'\"`’“”")

# ── Layer 0: Zero-Token Spam, Gibberish & Code Detector ──────────────────────
def is_gibberish_or_code(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, ""

    t = text.strip()

    # 1. Code / Script / Markup Detection
    code_patterns = [
        r'function\s*\w*\s*\(', r'def\s+\w+\s*\(', r'import\s+\w+', r'class\s+\w+',
        r'const\s+\w+\s*=', r'let\s+\w+\s*=', r'var\s+\w+\s*=', 
        r'<[a-zA-Z]+[^>@]*>', # Matches HTML like <div> but ignores emails like <brian@airbnb.com>
        r'public\s+static\s+void', r'console\.log', r'return\s+.*;', r'#include\s*<',
        r'select\s+.*\s+from\s+', r'{\s*".*":\s*".*"\s*}'
    ]
    for p in code_patterns:
        if re.search(p, t, re.IGNORECASE):
            return True, "Code snippet / script syntax detected in input"

    # 2. Keyboard Row Patterns (e.g. asdfgh, qwerty, zxcvbn)
    mash_patterns = [r'asdfgh', r'qwerty', r'zxcvbn', r'123456', r'hjklmn', r'dfghjk']
    for m in mash_patterns:
        if re.search(m, t, re.IGNORECASE):
            return True, "Keyboard mash pattern detected (e.g. 'asdf'/'qwerty')"

    # 3. Repeated Character Spam (e.g. "aaaaa")
    if re.search(r'(.)\1{4,}', t):
        return True, "Repeated character pattern / spam detected"

    # 4. Unusually long words without spaces or URLs
    words = t.split()
    for w in words:
        if len(w) > 18 and not w.startswith(("http://", "https://", "mailto:")) and "@" not in w:
            return True, "Unusually long string without spaces (keyboard mash)"

    # 5. Vowel Ratio Check for text over 15 characters
    letters = [c.lower() for c in t if c.isalpha()]
    if len(letters) > 15:
        vowels = [c for c in letters if c in 'aeiou']
        vowel_ratio = len(vowels) / len(letters)
        if vowel_ratio < 0.15 or vowel_ratio > 0.85:
            return True, f"Abnormal vowel ratio ({vowel_ratio:.1%}) - likely gibberish"

    return False, ""

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
BACKUP_APPROVED_FILE = "last_session_approved.csv"
BACKUP_AUDIT_FILE = "last_session_audit.csv"

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

# ── Session Backup Engine ─────────────────────────────────────────────────────
def save_session_backup(results_list: list):
    if not results_list:
        return
    export_rows = []
    for r in results_list:
        lead = r["lead"]
        fname = lead.get("first_name") or (lead.get("name", "").split()[0] if lead.get("name") else "N/A")
        lname = lead.get("last_name") or (" ".join(lead.get("name", "").split()[1:]) if len(lead.get("name", "").split()) > 1 else "N/A")

        export_rows.append({
            "First Name": fname,
            "Last Name": lname,
            "Email": lead.get("email", "N/A"),
            "Company": lead.get("company", "N/A"),
            "Sales Volume Tier": lead.get("annual_sales_volume", "N/A"),
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

    export_df.to_csv(BACKUP_AUDIT_FILE, index=False)
    approved_df.to_csv(BACKUP_APPROVED_FILE, index=False)

# ── Instant Route Template Callback ─────────────────────────────────────────
def update_route_template(idx: int):
    selected_route = st.session_state[f"route_override_{idx}"]
    item = st.session_state["triage_results"][idx]
    item["final_next_step"] = selected_route
    lead = item["lead"]
    fname = lead.get('first_name') or lead.get('name') or 'there'

    if selected_route == "Disqualify":
        new_template = "Thank you for reaching out to Peek Pro. Based on your current setup, our platform is not a fit at this time."
    elif selected_route == "Self-serve / PLG":
        new_template = f"Hi {fname},\n\nThanks for your interest! The best way to explore Peek Pro is to sign up for our self-serve platform directly on our website."
    elif selected_route == "Nurture":
        new_template = f"Hi {fname},\n\nThanks for reaching out! We'd love to stay in touch and keep you updated with the latest news and features from Peek Pro as you grow."
    elif selected_route == "Customer Support":
        new_template = f"Hi {fname},\n\nI've routed your message to our technical support team. They will be in touch shortly to assist you."
    elif selected_route == "Needs More Information":
        new_template = f"Hi {fname},\n\nThanks for reaching out! To help route you to the right team, could you share a bit more about your typical monthly booking volume?"
    elif selected_route == "Sales":
        new_template = f"Hi {fname},\n\nThanks for your interest in Peek Pro! We'd love to learn more about your setup. When is a good time to connect next week?"
    else:
        new_template = item["final_draft"]

    item["final_draft"] = new_template
    st.session_state[f"draft_edit_{idx}"] = new_template
    save_session_backup(st.session_state["triage_results"])

# ── Intelligence-First Enrichment Guardrails ───────────────────────────────
def should_enrich_lead(lead: dict) -> bool:
    if not lead.get("company") and not lead.get("email"):
        return False

    high_volume_tiers = [
        "150,000 USD to 1,500,000 USD", "$150,000 to $1,500,000",
        "1,500,000 USD to 4m USD", "$1,500,000 to $4m",
        "More than 4m USD", "More than $4m"
    ]
    if lead.get("annual_sales_volume") in high_volume_tiers:
        return True
    if lead.get("competitor_software") in ["FareHarbor", "Xola", "Anchor", "Resmark"]:
        return True

    if not lead.get("annual_sales_volume") or not lead.get("competitor_software") or lead.get("competitor_software") == "Other / Unknown":
        return True

    low_volume_tiers = ["New business / less than 1 year of sales", "less than 50,000 USD", "less than $50,000"]
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
    clean_key = sanitize_api_key(api_key)
    if not clean_key:
        return {}

    domain = email.split("@")[-1] if email and "@" in email else ""
    generic_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"]
    if domain in generic_domains:
        domain = ""

    url = "https://api.apollo.io/v1/organizations/enrich"
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
        "X-Api-Key": clean_key  # 👈 Fixed: Passed as Header (Apollo Requirement)
    }

    payload = {}
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

    raw_env_openai = os.environ.get("OPENAI_API_KEY", "")
    raw_env_apollo = os.environ.get("APOLLO_API_KEY", "")

    with st.expander("🔑 API Keys & Integrations", expanded=False):
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=sanitize_api_key(raw_env_openai),
            help="Paste your OpenAI API key here. Required for AI triage.",
        )
        apollo_key = st.text_input(
            "Apollo API Key (Optional)",
            type="password",
            value=sanitize_api_key(raw_env_apollo),
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

        if st.button("🗑️ Clear Queue", width="stretch"):
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
    st.subheader("💾 Session Recovery")
    st.caption("Recover data from your last session if you accidentally close or refresh the app.")

    has_backup = False
    if os.path.exists(BACKUP_APPROVED_FILE) and os.path.getsize(BACKUP_APPROVED_FILE) > 0:
        has_backup = True
        with open(BACKUP_APPROVED_FILE, "rb") as f_app:
            st.download_button(
                "💾 Recover Approved Leads (CSV)",
                data=f_app.read(),
                file_name=f"recovered_approved_leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
                width="stretch"
            )

    if os.path.exists(BACKUP_AUDIT_FILE) and os.path.getsize(BACKUP_AUDIT_FILE) > 0:
        has_backup = True
        with open(BACKUP_AUDIT_FILE, "rb") as f_aud:
            st.download_button(
                "💾 Recover Full Audit Log (CSV)",
                data=f_aud.read(),
                file_name=f"recovered_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="secondary",
                width="stretch"
            )

    if not has_backup:
        st.caption("No previous session backup found.")

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

STRICT GROUND TRUTH & EXTRACTION RULES:
- INFORMATION EXTRACTION (CRITICAL): If the lead input contains raw transcript or email text, explicitly extract any contact details present in the text:
  * "extracted_first_name": First name string or null
  * "extracted_last_name": Last name string or null
  * "extracted_email": Email address string or null
  * "extracted_company": Company name string or null
  * "extracted_annual_sales_volume": Standardized volume string (must map to one of: "New business / less than 1 year of sales", "less than 50,000 USD", "50,000 USD to 150,000 USD", "150,000 USD to 1,500,000 USD", "1,500,000 USD to 4m USD", "More than 4m USD") or null
  * "extracted_job_title": Title string or null
  * "extracted_competitor": Software/POS name mentioned (e.g. Anchor, FareHarbor, Xola) or null

- GENERAL RULE: Err on the safe side. It is preferable to assign "Unknown" to missing information. Never assume or infer; deduce only from facts.
- INDUSTRY EXCLUSION RULE (CRITICAL): Peek Pro is ONLY for Tour, Activity, Rental, Museum, and Experience operators. If the notes, company name, or enrichment data indicate they sell physical goods (e.g., paper, printers, retail, ecommerce), B2B services, software, or are otherwise entirely outside the experiences industry, you MUST score them low and route to "Disqualify", regardless of how high their revenue or job title is. Industry fit is absolute.
- INTENT OVERRIDE RULE (CRITICAL): If the `notes` or transcript clearly indicate they are a reporter, student, or vendor NOT looking to buy, route to "Disqualify". HOWEVER, do NOT disqualify based solely on an unusual job title (e.g., "Mayor", "Illusionist") if they express clear intent for booking software, mention conversion issues, or name a competitor. Intent trumps unusual titles.
- URGENCY & COMPETITOR RULE (CRITICAL): If the lead explicitly expresses HIGH URGENCY (e.g., "need to switch this week") OR mentions using a direct competitor (FareHarbor, Xola, Anchor, Resmark), they MUST be routed to "Sales" and given "High" priority, completely overriding low sales volume or unusual titles.
- FORMATTING RULE: NEVER use the '$' symbol for currency (e.g., write "150,000 USD" instead of "$150,000") to prevent UI rendering errors. 

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
- "extracted_first_name": string or null
- "extracted_last_name": string or null
- "extracted_email": string or null
- "extracted_company": string or null
- "extracted_annual_sales_volume": string or null
- "extracted_job_title": string or null
- "extracted_competitor": string or null
- "fit_score": integer 1-10
- "priority": "High", "Medium", or "Low"
- "persona": One of ["Owner-Operator", "Ops Manager", "Multi-Location Director", "Marketing / Revenue Lead", "Unknown"]
- "persona_rationale": 1 sentence explaining why this persona was selected
- "icp_signals": JSON object with ratings ("Strong", "Moderate", "Weak", "Unknown") for:
    "industry_fit", "business_model", "company_size", "pain_indicators", "booking_method", "growth_signals", "geo_market", "multi_channel", "tech_readiness"
- "buying_signals": list of strings
- "missing_info": list of strings
- "green_flags": list of strings (highlight strong ICP matches, high intent, or competitors)
- "red_flags": list of strings (highlight poor industry fit, low revenue, or disqualifying factors)
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

    raw_text = response.choices[0].message.content.strip()
    clean_text = raw_text.replace("—", ", ").replace("–", "-")
    clean_text = re.sub(r"(?i)\n*(Best regards|Best|Sincerely|Cheers|Looking forward to hearing from you)[,\s]*\n*.*", "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(r"\[Your Name\]|\[Name\]|\[Your Title\]|\[Peek Pro\]|\[Company Name\]", "", clean_text, flags=re.IGNORECASE)

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
            placeholder="Example:\nFrom: Brian Chesky <brian@airbnb.com>\nSubject: Exploring Peek Pro POS\n\nHi Peek team, I am looking to integrate new booking management software for our experiences division doing $4M+ annual volume. Need to connect with someone this week.",
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
                import csv
                import io
                content = uploaded_file.read().decode('utf-8', errors='ignore')
                reader = csv.reader(io.StringIO(content))
                data = list(reader)

                if data:
                    header = data[0]
                    max_cols = max(len(row) for row in data)

                    if max_cols > len(header):
                        header.extend([f"Unnamed_{i}" for i in range(len(header), max_cols)])

                    raw_df = pd.DataFrame(data[1:], columns=header)
                else:
                    raw_df = pd.DataFrame()
            else:
                raw_df = pd.read_excel(uploaded_file)

            st.info(f"Loaded {len(raw_df)} raw rows. Normalizing headers...")
            cleaned_leads = [normalize_csv_row(row.to_dict()) for _, row in raw_df.iterrows()]

            st.dataframe(pd.DataFrame(cleaned_leads), width="stretch")

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

    # Flush stale UI widget keys so dropdowns default clean
    for key in list(st.session_state.keys()):
        if key.startswith("route_override_") or key.startswith("draft_edit_") or key.startswith("notes_"):
            del st.session_state[key]

    clean_openai_key = sanitize_api_key(openai_key)
    clean_apollo_key = sanitize_api_key(apollo_key)

    if not clean_openai_key:
        st.error("⚠️ OpenAI API Key is missing or invalid! Please enter a valid key in the sidebar.")
        st.stop()

    client = OpenAI(api_key=clean_openai_key)
    results = []

    progress_bar = st.progress(0)
    queue_length = len(st.session_state["lead_queue"])

    for idx, raw_lead in enumerate(st.session_state["lead_queue"]):
        lead_payload = dict(raw_lead)

        raw_text_to_check = lead_payload.get("raw_transcript", "") or lead_payload.get("notes", "") or lead_payload.get("company", "")
        is_gibberish, gib_reason = is_gibberish_or_code(raw_text_to_check)

        if is_gibberish:
            stage1_res = {
                "fit_score": 1,
                "priority": "Low",
                "persona": "Unknown",
                "persona_rationale": "Raw input detected as invalid spam, script injection, or keyboard mashing.",
                "icp_signals": {
                    "industry_fit": "Weak", "business_model": "Weak", "company_size": "Weak",
                    "pain_indicators": "Weak", "booking_method": "Weak", "growth_signals": "Weak",
                    "geo_market": "Weak", "multi_channel": "Weak", "tech_readiness": "Weak"
                },
                "buying_signals": [],
                "missing_info": ["valid lead info", "company details", "buying intent"],
                "green_flags": [],
                "red_flags": [f"🚨 {gib_reason}", "Zero buying intent"],
                "recommended_next_step": "Disqualify",
                "explanation": f"Lead automatically disqualified at Layer 0 pre-filter ($0 AI cost): {gib_reason}."
            }
            draft = "Thank you for reaching out to Peek Pro. Based on your current setup, our platform is not a fit at this time."
            draft_source = "Layer 0 Auto-Disqualify ($0)"
            draft_tokens = 0
            apollo_status_msg = "Skipped (Spam / Code Input)"
            variation = "N/A"
            rec_step = "Disqualify"

        else:
            # 1. RUN AI EXTRACTION FIRST
            stage1_res = run_stage_1_scoring(client, lead_payload)

            # 2. AUTO-BACKFILL EXTRACTED FIELDS BEFORE CALLING APOLLO
            for field, target_key in [
                ("extracted_first_name", "first_name"),
                ("extracted_last_name", "last_name"),
                ("extracted_email", "email"),
                ("extracted_company", "company"),
                ("extracted_annual_sales_volume", "annual_sales_volume"),
                ("extracted_job_title", "title"),
                ("extracted_competitor", "competitor_software")
            ]:
                extracted_val = stage1_res.get(field)
                if extracted_val and str(extracted_val).lower() not in ["null", "none", "unknown", "n/a", ""]:
                    if not lead_payload.get(target_key):
                        lead_payload[target_key] = str(extracted_val).strip()

            if "first_name" in lead_payload and "last_name" in lead_payload and "name" not in lead_payload:
                lead_payload["name"] = f"{lead_payload['first_name']} {lead_payload['last_name']}".strip()

            # 3. CALL APOLLO WITH X-API-KEY HEADER
            apollo_status_msg = "Skipped (Auto-Rules)"
            if clean_apollo_key and should_enrich_lead(lead_payload):
                enrichment = enrich_with_apollo(lead_payload.get("email", ""), lead_payload.get("company", ""), clean_apollo_key)
                if enrichment:
                    lead_payload["apollo_enrichment"] = enrichment
                    apollo_status_msg = "Enriched Successfully"
                else:
                    apollo_status_msg = "Attempted (No Org Match Found)"
            elif not clean_apollo_key:
                apollo_status_msg = "No API Key Provided"

            # 4. CONTINUE WITH ROUTING & DRAFTING
            priority = stage1_res.get("priority", "Low")
            rec_step = stage1_res.get("recommended_next_step", "Disqualify")

            variation = random.choice(["A", "B"])

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

    save_session_backup(results)

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

            col_info, col_grid = st.columns([3, 1])

            with col_info:
                st.markdown(f"**Persona Rationale:** {s1.get('persona_rationale', '')}")
                st.markdown(f"**AI Explanation:** {s1.get('explanation', '')}")

                if s1.get('buying_signals'):
                    st.markdown(f"**Buying Signals:** {', '.join(s1.get('buying_signals', []))}")

                c_flags1, c_flags2 = st.columns(2)
                with c_flags1:
                    if s1.get("green_flags"):
                        st.markdown("**🟢 Green Flags:**")
                        for flag in s1.get("green_flags", []):
                            st.caption(f"✅ {flag}")
                with c_flags2:
                    if s1.get("red_flags"):
                        st.markdown("**🚩 Red Flags:**")
                        for flag in s1.get("red_flags", []):
                            st.caption(f"🚨 {flag}")

                if s1.get("missing_info"):
                    st.warning(f"Missing Info: {', '.join(s1.get('missing_info'))}")

                if lead.get("apollo_enrichment"):
                    st.info(f"ℹ️ **Apollo Verified:** {lead.get('apollo_enrichment')}")
                else:
                    st.caption(f"🔍 **Apollo Status:** {item.get('apollo_status', 'N/A')}")

                st.divider()

                c_step, c_notes = st.columns([1, 2])
                current_route = item["final_next_step"]

                route_options = ["Sales", "Self-serve / PLG", "Nurture", "Customer Support", "Disqualify", "Needs More Information"]
                default_route_idx = route_options.index(current_route) if current_route in route_options else 4

                c_step.selectbox(
                    "Override Next Step:",
                    route_options,
                    index=default_route_idx,
                    key=f"route_override_{idx}",
                    on_change=update_route_template,
                    args=(idx,)
                )

                if f"draft_edit_{idx}" not in st.session_state:
                    st.session_state[f"draft_edit_{idx}"] = item["final_draft"]

                new_draft = st.text_area("Outreach Email Draft (Editable):", key=f"draft_edit_{idx}", height=220)

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

            btn_app, btn_rej = st.columns(2)

            if btn_app.button("✅ Approve", type="primary", width="stretch", key=f"btn_approve_{idx}"):
                item["final_next_step"] = st.session_state[f"route_override_{idx}"]
                item["final_draft"] = new_draft
                item["reviewer_notes"] = rev_notes

                if new_draft.strip() != item["draft"].strip() or item["final_next_step"] != item["stage1"].get("recommended_next_step"):
                    item["human_edited"] = True
                    item["review_status"] = "APPROVED_WITH_EDITS"
                    save_correction(persona, item["final_next_step"], str(lead), rev_notes, item["draft"], new_draft)
                else:
                    item["human_edited"] = False
                    item["review_status"] = "APPROVED"

                save_session_backup(results_list)
                st.rerun()

            if btn_rej.button("🗑️ Delete (Spam / Test / Error)", width="stretch", key=f"btn_reject_{idx}"):
                item["review_status"] = "DELETED"
                save_session_backup(results_list)
                st.rerun()

    st.divider()

    # ── Step 4: Export Section ────────────────────────────────────────────────
    st.header("📊 Step 4: Export & Sync")

    all_processed = all(r["review_status"] != "PENDING" for r in results_list)

    export_rows = []
    for r in results_list:
        lead = r["lead"]
        fname = lead.get("first_name") or (lead.get("name", "").split()[0] if lead.get("name") else "N/A")
        lname = lead.get("last_name") or (" ".join(lead.get("name", "").split()[1:]) if len(lead.get("name", "").split()) > 1 else "N/A")

        export_rows.append({
            "First Name": fname,
            "Last Name": lname,
            "Email": lead.get("email", "N/A"),
            "Company": lead.get("company", "N/A"),
            "Sales Volume Tier": lead.get("annual_sales_volume", "N/A"),
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button(
        "✅ Download Approved Leads (CSV)",
        data=approved_df.to_csv(index=False).encode("utf-8"),
        file_name=f"approved_leads_{timestamp}.csv",
        mime="text/csv",
        type="primary",
        width="stretch"
    )

    col_dl2.download_button(
        "⬇️ Download Full Audit Log (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"all_triage_results_{timestamp}.csv",
        mime="text/csv",
        type="secondary",
        width="stretch",
        disabled=not all_processed
    )

    if not all_processed:
        st.info("💡 The Full Audit Log will unlock once all leads have been reviewed.")