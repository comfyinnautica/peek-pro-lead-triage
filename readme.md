# 🏔️ Peek Pro — Multi-Channel Lead Triage Engine

An AI-powered Revenue Operations (RevOps) lead scoring, A/B testing outreach, conditional enrichment, and Human-in-the-Loop (HITL) triage application built for Peek Pro.

---

## 🌟 Overview & System Architecture

This application automates inbound lead qualification and outreach drafting using a **Two-Stage AI Pipeline** with deterministic safety guardrails:

1. **Ingestion & Normalization (Layer 1):** Ingests leads via Web Form, Raw Text/Chat transcripts, or CSV/Excel upload. A zero-token deterministic normalizer standardizes column headers and sanitizes text inputs.
2. **Conditional Data Enrichment (Apollo.io):** Automatically evaluates leads against volume and intent guardrails. If a lead qualifies, the system calls the Apollo API to enrich the payload with firmographic data (estimated revenue, employee count, industry, location) before AI evaluation, reducing hallucinations and filling data gaps.
3. **Stage 1: Intent & ICP Qualification (GPT-4o Mini):** Evaluates lead firmographics and enriched data, extracts buyer personas, maps ICP signals, and assigns a fit score (1–10), priority (`High`, `Medium`, `Low`), and action route.
4. **Stage 2: Dynamic Outreach Drafting (GPT-4o / GPT-4o Mini):** Generates 1-to-1 personalized email drafts featuring automated A/B testing strategies (Strategy A: ROI/Fee Transparency vs. Strategy B: Social Proof/Ops Efficiency) and real-time SDR Session Bias injection.
5. **Step 3: Human Review Gate (HITL):** SDRs can review classifications, override next steps (which dynamically loads route templates), edit drafts, and approve or delete leads.
6. **Step 4: Learning Loop & Export:** Approved edits are logged locally (`corrections.jsonl`) to power few-shot learning for future drafts. Clean CSV audit logs are timestamped and exported.

---

## 🚀 Setup & Run Instructions

### Prerequisites
* Python 3.10+
* OpenAI API Key
* Apollo.io API Key (Optional, for lead enrichment)

### Installation

1. **Clone the repository:**
   git clone https://github.com/your-username/peek-pro-lead-triage.git
   cd peek-pro-lead-triage

2. **Set up a virtual environment:**
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies:**
   pip install -r requirements.txt

4. **Environment Configuration:**
   Copy `.env.example` to `.env` and insert your OpenAI API Key and Apollo API Key:
   cp .env.example .env

5. **Launch the Streamlit App:**
   streamlit run app.py

---

## 🧪 Testing the Application

### Option A: Peek Web Form
* Open the app, fill out the form under **📋 Peek Web Form**, and click **Queue Web Form Lead**.

### Option B: Raw Text / Chat Transcript
* Go to the **💬 Raw Text** tab and paste the following test transcript:

  From: Sarah Jenkins <sarah.jenkins@coloradoriverguides.com>
  Subject: Urgent: Looking for new booking software

  Hi Peek team, I'm the Ops Manager over at Colorado River Guides ($1.8M USD annual volume). We currently use Anchor, but staff scheduling is a nightmare for our 40+ guides. We need to switch before late August. Call me at 555-0199.

### Option C: Batch CSV Upload
* Upload any lead CSV under **📁 Batch Upload**. The normalizer dynamically handles malformed rows and extra commas.

---

## 📐 Short Design Summary

### 1. Workflow & Input Field Selection
The input interface was modeled directly after Peek Pro's web intake and real-world SDR workflows:
* **Multi-Channel Tabs:** Accommodates structured web forms, raw email/chat pastes, and bulk campaign CSV uploads.
* **Session Bias Field:** A 120-character input allowing SDRs to inject real-time context (e.g., *"It's Friday, wish them a great weekend"*).

### 2. AI vs. Fixed Rules
To optimize cost and prevent hallucinations, business logic is split between deterministic code and LLM intelligence:
* **Fixed Rules (Deterministic):**
  * Column alias normalization and CSV padding.
  * Hardcoded fallback templates for $0 cost on Low Priority / Disqualified leads.
  * Conditional enrichment triggers (e.g., only trigger Apollo if $150k+ volume or competitor present) to save API credits.
  * Regex post-processing to strip em-dashes, `$`, and `[Your Name]` placeholders.
* **AI Engine (LLM):**
  * Unstructured text parsing, buyer persona categorization, intent recognition, buying signal extraction, and personalized draft writing.

### 3. Prompt Design & Guardrails
Prompts utilize JSON Mode for structured outputs and enforce ground-truth rules:
* **Intent Override Rule:** Intent trumps unusual titles (e.g., a lead titled "Mayor" asking for conversion fixes is routed to `Needs More Information` or `Sales`, not `Disqualified`).
* **Urgency & Competitor Overrides:** High urgency ("switch this week") or direct competitor displacement automatically elevates leads to `High` priority / `Sales`.
* **Signature & Formatting Guardrails:** Strict prompt rules paired with Python post-processing prevent AI sign-offs, em-dashes, and currency formatting bugs.

### 4. Handling Unclear Leads & Errors
* **Unclear Info:** If key context or volume is missing, the AI assigns `persona: "Unknown"` and routes to `Needs More Information` with an open-ended discovery question.
* **Error Resilience:** Catch-all try/except wrappers gracefully handle API timeouts or missing environment variables.

### 5. Token & Cost Optimization
* **Tiered Routing:** GPT-4o-mini handles Stage 1 scoring for all leads and Stage 2 drafting for Medium priority leads. GPT-4o is strictly reserved for High priority leads.
* **Hardcoded Routing Templates:** Disqualified or low-fit leads utilize static `$0` response templates.
* **Character Capping:** Session Bias inputs are capped at 120 characters to preserve prompt window space.
* **Enrichment Throttling:** Apollo is only called for leads that pass initial deterministic checks, saving third-party API costs.

### 6. Future Improvements
* **Direct CRM Integration:** Replace CSV export buttons with live Webhooks to HubSpot/Salesforce.
* **Vector Store Learning Loop:** Upgrade `corrections.jsonl` to a RAG vector database (e.g., Pinecone/ChromaDB) for semantic few-shot correction retrieving.
* **Automated Web Scraping:** Auto-scrape lead websites on submission to verify actual booking engine infrastructure.

### 7. Measuring Success
* **Efficiency:** SDR time spent per inbound lead (Target: <30 seconds review time).
* **HITL Edit Rate:** Percentage of AI drafts approved without human edits (Target: >85% acceptance).
* **Conversion Rate:** Speed-to-lead and meeting booked rate comparison between AI-triaged leads vs. manual routing.