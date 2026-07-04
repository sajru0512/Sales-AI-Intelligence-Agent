"""
Generate the Sales Intelligence Agent Google Colab Notebook (.ipynb)
====================================================================
This script creates a comprehensive Jupyter notebook for the hackathon.
Run: python generate_notebook.py
"""

import json
import os

# ─────────────────────────────────────────────
# Helper functions to create notebook cells
# ─────────────────────────────────────────────

cells = []

def md(source):
    """Add a markdown cell."""
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    })

def code(source):
    """Add a code cell."""
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [source],
        "outputs": [],
        "execution_count": None
    })


# ═══════════════════════════════════════════════
# SECTION 1: TITLE & INTRODUCTION
# ═══════════════════════════════════════════════

md("""# \U0001f916 AI-Powered Sales Intelligence Agent
## XYZ Analytics Consulting \u2014 Automotive Industry

---

### \U0001f4cb Project Overview

This notebook implements an **AI-powered Sales Intelligence Agent** for **XYZ Analytics Consulting**, an automotive analytics consultancy specializing in:

| Service | Description |
|---------|-------------|
| \U0001f527 **Warranty Analytics** | Detect quality issues early, reduce claim costs, prevent recalls |
| \U0001f4e6 **Supply Chain Risk Prediction** | Monitor supplier health, anticipate disruptions, protect production |
| \U0001f3ea **Dealer & Field Service Intelligence** | Turn dealer data into actionable insights for sales & service |

### \U0001f3af What This Agent Does

1. **Market Research** \u2014 Analyzes the Indian automotive industry landscape
2. **Company Intelligence** \u2014 Deep-dives into OEMs, Tier-1 Suppliers, and Component Manufacturers
3. **Product Knowledge** \u2014 Maps XYZ's solutions to company challenges using RAG
4. **Prioritization** \u2014 Scores and ranks the Top 10-15 target companies
5. **Recommendations** \u2014 Generates actionable business proposals for each target

### \U0001f3d7\ufe0f Technology Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | CrewAI |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Google text-embedding-004 |
| Vector Store | ChromaDB |
| Web Search | Tavily API |
| Web Scraping | BeautifulSoup4 |
| PDF Parsing | pdfplumber |

---
**Author:** Sales Intelligence Team | **Date:** July 2026 | **Hackathon:** XYZ Analytics Consulting
""")


# ═══════════════════════════════════════════════
# SECTION 2: INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════

md("""## \U0001f4e6 Section 1: Install Dependencies

Install all required Python packages. This cell only needs to run once per Colab session.
""")

code("""# ============================================================
# Install Dependencies
# ============================================================
# Note: This may take 2-3 minutes on first run

!pip install -q crewai crewai-tools google-generativeai chromadb tavily-python pdfplumber beautifulsoup4 requests

print("\\u2705 All dependencies installed successfully!")
""")


# ═══════════════════════════════════════════════
# SECTION 3: IMPORTS & CONFIGURATION
# ═══════════════════════════════════════════════

md("""## \u2699\ufe0f Section 2: Configuration & API Keys

Configure the LLM, API keys, and core settings.

> \u26a0\ufe0f **Important:** You will need:
> - A **Google Gemini API Key** (free at [aistudio.google.com](https://aistudio.google.com))
> - A **Tavily API Key** (free at [tavily.com](https://tavily.com))
""")

code("""# ============================================================
# Imports
# ============================================================
import os
import json
import time
import getpass
import warnings
warnings.filterwarnings('ignore')

# CrewAI
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# Google AI
import google.generativeai as genai

# Vector Store
import chromadb

# Web Tools
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup

# PDF Processing
import pdfplumber

# Display
from IPython.display import display, Markdown, HTML

print("\\u2705 All libraries imported successfully!")
""")

code("""# ============================================================
# API Key Configuration
# ============================================================
# Keys are entered securely via getpass (never hardcoded)

try:
    # Try Google Colab secrets first
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    TAVILY_API_KEY = userdata.get('TAVILY_API_KEY')
    print("\\u2705 API keys loaded from Colab Secrets")
except:
    # Fallback to manual input
    print("Enter your API keys below (input is hidden):\\n")
    GOOGLE_API_KEY = getpass.getpass("\\U0001f511 Google Gemini API Key: ")
    TAVILY_API_KEY = getpass.getpass("\\U0001f511 Tavily API Key: ")
    print("\\n\\u2705 API keys configured!")

# Set environment variables
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

# Configure Google AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.3
)

print("\\u2705 LLM initialized: Google Gemini 2.0 Flash")
""")


# ═══════════════════════════════════════════════
# SECTION 4: PRODUCT KNOWLEDGE BASE (RAG)
# ═══════════════════════════════════════════════

md("""## \U0001f4da Section 3: Product Knowledge Base (RAG)

Load the **XYZ Analytics Consulting Product & Solutions Handbook**, create embeddings, and build a vector store for retrieval-augmented generation.

### RAG Pipeline:
```
PDF \\u2192 Text Extraction \\u2192 Chunking \\u2192 Embeddings \\u2192 ChromaDB Vector Store \\u2192 Query Tool
```
""")

code("""# ============================================================
# Upload the Product Knowledge Handbook PDF
# ============================================================

import os

# Check if file exists (for local runs)
handbook_path = None
possible_paths = [
    "XYZ Analytics Consulting \\u2013 Product & Solutions Handbook.pdf",
    "/content/XYZ Analytics Consulting \\u2013 Product & Solutions Handbook.pdf",
    "XYZ Analytics Consulting - Product & Solutions Handbook.pdf",
]

for p in possible_paths:
    if os.path.exists(p):
        handbook_path = p
        break

if handbook_path is None:
    # Upload via Colab
    from google.colab import files
    print("\\U0001f4e4 Please upload the Product & Solutions Handbook PDF file:")
    uploaded = files.upload()
    handbook_path = list(uploaded.keys())[0]

print(f"\\u2705 Handbook found: {handbook_path}")
""")

code("""# ============================================================
# Extract Text from PDF
# ============================================================

handbook_text = ""
page_texts = []

with pdfplumber.open(handbook_path) as pdf:
    print(f"\\U0001f4d6 Processing {len(pdf.pages)} pages...\\n")
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            handbook_text += text + "\\n\\n"
            page_texts.append({"page": i + 1, "text": text})
            print(f"  \\u2705 Page {i + 1}: {len(text)} characters extracted")

print(f"\\n\\U0001f4ca Total text extracted: {len(handbook_text):,} characters from {len(page_texts)} pages")
""")

code("""# ============================================================
# Chunk Text for Embeddings
# ============================================================

def chunk_text(text, chunk_size=800, overlap=150):
    '''Split text into overlapping chunks for better retrieval.'''
    chunks = []
    # Split by double newlines first (paragraph boundaries)
    paragraphs = text.split('\\n\\n')

    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\\n\\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para + "\\n\\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

chunks = chunk_text(handbook_text)
print(f"\\U0001f4e6 Created {len(chunks)} text chunks")
print(f"   Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} characters")
print(f"   Smallest chunk: {min(len(c) for c in chunks)} characters")
print(f"   Largest chunk: {max(len(c) for c in chunks)} characters")
""")

code("""# ============================================================
# Create Embeddings & Vector Store
# ============================================================

def get_embedding(text):
    '''Get embedding vector from Google text-embedding-004 model.'''
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return result['embedding']

# Initialize ChromaDB
chroma_client = chromadb.Client()

# Create collection (delete if exists from previous run)
try:
    chroma_client.delete_collection("xyz_product_knowledge")
except:
    pass

collection = chroma_client.create_collection(
    name="xyz_product_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# Add chunks with embeddings
print("\\U0001f504 Creating embeddings and building vector store...\\n")
for i, chunk in enumerate(chunks):
    embedding = get_embedding(chunk)
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[f"chunk_{i}"],
        metadatas=[{"source": "handbook", "chunk_index": i}]
    )
    if (i + 1) % 5 == 0 or i == len(chunks) - 1:
        print(f"  \\U0001f4cd Processed {i + 1}/{len(chunks)} chunks")
    time.sleep(0.2)  # Rate limit protection

print(f"\\n\\u2705 Vector store created with {collection.count()} chunks")
print("   Model: text-embedding-004 | Store: ChromaDB (in-memory)")
""")

code("""# ============================================================
# Test RAG Retrieval
# ============================================================

def search_knowledge_base(query, n_results=3):
    '''Search the product knowledge base.'''
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

# Test with sample queries
test_queries = [
    "What is Warranty Analytics and what are its key capabilities?",
    "How does Supply Chain Risk Prediction work?",
    "What are the pricing and engagement models?",
]

for query in test_queries:
    results = search_knowledge_base(query, n_results=1)
    print(f"\\U0001f50d Query: {query}")
    print(f"   Top result (first 150 chars): {results['documents'][0][0][:150]}...")
    print()

print("\\u2705 RAG retrieval is working correctly!")
""")


# ═══════════════════════════════════════════════
# SECTION 5: DEFINE TOOLS
# ═══════════════════════════════════════════════

md("""## \\U0001f527 Section 4: Define Agent Tools

Three custom tools power the agents:

| Tool | Purpose | Data Source |
|------|---------|-------------|
| \\U0001f50d **Web Search** | Search for market data, company info, news | Tavily API |
| \\U0001f310 **Web Scraper** | Extract content from specific websites | BeautifulSoup4 |
| \\U0001f4da **Product Knowledge Search** | Query the XYZ handbook via RAG | ChromaDB + Gemini Embeddings |
""")

code("""# ============================================================
# Tool 1: Web Search (Tavily)
# ============================================================

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def web_search(query: str) -> str:
    '''Search the web for information about Indian automotive companies, market trends,
    financial data, industry reports, and business news. Use specific, targeted queries
    for the best results. Good for finding company financials, recent news, market statistics,
    and industry analysis.'''
    try:
        results = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )

        output_parts = []
        answer = results.get('answer', '')
        if answer:
            output_parts.append(f"**Summary:** {answer}\\n")

        output_parts.append("**Sources:**")
        for r in results.get('results', []):
            title = r.get('title', 'N/A')
            content = r.get('content', '')[:400]
            url = r.get('url', '')
            output_parts.append(f"\\n- **{title}**\\n  {content}\\n  Source: {url}")

        return "\\n".join(output_parts)
    except Exception as e:
        return f"Search error: {str(e)}. Try a different or simpler query."

print("\\u2705 Web Search tool ready (Tavily API)")
""")

code("""# ============================================================
# Tool 2: Web Scraper
# ============================================================

@tool
def scrape_website(url: str) -> str:
    '''Scrape a website URL to extract text content. Useful for extracting information
    from company websites, IBEF automotive industry pages, SIAM statistics, ACMA data,
    Moneycontrol financial pages, and Screener.in company profiles. Provide the full URL.'''
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove non-content elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            tag.decompose()

        # Extract text
        text = soup.get_text(separator='\\n', strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split('\\n') if line.strip()]
        clean_text = '\\n'.join(lines)

        # Truncate to avoid token limits
        if len(clean_text) > 4000:
            clean_text = clean_text[:4000] + "\\n\\n[Content truncated for brevity]"

        return clean_text
    except Exception as e:
        return f"Scraping error for {url}: {str(e)}. The site may be blocking automated access."

print("\\u2705 Web Scraper tool ready (BeautifulSoup4)")
""")

code("""# ============================================================
# Tool 3: Product Knowledge Search (RAG)
# ============================================================

@tool
def search_product_knowledge(query: str) -> str:
    '''Search the XYZ Analytics Consulting Product and Solutions Handbook for information
    about their consulting services, solutions, capabilities, case studies, pricing models,
    implementation roadmap, KPIs, deliverables, and business value propositions.
    Use this tool to understand what XYZ offers and match services to company challenges.'''
    try:
        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4
        )

        if results and results['documents'] and results['documents'][0]:
            formatted_results = []
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append(f"**[Relevant Section {i+1}]**\\n{doc}")
            return "\\n\\n---\\n\\n".join(formatted_results)
        return "No relevant information found in the product handbook."
    except Exception as e:
        return f"Knowledge search error: {str(e)}"

print("\\u2705 Product Knowledge Search tool ready (RAG + ChromaDB)")
print("\\n" + "="*50)
print("\\U0001f3af All 3 tools are ready for the agents!")
print("="*50)
""")


# ═══════════════════════════════════════════════
# SECTION 6: DEFINE AI AGENTS
# ═══════════════════════════════════════════════

md("""## \\U0001f916 Section 5: Define AI Agents

Five specialized agents work together in a sequential pipeline:

```
Market Research Agent \\u2192 Company Research Agent \\u2192 Product Knowledge Agent
                                                           \\u2193
              Recommendation Agent  \\u2190  Prioritization Agent
```

Each agent has a specific **role**, **goal**, **backstory**, and set of **tools**.
""")

code("""# ============================================================
# Agent 1: Market Research Agent
# ============================================================

market_research_agent = Agent(
    role="Senior Automotive Market Research Analyst",
    goal=(
        "Research the Indian automotive industry comprehensively and identify "
        "25-30 potential target companies including Automotive OEMs, Tier-1 Suppliers, "
        "and Component Manufacturers that could benefit from data analytics and "
        "consulting services for warranty, supply chain, and dealer operations."
    ),
    backstory=(
        "You are a veteran market research analyst with 15+ years of experience "
        "in the Indian automotive industry. You have deep knowledge of the Indian auto "
        "market from major OEMs like Maruti Suzuki, Tata Motors, Mahindra and Mahindra, "
        "and Hyundai Motor India, to Tier-1 suppliers like Bosch India, Motherson Group, "
        "Bharat Forge, and Samvardhana Motherson, to component manufacturers like Minda "
        "Industries, Varroc Engineering, and Endurance Technologies. You understand market "
        "trends including the EV transition, supply chain challenges post-pandemic, rising "
        "warranty costs, and dealer network dynamics. You use authoritative sources like "
        "IBEF, SIAM, and ACMA for industry data."
    ),
    tools=[web_search, scrape_website],
    llm=llm,
    verbose=True,
    max_iter=15,
    allow_delegation=False
)

print("\\u2705 Agent 1: Market Research Agent created")
""")

code("""# ============================================================
# Agent 2: Company Research Agent
# ============================================================

company_research_agent = Agent(
    role="Corporate Intelligence and Financial Analyst",
    goal=(
        "Conduct detailed deep-dive research on each identified target company to build "
        "comprehensive company profiles including business overview, products and services, "
        "manufacturing locations, financial performance, recent news, growth initiatives, "
        "and business challenges related to warranty, supply chain, and dealer operations."
    ),
    backstory=(
        "You are an experienced corporate intelligence analyst specializing in automotive "
        "companies in India. You excel at gathering and synthesizing information from "
        "multiple sources: company websites, financial databases (NSE, BSE, Moneycontrol, "
        "Screener), industry reports (IBEF, SIAM, ACMA), and business news (Economic Times "
        "Auto, Autocar Professional). You produce detailed, structured company profiles "
        "that highlight both opportunities and challenges. You are particularly skilled at "
        "identifying operational pain points related to quality management, supply chain "
        "risk, and dealer/distribution performance."
    ),
    tools=[web_search, scrape_website],
    llm=llm,
    verbose=True,
    max_iter=20,
    allow_delegation=False
)

print("\\u2705 Agent 2: Company Research Agent created")
""")

code("""# ============================================================
# Agent 3: Product Knowledge Agent
# ============================================================

product_knowledge_agent = Agent(
    role="XYZ Analytics Consulting Solutions Architect",
    goal=(
        "Deeply understand XYZ Analytics Consultings three core service offerings: "
        "Warranty Analytics, Supply Chain Risk Prediction, and Dealer and Field Service "
        "Intelligence. Create a comprehensive solutions mapping guide that identifies "
        "what company challenges and symptoms indicate a need for each service."
    ),
    backstory=(
        "You are a solutions architect at XYZ Analytics Consulting with complete knowledge "
        "of the companys service portfolio. You understand the technical capabilities, "
        "business value, KPIs, deliverables, implementation roadmap, and ROI of each "
        "solution. You are skilled at matching a companys specific business challenges "
        "with the most appropriate consulting solution and articulating the expected "
        "business value. You always reference specific KPIs, case studies, and "
        "quantified benefits from the product handbook."
    ),
    tools=[search_product_knowledge],
    llm=llm,
    verbose=True,
    max_iter=10,
    allow_delegation=False
)

print("\\u2705 Agent 3: Product Knowledge Agent created")
""")

code("""# ============================================================
# Agent 4: Prioritization Agent
# ============================================================

prioritization_agent = Agent(
    role="Strategic Sales Prioritization Advisor",
    goal=(
        "Evaluate all researched companies and select the Top 10-15 that represent "
        "the best opportunities for XYZ Analytics Consulting. Use a structured scoring "
        "framework covering revenue potential, challenge alignment, analytics readiness, "
        "strategic value, and engagement likelihood. Ensure the final list includes a "
        "mix of OEMs, Tier-1 Suppliers, and Component Manufacturers."
    ),
    backstory=(
        "You are a strategic sales advisor specializing in B2B customer prioritization "
        "for analytics consulting firms. You evaluate companies based on multiple criteria: "
        "market position and revenue (ability to invest), urgency of analytics needs "
        "(specific pain points), alignment with the consultancys services, strategic "
        "value (brand, reference potential), and signals of openness to technology "
        "adoption. You use a rigorous scoring framework to ensure objectivity and "
        "provide clear, defensible justifications for each selection."
    ),
    tools=[search_product_knowledge],
    llm=llm,
    verbose=True,
    max_iter=10,
    allow_delegation=False
)

print("\\u2705 Agent 4: Prioritization Agent created")
""")

code("""# ============================================================
# Agent 5: Recommendation Agent
# ============================================================

recommendation_agent = Agent(
    role="Senior Business Development Consultant",
    goal=(
        "Create comprehensive, executive-ready business recommendations for each of "
        "the Top 10-15 prioritized companies. Each recommendation must include an "
        "executive summary, company snapshot, business challenges analysis, recommended "
        "XYZ consulting solution with specific use cases, expected business value with "
        "quantified benefits, and an engagement strategy."
    ),
    backstory=(
        "You are a senior business development consultant who creates compelling, "
        "data-backed recommendations for sales teams. You combine market intelligence, "
        "company research, and product knowledge to produce executive-ready reports "
        "that clearly articulate why a company is a strong prospect, which XYZ service "
        "is most relevant, and what business value the engagement would deliver. Your "
        "recommendations include specific ROI projections based on industry benchmarks "
        "and case studies. You are persuasive, structured, and actionable."
    ),
    tools=[search_product_knowledge, web_search],
    llm=llm,
    verbose=True,
    max_iter=15,
    allow_delegation=False
)

print("\\u2705 Agent 5: Recommendation Agent created")
print("\\n" + "="*50)
print("\\U0001f916 All 5 agents are ready!")
print("="*50)
""")


# ═══════════════════════════════════════════════
# SECTION 7: DEFINE TASKS
# ═══════════════════════════════════════════════

md("""## \\U0001f4dd Section 6: Define Tasks

Five sequential tasks form the intelligence pipeline. Each task builds on the output of previous tasks.

```
Task 1: Market Research  \\u2192  Task 2: Company Deep-Dive  \\u2192  Task 3: Product Knowledge
                                                                      \\u2193
                         Task 5: Recommendations  \\u2190  Task 4: Prioritization
```
""")

code("""# ============================================================
# Task 1: Market Research
# ============================================================

market_research_task = Task(
    description=(
        "Conduct comprehensive research on the Indian automotive industry and identify "
        "potential target companies for XYZ Analytics Consulting.\\n\\n"
        "Your research MUST cover:\\n\\n"
        "1. **Industry Overview:**\\n"
        "   - Market size (production volume, revenue, GDP contribution)\\n"
        "   - Growth trajectory and projections\\n"
        "   - Key segments: passenger vehicles, commercial vehicles, two-wheelers, EVs\\n"
        "   - Export performance\\n\\n"
        "2. **Key Market Trends:**\\n"
        "   - EV transition and electrification initiatives\\n"
        "   - Software-defined vehicles\\n"
        "   - Supply chain resilience challenges\\n"
        "   - Government policies (PLI scheme, FAME, scrappage policy)\\n\\n"
        "3. **Industry Challenges (relevant to XYZ services):**\\n"
        "   - Rising warranty costs (currently 2-4% of revenue)\\n"
        "   - Supply chain disruptions and logistics costs\\n"
        "   - Dealer network performance inconsistencies\\n\\n"
        "4. **Target Company Identification:**\\n"
        "   Identify 25-30 companies across THREE categories:\\n"
        "   - **Automotive OEMs** (8-10 companies): e.g., Maruti Suzuki, Tata Motors, "
        "Mahindra and Mahindra, Hyundai Motor India, Kia India, Toyota Kirloskar, "
        "Honda Cars India, MG Motor India, Bajaj Auto, TVS Motor, Hero MotoCorp, "
        "Ashok Leyland, Eicher Motors (Royal Enfield)\\n"
        "   - **Tier-1 Suppliers** (8-10 companies): e.g., Bosch India, Motherson Group, "
        "Bharat Forge, Sundaram-Clayton, ZF Group India, Continental India, "
        "Schaeffler India, SKF India, Cummins India, Wabco India\\n"
        "   - **Component Manufacturers** (8-10 companies): e.g., Minda Industries, "
        "Varroc Engineering, Endurance Technologies, Sona BLW, Suprajit Engineering, "
        "Jamna Auto, Balkrishna Industries, Apollo Tyres, CEAT, Exide Industries\\n\\n"
        "For each company provide: Company Name, Category (OEM/Tier-1/Component), "
        "Headquarters location, and a brief note on why they could benefit from analytics consulting."
    ),
    agent=market_research_agent,
    expected_output=(
        "A comprehensive market research report with:\\n"
        "1. Indian Automotive Industry Overview (market size, growth, key statistics)\\n"
        "2. Key Market Trends and Challenges\\n"
        "3. A structured list of 25-30 target companies organized by category "
        "(OEMs, Tier-1 Suppliers, Component Manufacturers) with company name, "
        "category, headquarters, and brief rationale for each"
    )
)

print("\\u2705 Task 1: Market Research defined")
""")

code("""# ============================================================
# Task 2: Company Deep-Dive Research
# ============================================================

company_research_task = Task(
    description=(
        "For each company identified in the market research, conduct a detailed "
        "deep-dive analysis. For EACH company, research and compile:\\n\\n"
        "1. **Company Overview:** Full name, founded year, headquarters, type "
        "(public/private), key leadership, number of employees\\n\\n"
        "2. **Products and Services:** Main product lines, market segments served, "
        "key customers, technology focus areas\\n\\n"
        "3. **Manufacturing Locations:** List of major plants/facilities in India\\n\\n"
        "4. **Business Growth:** Revenue trends (last 2-3 years), expansion plans, "
        "recent investments, new product launches, M and A activity\\n\\n"
        "5. **Financial Highlights:** Latest annual revenue (in INR Crores), "
        "operating margins, R and D spending as percent of revenue, market capitalization "
        "(for listed companies)\\n\\n"
        "6. **Recent News:** Significant developments in the last 6-12 months: "
        "partnerships, challenges, recalls, expansions, technology initiatives\\n\\n"
        "7. **Business Challenges:** Specific operational challenges related to:\\n"
        "   - Quality and warranty management issues\\n"
        "   - Supply chain risks and disruptions\\n"
        "   - Dealer network and field service challenges\\n"
        "   - Data and analytics maturity\\n\\n"
        "Use financial sources (Moneycontrol, Screener), industry sources (IBEF, "
        "SIAM, ACMA), and news sources (ET Auto, Autocar Professional) for data.\\n\\n"
        "FORMAT: Present each company as a structured profile card."
    ),
    agent=company_research_agent,
    expected_output=(
        "Detailed company profiles for all identified companies. Each profile must "
        "contain: Company Overview, Products and Services, Manufacturing Locations, "
        "Business Growth, Financial Highlights, Recent News, and Business Challenges. "
        "Format each as a structured profile card with clear headings."
    ),
    context=[market_research_task]
)

print("\\u2705 Task 2: Company Deep-Dive Research defined")
""")

code("""# ============================================================
# Task 3: Product Knowledge Analysis
# ============================================================

product_knowledge_task = Task(
    description=(
        "Thoroughly analyze the XYZ Analytics Consulting Product and Solutions Handbook "
        "to understand all three service offerings in complete detail.\\n\\n"
        "For EACH service, extract and summarize:\\n\\n"
        "**1. Warranty Analytics:**\\n"
        "   - What it does and how it works\\n"
        "   - Key capabilities (anomaly detection, clustering, prediction, etc.)\\n"
        "   - Business value and ROI (5-10 percent warranty cost reduction, $20M case study)\\n"
        "   - KPIs tracked (Defects/1000, MTBF, Warranty Spend/Unit, etc.)\\n"
        "   - Deliverables (dashboards, alerts, reports, ML models)\\n"
        "   - Ideal client profile: Who benefits most?\\n\\n"
        "**2. Supply Chain Risk Prediction:**\\n"
        "   - What it does and how it works\\n"
        "   - Key capabilities (graph analysis, risk scoring, disruption simulation)\\n"
        "   - Business value (15 percent logistics cost reduction, 35 percent inventory reduction)\\n"
        "   - KPIs tracked (Supplier Risk Score, Parts Availability, OTIF, etc.)\\n"
        "   - Deliverables (control tower, alerts, reports, predictive models)\\n"
        "   - Ideal client profile\\n\\n"
        "**3. Dealer and Field Service Intelligence:**\\n"
        "   - What it does and how it works\\n"
        "   - Key capabilities (dealer scorecards, demand forecasting, lead scoring)\\n"
        "   - Business value (15 percent warranty sales increase, 10-20 percent KPI improvement)\\n"
        "   - KPIs tracked (Sales/Dealer, Conversion Rate, Finance Penetration, etc.)\\n"
        "   - Deliverables (dashboards, alerts, reports, models)\\n"
        "   - Ideal client profile\\n\\n"
        "**4. Cross-cutting Information:**\\n"
        "   - Data requirements and sources\\n"
        "   - Technical approach and architecture\\n"
        "   - Implementation roadmap (3 phases over 9-12 months)\\n"
        "   - Pricing and engagement models\\n\\n"
        "Create a SOLUTIONS MAPPING GUIDE that maps specific company symptoms/challenges "
        "to the most appropriate XYZ service."
    ),
    agent=product_knowledge_agent,
    expected_output=(
        "A comprehensive solutions mapping guide containing:\\n"
        "1. Detailed analysis of each XYZ service with capabilities, value, KPIs\\n"
        "2. Ideal client profiles for each service\\n"
        "3. Challenge-to-solution mapping criteria (which symptoms lead to which service)\\n"
        "4. Expected ROI benchmarks for each service\\n"
        "5. Implementation and engagement considerations"
    )
)

print("\\u2705 Task 3: Product Knowledge Analysis defined")
""")

code("""# ============================================================
# Task 4: Company Prioritization
# ============================================================

prioritization_task = Task(
    description=(
        "Using the company profiles from Task 2 and the solutions mapping guide from "
        "Task 3, evaluate ALL researched companies and select the **Top 10-15** that "
        "represent the BEST opportunities for XYZ Analytics Consulting.\\n\\n"
        "**Scoring Framework** (rate each criterion 1-5):\\n\\n"
        "| Criterion | Weight | Description |\\n"
        "|-|-|-|\\n"
        "| Revenue Potential | 25% | Company size, revenue, ability to invest in analytics |\\n"
        "| Challenge Alignment | 30% | How well challenges match XYZ 3 services |\\n"
        "| Analytics Readiness | 15% | Data maturity, technology adoption signals |\\n"
        "| Strategic Value | 15% | Industry influence, brand value, reference potential |\\n"
        "| Engagement Likelihood | 15% | Signals of openness to consulting/technology |\\n\\n"
        "**For each selected company, provide:**\\n"
        "1. Scoring table with all criteria scores and weighted total\\n"
        "2. **Why Selected:** Specific business reasons (not generic)\\n"
        "3. **Primary Recommended Service:** Which of XYZ 3 services is most relevant\\n"
        "4. **Secondary Service (if applicable):** Additional opportunity\\n"
        "5. **Expected Business Value:** Quantified where possible using handbook benchmarks\\n"
        "6. **Customer Strength Assessment:** Why this company is a strong potential customer\\n\\n"
        "**Diversity Requirements:**\\n"
        "- At least 4-5 Automotive OEMs\\n"
        "- At least 3-4 Tier-1 Suppliers\\n"
        "- At least 2-3 Component Manufacturers\\n\\n"
        "Present the final list as a RANKED TABLE followed by detailed justifications."
    ),
    agent=prioritization_agent,
    expected_output=(
        "A prioritized list of Top 10-15 companies with:\\n"
        "1. Ranked scoring table showing all criteria and weighted total scores\\n"
        "2. For each company: detailed selection rationale, primary and secondary "
        "recommended services, quantified expected value, customer strength assessment\\n"
        "3. Balanced distribution across OEMs, Tier-1 Suppliers, and Component Manufacturers"
    ),
    context=[company_research_task, product_knowledge_task]
)

print("\\u2705 Task 4: Company Prioritization defined")
""")

code("""# ============================================================
# Task 5: Final Business Recommendations
# ============================================================

recommendation_task = Task(
    description=(
        "Create comprehensive, executive-ready business recommendations for each of "
        "the Top 10-15 prioritized companies. This is the FINAL deliverable for the "
        "sales team.\\n\\n"
        "For EACH company, produce a detailed recommendation with these sections:\\n\\n"
        "---\\n\\n"
        "### [Company Name] - [Category: OEM/Tier-1/Component]\\n\\n"
        "**1. Executive Summary** (2-3 sentences)\\n"
        "   Why this company should be a priority target for XYZ Analytics Consulting.\\n\\n"
        "**2. Company Snapshot**\\n"
        "   - Revenue, employees, market position\\n"
        "   - Key products and markets\\n"
        "   - Recent strategic moves\\n\\n"
        "**3. Business Challenges**\\n"
        "   - Specific operational challenges identified through research\\n"
        "   - Impact on business performance\\n"
        "   - Why analytics is the solution to these challenges\\n\\n"
        "**4. Recommended Consulting Solution**\\n"
        "   - **Primary:** [Warranty Analytics / Supply Chain Risk / Dealer Intelligence]\\n"
        "   - **Secondary:** [If applicable]\\n"
        "   - Specific use cases for this company\\n"
        "   - Expected KPI improvements (reference specific KPIs from the handbook)\\n\\n"
        "**5. Expected Business Value**\\n"
        "   - Quantified cost savings or revenue uplift (use handbook benchmarks)\\n"
        "   - ROI timeline (reference the 9-12 month implementation roadmap)\\n"
        "   - Strategic benefits beyond financial returns\\n\\n"
        "**6. Engagement Strategy**\\n"
        "   - Recommended approach: Pilot PoC vs. full engagement\\n"
        "   - Key stakeholders to target (CXO level)\\n"
        "   - Suggested talking points for initial outreach\\n"
        "   - Recommended engagement model from the handbook\\n\\n"
        "---\\n\\n"
        "FORMAT: Use clear markdown with headers, bullet points, and bold text. "
        "Make each recommendation self-contained and ready for the sales team to use "
        "directly in customer engagement preparation."
    ),
    agent=recommendation_agent,
    expected_output=(
        "A comprehensive sales intelligence report with detailed business recommendations "
        "for each of the Top 10-15 companies. Each recommendation includes: Executive "
        "Summary, Company Snapshot, Business Challenges, Recommended Solution (with specific "
        "use cases and KPIs), Expected Business Value (quantified), and Engagement Strategy. "
        "Formatted as a professional, executive-ready document."
    ),
    context=[prioritization_task, product_knowledge_task]
)

print("\\u2705 Task 5: Final Business Recommendations defined")
print("\\n" + "="*50)
print("\\U0001f4dd All 5 tasks are defined and ready!")
print("="*50)
""")


# ═══════════════════════════════════════════════
# SECTION 8: CREATE & EXECUTE THE CREW
# ═══════════════════════════════════════════════

md("""## \\U0001f680 Section 7: Execute the Sales Intelligence Agent

The crew orchestrates all 5 agents through the sequential task pipeline.

> \\u23f1\\ufe0f **Estimated Runtime:** 10-20 minutes (depending on API response times)

```
[Market Research] \\u2192 [Company Deep-Dive] \\u2192 [Product Knowledge] \\u2192 [Prioritization] \\u2192 [Recommendations]
```
""")

code("""# ============================================================
# Create the Crew
# ============================================================

sales_intelligence_crew = Crew(
    agents=[
        market_research_agent,
        company_research_agent,
        product_knowledge_agent,
        prioritization_agent,
        recommendation_agent
    ],
    tasks=[
        market_research_task,
        company_research_task,
        product_knowledge_task,
        prioritization_task,
        recommendation_task
    ],
    process=Process.sequential,
    verbose=True,
    memory=True,
    full_output=True
)

print("\\u2705 Sales Intelligence Crew assembled!")
print(f"   Agents: {len(sales_intelligence_crew.agents)}")
print(f"   Tasks: {len(sales_intelligence_crew.tasks)}")
print(f"   Process: Sequential")
print(f"   LLM: Google Gemini 2.0 Flash")
""")

code("""# ============================================================
# Execute the Crew
# ============================================================

print("\\U0001f680 " + "="*58)
print("\\U0001f680  STARTING SALES INTELLIGENCE AGENT")
print("\\U0001f680  XYZ Analytics Consulting - Automotive Industry")
print("\\U0001f680 " + "="*58)
print()

start_time = time.time()

# Run the crew
result = sales_intelligence_crew.kickoff()

elapsed = time.time() - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

print()
print("\\u2705 " + "="*58)
print(f"\\u2705  SALES INTELLIGENCE AGENT COMPLETED!")
print(f"\\u2705  Total execution time: {minutes}m {seconds}s")
print("\\u2705 " + "="*58)
""")


# ═══════════════════════════════════════════════
# SECTION 9: DISPLAY OUTPUTS
# ═══════════════════════════════════════════════

md("""---

# \\U0001f4ca OUTPUT REPORTS

The following sections display the complete output from the Sales Intelligence Agent.

---
""")

# Market Research Report
md("""## \\U0001f4ca Report 1: Indian Automotive Market Research""")

code("""# ============================================================
# Display Market Research Report
# ============================================================

market_research_output = market_research_task.output.raw

display(Markdown("---"))
display(Markdown(market_research_output))
display(Markdown("---"))
""")

# Company Profiles
md("""## \\U0001f3e2 Report 2: Company Deep-Dive Profiles""")

code("""# ============================================================
# Display Company Research Profiles
# ============================================================

company_research_output = company_research_task.output.raw

display(Markdown("---"))
display(Markdown(company_research_output))
display(Markdown("---"))
""")

# Product Knowledge
md("""## \\U0001f4da Report 3: XYZ Product Knowledge & Solutions Mapping""")

code("""# ============================================================
# Display Product Knowledge Analysis
# ============================================================

product_knowledge_output = product_knowledge_task.output.raw

display(Markdown("---"))
display(Markdown(product_knowledge_output))
display(Markdown("---"))
""")

# Prioritized Companies
md("""## \\U0001f3c6 Report 4: Top 10-15 Target Companies (Prioritized)""")

code("""# ============================================================
# Display Prioritized Company Rankings
# ============================================================

prioritization_output = prioritization_task.output.raw

display(Markdown("---"))
display(Markdown(prioritization_output))
display(Markdown("---"))
""")

# Business Recommendations
md("""## \\U0001f4bc Report 5: Business Recommendations & Solution Mapping""")

code("""# ============================================================
# Display Final Business Recommendations
# ============================================================

recommendation_output = recommendation_task.output.raw

display(Markdown("---"))
display(Markdown(recommendation_output))
display(Markdown("---"))
""")

# Combined Executive Summary
md("""## \\U0001f4cb Executive Summary - Complete Sales Intelligence Report""")

code("""# ============================================================
# Generate Combined Executive Summary
# ============================================================

executive_summary = f'''
# \\U0001f916 AI-Powered Sales Intelligence Report
## XYZ Analytics Consulting - Indian Automotive Industry

---

## \\U0001f4ca How This Report Was Generated

This report was generated by an **AI-powered Sales Intelligence Agent** that:

1. **Researched** the Indian automotive industry using web search and industry sources
2. **Identified** 25-30 potential target companies across OEMs, Tier-1 Suppliers, and Component Manufacturers
3. **Analyzed** each company business profile, financials, challenges, and growth trajectory
4. **Mapped** XYZ consulting solutions to company-specific challenges using RAG
5. **Prioritized** the Top 10-15 companies using a weighted scoring framework
6. **Generated** actionable business recommendations for the sales team

### Technology Stack
| Component | Technology |
|-----------|-----------|
| Agent Framework | CrewAI (5 specialized agents) |
| LLM | Google Gemini 2.0 Flash |
| Knowledge Retrieval | RAG (ChromaDB + text-embedding-004) |
| Web Search | Tavily API |
| Process | Sequential multi-agent pipeline |

### Execution Summary
| Metric | Value |
|--------|-------|
| Agents Used | 5 |
| Tasks Executed | 5 |
| Execution Time | {minutes}m {seconds}s |
| Companies Researched | 25-30 |
| Companies Recommended | Top 10-15 |

---

*Generated on: July 2026*
*Agent: XYZ Sales Intelligence Agent v1.0*
'''

display(Markdown(executive_summary))
""")


# ═══════════════════════════════════════════════
# SECTION 10: SAVE OUTPUTS
# ═══════════════════════════════════════════════

md("""## \\U0001f4be Save Reports""")

code("""# ============================================================
# Save All Reports to Files
# ============================================================

reports = {
    "01_Market_Research_Report.md": market_research_output,
    "02_Company_Profiles.md": company_research_output,
    "03_Product_Knowledge_Analysis.md": product_knowledge_output,
    "04_Top_Companies_Prioritized.md": prioritization_output,
    "05_Business_Recommendations.md": recommendation_output,
}

import os
os.makedirs("reports", exist_ok=True)

for filename, content in reports.items():
    filepath = os.path.join("reports", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\\U0001f4be Saved: {filepath}")

print(f"\\n\\u2705 All {len(reports)} reports saved to /reports/ directory!")

# Download reports (for Colab)
try:
    from google.colab import files
    import shutil
    shutil.make_archive("Sales_Intelligence_Reports", 'zip', "reports")
    files.download("Sales_Intelligence_Reports.zip")
    print("\\U0001f4e5 Reports ZIP downloaded!")
except:
    print("\\U0001f4c1 Reports are available in the /reports/ directory")
""")


# ═══════════════════════════════════════════════
# SECTION 11: ARCHITECTURE DIAGRAM
# ═══════════════════════════════════════════════

md("""---

## \\U0001f3d7\\ufe0f Architecture Diagram

### Agent Workflow & System Architecture

```
+=========================================================================+
|                    AI SALES INTELLIGENCE AGENT                           |
|                    XYZ Analytics Consulting                              |
+=========================================================================+
|                                                                         |
|  +--------------+                                                       |
|  |  User Input  |  "Research Indian automotive industry and recommend   |
|  |  (Trigger)   |   target companies for XYZ Analytics Consulting"      |
|  +------+-------+                                                       |
|         |                                                               |
|         v                                                               |
|  +------------------------------------------------------+              |
|  |            CREWAI ORCHESTRATOR                        |              |
|  |     Sequential Process | Memory Enabled               |              |
|  |     LLM: Google Gemini 2.0 Flash                      |              |
|  +------+----------+----------+----------+---------+----+              |
|         |          |          |          |         |                     |
|         v          v          v          v         v                     |
|  +----------++----------++----------++----------++----------+           |
|  | Agent 1  || Agent 2  || Agent 3  || Agent 4  || Agent 5  |           |
|  | Market   || Company  || Product  || Priority || Recommend|           |
|  | Research || Research ||Knowledge || -ization || -ation   |           |
|  | Analyst  || Analyst  || Expert   || Advisor  ||Consultant|           |
|  +----+-----++----+-----++----+-----++----+-----++----+-----+           |
|       |           |           |           |           |                  |
|  +----+-----------+---+  +----+---+  +----+-----------+---+             |
|  |     TOOL LAYER     |  |  RAG   |  |     TOOL LAYER     |             |
|  |                    |  |  TOOL  |  |                    |             |
|  | +------+ +------+ |  |        |  | +------+ +------+ |             |
|  | |Tavily| | Web  | |  |ChromaDB|  | |Tavily| |  RAG | |             |
|  | |Search| |Scrape| |  |Vectors |  | |Search| |      | |             |
|  | +------+ +------+ |  |        |  | +------+ +------+ |             |
|  +--------------------+  +----+---+  +--------------------+             |
|                               |                                         |
|                    +----------+----------+                              |
|                    |   KNOWLEDGE BASE    |                              |
|                    |                     |                              |
|                    |  XYZ Product &      |                              |
|                    |  Solutions Handbook  |                              |
|                    |  (PDF > Chunks >    |                              |
|                    |   Embeddings >      |                              |
|                    |   ChromaDB)         |                              |
|                    +---------------------+                              |
|                                                                         |
|  +----------------------- OUTPUTS --------------------------+           |
|  |                                                           |           |
|  |  Market Research Report                                   |           |
|  |  Company Deep-Dive Profiles (25-30 companies)            |           |
|  |  Product Knowledge & Solutions Mapping Guide              |           |
|  |  Top 10-15 Prioritized Companies (Scored & Ranked)       |           |
|  |  Business Recommendations (Executive-Ready)               |           |
|  |                                                           |           |
|  +-----------------------------------------------------------+           |
|                                                                         |
+=========================================================================+
```

### Data Flow

```
                    +------------------+
                    |  DATA SOURCES    |
                    +--------+---------+
                             |
            +----------------+----------------+
            |                |                |
            v                v                v
    +---------------+ +------------+ +--------------+
    |  Web Search   | |  Web       | |  Product     |
    |  (Tavily)     | |  Scraping  | |  Handbook    |
    |               | |            | |  (PDF/RAG)   |
    | - IBEF        | | - Company  | |              |
    | - SIAM        | |   websites | | - Services   |
    | - ACMA        | | - NSE/BSE  | | - KPIs       |
    | - ET Auto     | | - Screener | | - Case       |
    | - Moneycontrol| | - LinkedIn | |   Studies    |
    | - Google News | |            | | - Pricing    |
    +-------+-------+ +-----+------+ +------+-------+
            |               |               |
            +---------------+---------------+
                            |
                            v
                +-----------------------+
                |  AI AGENT PIPELINE    |
                |                       |
                |  1. Market Research   |
                |  2. Company Analysis  |
                |  3. Solution Mapping  |
                |  4. Prioritization    |
                |  5. Recommendations   |
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                |  DELIVERABLES         |
                |                       |
                |  - Market Report      |
                |  - Company Profiles   |
                |  - Top 10-15 List     |
                |  - Recommendations    |
                |  - Solution Mapping   |
                +-----------------------+
```

### Knowledge Retrieval Process (RAG)

```
+----------+    +--------------+    +--------------+    +--------------+
|  PDF     |--->|  Text        |--->|  Embeddings  |--->|  ChromaDB    |
|  Handbook|    |  Extraction  |    |  (Google     |    |  Vector      |
|          |    |  (pdfplumber)|    |   text-      |    |  Store       |
+----------+    +--------------+    |   embedding  |    +------+-------+
                                    |   -004)      |           |
                                    +--------------+           |
                                                               |
+----------+    +--------------+    +--------------+           |
|  Agent   |--->|  Query       |--->|  Cosine      |<----------+
|  Query   |    |  Embedding   |    |  Similarity  |
|          |    |              |    |  Search      |
+----------+    +--------------+    +------+-------+
                                           |
                                           v
                                    +--------------+
                                    |  Top-K       |
                                    |  Relevant    |
                                    |  Chunks      |
                                    |  Returned    |
                                    +--------------+
```
""")


# ═══════════════════════════════════════════════
# SECTION 12: CONCLUSION
# ═══════════════════════════════════════════════

md("""---

## \\u2705 Conclusion

This AI-powered Sales Intelligence Agent demonstrates how **multi-agent AI systems** can transform B2B sales research and customer acquisition for consulting firms.

### Key Achievements:
- \\u2705 **Automated Market Research** - Comprehensive Indian automotive industry analysis
- \\u2705 **Intelligent Company Profiling** - Deep-dive analysis of 25-30 companies
- \\u2705 **RAG-based Product Matching** - Solutions mapped to specific company challenges
- \\u2705 **Data-Driven Prioritization** - Structured scoring framework for objective ranking
- \\u2705 **Actionable Recommendations** - Executive-ready proposals for the sales team

### Business Impact:
- Reduced research time from weeks to minutes
- Improved targeting accuracy through data-driven prioritization
- Better solution-fit via RAG-based product knowledge matching
- Sales-ready outputs that enable faster customer engagement

### Future Enhancements:
- Real-time data refresh and continuous monitoring
- CRM integration for automated lead scoring
- Expansion to global automotive markets
- Integration with LinkedIn for contact identification

---
*Built for the XYZ Analytics Consulting Hackathon | July 2026*
""")


# ═══════════════════════════════════════════════
# ASSEMBLE & WRITE THE NOTEBOOK
# ═══════════════════════════════════════════════

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "name": "Sales_Intelligence_Agent.ipynb",
            "toc_visible": True
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sales_Intelligence_Agent.ipynb")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook generated: {output_path}")
print(f"   Total cells: {len(cells)}")
print(f"   Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"   Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
