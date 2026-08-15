# 🛠️ Step-by-Step Guide: Getting CRAG-Ops Running

This walks through everything from zero — no prior assumption that you've set
up Python projects, API keys, or Pinecone before. Follow it top to bottom in
order; each step has a way to check it actually worked before moving on.

Total time: ~20-30 minutes (mostly waiting on signups and the one-time ingest).

---

## Step 0: Prerequisites

You need:
- **Python 3.10+** installed. Check with:
  ```bash
  python3 --version
  ```
  If that fails or shows something older than 3.10, install Python from
  https://www.python.org/downloads/ first.
- A terminal (macOS/Linux: Terminal; Windows: PowerShell or the VS Code
  integrated terminal).
- A code editor (VS Code is fine) — optional but helpful for editing `.env`.

---

## Step 1: Get the project files onto your machine

1. Create a folder for the project and put the files I generated
   (`config.py`, `ingest.py`, `agent.py`, `app.py`, `requirements.txt`,
   `.env.example`, `README.md`, `.gitignore`) into it — or unzip the
   `crag-ops-agentic-rag.zip` I gave you directly into a folder.
2. Open a terminal and `cd` into that folder:
   ```bash
   cd path/to/crag-ops
   ls
   ```
   You should see all the files listed above.
3. Create an empty `data/` folder inside it (for your 10-K PDFs):
   ```bash
   mkdir -p data
   ```

**Checkpoint:** running `ls` shows `app.py`, `agent.py`, `ingest.py`,
`config.py`, `requirements.txt` in the current folder.

---

## Step 2: Create and activate a virtual environment

This keeps this project's Python packages isolated from everything else on
your machine.

```bash
python3 -m venv venv
```

Activate it:
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate.ps1
  ```
- **Windows (cmd.exe):**
  ```cmd
  venv\Scripts\activate.bat
  ```

**Checkpoint:** your terminal prompt now starts with `(venv)`.

> You'll need to re-run the activation command every time you open a new
> terminal to work on this project. If you close and reopen your terminal
> later, just `cd` back into the folder and reactivate.

---

## Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This takes 1-3 minutes.

**Checkpoint:** run this — it should print version numbers with no errors:
```bash
python3 -c "import streamlit, langgraph, langchain_pinecone, langchain_tavily; print('all imports OK')"
```

---

## Step 4: Get your three API keys

You need keys from **OpenAI**, **Pinecone**, and **Tavily**. All three have
free or pay-as-you-go tiers sufficient for this project.

### 4a. OpenAI API key
1. Go to https://platform.openai.com and sign in / sign up.
2. Go to **Settings → API keys** (or https://platform.openai.com/api-keys).
3. Click **Create new secret key**, name it (e.g. `crag-ops`), and copy the
   key immediately — it's shown only once. It starts with `sk-`.
4. Under **Billing**, make sure you have a payment method or available
   credit — `gpt-4o-mini` and `text-embedding-3-small` are both inexpensive,
   but the account needs billing enabled to make API calls at all.

### 4b. Pinecone API key
1. Go to https://www.pinecone.io and sign up for a free account.
2. In the Pinecone console, go to **API Keys**.
3. Copy the **default** API key (starts with `pcsk_`).
4. You do **not** need to manually create an index — `config.py` creates the
   Serverless index automatically the first time the app or `ingest.py` runs.

### 4c. Tavily API key
1. Go to https://tavily.com and sign up (free tier gives you a generous
   number of monthly searches — plenty for testing).
2. From the dashboard, copy your API key (starts with `tvly-`).

**Checkpoint:** you have three strings copied somewhere safe:
`sk-...`, `pcsk_...`, `tvly-...`.

---

## Step 5: Configure your environment file

```bash
cp .env.example .env
```

Open `.env` in your editor and fill in the three keys from Step 4:

```env
OPENAI_API_KEY=sk-your-real-key-here
PINECONE_API_KEY=pcsk_your-real-key-here
PINECONE_INDEX_NAME=crag-ops-10k
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
TAVILY_API_KEY=tvly-your-real-key-here
```

Save the file. Leave `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, and
`PINECONE_REGION` as-is unless you have a reason to change them.

**Checkpoint:** run this to confirm the app can actually read your keys
(it will error out and tell you exactly which one is missing if not):
```bash
python3 -c "import config; config._validate_env(); print('env OK')"
```

---

## Step 6: Add your source documents

Follow the **"Task 1 — Sourcing sample 10-K PDFs from SEC EDGAR"** section of
`README.md` to download 2-3 real 10-K filings (Apple, Microsoft, NVIDIA are
good test picks). Drop the PDFs into the `data/` folder:

```
data/apple_10k_2024.pdf
data/microsoft_10k_2024.pdf
data/nvidia_10k_2024.pdf
```

**Checkpoint:**
```bash
ls data/
```
shows your PDF files.

---

## Step 7: Run the one-time ingestion

This is the step that actually embeds your documents into Pinecone.

```bash
python3 ingest.py
```

Expected output looks like:
```
Starting bulk ingestion from 'data/' ...
✅ INGESTED | apple_10k_2024.pdf                      | Embedded 214 new chunks.
✅ INGESTED | microsoft_10k_2024.pdf                   | Embedded 198 new chunks.
✅ INGESTED | nvidia_10k_2024.pdf                      | Embedded 176 new chunks.
Done.
```

This can take 1-2 minutes depending on filing size — it's chunking each PDF
and calling the OpenAI embeddings API for every chunk.

**Now run it again immediately**, without changing anything:
```bash
python3 ingest.py
```
You should see every file logged as **SKIPPED** this time:
```
⏭️  SKIPPED | apple_10k_2024.pdf                      | Document already exists in Pinecone -- not re-embedded.
```
This confirms the cloud-side deduplication is working — this is the
behavior that also protects you when the Streamlit app itself restarts.

**Optional visual checkpoint:** log into https://app.pinecone.io, open your
`crag-ops-10k` index, and check the **vector count** — it should roughly
match the total chunk count printed above.

---

## Step 8: Launch the app

```bash
streamlit run app.py
```

Your terminal will print a local URL, typically:
```
Local URL: http://localhost:8501
```
It should also open automatically in your browser. If not, open that URL
manually.

**Checkpoint:** you see the "💹 CRAG-Ops" title, a sidebar listing your three
ingested filings under "Currently indexed filings", and a chat box at the
bottom.

---

## Step 9: Test it

Try a question that should be answerable straight from your PDFs, e.g.:
> *What was Apple's total net sales in its most recent fiscal year?*

Watch the status panel above the answer — you should see it walk through
`🔍 Retrieving...` → `🧐 Grading document relevance...` → `🤖 Generating an
answer...`. Expand **"Sources used"** under the answer to confirm it's citing
your actual PDF filename, not a web search.

Now try something your PDFs almost certainly *don't* cover, e.g.:
> *What was NVIDIA's stock price yesterday?*

This time the status panel should show the self-correction path:
`🧐 Grading...` → `✍️ Rewriting the query...` → `🌐 Falling back to a live web
search...` → `🤖 Generating...`. The "Sources used" list should now show
`web_search (Tavily)`.

---

## Step 10: Test dynamic ingestion via the UI

1. In the sidebar, use **"Upload a new 10-K PDF"** to add a filing you didn't
   put in `data/` (e.g. a different company, or a different fiscal year).
2. Click **"Ingest uploaded file(s)"**.
3. You should see a green success toast and the new filename appear in
   "Currently indexed filings" within a few seconds.
4. Immediately ask a question about that new filing — no restart required.

---

## Step 11: Confirm restart-safety (the "zero re-embedding" requirement)

1. Stop the app (`Ctrl+C` in the terminal).
2. Restart it: `streamlit run app.py`.
3. Ask a question again — it should answer instantly using existing Pinecone
   data, with **no re-ingestion happening**. You can further confirm by
   re-running `python3 ingest.py` in a separate terminal — everything should
   log as `SKIPPED`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `EnvironmentError: Missing required environment variable(s)` | `.env` wasn't filled in, or you're running from the wrong folder | Re-check Step 5; make sure `.env` is in the same folder as `app.py` |
| `AuthenticationError` from OpenAI | Bad/expired key, or no billing set up | Regenerate the key; check Billing on platform.openai.com |
| Pinecone `401` / `403` errors | Wrong `PINECONE_API_KEY`, or index region mismatch | Re-copy the key from the Pinecone console; leave `PINECONE_REGION` on a supported Serverless region |
| `ingest.py` runs but 0 chunks embedded | PDF has no extractable text (scanned image, not real text) | Try a different filing, or OCR the PDF first |
| Streamlit shows a blank sidebar file list right after ingest | The 30-second cache (`ttl=30`) hasn't refreshed | Wait a moment or click the file uploader area again; it clears the cache after every upload |
| `ModuleNotFoundError` for any package | Virtual environment not activated, or `pip install -r requirements.txt` wasn't run in it | Re-check Step 2 and Step 3 — your prompt should show `(venv)` |
| App works locally but fails after deploying to Streamlit Cloud | Secrets not set in the cloud dashboard | See the README's "Deploying to Streamlit Community Cloud" section — secrets go in **App settings → Secrets**, not a `.env` file |

---

## Step 12 (optional): Deploy it publicly

Once it's working locally, follow the **"Deploying to Streamlit Community
Cloud"** section in `README.md` — it's a 5-minute process once your GitHub
repo is pushed, and no code changes are needed since everything already reads
from environment variables.
