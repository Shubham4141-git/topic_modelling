# Content Topics, Coverage Gaps and a Research Assistant

## Summary

A research and advisory firm writes for business leaders about technology, AI, cybersecurity, cloud, data and digital transformation. Their content team wants to know three things:

- what the market is writing about
- where their own library is thin
- how to search their past work quickly, with sources they can trust

I built a working solution for all three, in three notebooks:

1. **Topic discovery:** finds the main tech topics in external news.
2. **Coverage and gaps:** checks how much of our own content covers each topic, and ranks the gaps.
3. **Research assistant:** answers questions from our library with citations, and says "Not found" when it can't back up an answer. I tested it on 10 questions, checking each answer by hand.

| | Result |
|---|---|
| Topics found | 32 tech topics, from 31,007 external tech news items |
| Biggest gaps | Intel/AMD processors, Oracle, Windows Server, Windows XP security |
| Most over-covered | Cybersecurity: 6% of external news, but 23% of our library |
| Assistant | Right article in the top 5 for 7 of 7 questions, 86% of citations correct, 0 hallucinations, 2 of 2 out-of-scope questions declined |

The rest of this document goes step by step: the assumptions first, then each step of the build (what I did, why, and how I checked it), then the evaluation, the findings, my recommendations for the business, and what I'd improve next.

---

## 1. Assumptions

The case study gives three public datasets but doesn't say which one is "internal". Everything else depends on this choice, so I made it first:

- **External content = AG News** (Business and Sci/Tech rows). It collects short news items from over 2,000 sources, so it's a good stand-in for what the market writes about.
- **Internal content = CNN/DailyMail.** It has full-length articles with short summaries. That's the closest thing to a firm's own library, and the assistant needs full text to quote from.
- **GDELT is not used.** Its free API only gives headlines from the last 3 months, with no article text. It also comes from a completely different time period from the other two, so comparing it with our library wouldn't be fair.

Other assumptions:

- **"External demand" means a topic's share of external tech news.** The brief doesn't define demand, so I used the simplest measure I could explain and defend.
- **"Relevant" means technology in the broad sense,** following the firm's focus areas. Oil prices, stock markets and sport are out.
- **The data is historical.** AG News is from about 2004–05 and CNN/DailyMail from 2007–15, and neither has dates. So the topics belong to that era and I can't measure what's "emerging" over time. The method itself works the same on current data.
- **The assistant searches a fixed sample of 15,000 internal tech articles,** sized so the whole pipeline runs on a laptop in well under an hour.

---

## 2. Approach and methodology

### Step 1: Clean and explore the data

**What I did**
- **AG News:** kept Business and Sci/Tech rows, cleaned the text and removed 404 duplicates. That left 63,396 rows.
- **CNN/DailyMail:** removed 5,058 duplicate summaries, leaving 306,913 articles.
- **Text lengths:**

  | Text | Median length |
  |---|---|
  | AG News item | 36 words |
  | CNN/DailyMail summary | 49 words |
  | CNN/DailyMail full article | 624 words |

**Why it matters:** the lengths decided how I use the internal data. I match topics using the CNN **summaries**, because they're about the same length as AG items, so the comparison is like for like. I keep the **full articles** for the assistant, because that's where the detail is.

**What I checked:** the row counts looked fine, but when I read actual rows I found leftover HTML (tags like `<b>` and `<A HREF>`, broken codes like `amp;`) in about 6% of AG items. It was leaking junk words like "href" and "fullquote" into the topic keywords. I fixed the cleaning, re-scanned the whole dataset to confirm nothing was left, and rebuilt everything after it.

### Step 2: Keep only tech content, on both sides

**What I did:** I wrote a short description for each of four groups: tech, business, science and general news. Each news item is compared with the four descriptions using sentence embeddings (`all-MiniLM-L6-v2`) and goes to the closest one. Only "tech" items are kept. That left 31,007 external items and 50,873 internal articles.

**Why:** the firm only writes about technology, and AG News's own labels are too broad. For example, "Business" mixes Oracle's takeover bid with oil prices. Using the same filter on both datasets keeps the comparison fair. I chose this method because there's no labelled "is this tech?" data to train a classifier on, and a zero-shot model like `bart-large-mnli` would take hours on a laptop.

**What I checked:** in the first version, 7 of 39 topics were clearly not tech (earnings, layoffs, fraud). I traced the cause to the wording of my tech description, which mentioned "earnings", and rewrote the descriptions. After that, only 2 of 32 topics were non-tech. I flagged those two and left them out of the recommendations.

### Step 3: Discover topics (Part 1)

**What I did:** I ran BERTopic on the 31,007 tech items.
1. Each item becomes an embedding.
2. UMAP reduces the dimensions.
3. HDBSCAN groups similar items into clusters.
4. The most typical words of each cluster become its keywords.

About 35% of items didn't fall clearly into a cluster; I assigned them to their closest topic so every item counts. GPT-4o-mini then gave each topic a readable name, based on its keywords and three example items.

**Why BERTopic:** it covers the methods the brief lists (embeddings, clustering and keyword extraction) in one tool, and it works well on short news text.

**Result:** 32 topics, for example mobile and wireless, cybersecurity, Apple and iTunes, search engines, and Intel/AMD processors. Two leftover non-tech topics (Stock Market Movements and Ancient Fossil Discoveries) are kept in the data but excluded from recommendations.

### Step 4: Measure coverage and rank the gaps (Parts 2 and 3)

**What I did:**
- **Same topics on both sides.** Running BERTopic separately on our library would produce different topics, and they couldn't be compared. Instead, each of the 32 topics gets a centre point: the average embedding of its items. Every tech item, external or internal, goes to the closest topic if it's similar enough (a similarity score of 0.45 or more). Otherwise it stays unmatched.
- **Per topic:**
  - its share of external items and its share of internal items
  - coverage ratio = internal share ÷ external share: **Low** under 0.5, **Medium** 0.5–1.5, **High** over 1.5
  - gap = external share − internal share
  - Topics are ranked by gap, biggest first.

**Why shares and not counts:** the two sides are very different sizes (22,547 external vs 4,562 internal matched items). Comparing raw counts would just say "external is bigger" for every topic.

**What I checked:** instead of guessing the threshold, I ran three values end to end and read sample matches for each:

| Threshold | External matched | Internal matched | What I saw |
|---|---|---|---|
| 0.40 | 84% | 15% | 3 of 10 sample matches were wrong |
| **0.45 (chosen)** | **72.7%** | **9.0%** | Matches looked right, and enough internal articles per topic |
| 0.50 | 58% | 4.7% | Clean, but too few internal articles to trust the numbers |

### Step 5: Build the research assistant (Part 4)

**What I did:**
- **Ingestion:** took 15,000 internal tech articles and split them into chunks of 150 words, overlapping by 30 words. That gave 86,334 chunks, stored as embeddings in a FAISS index (exact search).
- **Retrieval:** for each question, take the 20 closest chunks, keep only the best chunk per article, and pass the top 5 articles on.
- **Answer:** GPT-4o-mini (temperature 0) writes the answer **only** from those 5 sources and cites them as [1], [2]. Each number links to an article id and chunk id. If the sources don't contain the answer, it replies "Not found in the library."
- **Gap questions:** a question like "where are our biggest gaps?" goes to a second tool, which reads the gap table from Step 4. The model decides which tool fits each question.

**Why these choices:**
- **Chunk size:** the embedding model only reads about 256 tokens, so a whole article would be cut off. 150 words fits, and the overlap stops a fact from being split between two chunks.
- **One chunk per article:** stops one long article from filling all 5 slots.
- **Separate gap tool:** search finds text, but it can't rank or count. A small table answers those questions exactly.

**What I checked:**
- **Consistency with Part 3.** The gap tool's first version calculated its own numbers from raw counts, which made cybersecurity look like a gap, while Step 4 shows it's the most over-covered topic. I caught this by comparing the assistant's answer with the gap table. The tool now reads Step 4's table directly, so they can never disagree.
- **Unanswerable questions.** On a question the library couldn't answer, the model kept retrying searches until the test run crashed. I added a clear rule to the prompt and a safe fallback, so it now always ends with "Not found in the library."

---

## 3. Evaluation framework (Part 5)

I wrote 10 test questions, each after reading its source article:
- **7 fact questions,** each with the article that contains the answer
- **1 gap question,** checked against the Step 4 ranking
- **2 questions the library can't answer** (NASA's 2024 Artemis crew, OpenAI's 2025 funding), to see whether the assistant makes things up

| What I measured | How | Result |
|---|---|---|
| Retrieval | Is the correct article in the top 5? (checked automatically) | **7 of 7** |
| Citation accuracy | Does each cited article really say what the answer claims? (checked by hand) | **86%** (6 of 7) |
| Hallucination | Does the answer say anything the sources don't support, or answer when it should decline? | **0 of 10** |
| Declining | Do out-of-scope questions get "Not found in the library"? | **2 of 2** |

- **Why I labelled by hand:** with 10 questions I could read every answer against its source, which I trust more than asking another model to grade.
- **The one citation miss** was a correct answer that left out its [n] marker.
- **The gap question** matched the Step 4 ranking exactly.
- **Wording varies between runs:** LLM wording changes slightly from run to run, so I re-checked every label against the final run's answers before reporting these numbers.

All 10 questions, with answers, citations and labels, are in `outputs/rag_demo.md`.

---

## 4. Key findings

1. **What the market writes about most:** mobile and wireless, cybersecurity, Apple and digital music, and search engines.
2. **Biggest gaps: enterprise software and hardware.**
   - The top 5 are Intel/AMD processors, the Oracle takeover, Oracle enterprise software, Windows Server, and Windows XP security.
   - Next come IBM's PC business, Linux and open source, HP servers, and Sun/Solaris.
   - These matter to a business-leader audience, yet our library barely covers them. This is the clearest content opportunity.
3. **Covered about right:** Apple iTunes, search engines and mobile (ratio between 1.07 and 1.18).
4. **Over-covered:**
   - Cybersecurity, at 3.8 times the external share. Part of this is because the topic is broad and also catches surveillance and crime stories.
   - CEO appointments, gaming consoles and media-centre PCs are also above demand.
   - Some effort here could move to the gap topics.
5. **How to read the ranking:** because the external news is older than our library, some gaps (like Windows XP) are partly a timing effect. The ranking is best used as a prioritised shortlist for the content team to review. Run on current data, the same pipeline gives a current shortlist.

---

## 5. Business recommendations

**Top content opportunities**
1. **An enterprise software series** covering Oracle, Windows Server, and Linux and open source. This is the largest under-covered area. For example, the Oracle takeover is 3.9% of market news and almost 0% of our library.
2. **Chips and hardware:** Intel and AMD processors (5.1% of the market vs 0.6% of our library), plus servers and storage (HP, IBM, Sun).
3. **Rebalance cybersecurity.** It's 23% of our library but only 6% of the market. Keep the quality, and move some of that effort to the gaps above.

**Proposed research investments**
- **A current, dated news feed** (GDELT or a news API). It removes the timing effect and lets the team track emerging topics month by month. This is the biggest single upgrade.
- **Connect the firm's real report library,** with titles and dates, so coverage is measured on our actual work.
- **A stronger tech filter and embedding model,** so topics and gaps get sharper.

**Suggested enhancements to the AI assistant**
- **Better search:** combine keyword and vector search, and add a reranker, so it finds the right article more often.
- **Coverage questions:** a new "how much do we have on X?" table, so the team can check coverage for any topic.
- **Citations:**
  - Re-ask automatically when an answer comes back without citations.
  - Show the exact supporting sentence for each source.
- **A bigger test set,** built with the content team, to measure quality on the questions they really ask.

The technical details behind these are in section 6.

---

## 6. Recommended improvements

**Data**
- Add a current, dated news source (GDELT or a news API), so the gaps reflect today's demand and emerging topics can be tracked month by month.
- Replace the stand-in library with the firm's real reports, including titles, dates and authors.

**Tech filter (Step 2)**
- Use a zero-shot classifier (`bart-large-mnli`), or label a few hundred items and train a small classifier.
- Drop items that don't clearly belong to any group, instead of forcing them into the closest one. This would stop sport and celebrity stories from reaching the internal "tech" pool.
- Measure the filter's accuracy on a labelled sample and report it as a number.

**Embeddings (all steps)**
- Try a stronger embedding model (for example from the BGE or E5 families). It should separate close topics better, and some read longer text, so less chunking would be needed.

**Topic discovery (Step 3)**
- Tune BERTopic (cluster size, merging similar topics) to get a tighter set of about 15–25 topics.
- Add news filler words like "said" and "new" to the stop-word list.
- Keep poorly fitting items as "other" instead of forcing them into a topic.
- Group topics into a hierarchy, for example "Enterprise software" → "Oracle", "Windows Server".
- Have an editor review the topic names.

**Coverage and gaps (Step 4)**
- Use a separate threshold for each topic, so broad topics like cybersecurity don't pull in nearby stories.
- Allow an article to belong to more than one topic.
- Compare each article with a topic's individual items, not only its average, for sharper matching.
- Flag topics with very few internal articles, so nobody over-reads their ratios.
- With dated data, weight the gap by how fast a topic is growing.

**Research assistant (Step 5)**
- **Retrieval:**
  - Combine keyword search (BM25) with vector search, so exact names and product codes are found reliably.
  - Add a reranker, a model that re-scores the top results before they go to the LLM.
  - Index all 50k tech articles once the filter is cleaner.
- **New questions it could answer:**
  - Add a coverage table so it can answer "how much do we have on X?" for any topic.
  - Add filters by topic and date.
- **Answers:**
  - Automatically re-ask the model when an answer has no citation.
  - Highlight the exact supporting sentence in each cited source.

**Evaluation**
- Build a larger, harder test set together with the content team, including reworded and multi-article questions.
- Run each question several times to measure how much answers vary.
- Automatically check that gap answers match the gap table.
- Once there are enough hand labels to check it against, use an LLM judge to scale up grading.
- Track cost and response time per question.

---

## How to run

1. Use Python 3.11 and run `pip install -r requirements.txt`.
2. Add `OPENAI_API_KEY=...` to a `.env` file in the project folder.
3. Run the notebooks in order: `01_eda_topics` → `02_coverage_gap` → `03_rag_eval`.
   - The first run takes about 30–40 minutes on a MacBook Air M1, mostly creating embeddings.
   - After that, saved results in `cache/` and `data/` are reused. Delete a file to rebuild it.
4. Optional: `streamlit run app.py` opens a small page for asking the assistant questions.

| File | What's in it |
|---|---|
| `outputs/topics.csv` | The 32 topics: name, keywords, size |
| `outputs/gap_ranking.csv`, `gap_chart.png` | Coverage and gap for every topic |
| `outputs/rag_demo.md` | All 10 test questions with answers, citations and labels |
| `eval/` | The test questions and my labels |
