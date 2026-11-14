# Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Vector store | NumPy matrix + cosine similarity | Corpus is small (hundreds of chunks); avoids FAISS/Chroma install issues on free hosting |
| Embeddings | all-MiniLM-L6-v2, local | Free, fast on CPU, no extra API key |
| LLM | Groq `openai/gpt-oss-120b` | Free tier, fast, strong instruction following. Model ids change over time: edit the sidebar field if Groq retires it |
| Chunk size | 900 chars, 150 overlap | Keeps a fact and its context together without exceeding the prompt budget |
| Memory | Last 6 messages + question rewriting | Retrieval needs a standalone query; "and last year?" would match nothing on its own |
| "Not found" handling | Low relevance threshold (0.05) + strict system prompt | A hard threshold alone rejected valid vague questions ("what is the name in the document?"), so the LLM makes the final call |
| First-chunk inclusion | Add each matched document's first chunk | Titles, names and dates usually sit at the top of a document and score low on generic questions |
| Reasoning model setting | `reasoning_effort="low"`, `max_tokens=2500` | Reasoning tokens count toward the limit; a small limit can produce empty answers |

## Request flow (`RAGEngine.answer`)
1. No documents indexed -> ask the user to upload.
2. `condense()` rewrites follow-ups using the last 6 messages (falls back to the raw question on error).
3. Embed the standalone question, take the top-k chunks, drop those below the threshold.
4. No chunks -> "not found". Otherwise send labelled context + history to the LLM.
5. If the LLM replies with the not-found sentence, no sources are shown.

## Known limitations
See section 10 of the README.
