// server.js
import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import natural from "natural";
import fs from "fs";
import Fuse from "fuse.js"; // <-- fuzzy matching library

const app = express();
app.use(cors());
app.use(bodyParser.json());

const TfIdf = natural.TfIdf;
const tfidf = new TfIdf();
let faqs = [];
let fuse = null; // Fuse index for fuzzy matching

// --- Track consecutive failed attempts ---
let failCount = 0;
const FAIL_LIMIT = 3;

// --- Setup WordNet for synonyms (English only) ---
const wordnet = new natural.WordNet();
const stemmer = natural.PorterStemmer;

// --- Load FAQ from text file ---
function loadFaqs() {
  const raw = fs.readFileSync("faq.txt", "utf-8");
  const text = raw.replace(/\r\n/g, "\n");

  // Regex to parse Q/A blocks: Q: ... A: ...
  const regex = /Q:\s*([^\n]+)\nA:\s*([\s\S]*?)(?=\nQ:|$)/g;
  faqs = [];

  let match;
  while ((match = regex.exec(text)) !== null) {
    const q = match[1].trim();
    const a = match[2].trim();
    faqs.push({ q, a });
  }

  // --- Build TF-IDF index (use preprocessed question text) ---
  tfidf.documents = [];
  faqs.forEach((faq) => {
    tfidf.addDocument(preprocess(faq.q));
  });

  // --- Build Fuse fuzzy index on original questions (keeps original text for better fuzzy) ---
  const fuseOptions = {
    keys: ["q"],
    includeScore: true,
    threshold: 0.45, // 0.0 = exact, 1.0 = very fuzzy. Tune as needed.
    ignoreLocation: true,
    minMatchCharLength: 2,
  };
  fuse = new Fuse(faqs, fuseOptions);

  console.log(`📚 Loaded ${faqs.length} FAQs (TF-IDF + Fuse index rebuilt)`);
}

// --- Preprocess text ---
// Normalize unicode, keep english and bangla characters, tokenize, apply english stemming if appropriate.
function preprocess(text) {
  if (!text) return "";
  const normalized = text.normalize("NFC").toLowerCase();
  // Replace punctuation with spaces, but allow a-z, ০-৯ digits, and Bangla unicode block \u0980-\u09FF
  const cleaned = normalized.replace(/[^a-z\u0980-\u09FF0-9\s]/g, " ");
  const tokens = cleaned.split(/\s+/).filter(Boolean);

  const processed = tokens
    .map((tok) => {
      // If token looks like English, remove English stopwords and stem
      if (/[a-z]/.test(tok)) {
        if (natural.stopwords.includes(tok)) return ""; // drop English stopwords
        try {
          return stemmer.stem(tok);
        } catch {
          return tok;
        }
      }
      // For Bangla (or other scripts) leave token as-is
      return tok;
    })
    .filter(Boolean);

  return processed.join(" ");
}

// --- Expand query with English synonyms (WordNet) ---
// Only attempt expansion for English words (WordNet is English-only).
async function expandQuery(rawQuery) {
  if (!rawQuery) return "";
  const normalized = rawQuery.normalize("NFC").toLowerCase();
  // extract english words (letters a-z)
  const words = normalized
    .replace(/[^a-z\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  const expanded = new Set(words);

  for (const word of words) {
    // Lookup synonyms in WordNet (async wrapper)
    await new Promise((resolve) => {
      wordnet.lookup(word, (results) => {
        results.forEach((result) => {
          result.synonyms.forEach((syn) => {
            if (!syn.includes(" ")) {
              expanded.add(syn.toLowerCase());
            }
          });
        });
        resolve();
      });
    }).catch(() => {
      /* ignore errors from WordNet */
    });
  }

  return Array.from(expanded).join(" ");
}

// --- Find best answers (TF-IDF primary, Fuse fuzzy fallback) ---
async function findAnswers(userQuestion, k = 3) {
  if (!userQuestion || !userQuestion.trim()) {
    return { answer: null, suggestions: [] };
  }

  // Preprocess user question for TF-IDF
  const processedQuestion = preprocess(userQuestion);

  // Expand English query via WordNet and incorporate (raw expansion will be preprocessed before TF-IDF)
  const expansionRaw = await expandQuery(userQuestion);
  const expansionProcessed = preprocess(expansionRaw);

  // Combine processed question + processed expansion (gives TF-IDF more material)
  const queryForTfidf = `${processedQuestion} ${expansionProcessed}`.trim();

  // Compute TF-IDF scores
  let scores = [];
  tfidf.tfidfs(queryForTfidf, (i, measure) => {
    scores.push({ index: i, score: measure });
  });

  // Sort high → low
  scores.sort((a, b) => b.score - a.score);
  const best = scores.slice(0, k);

  const TFIDF_THRESHOLD = 0.15; // tune this if needed

  // If TF-IDF found a confident match, return it
  if (best.length > 0 && best[0].score >= TFIDF_THRESHOLD) {
    return {
      answer: faqs[best[0].index].a,
      suggestions: best.map((s) => faqs[s.index]?.q).filter(Boolean),
    };
  }

  // ---------- TF-IDF low confidence: use Fuse fuzzy matching as fallback ----------

  // If Fuse index isn't built (shouldn't happen), return top-k raw suggestions
  if (!fuse) {
    return { answer: null, suggestions: faqs.slice(0, k).map((f) => f.q) };
  }

  // Normalize user question for Fuse search (keep original script; Fuse is unicode-friendly)
  const fuseResults = fuse.search(userQuestion, { limit: k });

  // Prepare suggestions (top k question texts)
  const suggestions = fuseResults.map((r) => r.item.q).slice(0, k);

  // Decide whether to accept top Fuse result as the answer:
  // Fuse scores are 0 (best) → 1 (worst). Choose an acceptance threshold.
  const FUSE_ACCEPT_THRESHOLD = 0.35; // lower => stricter
  if (fuseResults.length > 0 && fuseResults[0].score <= FUSE_ACCEPT_THRESHOLD) {
    // Good fuzzy match — return its answer and also suggestions list
    return {
      answer: fuseResults[0].item.a,
      suggestions,
    };
  }

  // Otherwise return no exact answer but give suggestions (from Fuse)
  return {
    answer: null,
    suggestions: suggestions.length ? suggestions : faqs.slice(0, k).map((f) => f.q),
  };
}

// --- API endpoint ---
app.post("/ask", async (req, res) => {
  const { question } = req.body;
  try {
    const result = await findAnswers(question);

    if (result.answer) {
      // Found an answer → reset fail count
      failCount = 0;
      return res.json(result);
    } else {
      // No exact match → increment fail count
      failCount++;

      if (failCount >= FAIL_LIMIT) {
        // Special fallback after 3 fails
        failCount = 0; // reset so it doesn’t repeat forever
        return res.json({
          answer:
            "Sorry, I wasn't able to answer your question after multiple tries. Call 16221 (toll free) to reach our support team for further detailed help.",
          suggestions: [],
        });
      }

      // Regular suggestion response
      return res.json(result);
    }
  } catch (err) {
    console.error("Error in /ask:", err);
    return res.status(500).json({ answer: null, suggestions: [] });
  }
});

// --- Reload FAQs without restart ---
app.get("/reload", (req, res) => {
  try {
    loadFaqs();
    res.json({ status: "FAQ reloaded" });
  } catch (err) {
    console.error("Reload error:", err);
    res.status(500).json({ status: "reload failed" });
  }
});

// --- Start server ---
app.listen(5000, () => console.log("🚀 Backend running at http://localhost:5000"));
loadFaqs();
