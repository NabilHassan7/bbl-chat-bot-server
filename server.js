// server.js
import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import natural from "natural";
import fs from "fs";
import Fuse from "fuse.js";

const app = express();
app.use(cors());
app.use(bodyParser.json());

/* -------------------- NLP SETUP -------------------- */

const TfIdf = natural.TfIdf;
const tfidf = new TfIdf();
const stemmer = natural.PorterStemmer;
const wordnet = new natural.WordNet();

let faqs = [];
let fuse = null;

/* -------------------- FAIL TRACKING -------------------- */

let failCount = 0;
const FAIL_LIMIT = 3;

/* -------------------- LOAD FAQ FILE -------------------- */

function loadFaqs() {
  const raw = fs.readFileSync("faq.txt", "utf-8");
  const text = raw.replace(/\r\n/g, "\n");

  const regex = /Q:\s*([^\n]+)\nA:\s*([\s\S]*?)(?=\nQ:|$)/g;
  faqs = [];
  tfidf.documents = [];

  let match;
  while ((match = regex.exec(text)) !== null) {
    const q = match[1].trim();
    const a = match[2].trim();
    faqs.push({ q, a });
    tfidf.addDocument(preprocess(q));
  }

  fuse = new Fuse(faqs, {
    keys: ["q"],
    includeScore: true,
    threshold: 0.45,
    ignoreLocation: true,
    minMatchCharLength: 2,
  });

  console.log(`📚 Loaded ${faqs.length} FAQs`);
}

/* -------------------- TEXT PREPROCESSING -------------------- */

function preprocess(text) {
  if (!text) return "";

  const normalized = text.normalize("NFC").toLowerCase();
  const cleaned = normalized.replace(/[^a-z\u0980-\u09FF0-9\s]/g, " ");
  const tokens = cleaned.split(/\s+/).filter(Boolean);

  const processed = tokens
    .map((tok) => {
      if (/[a-z]/.test(tok)) {
        if (natural.stopwords.includes(tok)) return "";
        return stemmer.stem(tok);
      }
      return tok;
    })
    .filter(Boolean);

  return processed.join(" ");
}

/* -------------------- WORDNET QUERY EXPANSION -------------------- */

async function expandQuery(query) {
  if (!query) return "";

  const words = query
    .toLowerCase()
    .replace(/[^a-z\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  const expanded = new Set(words);

  for (const word of words) {
    await new Promise((resolve) => {
      wordnet.lookup(word, (results) => {
        results.forEach((r) =>
          r.synonyms.forEach((s) => {
            if (!s.includes(" ")) expanded.add(s.toLowerCase());
          }),
        );
        resolve();
      });
    }).catch(() => {});
  }

  return Array.from(expanded).join(" ");
}

/* -------------------- CORE MATCHING LOGIC -------------------- */

async function findAnswers(userQuestion, k = 3) {
  if (!userQuestion || !userQuestion.trim()) {
    return { answer: null, suggestions: [] };
  }

  const processed = preprocess(userQuestion);
  const expansion = preprocess(await expandQuery(userQuestion));
  const query = `${processed} ${expansion}`.trim();

  let scores = [];
  tfidf.tfidfs(query, (i, score) => {
    scores.push({ index: i, score });
  });

  scores.sort((a, b) => b.score - a.score);
  const best = scores.slice(0, k);

  /* ---------- CONFIDENCE THRESHOLDS ---------- */

  const TFIDF_STRONG = 0.3;
  const TFIDF_WEAK = 0.2;
  const TFIDF_SCORE_GAP = 0.08;

  if (best.length > 0) {
    const top = best[0];
    const second = best[1];
    const gap = second ? top.score - second.score : top.score;

    if (top.score >= TFIDF_STRONG && gap >= TFIDF_SCORE_GAP) {
      return {
        answer: faqs[top.index].a,
        suggestions: [],
      };
    }

    if (top.score >= TFIDF_WEAK) {
      return {
        answer: faqs[top.index].a,
        suggestions: best.map((b) => faqs[b.index].q),
      };
    }
  }

  /* ---------- FUZZY FALLBACK ---------- */

  const fuseResults = fuse.search(userQuestion, { limit: k });
  const suggestions = fuseResults.map((r) => r.item.q);

  const FUSE_ACCEPT_THRESHOLD = 0.3;

  if (fuseResults.length > 0 && fuseResults[0].score <= FUSE_ACCEPT_THRESHOLD) {
    if (
      fuseResults.length > 1 &&
      Math.abs(fuseResults[0].score - fuseResults[1].score) < 0.05
    ) {
      return { answer: null, suggestions };
    }

    return {
      answer: fuseResults[0].item.a,
      suggestions,
    };
  }

  return { answer: null, suggestions };
}

/* -------------------- API ENDPOINT -------------------- */

app.post("/ask", async (req, res) => {
  const { question } = req.body;

  try {
    const result = await findAnswers(question);

    if (result.answer) {
      failCount = 0;
      return res.json(result);
    }

    failCount++;
    if (failCount >= FAIL_LIMIT) {
      failCount = 0;
      return res.json({
        answer:
          "Sorry, I wasn't able to answer your question after multiple tries. Call 16221 (toll free) to reach our support team for further detailed help.",
        suggestions: [],
      });
    }

    res.json(result);
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ answer: null, suggestions: [] });
  }
});

/* -------------------- RELOAD ENDPOINT -------------------- */

app.get("/reload", (req, res) => {
  try {
    loadFaqs();
    res.json({ status: "FAQ reloaded" });
  } catch {
    res.status(500).json({ status: "reload failed" });
  }
});

/* -------------------- START SERVER -------------------- */

app.listen(5000, () =>
  console.log("🚀 Backend running at http://localhost:5000"),
);

loadFaqs();
