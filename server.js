import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import natural from "natural";
import fs from "fs";

const app = express();
app.use(cors());
app.use(bodyParser.json());

const TfIdf = natural.TfIdf;
const tfidf = new TfIdf();
let faqs = [];

// --- Track consecutive failed attempts ---
let failCount = 0;
const FAIL_LIMIT = 3;

// --- Setup WordNet for synonyms ---
const wordnet = new natural.WordNet();
const stemmer = natural.PorterStemmer;

// --- Load FAQ from text file ---
function loadFaqs() {
  const raw = fs.readFileSync("faq.txt", "utf-8");
  const text = raw.replace(/\r\n/g, "\n");

  // Regex to parse Q/A blocks
  const regex = /Q:\s*([^\n]+)\nA:\s*([\s\S]*?)(?=\nQ:|$)/g;
  faqs = [];

  let match;
  while ((match = regex.exec(text)) !== null) {
    const q = match[1].trim();
    const a = match[2].trim();
    faqs.push({ q, a });
  }

  // Reset TF-IDF index
  tfidf.documents = [];
  faqs.forEach((faq) => {
    tfidf.addDocument(preprocess(faq.q)); // doc = preprocessed question
  });

  console.log(`📚 Loaded ${faqs.length} FAQs`);
}

// --- Preprocess text ---
function preprocess(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "") // remove punctuation
    .split(/\s+/)
    .filter((word) => !natural.stopwords.includes(word))
    .map((word) => stemmer.stem(word)) // apply stemming
    .join(" ");
}

// --- Expand query with synonyms ---
async function expandQuery(query) {
  const words = query.split(" ");
  const expanded = new Set(words);

  for (const word of words) {
    await new Promise((resolve) => {
      wordnet.lookup(word, (results) => {
        results.forEach((result) => {
          result.synonyms.forEach((syn) => {
            if (syn.indexOf(" ") === -1) {
              expanded.add(stemmer.stem(syn.toLowerCase()));
            }
          });
        });
        resolve();
      });
    }).catch(() => {});
  }

  return Array.from(expanded).join(" ");
}

// --- Find best answers (top-k) ---
async function findAnswers(userQuestion, k = 3) {
  const processedQuestion = preprocess(userQuestion);
  const expanded = await expandQuery(processedQuestion);

  let scores = [];
  tfidf.tfidfs(expanded, (i, measure) => {
    scores.push({ index: i, score: measure });
  });

  // Sort high → low
  scores.sort((a, b) => b.score - a.score);
  const best = scores.slice(0, k);

  if (best.length === 0 || best[0].score < 0.15) {
    return { answer: null, suggestions: faqs.slice(0, k).map((f) => f.q) };
  }

  return {
    answer: faqs[best[0].index].a,
    suggestions: best.map((s) => faqs[s.index]?.q).filter(Boolean),
  };
}

// --- API endpoint ---
app.post("/ask", async (req, res) => {
  const { question } = req.body;
  const result = await findAnswers(question);

  if (result.answer) {
    // ✅ Found an answer → reset fail count
    failCount = 0;
    return res.json(result);
  } else {
    // ❌ No exact match → increment fail count
    failCount++;

    if (failCount >= FAIL_LIMIT) {
      // 🚨 Special fallback after 3 fails
      failCount = 0; // reset so it doesn’t repeat forever
      return res.json({
        answer:
          "🙇 Sorry, I wasn’t able to answer your question after multiple tries. Please contact our support team for further help.",
        suggestions: [],
      });
    }

    // Regular suggestion response
    return res.json(result);
  }
});

// --- Reload FAQs without restart ---
app.get("/reload", (req, res) => {
  loadFaqs();
  res.json({ status: "✅ FAQ reloaded" });
});

// --- Start server ---
app.listen(5000, () =>
  console.log("🚀 Backend running at http://localhost:5000")
);
loadFaqs();
