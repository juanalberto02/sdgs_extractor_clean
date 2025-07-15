import fitz  # PyMuPDF
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)

def extract_title_improved(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    text_elements = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text, size, bbox = span["text"].strip(), span["size"], span["bbox"]
                    x0, y0, x1, y1 = bbox
                    if y0 < page.rect.height * 0.33 and x0 > 30:
                        if (
                            text and len(text.split()) > 3 and
                            not any(word in text for word in ["arXiv", "doi", "[cs", "[pdf"])
                        ):
                            text_elements.append((text, size, y0))
    if not text_elements:
        return "Title not found"
    max_size = max(size for _, size, _ in text_elements)
    candidates = [text for text, size, _ in text_elements if size == max_size]
    return " ".join(candidates).strip() if candidates else "Title not found"

def extract_abstract(text):
    match = re.search(r"(?i)\bAbstract(?:—|:)?\s*", text)
    if not match:
        return "Abstract not found"
    abstract_start = match.end()
    text_after_abstract = text[abstract_start:].strip()
    valid_headings = [
        "INTRODUCTION", "METHODS", "RESULTS", "CONCLUSION",
        "DISCUSSION", "REFERENCES", "ACKNOWLEDGMENT", "MATERIALS AND METHODS",
        "ARTICLE HISTORY", "KEYWORDS", "Key words"
    ]
    match_end = re.search(r"\b(?:Keywords(?:—|:|\s)|Index(?:—|:|\s)|" + "|".join(valid_headings) + r")\b", text_after_abstract)
    abstract = text_after_abstract[:match_end.start()] if match_end else text_after_abstract
    return re.sub(r"\s+", " ", re.sub(r"-\n", "", abstract)).strip()

def extract_keywords(text):
    match = re.search(r"(?i)\b(?:Key\s*words|Index\s*Terms)[\s—:-]*(.*)", text)
    if not match:
        return "Keywords not found"
    keywords_start = match.start()
    text_after_keywords = text[keywords_start:]
    match_paren = re.search(r"\(", text_after_keywords)
    match_heading = re.search(
        r"\n(?:[IVXLCDM]+\.\s+|INTRODUCTION|METHODS|RESULTS|CONCLUSION|REFERENCES|DISCUSSION|ACKNOWLEDGMENT|MATERIALS AND METHODS|ARTICLE HISTORY|CONTACT|Introduction|I)\b",
        text_after_keywords, flags=re.IGNORECASE)
    end_positions = [m.start() for m in [match_paren, match_heading] if m]
    end_position = min(end_positions) if end_positions else None
    keywords = text_after_keywords[:end_position] if end_position else text_after_keywords
    keywords = re.sub(r"\s+", " ", re.sub(r"-\n", "", keywords)).strip()
    return re.sub(r"(?i)\b(?:Key\s*words|Index\s*Terms)[\s—:-]*", "", keywords, count=1).strip()

def compute_similarity(text, rules_df):
    all_texts = [text] + list(rules_df["inc"])
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    rules_df = rules_df.copy()
    rules_df["similarity"] = similarities
    return rules_df.sort_values(by="similarity", ascending=False)

# --- FUNGSI EVALUASI RULE ---
def get_groups(expr):
    stack = []
    groups = []
    for i, c in enumerate(expr):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                start = stack.pop()
                groups.append((start, i))
    return groups

def split_main_and_or(expr):
    expr = expr.strip()
    groups = get_groups(expr)
    protected = []
    for s, e in groups:
        protected.extend(list(range(s, e+1)))
    result = []
    curr = ''
    i = 0
    while i < len(expr):
        if i in protected:
            curr += expr[i]
            i += 1
        elif expr[i:i+3] == 'AND':
            result.append(curr.strip())
            result.append('AND')
            curr = ''
            i += 3
        elif expr[i:i+2] == 'OR':
            result.append(curr.strip())
            result.append('OR')
            curr = ''
            i += 2
        else:
            curr += expr[i]
            i += 1
    if curr.strip():
        result.append(curr.strip())
    return result

def extract_func_kw(expr):
    m = re.match(r'([A-Z\- ]+)\(["\'](.+?)["\']\)', expr.strip())
    if m:
        return m.group(1).replace(" ", ""), m.group(2)
    return None, None

def check_func_kw(func, kw, title, abstract, keywords):
    kw = kw.lower()
    if func == "AUTHKEY":
        return kw in keywords.lower()
    elif func == "TITLE":
        return kw in title.lower()
    elif func == "TITLE-ABS":
        return (kw in title.lower()) or (kw in abstract.lower())
    else:
        return any(kw in x.lower() for x in [title, abstract, keywords])

def eval_expr(expr, title, abstract, keywords):
    expr = expr.strip()
    if expr.startswith('(') and expr.endswith(')'):
        return eval_expr(expr[1:-1], title, abstract, keywords)
    parts = split_main_and_or(expr)
    if 'AND' in parts:
        idxs = [i for i, v in enumerate(parts) if v == 'AND']
        subparts = []
        last = 0
        for idx in idxs:
            subparts.append(parts[last:idx])
            last = idx+1
        subparts.append(parts[last:])
        return all(eval_expr(' '.join(p), title, abstract, keywords) for p in subparts)
    if 'OR' in parts:
        idxs = [i for i, v in enumerate(parts) if v == 'OR']
        subparts = []
        last = 0
        for idx in idxs:
            subparts.append(parts[last:idx])
            last = idx+1
        subparts.append(parts[last:])
        return any(eval_expr(' '.join(p), title, abstract, keywords) for p in subparts)
    func, kw = extract_func_kw(expr)
    if func:
        return check_func_kw(func, kw, title, abstract, keywords)
    return False

def missing_keywords(expr, title, abstract, keywords):
    expr = expr.strip()
    miss = []
    if expr.startswith('(') and expr.endswith(')'):
        return missing_keywords(expr[1:-1], title, abstract, keywords)
    parts = split_main_and_or(expr)
    if 'AND' in parts:
        idxs = [i for i, v in enumerate(parts) if v == 'AND']
        subparts = []
        last = 0
        for idx in idxs:
            subparts.append(parts[last:idx])
            last = idx+1
        subparts.append(parts[last:])
        for p in subparts:
            miss += missing_keywords(' '.join(p), title, abstract, keywords)
        return miss
    if 'OR' in parts:
        idxs = [i for i, v in enumerate(parts) if v == 'OR']
        subparts = []
        last = 0
        for idx in idxs:
            subparts.append(parts[last:idx])
            last = idx+1
        subparts.append(parts[last:])
        if any(eval_expr(' '.join(p), title, abstract, keywords) for p in subparts):
            return []
        for p in subparts:
            miss += missing_keywords(' '.join(p), title, abstract, keywords)
        return miss
    func, kw = extract_func_kw(expr)
    if func:
        if not check_func_kw(func, kw, title, abstract, keywords):
            miss.append((func, kw))
    return miss

def check_required_keywords(rule_expr, title, abstract, keywords):
    FUNC_LABEL_MAP = {
        "AUTHKEY": "Keywords",
        "TITLE": "Title",
        "TITLE-ABS": "Title and Abstract"
    }
    miss = missing_keywords(rule_expr, title, abstract, keywords)
    if not miss:
        return "All required words present"
    else:
        return f"{', '.join([f'{FUNC_LABEL_MAP.get(f, f)}: {k}' for f, k in miss])}"

    
def find_unnecessary_words(exc_raw, title, abstract, keywords):
    matches = re.findall(r'"(.*?)"', exc_raw or "")
    unnecessary = []
    for word in matches:
        word_lower = word.lower()
        if word_lower in (title or "").lower() or word_lower in (abstract or "").lower() or word_lower in (keywords or "").lower():
            unnecessary.append(word)
    return ", ".join(unnecessary) if unnecessary else "No unnecessary words found"


# === FUNGSI UTAMA UNTUK FASTAPI ===
def detect_from_pdf_with_rules(pdf_path, rules_df):
    full_text = extract_text_from_pdf(pdf_path)
    title = extract_title_improved(pdf_path)
    abstract = extract_abstract(full_text)
    keywords = extract_keywords(full_text)
    combined_text = f"{title}. {abstract}. {keywords}"
    similar_results = compute_similarity(combined_text, rules_df)
    similar_results['required_words'] = similar_results['inc_raw'].apply(
        lambda x: check_required_keywords(x, title, abstract, keywords)
    )
    # Get only the highest similarity per SDG, among the top 3 SDG with the highest max similarity
    top_sdgs = (
        similar_results.groupby("sdg")["similarity"]
        .max()
        .reset_index()
        .sort_values(by="similarity", ascending=False)
        .head(3)
    )
    top_rules_per_sdgs = []

    for sdg in top_sdgs["sdg"]:
        best_rule = (
            similar_results[similar_results["sdg"] == sdg]
            .sort_values("similarity", ascending=False)
            .iloc[0]
        )
        best_rule_dict = best_rule[["no", "inc_raw", "exc_raw", "sdg", "similarity", "required_words"]].to_dict()
        # Tambahkan pengecekan exclusion
        best_rule_dict["unnecessary_words"] = find_unnecessary_words(
            best_rule_dict["exc_raw"], title, abstract, keywords
        )
        top_rules_per_sdgs.append(best_rule_dict)

        print("  - SDG:", best_rule_dict["sdg"], "Similarity:", best_rule_dict["similarity"])

    return {
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "top_rules": top_rules_per_sdgs  # <--- Selalu list dgn maksimal 3 item, 1 per SDG
    }
