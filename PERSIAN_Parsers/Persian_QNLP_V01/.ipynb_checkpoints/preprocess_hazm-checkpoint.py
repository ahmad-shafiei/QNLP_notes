# preprocess_hazm.py
# -*- coding: utf-8 -*-
"""
preprocess_hazm.py
Hazm-based preprocessing with:
 - multiword merging (e.g. 'نرم افزار' -> 'نرم_افزار')
 - POS tagging via Hazm
 - merging light verbs into compound verbs (e.g. 'اجرا' + 'کرد' -> 'اجرا_کرد' as VERB)
 - simple role labeling (subj/obj/adj/verb/ra/other)
"""

import os
from hazm import Normalizer, POSTagger, word_tokenize

# ------------------------
# مدل‌ها و ابزارها
# ------------------------
MODEL_DIR = "resources"
POS_MODEL = os.path.join(MODEL_DIR, "pos_tagger.model")

if not os.path.exists(POS_MODEL):
    raise FileNotFoundError(f"POS model not found at {POS_MODEL}. Put pos_tagger.model in resources/")

normalizer = Normalizer()
tagger = POSTagger(model=POS_MODEL)

def process_sentence(text: str):
    """Full pipeline: normalize + tokenize + role labeling."""
    norm = normalizer.normalize(text)   # 👈 استفاده از نرمالایزر هضم
    tokens = word_tokenize(norm)        # توکنایزر هضم
    labeled = label_roles(tokens)
    return labeled
# ------------------------
# multiword expressions (configurable)
# ------------------------
MULTIWORDS = [
    "نرم افزار",
    "اشکال زدایی",
    "یادگیری ماشین",
    "هوش مصنوعی",
    "شبکه عصبی",
]

def merge_multiwords(tokens, multiwords_list=None):
    """
    Merge multiword expressions in tokens.
    Keep spaces intact, do not convert to underscores.
    """
    if multiwords_list is None:
        multiwords_list = MULTIWORDS

    merged = []
    skip = 0
    i = 0
    while i < len(tokens):
        if skip > 0:
            skip -= 1
            i += 1
            continue
        matched = False
        for mw in multiwords_list:
            L = len(mw.split())
            if i + L <= len(tokens) and tokens[i:i+L] == mw.split():
                merged.append(" ".join(mw.split()))  # keep spaces
                skip = L - 1
                matched = True
                break
        if not matched:
            merged.append(tokens[i])
        i += 1
    return merged

import unicodedata

def strip_punct_tokens(tokens):
    """Remove tokens that are pure punctuation marks."""
    clean = []
    for tok in tokens:
        # حذف اگر همه‌ی کاراکترهای توکن جزو Punctuation باشند
        if all(unicodedata.category(ch).startswith('P') for ch in tok):
            continue
        clean.append(tok)
    return clean
# ------------------------
# light verbs (افعال مرکب)
# ------------------------
LIGHT_VERBS = ['کرد', 'داد', 'گرفت', 'زد', 'شد', 'است', 'بود', 'خواهد', 'می‌کند', 'می‌رود']

def merge_light_verbs(tagged_tokens, light_verbs=None):
    """
    Keep compound verbs as two tokens for Lambeq wiring,
    but combine them into one string for display.
    Returns list of (token, pos)
    """
    if light_verbs is None:
        light_verbs = LIGHT_VERBS

    merged = []
    i = 0
    n = len(tagged_tokens)
    while i < n:
        w, pos = tagged_tokens[i]
        if i + 1 < n:
            nw, npos = tagged_tokens[i+1]
            if nw in light_verbs:
                # combine for display only
                # new_word_display = f"{w} {nw}"
                # append separately for Lambeq wiring
                new_word = f"{w} {nw}"
                # merged.append((w, pos))
                merged.append((new_word, 'VERB'))
                i += 2
                continue
        merged.append((w, pos))
        i += 1
    return merged

# ------------------------
# نقش‌دهی ساده بر اساس POS و particle 'را'
# ------------------------
def label_roles_from_tagged(tagged_tokens):
    """
    tagged_tokens: list of (word, pos) - pos are Hazm tags like 'NOUN,EZ' or 'VERB' etc.
    Returns: list of (word, role) where role in {'subj','obj','adj','verb','ra','other'}
    Rules:
      - mark 'را' tokens as 'ra'
      - mark tokens with pos starting with V as 'verb'
      - mark tokens with pos starting with ADJ as 'adj'
      - find 'را' and mark the noun(s) before it as obj (skip post-nominal adjectives)
      - first remaining noun -> subj
      - remaining nouns default to 'other' (or 'subj/obj' if you prefer)
    """
    words = [w for w, _ in tagged_tokens]
    pos_tags = [p for _, p in tagged_tokens]
    n = len(tagged_tokens)
    roles = ['other'] * n

    # initial marking
    for i, (w, p) in enumerate(tagged_tokens):
        up = (p or "").upper()
        if w == 'را' or up.startswith('ADP'):
            roles[i] = 'ra'
        elif up.startswith('V'):
            roles[i] = 'verb'
        elif up.startswith('ADJ'):
            roles[i] = 'adj'
        else:
            roles[i] = 'other'

    # handle 'را' -> mark object span before 'را'
    ra_indices = [i for i, w in enumerate(words) if w == 'را']
    for r in ra_indices:
        if r == 0:
            continue
        # scan left to find head (skip trailing adjectives)
        j = r - 1
        # skip adjectives to find head
        while j > 0 and ((tagged_tokens[j][1] or "").upper().startswith('ADJ')):
            j -= 1
        head = j
        # mark head..r-1
        for k in range(head, r):
            if k == head:
                roles[k] = 'obj'
            else:
                # if tag is adjective mark as adj, else mark as part-of-obj (keep 'other' or 'obj-part')
                if (tagged_tokens[k][1] or "").upper().startswith('ADJ'):
                    roles[k] = 'adj'
                else:
                    roles[k] = 'other'

    # mark first noun (not already obj) as subj
    for i, (w, p) in enumerate(tagged_tokens):
        up = (p or "").upper()
        if roles[i] == 'other' and up.startswith('N'):
            roles[i] = 'subj'
            break

    # finalize list
    result = [(tagged_tokens[i][0], roles[i]) for i in range(n)]
    return result

# ------------------------
# تابع اصلی پردازش جمله
# ------------------------
def process_sentence(text: str):
    """
    Full pipeline:
      - normalize
      - tokenize (Hazm)
      - merge multiwords (config list)
      - POS tag (Hazm)
      - merge light verbs (compound verbs)
      - role labeling (ra rule + first-noun-as-subj)
    Returns:
      list of (token, role)   # سازگار با خروجی قدیمی preprocess.py
    """
    norm = normalizer.normalize(text)
    tokens = word_tokenize(norm)
    tokens = strip_punct_tokens(tokens)    # 👈 وظیفه حذف علائم و نقطه  در جمله
    tokens = merge_multiwords(tokens)              # e.g. 'نرم افزار' -> 'نرم_افزار'
    tagged = tagger.tag(tokens)                    # list of (word, pos)

    # merge light verbs into compound verbs (affects tagged list)
    merged_tagged = merge_light_verbs(tagged)

    # label roles based on merged tags
    roles = label_roles_from_tagged(merged_tagged)

    # خروجی مشابه نسخه قدیمی: [(token, role), ...]
    labeled = [(w, r) for (w, _), (_, r) in zip(merged_tagged, roles)]
    return labeled

def read_data(filename, sep=" "):
    """Read dataset with labels and sentences (binary labels as [t, 1-t])."""
    labels, sentences = [], []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # جدا کردن لیبل از جمله
            parts = line.split(sep, 1)
            if len(parts) < 2:
                continue
            label_str, sentence = parts
            t = int(label_str)
            # مشابه نسخه انگلیسی: برچسب باینری دو بعدی
            labels.append([t, 1 - t])
            sentences.append(sentence.strip())
    return labels, sentences
# ------------------------
# تست سریع
# ------------------------
if __name__ == "__main__":
    tests = [
        "مرد غذای خوشمزه را پخت",
        "شخص ماهر غذا را پخت",
        "مرد نرم افزار را اجرا کرد",
        "او تصمیم گرفت برنامه را اجرا کرد"
    ]
    for sent in tests:
        res = process_sentence(sent)
        print("===\nجمله:", sent)
        print("Tokens:", res["tokens"])
        print("POS Tags:", res["pos_tags"])
        print("Roles:", res["roles"])



# # -*- coding: utf-8 -*-
# """
# preprocess_hazm.py
# نسخه ارتقایافته با استفاده از Hazm + Universal Dependency Parser (در صورت موجود بودن)
# """

# import os
# from hazm import Normalizer, word_tokenize, POSTagger, DependencyParser

# try:
#     from hazm import DependencyParser
#     HAS_DEP_PARSER = True
# except ImportError:
#     HAS_DEP_PARSER = False

# # ------------------------ پیکربندی مدل‌ها ------------------------
# MODEL_DIR = "resources"
# POS_TAGGER_MODEL = os.path.join(MODEL_DIR, "pos_tagger.model")
# DEP_PARSER_MODEL = os.path.join(MODEL_DIR, "universal_dependency_parser")  # باید unzip شده باشه

# # ------------------------ ابزارهای Hazm ------------------------
# normalizer = Normalizer()
# pos_tagger = POSTagger(model=POS_TAGGER_MODEL)

# dep_parser = None
# if HAS_DEP_PARSER and os.path.exists(DEP_PARSER_MODEL):
#     try:
#         dep_parser = DependencyParser(model=DEP_PARSER_MODEL)
#     except Exception as e:
#         print(f"[WARN] DependencyParser load failed: {e}")
#         dep_parser = None

# # ------------------------ multiwords ------------------------
# DEFAULT_MULTIWORDS = [
#     ["نرم", "افزار"],
#     ["هوش", "مصنوعی"],
#     ["یادگیری", "ماشین"],
#     ["اشکال", "زدایی"],
# ]

# def merge_multiwords(tokens, multiwords_list=None):
#     if multiwords_list is None:
#         multiwords_list = DEFAULT_MULTIWORDS
#     merged, skip, i = [], 0, 0
#     while i < len(tokens):
#         if skip > 0:
#             skip -= 1
#             i += 1
#             continue
#         matched = False
#         for mw in multiwords_list:
#             L = len(mw)
#             if i + L <= len(tokens) and tokens[i:i+L] == mw:
#                 merged.append("_".join(mw))
#                 skip = L - 1
#                 matched = True
#                 break
#         if not matched:
#             merged.append(tokens[i])
#         i += 1
#     return merged

# # ------------------------ نقش‌دهی بر اساس POS ------------------------
# def label_roles_with_pos(tokens):
#     tagged = pos_tagger.tag(tokens)
#     roles = []
#     for i, (tok, pos) in enumerate(tagged):
#         role = "other"
#         if pos.startswith("N"):
#             role = "subj" if i == 0 else "obj"
#         elif pos.startswith("V"):
#             role = "verb"
#         elif pos.startswith("ADJ"):
#             role = "adj"
#         roles.append((tok, role))
#     return roles

# # ------------------------ نقش‌دهی بر اساس DependencyParser ------------------------
# def label_roles_with_dep(tokens):
#     """استفاده از وابستگی نحوی برای نقش‌دهی دقیق‌تر"""
#     tagged = pos_tagger.tag(tokens)  # نیاز parser به POS
#     tree = dep_parser.parse(tagged)
#     roles = []
#     for node in tree.nodes.values():
#         if node["word"] is None:
#             continue
#         rel = node["rel"]  # نوع وابستگی مثل nsubj, obj, amod
#         role = "other"
#         if rel == "nsubj":
#             role = "subj"
#         elif rel in ["obj", "dobj"]:
#             role = "obj"
#         elif rel.startswith("amod"):
#             role = "adj"
#         elif rel.startswith("root") or rel.startswith("cop") or rel.startswith("aux"):
#             role = "verb"
#         roles.append((node["word"], role))
#     return roles

# # ------------------------ پایپلاین اصلی ------------------------
# def process_sentence(text: str):
#     """پایپلاین: نرمال‌سازی + توکنیزیشن + multiword merge + نقش‌دهی"""
#     norm = normalizer.normalize(text)
#     tokens = word_tokenize(norm)
#     tokens = merge_multiwords(tokens)

#     if dep_parser is not None:
#         return label_roles_with_dep(tokens)
#     else:
#         return label_roles_with_pos(tokens)

# # ------------------------ تست ------------------------
# if __name__ == "__main__":
#     sent = "مرد نرم افزار را اجرا کرد"
#     result = process_sentence(sent)
#     print("Input:", sent)
#     print("Output:", result)




# # -*- coding: utf-8 -*-
# """
# preprocess_hazm.py  (Normalization & POS tagging using Hazm)
# این فایل مشابه preprocess.py عمل می‌کند اما از قابلیت‌های Hazm استفاده می‌کند:
# - نرمال‌سازی متن
# - توکنیزه کردن
# - POS Tagging و Dependency parsing
# - حفظ قابلیت‌های merge کلمات چندبخشی و افعال مرکب
# - نقش‌گذاری پیش‌گروه (subj, obj, verb, adj)
# """

# from hazm import Normalizer, word_tokenize, POSTagger, DependencyParser
# import re

# # ------------------------ Hazm objects ------------------------
# normalizer = Normalizer()
# pos_tagger = POSTagger(model='resources/pos_tagger.model')  # مسیر مدل Hazm
# # dep_parser = DependencyParser(model='resources/parser.model')  # مسیر مدل Hazm

# # ------------------------ لیست صفت‌ها ------------------------
# COMMON_ADJECTIVES = {
#     "ماهر", "خوشمزه","مناسب", "زیبا", "جدید", "قدیمی", "بزرگ", "کوچک"
# }

# def is_adjective(token: str) -> bool:
#     return token in COMMON_ADJECTIVES

# # ------------------------ لیست کلمات چندبخشی ------------------------
# DEFAULT_MULTIWORDS = [
#     ["نرم", "افزار"],
#     ["هوش", "مصنوعی"],
#     ["یادگیری", "ماشین"],
#     ["اشکال","زدایی"]
# ]

# # ------------------------ توابع دستی برای multiwords و compound verbs ------------------------
# def merge_multiwords(tokens, multiwords_list=None):
#     if multiwords_list is None:
#         multiwords_list = DEFAULT_MULTIWORDS
#     merged = []
#     skip = 0
#     i = 0
#     while i < len(tokens):
#         if skip > 0:
#             skip -= 1
#             i += 1
#             continue
#         matched = False
#         for mw in multiwords_list:
#             L = len(mw)
#             if i + L <= len(tokens) and tokens[i:i+L] == mw:
#                 merged.append(" ".join(mw))
#                 skip = L - 1
#                 matched = True
#                 break
#         if not matched:
#             merged.append(tokens[i])
#         i += 1
#     return merged

# def merge_compound_verbs(tokens, light_verbs=None):
#     if light_verbs is None:
#         light_verbs = ['کرد', 'داد', 'گرفت', 'زد', 'شد', 'است', 'بود', 'خواهد', 'می‌کند', 'می‌رود']
#     merged_tokens = []
#     skip_next = False
#     for i in range(len(tokens)):
#         if skip_next:
#             skip_next = False
#             continue
#         if i < len(tokens) - 1 and tokens[i+1] in light_verbs:
#             merged_tokens.append(tokens[i] + ' ' + tokens[i+1])
#             skip_next = True
#         else:
#             merged_tokens.append(tokens[i])
#     return merged_tokens

# # ------------------------ نرمال‌سازی ------------------------
# def normalize_text(text: str) -> str:
#     return normalizer.normalize(text)

# # ------------------------ توکنیزه کردن ------------------------
# def tokenize_text(text: str):
#     return word_tokenize(text)

# # ------------------------ برچسب‌گذاری نقش‌ها با Hazm ------------------------
# def label_roles(tokens, multiwords_list=None, light_verbs=None):
#     """
#     Assign syntactic roles using Hazm POS Tagger + Dependency Parser
#     Returns: list of tuples (token, role)
#     """
#     # 1) merge multiwords & compound verbs
#     merged_tokens = merge_multiwords(tokens, multiwords_list)
#     merged_tokens = merge_compound_verbs(merged_tokens, light_verbs=light_verbs)

#     # 2) Hazm POS tagging
#     tagged = pos_tagger.tag(merged_tokens)  # [(token, POS), ...]

#     # 3) نقش‌ها
#     roles = ['other'] * len(merged_tokens)
#     n = len(merged_tokens)
#     if n == 0:
#         return []

#     # فاعل و فعل اولیه
#     roles[0] = 'subj'
#     roles[-1] = 'verb'

#     # 4) شناسایی مفعول با particle "را"
#     if 'را' in merged_tokens:
#         ra_idx = merged_tokens.index('را')
#         # scan left for head of object phrase
#         head_idx = ra_idx - 1
#         while head_idx > 0 and is_adjective(merged_tokens[head_idx]):
#             head_idx -= 1
#         for k in range(head_idx, ra_idx):
#             if k == head_idx:
#                 roles[k] = 'obj'
#             else:
#                 roles[k] = 'adj'
#         # "را" خودش به عنوان other باقی می‌ماند

#     # 5) برچسب‌گذاری سایر صفت‌ها
#     for i, (tok, pos) in enumerate(tagged):
#         if roles[i] == 'other' and pos in ['ADJ', 'JJ']:
#             roles[i] = 'adj'

#     # 6) نتیجه نهایی
#     return [(merged_tokens[i], roles[i]) for i in range(n)]

# # ------------------------ پردازش کامل جمله ------------------------
# def process_sentence(text: str):
#     """Full pipeline: normalize + tokenize + role labeling."""
#     norm = normalize_text(text)
#     tokens = tokenize_text(norm)
#     labeled = label_roles(tokens)
#     return labeled

# # ------------------------ لود دیتاست ------------------------
# def read_data(filename, sep=" "):
#     """Read dataset with labels and sentences (binary labels as [t, 1-t])."""
#     labels, sentences = [], []
#     with open(filename, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.split(sep, 1)
#             if len(parts) < 2:
#                 continue
#             label_str, sentence = parts
#             t = int(label_str)
#             labels.append([t, 1 - t])
#             sentences.append(sentence.strip())
#     return labels, sentences
