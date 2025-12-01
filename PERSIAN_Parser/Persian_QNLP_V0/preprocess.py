# -*- coding: utf-8 -*-
"""
preprocess.py  (Normalization utilities - Persian)
این فایل شامل توابع پایه‌ای نرمال‌سازی و برچسب‌گذاری متن فارسی است که برای
پردازش بعدی (توکنیزاسیون، تشخیص افعال مرکب، نگاشت به پری‌گروپ و ...) مفید خواهند بود.
قابلیت‌ها:
- یک‌دست‌سازی حروف (آلف‌ها، ی/ك، کشیدهٔ تطویل)
- حذفِ «تَشکیل» (حركات) و دیگر نشانگرهای ترکیبی
- convert Persian/Arabic digits to ASCII
- یک‌دست‌سازی فاصله‌ها و حذف فضای زائد
- تبدیلِ علائم‌نگارِ لاتین به معادل فارسی (اختیاری)
- توکنیزه ساده
- شناسایی و ترکیب افعال مرکب
- برچسب‌گذاری نقش‌های نحوی (فاعل، مفعول، فعل)

تابع اصلی: 
    normalize_text(text)
    tokenize_text(text)
    merge_compound_verbs(tokens)
    label_roles(tokens)
    process_sentence(text)
    load_labeled_dataset(path)
"""
import re
import unicodedata
# ------------------------ تنظیمات یونی‌کد ------------------------
ARABIC_YEHS = ['\u064A', '\u0649']   # Arabic Yeh, Alef Maksura
PERSIAN_YEH = '\u06CC'               # Persian Yeh (ی)
ARABIC_KAF = '\u0643'
PERSIAN_KAF = '\u06A9'               # Persian Kaf (ک)
ALEF_VARIANTS = {'\u0622':'\u0627', '\u0623':'\u0627', '\u0625':'\u0627'}
TATWEEL = '\u0640'                   # کِشیده (ـ)
# محدودهٔ اعراب/تَشکیل
DIACRITICS_RE = re.compile(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]+')
# نگاشت ارقام (Persian & Arabic-Indic -> ASCII)
PERSIAN_DIGITS = '۰۱۲۳۴۵۶۷۸۹'
ARABIC_INDIC_DIGITS = '٠١٢٣٤٥٦٧٨٩'
ASCII_DIGITS = '0123456789'
TO_ASCII_DIGITS = {ord(p): ord(a) for p,a in zip(PERSIAN_DIGITS, ASCII_DIGITS)}
TO_ASCII_DIGITS.update({ord(a): ord(b) for a,b in zip(ARABIC_INDIC_DIGITS, ASCII_DIGITS)})

# نگاشت علائم (اختیاری) - از ascii -> punctuation فارسی
PUNCT_TRANSL = str.maketrans({
    ',': '،',
    '?': '؟',
    ';': '؛',
})
# ------------------------ لیست صفت‌ها ------------------------
COMMON_ADJECTIVES = {
    "ماهر", "خوشمزه","مناسب", "زیبا", "جدید", "قدیمی", "بزرگ", "کوچک"
}

def is_adjective(token: str) -> bool:
    return token in COMMON_ADJECTIVES

# ------------------------ لیست کلمات چندبخشی ------------------------
DEFAULT_MULTIWORDS = [
    ["نرم", "افزار"],
    ["هوش", "مصنوعی"],
    ["یادگیری", "ماشین"],
    ["اشکال","زدایی"]
]
# ------------------------ توابع کمکی ------------------------
def _remove_diacritics(text: str) -> str:
    return DIACRITICS_RE.sub('', text)

def _replace_arabic_chars(text: str) -> str:
    for k,v in ALEF_VARIANTS.items():
        text = text.replace(k, v)
    text = text.replace(ARABIC_KAF, PERSIAN_KAF)
    for ay in ARABIC_YEHS:
        text = text.replace(ay, PERSIAN_YEH)
    text = text.replace(TATWEEL, '')
    return text
def _normalize_spaces(text: str) -> str:
    text = text.replace('\u00A0', ' ').replace('\u200B', ' ')
    punctuation = '،؟!;:\.،\,\?'
    text = re.sub(r"\s+([{}])".format(punctuation), r"\1", text)
    text = re.sub(r'([\.,؛:!؟\)\]\}])(?=\S)', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------ تابع اصلی ------------------------
def normalize_text(text: str,
                   remove_diacritics: bool = True,
                   unify_chars: bool = True,
                   digits_to_ascii: bool = True,
                   convert_punctuation: bool = False,
                   preserve_zwnj: bool = True) -> str:
    if text is None:
        return text
    text = unicodedata.normalize('NFC', text)
    if unify_chars:
        text = _replace_arabic_chars(text)
    if remove_diacritics:
        text = _remove_diacritics(text)
    if digits_to_ascii:
        text = text.translate(TO_ASCII_DIGITS)
    if convert_punctuation:
        text = text.translate(PUNCT_TRANSL)
    if not preserve_zwnj:
        text = text.replace('\u200C', ' ')
    text = _normalize_spaces(text)
    return text

#======================== Tokenization ============================
def tokenize_text(text: str):
    return re.findall(r"[\w‌]+", text)
#======================== کلمات چندبخشی ===========================
def merge_multiwords(tokens, multiwords_list=None):
    if multiwords_list is None:
        multiwords_list = DEFAULT_MULTIWORDS
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
            L = len(mw)
            if i + L <= len(tokens) and tokens[i:i+L] == mw:
                merged.append("_".join(mw))
                skip = L - 1
                matched = True
                break
        if not matched:
            merged.append(tokens[i])
        i += 1
    return merged
#======================== افعال مرکب ==============================
def merge_compound_verbs(tokens, light_verbs=None):
    """Merge tokens into compound verbs using a list of light verbs."""
    if light_verbs is None:
        light_verbs = ['کرد', 'داد', 'گرفت', 'زد', 'شد', 'است', 'بود', 'خواهد', 'می‌کند', 'می‌رود']
    merged_tokens = []
    skip_next = False
    for i in range(len(tokens)):
        if skip_next:
            skip_next = False
            continue
        if i < len(tokens) - 1 and tokens[i+1] in light_verbs:
            merged_tokens.append(tokens[i] + '_' + tokens[i+1])
            skip_next = True
        else:
            merged_tokens.append(tokens[i])
    return merged_tokens
#======================== برچسب‌گذاری نقش‌ها ========================
def label_roles(tokens, light_verbs=None, multiwords_list=None):
    """
    Assign syntactic roles (subj, obj, verb, adj, other) to tokens.
    Improved handling for noun + adjective ... را  (e.g. 'غذا خوشمزه را').
    """
    # 1. merge multiwords & compound verbs (assumes these functions exist)
    merged_tokens = merge_multiwords(tokens, multiwords_list) if 'merge_multiwords' in globals() else tokens[:]
    merged_tokens = merge_compound_verbs(merged_tokens, light_verbs=light_verbs)

    n = len(merged_tokens)
    if n == 0:
        return []

    # 2. init roles
    roles = ['other'] * n
    roles[0] = 'subj'
    if n > 1:
        roles[-1] = 'verb'

    # 3. find indices of 'را' and mark object spans
    ra_indices = [i for i, t in enumerate(merged_tokens) if t == 'را']
    for r in ra_indices:
        if r == 0:
            continue
        # scan left to find nearest non-adjective token (stop at subj index 0)
        j = r - 1
        # skip trailing adjectives (e.g. 'غذا' 'خوشمزه' => will skip 'خوشمزه' then find 'غذا')
        while j > 0 and is_adjective(merged_tokens[j]):
            j -= 1
        # head candidate
        head = j if j > 0 else (r - 1)  # fallback to token before 'را'
        # mark span from head .. r-1 as the object phrase
        for k in range(head, r):
            if k == head:
                roles[k] = 'obj'
            else:
                # tokens between head and 'را' normally are post-nominal adjectives
                if is_adjective(merged_tokens[k]):
                    roles[k] = 'adj'
                else:
                    # if not adjective (rare), keep as 'other' or mark as obj-part
                    roles[k] = 'other'

        # leave roles[r] as 'other' (the particle 'را')

    # 4. mark remaining adjectives as 'adj' (subject modifiers etc.)
    for i in range(1, n-1):
        if roles[i] == 'other' and is_adjective(merged_tokens[i]):
            roles[i] = 'adj'

    # 5. produce result list of tuples
    result = [(merged_tokens[i], roles[i]) for i in range(n)]
    return result

#======================== پردازش جمله ==============================
def process_sentence(text: str):
    """Full pipeline: normalize + tokenize + role labeling."""
    norm = normalize_text(text)
    tokens = tokenize_text(norm)
    labeled = label_roles(tokens)
    return labeled
#======================== لود دیتاست ===============================
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