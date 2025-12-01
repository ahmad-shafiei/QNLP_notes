# persian_cat_parser_v5.py
# -*- coding: utf-8 -*-
import arabic_reshaper
from bidi.algorithm import get_display
from lambeq import AtomicType
from lambeq.backend.grammar import Cup, Word, Diagram, Id, Swap, Ty
# از preprocess استفاده می‌کنیم تا نقش‌ها را دقیق بگیریم
from preprocess import process_sentence
N = AtomicType.NOUN
S = AtomicType.SENTENCE
O = Ty('o')  # برای "را"
def fix_persian(text: str) -> str:
    """reshape + bidi برای کل رشته (فقط یکبار روی رشته کامل اجرا شود)."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)
class PersianCatParser:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    def sentence2diagram(self, sentence: str) -> Diagram:
        """Parse a Persian sentence (using preprocess.process_sentence) into a lambeq Diagram."""
        # 1) از preprocess نقش‌ها را بگیر
        labeled = process_sentence(sentence)  # [(token, role), ...]
        if len(labeled) < 2:
            raise ValueError(f"Too short sentence for parsing: {labeled}")
        # آیا 'را' در جمله هست؟
        has_ra = any(tok == 'را' or role == 'ra' for tok, role in labeled)
        # 2) ساخت لیست Wordها با ادغام spans (وقتی لازم است)
        words = []
        subj = obj = ra = verb = None
        i = 0
        n = len(labeled)
        while i < n:
            tok, role = labeled[i]
            # ---- فاعل: اگر نقش subj است، همه‌ی صفات بعدی را بچسبان
            if role == 'subj':
                parts = [tok]
                j = i + 1
                while j < n and labeled[j][1] == 'adj':
                    parts.append(labeled[j][0])
                    j += 1
                combined = " ".join(parts)
                w = Word(fix_persian(combined), N)
                subj = w
                words.append(w)
                i = j
                continue
            # ---- اگر توکن 'را' تشخیص داده شد
            if tok == 'را' or role == 'ra':
                w = Word(fix_persian(tok), N.r @ O)
                ra = w
                words.append(w)
                i += 1
                continue
            # ---- مفعول: اگر این موقعیت شروع مفعول است (obj) یا
            #      اگر یک توکن other قبل از 'را' است (مثال: 'غذا' ممکن است other باشد)
            if role == 'obj' or (role == 'other' and any(labeled[k][0]=='را' or labeled[k][1]=='ra' for k in range(i+1, n))):
                # اگر 'را' بعدی وجود دارد، ادغام کن تا 'را' (exclusive)
                # در غیر این صورت، ادغام کن شامل صفات بعدی
                # پیدا کردن اندیس 'را' بعدی (اگر هست)
                ra_idx = None
                for k in range(i+1, n):
                    if labeled[k][0] == 'را' or labeled[k][1] == 'ra':
                        ra_idx = k
                        break
                parts = []
                j = i
                if ra_idx is not None:
                    # ادغام از i تا ra_idx-1
                    while j < ra_idx:
                        parts.append(labeled[j][0])
                        j += 1
                    # i را به ra_idx می‌بریم تا در حلقه بعدی 'را' پردازش شود
                    i = ra_idx
                else:
                    # هیچ 'را'ای وجود ندارد: فقط ادغام اسم + صفات بعدی
                    while j < n and labeled[j][1] in ('adj', 'other', 'obj'):
                        parts.append(labeled[j][0])
                        j += 1
                    i = j
                combined = " ".join(parts)
                w = Word(fix_persian(combined), N)
                obj = w
                words.append(w)
                continue
            # ---- فعل
            if role == 'verb':
                if has_ra:
                    w = Word(fix_persian(tok), O.r @ N.r @ S)
                else:
                    w = Word(fix_persian(tok), N.r @ N.l @ S)
                verb = w
                words.append(w)
                i += 1
                continue
            # ---- اگر توکن صفت (adj) مانده (نادر) یا other باقی مانده، به عنوان N اضافه کن
            if role in ('adj', 'other'):
                w = Word(fix_persian(tok), N)
                words.append(w)
                i += 1
                continue
            # fallback
            i += 1
        if subj is None or verb is None:
            raise ValueError(f"Sentence must have at least subj and verb. Labeled: {labeled}")

        # 3) ساخت دیاگرام از words
        diagram = words[0]
        for w in words[1:]:
            diagram = diagram @ w
        # 4) کاهش‌ها — دو حالت: با 'را' یا بدون 'را'
        if has_ra and obj is not None and ra is not None:
            # انتظار ترتیب: subj (N) ⊗ obj (N) ⊗ ra (N.r @ O) ⊗ verb (O.r @ N.r @ S)
            diagram = diagram >> (Id(N) @ Cup(N, N.r) @ Id(O @ O.r @ N.r @ S))
            diagram = diagram >> (Id(N) @ Cup(O, O.r) @ Id(N.r @ S))
            diagram = diagram >> (Cup(N, N.r) @ Id(S))
        else:
            # fallback با Swap (برای ساختارهای قدیمی)
            if obj is not None:
                diagram = diagram >> (Id(N) @ Cup(N, N.r) @ Id(N.l @ S))
                diagram = diagram >> (Swap(N, N.l) @ Id(S))
                diagram = diagram >> (Cup(N.l, N) @ Id(S))
            else:
                diagram = diagram >> (Cup(N, N.l) @ Id(S))
        return diagram
    def sentences2diagrams(self, sentences):
        return [self.sentence2diagram(s) for s in sentences]