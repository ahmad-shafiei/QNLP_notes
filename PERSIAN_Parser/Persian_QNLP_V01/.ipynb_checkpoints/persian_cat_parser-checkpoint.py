# persian_cat_parser.py
# -*- coding: utf-8 -*-
import arabic_reshaper
from bidi.algorithm import get_display
from lambeq import AtomicType
from lambeq.backend.grammar import Cup, Word, Diagram, Id, Swap, Ty

from preprocess_hazm import process_sentence

# تایپ‌ها
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
        labeled = process_sentence(sentence)  # [(token, role), ...]
        if len(labeled) < 2:
            raise ValueError(f"Too short sentence for parsing: {labeled}")

        has_ra = any(tok == 'را' or role == 'ra' for tok, role in labeled)

        words = []                # لیست اشیاء Word (برای ساخت دیاگرام)
        word_role_pairs = []      # [(display_text, role), ...] برای بازرسی/پرینت
        subj = obj = ra = verb = None
        i = 0
        n = len(labeled)

        while i < n:
            tok, role = labeled[i]

            # ---- فاعل: ادغام صفات بعدی
            if role == 'subj':
                parts = [tok]
                j = i + 1
                while j < n and labeled[j][1] == 'adj':
                    parts.append(labeled[j][0])
                    j += 1
                combined = " ".join(parts)
                display = fix_persian(combined)
                w = Word(display, N)
                subj = w
                words.append(w)
                word_role_pairs.append((combined, 'subj'))
                i = j
                continue

            # ---- 'را'
            if tok == 'را' or role == 'ra':
                display = fix_persian(tok)
                w = Word(display, N.r @ O)
                ra = w
                words.append(w)
                word_role_pairs.append((tok, 'ra'))
                i += 1
                continue

            # ---- مفعول
            if role == 'obj' or (role == 'other' and any(labeled[k][0]=='را' or labeled[k][1]=='ra' for k in range(i+1, n))):
                ra_idx = None
                for k in range(i+1, n):
                    if labeled[k][0] == 'را' or labeled[k][1] == 'ra':
                        ra_idx = k
                        break

                parts = []
                j = i
                if ra_idx is not None:
                    while j < ra_idx:
                        parts.append(labeled[j][0])
                        j += 1
                    i = ra_idx
                else:
                    while j < n and labeled[j][1] in ('adj', 'other', 'obj'):
                        parts.append(labeled[j][0])
                        j += 1
                    i = j

                combined = " ".join(parts)
                display = fix_persian(combined)
                w = Word(display, N)
                obj = w
                words.append(w)
                word_role_pairs.append((combined, 'obj'))
                continue

            # ---- فعل
            if role == 'verb':
                display = fix_persian(tok)
                if has_ra:
                    w = Word(display, O.r @ N.r @ S)
                else:
                    w = Word(display, N.r @ N.l @ S)
                verb = w
                words.append(w)
                word_role_pairs.append((tok, 'verb'))
                i += 1
                continue

            # ---- صفت یا other -> به عنوان N اضافه کن
            if role in ('adj', 'other'):
                display = fix_persian(tok)
                w = Word(display, N)
                words.append(w)
                word_role_pairs.append((tok, role))
                i += 1
                continue

            i += 1

        if subj is None or verb is None:
            raise ValueError(f"Sentence must have at least subj and verb. Labeled: {labeled}")

        # ---- ساخت دیاگرام
        diagram = words[0]
        for w in words[1:]:
            diagram = diagram @ w

        # ---- کاهش‌ها
        if has_ra and obj is not None and ra is not None:
            # ترتیب: subj ⊗ obj ⊗ ra ⊗ verb
            diagram = diagram >> (Id(N) @ Cup(N, N.r) @ Id(O @ O.r @ N.r @ S))
            diagram = diagram >> (Id(N) @ Cup(O, O.r) @ Id(N.r @ S))
            diagram = diagram >> (Cup(N, N.r) @ Id(S))
        else:
            if obj is not None:
                diagram = diagram >> (Id(N) @ Cup(N, N.r) @ Id(N.l @ S))
                diagram = diagram >> (Swap(N, N.l) @ Id(S))
                diagram = diagram >> (Cup(N.l, N) @ Id(S))
            else:
                diagram = diagram >> (Cup(N, N.l) @ Id(S))

        # ---- attach convenience attributes for inspection/debugging
        # diagrams are lambeq objects, but adding these attrs is harmless and useful
        diagram.words = words                      # list of Word objects (for advanced inspection)
        diagram.word_role_pairs = word_role_pairs  # [(display_text, role), ...]

        return diagram

    def sentences2diagrams(self, sentences):
        return [self.sentence2diagram(s) for s in sentences]
