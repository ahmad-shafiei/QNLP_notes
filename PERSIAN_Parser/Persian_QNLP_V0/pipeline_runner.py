# pipeline_runner.py
# ----------------------------
# This file converts sentences into lambeq diagrams
# using our simple parser defined in preprocess.py.
from lambeq import AtomicType
from lambeq.backend.grammar import Cup, Word, Diagram, Ty, Id, Swap
from preprocess import process_sentence
from lambeq import RemoveCupsRewriter 
from persian_cat_parser import fix_persian
N = AtomicType.NOUN
S = AtomicType.SENTENCE
def simple_pregroup_parser(labeled_tokens):
    """
    Build a pregroup diagram for simple Persian SOV sentences with adjectives.
    Input:
        labeled_tokens : list of (token, role)
            - token (string)
            - role (subj/obj/verb/adj)

    Output:
        diagram : lambeq.Diagram
    """
    words = []
    subj = obj = verb = None
    for tok, role in labeled_tokens:
        display_tok = fix_persian(tok)
        if role == 'subj':
            w = Word(display_tok, N)
            subj = w
            words.append(w)
        elif role == 'obj':
            w = Word(display_tok, N)
            obj = w
            words.append(w)
        elif role == 'verb':
            w = Word(display_tok, N.r @ N.l @ S)
            verb = w
            words.append(w)
        elif role == 'adj':
            # صفت به عنوان N برای ترکیب با اسم
            w = Word(display_tok, N)
            words.append(w)
        else:
            # other token (e.g., 'را') را در دیاگرام نادیده می‌گیریم
            continue

    if subj is None or verb is None:
        raise ValueError("Sentence must contain at least subj and verb")

    # ساخت دیاگرام
    diagram = words[0]
    for w in words[1:]:
        diagram = diagram @ w

    # کاهش‌ها
    # اگر مفعول داریم
    if obj is not None:
        # find indices
        subj_idx = words.index(subj)
        obj_idx = words.index(obj)
        verb_idx = words.index(verb)

        diagram = diagram >> (Id(N) @ Cup(N, N.r) @ Id(N.l @ S))
        diagram = diagram >> (Swap(N, N.l) @ Id(S))
        diagram = diagram >> (Cup(N.l, N) @ Id(S))
    else:
        # جمله بدون مفعول
        diagram = diagram >> (Cup(N, N.l) @ Id(S))
    return diagram


def dataset_to_diagrams_with_labels(dataset, verbose=True):
    """
    Convert an entire dataset of sentences to diagrams.

    Input:
        dataset : list of (sentence, label)
        verbose : bool
            If True, print errors and progress

    Output:
        diagrams_with_labels : list of (diagram, label)
        errors : list of (index, sentence, label, error_message)
    """
    diagrams_with_labels = []
    errors = []
    for i, (sent, label) in enumerate(dataset):
        try:
            d = sentence_to_diagram(sent)
            diagrams_with_labels.append((d, label))
        except Exception as e:
            errors.append((i, sent, label, str(e)))
            if verbose:
                print(f"[ERROR] idx={i} sent={sent} label={label} -> {e}")
    if verbose:
        print(f"Converted {len(diagrams_with_labels)} / {len(dataset)} sentences. Failures: {len(errors)}")
    return diagrams_with_labels, errors

