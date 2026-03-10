"""Spanish syllabification, stress rules, and vos detection.

Rule-based syllable splitting for Spanish text, following standard
Spanish phonological conventions. Used for stress pattern analysis
and Rioplatense vos conjugation detection.
"""

import unicodedata

# Vowel categories
_STRONG = set("aeoáéó")
_WEAK = set("iuü")
_WEAK_ACCENTED = set("íú")
_VOWELS = _STRONG | _WEAK | _WEAK_ACCENTED
_ACCENTED = set("áéíóú")

# Inseparable onset clusters: stop/f + liquid (l/r)
_ONSET_CLUSTERS = frozenset({
    "bl", "br", "cl", "cr", "dr", "fl", "fr",
    "gl", "gr", "pl", "pr", "tr",
})

# Digraphs that are single phonemes (never split)
_DIGRAPHS = frozenset({"ch", "ll", "rr", "qu", "gu"})


def strip_accents(text: str) -> str:
    """Remove diacritical marks from Spanish text, preserving ñ/Ñ.

    á→a, é→e, í→i, ó→o, ú→u, ü→u, Á→A, etc.
    The letter ñ is kept as-is.
    """
    result: list[str] = []
    for ch in text:
        if ch in ("ñ", "Ñ"):
            result.append(ch)
        else:
            # NFD decomposes e.g. 'á' → 'a' + combining acute
            decomposed = unicodedata.normalize("NFD", ch)
            stripped = "".join(
                c for c in decomposed
                if unicodedata.category(c) != "Mn"  # Mn = Mark, Nonspacing
            )
            result.append(stripped)
    return "".join(result)


def _is_vowel(ch: str) -> bool:
    return ch.lower() in _VOWELS


def _forms_diphthong(a: str, b: str) -> bool:
    """Return True if vowels a and b form a diphthong (stay in same syllable).

    Diphthong rules:
      - strong + weak (unaccented)  → diphthong  (ai, au, ei, eu, oi, ou)
      - weak (unaccented) + strong  → diphthong  (ia, ie, io, ua, ue, uo)
      - weak + weak                 → diphthong  (iu, ui, üí, etc.)
      - strong + strong             → hiatus     (ae, ao, ea, eo, oa, oe)
      - accented weak + strong      → hiatus     (ía, úe, etc.)
      - strong + accented weak      → hiatus     (aí, eú, etc.)

    Note: two weak vowels ALWAYS form a diphthong, even if one carries
    an accent mark (e.g., lingüística → güís stays together). The accent
    on a weak vowel only creates hiatus when paired with a strong vowel.
    """
    al = a.lower()
    bl = b.lower()

    a_strong = al in _STRONG
    b_strong = bl in _STRONG
    a_weak = al in _WEAK or al in _WEAK_ACCENTED
    b_weak = bl in _WEAK or bl in _WEAK_ACCENTED

    # Two strong vowels → hiatus
    if a_strong and b_strong:
        return False

    # Two weak vowels → always diphthong (even with accent)
    if a_weak and b_weak:
        return True

    # One strong + one weak: hiatus if the weak one has accent
    if al in _WEAK_ACCENTED or bl in _WEAK_ACCENTED:
        return False

    # strong+weak(unaccented) or weak(unaccented)+strong → diphthong
    return True


def syllabify_spanish(word: str) -> list[str]:
    """Split a Spanish word into syllables using rule-based phonology.

    Returns a list of syllable strings preserving the original characters
    (including accent marks).
    """
    w = word.lower()
    n = len(w)

    if n == 0:
        return []

    # Step 1: tokenize into "phoneme units" (handling digraphs)
    # Each unit is a string of 1 or 2 characters.
    units: list[str] = []
    i = 0
    while i < n:
        if i + 1 < n:
            pair = w[i:i + 2]
            if pair in ("ch", "ll", "rr"):
                units.append(pair)
                i += 2
                continue
            # qu: treat as single onset unit, but we need to check
            # if followed by vowel (que, qui). The 'u' is silent.
            if pair == "qu" and i + 2 < n and _is_vowel(w[i + 2]):
                units.append("qu")
                i += 2
                continue
            # gu before e/i: the 'u' is silent (guerra, guiso)
            # but güe/güi: the 'ü' is pronounced (vergüenza)
            if pair == "gu" and i + 2 < n and w[i + 2] in "ei":
                units.append("gu")
                i += 2
                continue
        units.append(w[i])
        i += 1

    # Step 2: classify each unit as vowel (V) or consonant (C)
    labels: list[str] = []  # 'V' or 'C'
    for u in units:
        # A unit is a vowel if its first char is a vowel
        # (digraphs ch, ll, rr, qu, gu are all consonantal onsets)
        if len(u) == 1 and _is_vowel(u):
            labels.append("V")
        elif len(u) == 2 and u[0] in ("ü",):
            # ü alone as a unit — shouldn't normally happen but just in case
            labels.append("V")
        else:
            labels.append("C")

    # Step 3: group into syllables
    # Strategy: build syllable by consuming units left-to-right.
    # We track boundaries by finding vowel nuclei and distributing
    # consonants according to onset-maximization with cluster rules.

    # First, merge diphthongs: consecutive V units that form diphthongs
    # should be treated as a single nucleus.
    # We do this by creating "nucleus groups": indices of units that
    # form a single vowel nucleus.

    # Find vowel positions
    vowel_positions: list[int] = [i for i, l in enumerate(labels) if l == "V"]

    if not vowel_positions:
        # No vowels — return the whole thing as one syllable
        return ["".join(units)]

    # Group consecutive vowels into nuclei, splitting on hiatus
    nuclei: list[list[int]] = []
    current_nucleus = [vowel_positions[0]]

    for j in range(1, len(vowel_positions)):
        prev_idx = vowel_positions[j - 1]
        curr_idx = vowel_positions[j]

        # They must be adjacent units (no consonant between them)
        if curr_idx == prev_idx + 1:
            prev_vowel = units[prev_idx]
            curr_vowel = units[curr_idx]
            if _forms_diphthong(prev_vowel, curr_vowel):
                # Check triphthong: if we already have 2 vowels in nucleus,
                # only add if all three form a valid group
                if len(current_nucleus) >= 2:
                    # Triphthongs are rare; just add if the last two
                    # still form diphthong
                    current_nucleus.append(curr_idx)
                else:
                    current_nucleus.append(curr_idx)
            else:
                # Hiatus — start new nucleus
                nuclei.append(current_nucleus)
                current_nucleus = [curr_idx]
        else:
            # Consonant(s) between them — definitely separate nuclei
            nuclei.append(current_nucleus)
            current_nucleus = [curr_idx]

    nuclei.append(current_nucleus)

    # Now we have nuclei (each a list of unit indices that form one
    # vowel nucleus). We need to assign consonants to syllables.
    # Strategy: for consonants between two nuclei, determine the split.

    syllables: list[str] = []

    for ni in range(len(nuclei)):
        # Determine the start of this syllable's consonant onset
        if ni == 0:
            # First nucleus: all leading consonants belong to it
            syl_start = 0
        else:
            # Consonants between previous nucleus end and this nucleus start
            prev_nuc_end = nuclei[ni - 1][-1]  # last unit of prev nucleus
            this_nuc_start = nuclei[ni][0]      # first unit of this nucleus

            # Consonant units between them
            cons_start = prev_nuc_end + 1
            cons_end = this_nuc_start  # exclusive

            consonants_between = list(range(cons_start, cons_end))
            num_cons = len(consonants_between)

            if num_cons == 0:
                # Adjacent vowels with hiatus — syllable starts at nucleus
                syl_start = this_nuc_start
            elif num_cons == 1:
                # VCV → V.CV (consonant goes to next syllable)
                syl_start = consonants_between[0]
            else:
                # Multiple consonants: check if last two form an onset cluster
                last_two_units = units[consonants_between[-2]] + units[consonants_between[-1]]
                if last_two_units in _ONSET_CLUSTERS:
                    # Those two go with the next syllable
                    syl_start = consonants_between[-2]
                else:
                    # Only last consonant goes with next syllable
                    syl_start = consonants_between[-1]

        # Determine end of this syllable
        if ni == len(nuclei) - 1:
            # Last nucleus: all trailing consonants belong to it
            syl_end = len(units)
        else:
            # End is determined by next syllable's start (computed next iteration)
            # For now, mark end as nucleus end; we'll trim in the next pass
            syl_end = None  # placeholder

        # Store syl_start for this syllable
        if ni == 0:
            starts = [syl_start]
        else:
            starts.append(syl_start)

    # Now build syllables from consecutive starts
    for si in range(len(starts)):
        s = starts[si]
        e = starts[si + 1] if si + 1 < len(starts) else len(units)
        syl = "".join(units[s:e])
        syllables.append(syl)

    return syllables


def expected_stress_index(word: str) -> int:
    """Return the 0-indexed position of the stressed syllable.

    Rules:
    1. If any syllable contains a tilde (á é í ó ú), that syllable is stressed.
    2. If no tilde and word ends in vowel, n, or s → penultimate (second-to-last).
    3. If no tilde and word ends in other consonant → last syllable.
    4. Monosyllabic words → 0.
    """
    syls = syllabify_spanish(word)

    if len(syls) <= 1:
        return 0

    # Check for explicit tilde
    for i, syl in enumerate(syls):
        for ch in syl:
            if ch in _ACCENTED:
                return i

    # No tilde — apply positional rules
    w = word.lower()
    last_char = w[-1]

    if last_char in _VOWELS or last_char in ("n", "s"):
        # Llana (paroxytone): penultimate
        return len(syls) - 2
    else:
        # Aguda (oxytone): last
        return len(syls) - 1


def is_vos_form(word: str) -> bool:
    """Detect if a word matches Rioplatense vos conjugation patterns.

    Patterns:
    - Ends in -ás (present indicative -ar verbs: hablás, estás)
    - Ends in -és (present indicative -er verbs: tenés, querés)
    - Ends in -ís (present indicative -ir verbs: vivís, decís)
    - Special: "sos" (ser)
    - Must be at least 3 characters (except "sos" which is exactly 3)
    """
    if not word:
        return False

    w = word.lower()

    # Special case
    if w == "sos":
        return True

    # Must be at least 4 characters (excludes adverbs like "más")
    if len(w) < 4:
        return False

    # Check accented endings
    if w.endswith("ás") or w.endswith("és") or w.endswith("ís"):
        return True

    return False
