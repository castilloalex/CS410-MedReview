# AI-assisted code
# Model: Fable 5.2 (medium effort)
"""Keyword lexicons. Edit these lists to tune the keyword baseline.

SIDE_EFFECT_TERMS maps a canonical side-effect name to regex alternatives
(matched case-insensitively on word boundaries).
"""

SIDE_EFFECT_TERMS: dict[str, list[str]] = {
    "nausea": [r"nause\w*", r"queasy", r"sick to my stomach"],
    "vomiting": [r"vomit\w*", r"threw up", r"throwing up", r"throw up", r"puk\w+"],
    "headache": [r"headaches?", r"migraines?"],
    "dizziness": [r"dizz\w+", r"light[- ]?headed\w*", r"vertigo"],
    "drowsiness": [r"drows\w+", r"sleepy", r"sleepiness", r"grogg\w+", r"sedat\w+", r"zombie"],
    "fatigue": [r"fatigued?", r"tired(ness)?", r"exhaust\w+", r"letharg\w+", r"no energy", r"sluggish"],
    "insomnia": [r"insomnia", r"can'?t sleep", r"couldn'?t sleep", r"trouble sleeping",
                 r"unable to sleep", r"sleepless\w*"],
    "weight gain": [r"weight gain", r"gain(ed|ing)? (\w+ ){0,3}(weight|lbs?|pounds)", r"put on (\w+ ){0,3}(weight|lbs?|pounds)"],
    "weight loss": [r"weight loss", r"lost (\w+ ){0,3}(weight|lbs?|pounds)", r"los(e|ing) (\w+ ){0,3}weight"],
    "increased appetite": [r"increased appetite", r"always hungry", r"constantly hungry", r"cravings?"],
    "loss of appetite": [r"loss of appetite", r"no appetite", r"decreased appetite", r"lost my appetite", r"not hungry"],
    "dry mouth": [r"dry mouth", r"cotton ?mouth"],
    "constipation": [r"constipat\w+"],
    "diarrhea": [r"diarrh?ea", r"loose stools?"],
    "stomach pain": [r"stomach (pain|ache|aches|cramps?|upset|issues|problems)", r"upset stomach",
                     r"stomach ?ache", r"abdominal (pain|cramps?)", r"heartburn", r"indigestion", r"acid reflux"],
    "bloating": [r"bloat\w+", r"gass?y", r"\bgas\b"],
    "cramps": [r"cramps?", r"cramping"],
    "anxiety": [r"anxiety", r"anxious", r"panic attacks?", r"jitter\w+", r"nervous(ness)?", r"on edge"],
    "depression": [r"depress\w+", r"suicidal( thoughts)?"],
    "mood swings": [r"mood ?swings?", r"moody", r"irritab\w+", r"\brage\b", r"emotional", r"crying"],
    "low libido": [r"(no|low|decreased|loss of|lost my|zero) (sex drive|libido)", r"sex drive", r"libido",
                   r"sexual (side effects?|dysfunction)"],
    "acne": [r"acne", r"break(ing|s)? ?out", r"breakouts?", r"pimples?"],
    "rash": [r"rash(es)?", r"hives", r"itch\w*"],
    "hair loss": [r"hair loss", r"hair (is |was |started )?falling out", r"losing (my )?hair", r"hair thinning"],
    "sweating": [r"sweat\w*", r"night sweats", r"hot flashes?"],
    "tremor": [r"tremors?", r"shak(y|ing|es)", r"twitch\w*"],
    "heart palpitations": [r"palpitations?", r"heart (was )?(racing|pounding)", r"racing heart", r"rapid heart ?(beat|rate)"],
    "blurred vision": [r"blurr\w+ vision", r"vision (was |got |is )?blurr\w+"],
    "bleeding/spotting": [r"spotting", r"bleeding", r"bled", r"irregular periods?", r"heavy periods?"],
    "brain fog": [r"brain fog", r"foggy", r"memory (loss|problems|issues)", r"forgetful\w*", r"confus\w+",
                  r"can'?t concentrate", r"trouble concentrating"],
    "vivid dreams": [r"nightmares?", r"(vivid|weird|strange|crazy) dreams"],
    "muscle/joint pain": [r"(muscle|joint|body|back|leg) (pain|aches?)", r"sore(ness)?", r"achy"],
    "numbness": [r"numb(ness)?", r"tingl\w+"],
    "swelling": [r"swelling", r"swollen", r"swelled"],
    "breast tenderness": [r"(breast|boob)s? (tenderness|pain|are sore|were sore)", r"(sore|tender) (breasts?|boobs)"],
    "restlessness": [r"restless\w*", r"can'?t sit still"],
    "bad taste": [r"(bad|metallic|bitter|awful|horrible|terrible) taste"],
    "chest pain": [r"chest (pain|tightness)"],
    "yeast infection": [r"yeast infections?"],
    "burning": [r"burn(ing|ed|s)?"],
    "withdrawal": [r"withdrawals?", r"brain zaps?"],
}

# Generic cues that a sentence is about side effects (no specific symptom needed).
SIDE_EFFECT_CUES: list[str] = [
    r"side[- ]?effects?", r"adverse", r"reaction", r"made me (feel )?\w+", r"makes me (feel )?\w+",
    r"caused?( me)?", r"causing", r"gave me", r"gives me", r"experienc(ed|ing)", r"did experience", r"experience (some|any|a|the|no|side)", r"suffer\w+",
    r"couldn'?t (function|get out of bed)", r"felt (like|awful|terrible|horrible|sick|weird|off)",
    r"feel (awful|terrible|horrible|sick|weird|off)", r"unbearable", r"had to stop", r"stopped taking",
    r"went away after", r"subsided", r"wore off",
]

# Cues that a sentence is about whether the drug works.
EFFECTIVENESS_CUES: list[str] = [
    # "work" only when it is about the drug, not a job ("I work retail").
    r"(it|this|that|which|these|they|medicine|medication|med|meds|drug|pills?|stuff|product|dose|treatment|cream|shot|patch)"
    r"( \w+){0,3} work(s|ed|ing)?",
    r"work(s|ed|ing) (\w+ )?(great|well|wonders|fine|for|like|better|best|good|fast|quick\w*|amazing\w*|perfect\w*|"
    r"immediately|right away|within|wonderfully|miracles|so)",
    r"(not|n'?t|never|stopped|seems? to|seemed to|does|did|really|still|actually|finally|only thing that) work(s|ed|ing)?",
    r"help(s|ed|ing|ful)?", r"effective\w*", r"ineffective", r"relie(f|ve|ves|ved|ving)",
    r"improv\w+", r"cured?", r"clear(ed|ing|s)? (up|my)", r"cleared", r"(is|are|was|were|has|have) gone", r"went away",
    r"no longer", r"no more", r"under control", r"controls?", r"controlled", r"manag(e|es|ed|ing)", r"reduc\w+",
    r"decreas\w+", r"lessen\w*", r"eased?", r"subsided", r"disappear\w*",
    r"life ?saver", r"miracle", r"wonders", r"god ?send", r"game ?changer", r"saved my life", r"changed my life",
    r"did(n'?t| not) (do|help|work)", r"does(n'?t| not) (do|help|work)", r"did nothing", r"does nothing", r"useless",
    r"no (relief|difference|change|improvement|effect)", r"difference",
    r"(feel|feeling|felt) (so |much |a lot |way )?(better|normal|great|like myself|like a new person)",
    r"better", r"stopp?(ed|s) (my|the)", r"(pain|symptoms?|seizure|migraine|acne)[- ]free", r"kick(s|ed) in", r"results?",
    r"lowered", r"brought (my|it) down", r"back to normal", r"success\w*", r"(finally|now) (able|can)",
    r"(was|am|been|being) able to", r"keeps? (me|my|it|them)", r"kept (me|my|it|them)", r"symptoms", r"took the edge off",
    r"takes the edge off", r"magic\w*", r"waste of (money|time)", r"(have|had) not had (a|an|any)", r"haven'?t had (a|an|any)",
]

# Verbs that mark a following symptom as the thing being treated.
TREATING_VERBS: list[str] = [
    r"treat\w*", r"help(s|ed|ing)?( with)?", r"reliev\w+", r"relief (from|of|for)", r"control\w*", r"manag\w+", r"cur\w+",
    r"stopp?(ed|s|ing)?", r"reduc\w+", r"decreas\w+", r"lessen\w*", r"eas(e|es|ed|ing)", r"got rid of", r"gets? rid of",
    r"(taken|took|takes|take) away", r"prescribed( \w+){0,2} for", r"diagnosed with", r"suffer\w* (from|with)",
    r"history of", r"struggl\w+ with", r"deal(t|ing|s)? with", r"battl\w+", r"prevent\w*", r"cleared( up)?",
    r"(taking|take|took|use|used|using|on) (it|this) for", r"no more", r"no longer (have|had|get|suffer)",
]

NEGATIONS = [r"no", r"not", r"never", r"without", r"zero", r"none", r"n'?t", r"free of", r"nor", r"lack of", r"absence of",
             r"haven'?t had", r"didn'?t have", r"don'?t have"]

# Words in a review's condition field that mean a lexicon term is probably the
# problem being treated rather than a side effect (e.g. "anxiety" in reviews of an
# anxiety drug). The canonical name itself is always checked as well.
CONDITION_HINTS: dict[str, list[str]] = {
    "anxiety": ["anxiety", "panic", "stress"],
    "depression": ["depress", "bipolar"],
    "mood swings": ["bipolar", "premenstrual", "mood"],
    "insomnia": ["insomnia"],
    "drowsiness": ["narcolepsy"],
    "fatigue": ["fatigue", "narcolepsy"],
    "weight loss": ["obesity", "weight loss"],
    "acne": ["acne"],
    "headache": ["migraine", "headache"],
    "bleeding/spotting": ["bleeding", "menstrual", "period", "endometriosis"],
    "cramps": ["cramp", "period", "menstrual", "endometriosis"],
    "muscle/joint pain": ["pain", "arthritis", "fibromyalgia", "muscle spasm"],
    "nausea": ["nausea"],
    "vomiting": ["vomiting"],
    "constipation": ["constipation"],
    "diarrhea": ["diarrhea", "irritable bowel"],
    "stomach pain": ["gerd", "reflux", "irritable bowel", "indigestion", "ulcer"],
    "bloating": ["irritable bowel", "bloating"],
    "rash": ["rash", "urticaria", "itch", "dermatitis", "eczema", "psoriasis", "allerg"],
    "tremor": ["tremor", "parkinson"],
    "sweating": ["hot flashes", "menopaus", "hyperhidrosis"],
    "restlessness": ["restless legs"],
    "yeast infection": ["yeast", "candid"],
    "low libido": ["erectile", "sexual"],
    "hair loss": ["alopecia", "hair loss"],
    "swelling": ["edema"],
    "numbness": ["neuropathy"],
    "heart palpitations": ["tachycardia", "fibrillation"],
    "chest pain": ["angina"],
    "withdrawal": ["withdrawal", "dependence"],
    "burning": ["urinary tract", "bladder infection"],
}
