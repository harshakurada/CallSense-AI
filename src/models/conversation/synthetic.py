"""Synthetic multi-turn conversations with conversation-level labels.

No public dataset has real resolution/satisfaction/escalation labels for
customer-service calls (verified: MultiWOZ and SPADE have goal/task-success
annotations for booking-style dialogue, not this; DailyDialog has only
per-utterance emotion/act labels, nothing at dialogue level — see
docs/CONVERSATION_MODEL.md). This generates labeled conversations instead,
explicitly documented as synthetic, never presented as real.

To avoid the exercise being trivial (and to avoid data leakage in the
literal sense — a label that's a deterministic function of a shallow
surface feature), each conversation is built from one of five outcome
templates, each with several phrasing variants, and ~12% of conversations
get a "noisy" ending that doesn't cleanly match their nominal outcome
(a resolved conversation ending flatly, an unresolved one ending politely)
— so per-utterance sentiment/emotion are correlated with, but not a
perfect proxy for, the conversation-level label.

The customer's opening complaint is real text from the Bitext dataset
(Module 4) for a real category label — only the surrounding conversation
and the outcome are synthesized.
"""
import random
import re

from datasets import load_dataset
from faker import Faker

fake = Faker()

# Bitext's raw `instruction` text sometimes retains unfilled "{{Order
# Number}}"-style template slots (verified against the full 26,872-row
# dataset — these 9 are the complete set that occur). Filled with
# real-looking Faker values for coherent synthetic conversation turns;
# Module 4's intent classifier didn't need this cleanup since it only
# classifies the raw text, but a spoken-sounding conversation turn does.
_PLACEHOLDER_FILLERS = {
    "Order Number": lambda: str(fake.random_number(digits=5, fix_len=True)),
    "Invoice Number": lambda: f"INV-{fake.random_number(digits=6, fix_len=True)}",
    "Person Name": lambda: fake.name(),
    "Account Category": lambda: random.choice(["premium", "standard", "basic"]),
    "Account Type": lambda: random.choice(["checking", "savings", "business"]),
    "Currency Symbol": lambda: "$",
    "Delivery City": lambda: fake.city(),
    "Delivery Country": lambda: fake.country(),
    "Refund Amount": lambda: f"{fake.pydecimal(left_digits=3, right_digits=2, positive=True)}",
}
_PLACEHOLDER_RE = re.compile(r"\{\{([^}]+)\}\}")


def _clean_bitext_text(text: str) -> str:
    def _replace(match: re.Match) -> str:
        filler = _PLACEHOLDER_FILLERS.get(match.group(1))
        return filler() if filler else match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, text)

OUTCOMES = ["RESOLVED_SATISFIED", "RESOLVED_NEUTRAL", "PARTIALLY_RESOLVED", "UNRESOLVED_DISSATISFIED", "UNRESOLVED_ESCALATED"]

RESOLUTION_MAP = {
    "RESOLVED_SATISFIED": "Resolved",
    "RESOLVED_NEUTRAL": "Resolved",
    "PARTIALLY_RESOLVED": "Partially Resolved",
    "UNRESOLVED_DISSATISFIED": "Unresolved",
    "UNRESOLVED_ESCALATED": "Unresolved",
}
SATISFACTION_MAP = {
    "RESOLVED_SATISFIED": "Satisfied",
    "RESOLVED_NEUTRAL": "Neutral",
    "PARTIALLY_RESOLVED": "Neutral",
    "UNRESOLVED_DISSATISFIED": "Dissatisfied",
    "UNRESOLVED_ESCALATED": "Dissatisfied",
}
ESCALATION_MAP = {
    "RESOLVED_SATISFIED": "No",
    "RESOLVED_NEUTRAL": "No",
    "PARTIALLY_RESOLVED": "No",
    "UNRESOLVED_DISSATISFIED": "No",
    "UNRESOLVED_ESCALATED": "Yes",
}

_AGENT_GREETINGS = [
    "Thank you for calling support, how can I help you today?",
    "Hi there, thanks for reaching out. What's going on?",
    "Hello, I'm happy to help — what can I do for you?",
]
_AGENT_CLARIFY = [
    "I see. Can you tell me a bit more about that?",
    "Got it, let me pull that up. Can you confirm a few details?",
    "Understood. Do you have your order or account number handy?",
]
_CUSTOMER_DETAIL = [
    "Sure, it's been going on for a couple of days now.",
    "Yeah, my account is {ACCOUNT_ID} and this started yesterday.",
    "It's regarding order {ORDER_ID}, placed last week.",
]
_AGENT_ATTEMPT = [
    "Okay, I've made an update on my end — can you check now?",
    "I've issued a fix for that. Let me know if it looks right.",
    "I've escalated a note internally and adjusted your account.",
]

_CUSTOMER_REACTION = {
    "RESOLVED_SATISFIED": [
        "Oh perfect, that's exactly it — thank you so much!",
        "That worked! I really appreciate the quick help.",
        "Yes! That's fixed it, you're a lifesaver.",
        "Amazing, thank you for sorting that out so fast.",
    ],
    "RESOLVED_NEUTRAL": [
        "Okay, that works.",
        "Alright, that seems fine now.",
        "Yep, looks good on my end.",
        "Okay, I think that's taken care of.",
    ],
    "PARTIALLY_RESOLVED": [
        "I guess that helps a little, but not the whole issue.",
        "That's something, but part of it is still not fixed.",
        "Okay, half of it's sorted at least.",
        "That covers one part, but the rest is still a problem.",
    ],
    "UNRESOLVED_DISSATISFIED": [
        "That doesn't really solve my problem at all.",
        "Hmm, that's not it — the issue is still there.",
        "No, that's not what I needed at all.",
        "That didn't help, honestly.",
    ],
    "UNRESOLVED_ESCALATED": [
        "This is ridiculous, I want to speak to a manager right now!",
        "Unbelievable. Get me your supervisor, this is unacceptable.",
        "I've had enough of this, put your manager on the phone.",
        "This is absurd, I demand to talk to someone in charge.",
    ],
}
_AGENT_CLOSING = {
    "RESOLVED_SATISFIED": [
        "Wonderful, glad I could help! Have a great day.",
        "Perfect, you're all set!",
        "Great to hear, take care!",
        "Fantastic, glad that worked out.",
    ],
    "RESOLVED_NEUTRAL": [
        "Great, glad that's sorted. Anything else?",
        "Good, let me know if anything comes up.",
        "Alright, glad that's settled.",
        "Okay, feel free to reach out again if needed.",
    ],
    "PARTIALLY_RESOLVED": [
        "I understand, let me look into the rest of it further.",
        "I'll keep digging into the remaining part.",
        "Let me escalate the rest of this for you.",
        "I'll follow up once I have more on the other part.",
    ],
    "UNRESOLVED_DISSATISFIED": [
        "I'm sorry to hear that, let me see what else I can do.",
        "I apologize, let's try another approach.",
        "I understand your frustration, give me a moment to look further.",
        "Sorry about that, let me try something else.",
    ],
    "UNRESOLVED_ESCALATED": [
        "I understand your frustration, I'm connecting you to a supervisor now.",
        "I'm very sorry, transferring you right away.",
        "Of course, let me get my supervisor on the line immediately.",
        "I apologize for the trouble, connecting you now.",
    ],
}
# A fraction of conversations get an ending that doesn't match the nominal
# outcome as cleanly — a politeness mismatch real conversations also have.
# Applied to BOTH the customer's reaction and the agent's closing: an
# earlier version only noised the customer line while the agent's closing
# always stayed outcome-specific ("I'm connecting you to a supervisor" is
# an unambiguous escalation tell) — the model reached literal 100% on
# resolution/satisfaction/escalation by reading just that one always-honest
# final turn, regardless of the customer noise. See docs/CONVERSATION_MODEL.md.
_NOISY_CUSTOMER_ENDING = [
    "Okay, thanks anyway, I suppose.",
    "Fine, whatever, I have to go.",
    "Alright, sure, thanks.",
    "Sure, I guess that's it then.",
    "Right, okay, moving on.",
]
_NOISY_AGENT_CLOSING = [
    "Alright, thanks for calling in.",
    "Okay, take care.",
    "Understood, have a good one.",
    "Noted, thanks for your patience.",
]

_NOISE_RATE = 0.25


def _load_bitext_by_category(seed: int = 42) -> dict[str, list[str]]:
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    by_category: dict[str, list[str]] = {}
    for row in ds:
        by_category.setdefault(row["category"], []).append(row["instruction"])
    rng = random.Random(seed)
    for texts in by_category.values():
        rng.shuffle(texts)
    return by_category


def _fill(template: str) -> str:
    return template.replace("{ACCOUNT_ID}", f"ACC{fake.random_number(digits=6, fix_len=True)}").replace(
        "{ORDER_ID}", str(fake.random_number(digits=5, fix_len=True))
    )


def generate_conversation(category: str, opening_text: str, outcome: str, rng: random.Random) -> dict:
    turns = []
    t = 0.0

    def add(speaker: str, text: str, duration: float):
        nonlocal t
        turns.append({"speaker": speaker, "start": t, "end": t + duration, "text": text})
        t += duration + 0.4

    add("AGENT", rng.choice(_AGENT_GREETINGS), 3.0)
    add("CUSTOMER", opening_text, 4.5)
    add("AGENT", rng.choice(_AGENT_CLARIFY), 3.0)
    add("CUSTOMER", _fill(rng.choice(_CUSTOMER_DETAIL)), 4.0)

    # extra back-and-forth for some conversations — varies conversation length
    if rng.random() < 0.4:
        add("AGENT", "Thanks for confirming. One moment while I check.", 2.5)
        add("CUSTOMER", "Sure, take your time.", 1.5)

    add("AGENT", rng.choice(_AGENT_ATTEMPT), 3.0)

    is_noisy = rng.random() < _NOISE_RATE
    reaction = rng.choice(_NOISY_CUSTOMER_ENDING) if is_noisy else rng.choice(_CUSTOMER_REACTION[outcome])
    add("CUSTOMER", reaction, 3.0)
    closing = rng.choice(_NOISY_AGENT_CLOSING) if is_noisy else rng.choice(_AGENT_CLOSING[outcome])
    add("AGENT", closing, 2.5)

    return {
        "conversation": turns,
        "labels": {
            "category": category,
            "resolution": RESOLUTION_MAP[outcome],
            "satisfaction": SATISFACTION_MAP[outcome],
            "escalation": ESCALATION_MAP[outcome],
        },
        "outcome_template": outcome,  # for analysis only — never used as a model input feature
        "is_noisy_ending": is_noisy,
    }


def generate_dataset(n_examples: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    Faker.seed(seed)
    by_category = _load_bitext_by_category(seed=seed)
    categories = list(by_category.keys())
    cursors = {c: 0 for c in categories}

    examples = []
    for _ in range(n_examples):
        category = rng.choice(categories)
        texts = by_category[category]
        opening_text = _clean_bitext_text(texts[cursors[category] % len(texts)])
        cursors[category] += 1

        outcome = rng.choice(OUTCOMES)
        examples.append(generate_conversation(category, opening_text, outcome, rng))

    return examples
