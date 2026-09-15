"""Synthetic training data for entity types no public NER dataset labels:
ORDER_ID, ACCOUNT_ID, INVOICE_ID, PHONE, EMAIL, MONEY, DATE.

Every example here is template-generated with Faker-produced fake values.
None of it is real customer data, real orders, real phone numbers, or real
people — this is stated explicitly wherever this data is used (training
reports, docs/NER.md) so it is never mistaken for real-world evaluation
data. This exists because the spec's own instruction is explicit: "If
custom customer-service entities such as ORDER_ID are not available,
design a synthetic augmentation strategy using clearly documented
generated examples. Do not pretend synthetic data is real-world data."
"""
import random
import re

from faker import Faker

fake = Faker()
Faker.seed(42)

# {value} is replaced with a generated value; everything else is literal
# template text. One sentence may contain several entities.
TEMPLATES = [
    "Hi, my name is {PERSON} and I'm calling about order {ORDER_ID}.",
    "Can you check the status of invoice {INVOICE_ID} for account {ACCOUNT_ID}?",
    "I was charged {MONEY} on {DATE} but never received order {ORDER_ID}.",
    "Please call me back at {PHONE} or email me at {EMAIL}.",
    "My account number is {ACCOUNT_ID} and I placed the order on {DATE}.",
    "The invoice {INVOICE_ID} shows {MONEY} which seems incorrect.",
    "This is {PERSON}, reachable at {PHONE}, regarding invoice {INVOICE_ID}.",
    "I need a refund of {MONEY} for order {ORDER_ID} placed on {DATE}.",
    "Please update my account {ACCOUNT_ID} with my new email {EMAIL}.",
    "Order {ORDER_ID} was supposed to arrive on {DATE} but hasn't.",
    "You can reach {PERSON} at {EMAIL} or {PHONE} about account {ACCOUNT_ID}.",
    "I paid {MONEY} for invoice {INVOICE_ID} on {DATE}, please confirm receipt.",
]


def _order_id() -> str:
    return random.choice([str(fake.random_number(digits=5, fix_len=True)), f"ORD-{fake.random_number(digits=6, fix_len=True)}"])


def _account_id() -> str:
    return random.choice([f"ACC{fake.random_number(digits=6, fix_len=True)}", f"A-{fake.random_number(digits=5, fix_len=True)}"])


def _invoice_id() -> str:
    return f"INV-{fake.year()}-{fake.random_number(digits=4, fix_len=True)}"


def _money() -> str:
    return f"${fake.pydecimal(left_digits=3, right_digits=2, positive=True)}"


def _date() -> str:
    return fake.date(pattern=random.choice(["%B %d", "%m/%d/%Y", "%B %d, %Y"]))


def _phone() -> str:
    return fake.phone_number()


def _email() -> str:
    return fake.email()


def _person() -> str:
    return fake.name()


_GENERATORS = {
    "ORDER_ID": _order_id,
    "ACCOUNT_ID": _account_id,
    "INVOICE_ID": _invoice_id,
    "MONEY": _money,
    "DATE": _date,
    "PHONE": _phone,
    "EMAIL": _email,
    "PERSON": _person,
}

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _tokenize(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer — deliberately simple, matching
    what the fine-tuning tokenizer will re-split into subwords anyway."""
    return re.findall(r"\w+(?:[-/.]\w+)*|\S", text)


def generate_example() -> tuple[list[str], list[str]]:
    """Returns (tokens, bio_tags) for one synthetic sentence."""
    template = random.choice(TEMPLATES)

    tokens: list[str] = []
    tags: list[str] = []
    last_end = 0

    for match in _PLACEHOLDER_RE.finditer(template):
        literal = template[last_end : match.start()]
        for tok in _tokenize(literal):
            tokens.append(tok)
            tags.append("O")

        entity_type = match.group(1)
        value = _GENERATORS[entity_type]()
        value_tokens = _tokenize(value)
        for i, tok in enumerate(value_tokens):
            tokens.append(tok)
            tags.append(f"B-{entity_type}" if i == 0 else f"I-{entity_type}")

        last_end = match.end()

    for tok in _tokenize(template[last_end:]):
        tokens.append(tok)
        tags.append("O")

    return tokens, tags


def generate_dataset(n_examples: int, seed: int = 42) -> list[tuple[list[str], list[str]]]:
    random.seed(seed)
    return [generate_example() for _ in range(n_examples)]
