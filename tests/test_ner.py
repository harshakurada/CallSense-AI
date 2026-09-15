"""NER tests. Rules, synthetic data generation, PII masking, normalization,
and structured extraction are pure logic and always run. spaCy-baseline
tests need only the downloaded en_core_web_sm model (no training). The
fine-tuned Transformer's tests are skipped until it's trained."""
from pathlib import Path

import pytest

from src.nlp.ner.baseline import Entity, extract_baseline_entities
from src.nlp.ner.extract import extract_structured_fields
from src.nlp.ner.normalize import normalize_entity
from src.nlp.ner.pii import mask_pii
from src.nlp.ner.rules import extract_rule_based_entities
from src.nlp.ner.synthetic import generate_dataset

NER_MODEL = Path("models/nlp/ner/transformer")
requires_ner_model = pytest.mark.skipif(
    not (NER_MODEL / "config.json").exists(), reason="run: python scripts/train_ner_model.py"
)


# --- rules: names, order numbers, money, dates, emails, phone numbers ---


def test_rules_extract_email():
    entities = extract_rule_based_entities("reach me at jane.doe@example.com please")
    assert any(e.label == "EMAIL" and e.text == "jane.doe@example.com" for e in entities)


def test_rules_extract_phone():
    entities = extract_rule_based_entities("call 555-123-4567 today")
    assert any(e.label == "PHONE" for e in entities)


def test_rules_extract_order_id_from_spec_example():
    entities = extract_rule_based_entities("John Smith called regarding order 45821.")
    assert any(e.label == "ORDER_ID" and e.text == "45821" for e in entities)


def test_rules_extract_money():
    entities = extract_rule_based_entities("that invoice was for $1,240.50")
    assert any(e.label == "MONEY" and e.text == "$1,240.50" for e in entities)


def test_rules_extract_invoice_and_account_id():
    entities = extract_rule_based_entities("invoice INV-2024-0091 for account ACC123456")
    labels = {e.label: e.text for e in entities}
    assert labels["INVOICE_ID"] == "INV-2024-0091"
    assert labels["ACCOUNT_ID"] == "ACC123456"


def test_rules_no_false_positive_on_plain_text():
    entities = extract_rule_based_entities("Thank you for calling, how can I help you today?")
    assert entities == []


def test_rules_specific_ids_take_priority_over_phone_pattern():
    # ORD-99887 could otherwise be partially matched by a looser pattern —
    # confirms no double-claim/overlap corruption
    entities = extract_rule_based_entities("order ORD-99887 confirmed")
    assert len(entities) == 1
    assert entities[0].label == "ORDER_ID"


# --- baseline (spaCy + rules combined) ---


def test_baseline_extracts_person_org_date():
    text = "John Smith from Acme Corp called on March 5th."
    entities = extract_baseline_entities(text)
    labels = {e.label for e in entities}
    assert "PERSON" in labels
    assert "ORGANIZATION" in labels
    assert "DATE" in labels


def test_baseline_rule_wins_over_spacy_misclassification():
    # spaCy alone mislabels a bare order number as DATE (verified manually
    # during development) — the combined baseline must not.
    text = "regarding order 45821"
    entities = extract_baseline_entities(text)
    assert any(e.label == "ORDER_ID" and e.text == "45821" for e in entities)
    assert not any(e.label == "DATE" for e in entities)


def test_baseline_missing_entities_returns_empty():
    entities = extract_baseline_entities("Hello there.")
    assert entities == []


def test_baseline_noisy_transcript_does_not_crash():
    noisy = "uh so like my um order number is uh 45821 i think and uh my email is uh j@x.com yeah"
    entities = extract_baseline_entities(noisy)
    labels = {e.label for e in entities}
    assert "ORDER_ID" in labels
    assert "EMAIL" in labels


# --- synthetic data generation ---


def test_synthetic_generates_valid_bio_tags():
    examples = generate_dataset(20)
    assert len(examples) == 20
    for tokens, tags in examples:
        assert len(tokens) == len(tags)
        for tag in tags:
            assert tag == "O" or tag[0] in ("B", "I")


def test_synthetic_covers_business_entity_types():
    examples = generate_dataset(100)
    all_tags = {tag[2:] for _, tags in examples for tag in tags if tag != "O"}
    assert {"ORDER_ID", "ACCOUNT_ID", "INVOICE_ID", "PHONE", "EMAIL", "MONEY", "DATE"} <= all_tags


# --- PII masking (spec's exact worked example) ---


def test_pii_masking_matches_spec_example():
    text = "John Smith called regarding order 45821."
    entities = extract_baseline_entities(text)
    assert mask_pii(text, entities) == "[PERSON] called regarding order [ORDER_ID]."


def test_pii_masking_leaves_unlisted_types_untouched():
    entities = [Entity(label="PRODUCT", text="Widget", start=0, end=6, source="spacy")]
    assert mask_pii("Widget broke", entities) == "Widget broke"


# --- normalization ---


def test_normalize_money():
    entity = Entity(label="MONEY", text="$1,240.50", start=0, end=0, source="rule")
    assert normalize_entity(entity)["normalized"] == 1240.50


def test_normalize_phone():
    entity = Entity(label="PHONE", text="(415) 555-0199", start=0, end=0, source="rule")
    assert normalize_entity(entity)["normalized"] == "4155550199"


def test_normalize_unparseable_date_keeps_raw_text_with_none_normalized():
    entity = Entity(label="DATE", text="next Tuesday", start=0, end=0, source="spacy")
    result = normalize_entity(entity)
    assert result["normalized"] is None
    assert result["text"] == "next Tuesday"


def test_normalize_person_has_no_normalizer():
    entity = Entity(label="PERSON", text="John Smith", start=0, end=0, source="spacy")
    assert normalize_entity(entity)["normalized"] is None


# --- structured field extraction ---


def test_extract_structured_fields_populates_available_fields():
    entities = [
        Entity(label="PERSON", text="John Smith", start=0, end=10, source="spacy"),
        Entity(label="ORDER_ID", text="45821", start=20, end=25, source="rule"),
        Entity(label="MONEY", text="$120", start=30, end=34, source="rule"),
    ]
    fields = extract_structured_fields(entities)
    assert fields["customer"] == "John Smith"
    assert fields["order_id"] == "45821"
    assert fields["amount"] == "$120"


def test_extract_structured_fields_missing_entities_stay_none():
    fields = extract_structured_fields([])
    assert fields == {"customer": None, "order_id": None, "product": None, "issue": None, "date": None, "amount": None}


def test_extract_structured_fields_never_fabricates_issue():
    entities = [Entity(label="PERSON", text="Jane", start=0, end=4, source="spacy")]
    assert extract_structured_fields(entities)["issue"] is None


# --- fine-tuned transformer (skipped until trained) ---


@requires_ner_model
def test_transformer_predicts_entities_with_valid_schema():
    from src.nlp.ner.model import predict_entities

    entities = predict_entities(NER_MODEL, "Please call me at 555-123-4567 about order 45821")
    for e in entities:
        assert 0.0 <= e["confidence"] <= 1.0
        assert e["start"] < e["end"]


@requires_ner_model
def test_inference_prefers_longer_span_on_overlap():
    """Regression test: with more surrounding context, the fine-tuned
    model sometimes truncates a structured identifier ("45" instead of
    "45821") while the regex rule still gets the full span. The combiner
    must keep the longer, correct match rather than always trusting the
    model on overlap."""
    from src.nlp.ner.inference import extract_entities

    text = "This is ridiculous, my order 45821 never arrived and I want a refund. Get me a manager!"
    entities = extract_entities(text)
    order_ids = [e for e in entities if e.label == "ORDER_ID"]
    assert order_ids
    assert order_ids[0].text == "45821"


@requires_ner_model
def test_full_pipeline_integration():
    from src.nlp.ner.pipeline import analyze_text

    result = analyze_text("John Smith called regarding order 45821.")
    assert "[" in result["masked_text"]
    assert isinstance(result["entities"], list)
    assert "customer" in result["structured_fields"]
