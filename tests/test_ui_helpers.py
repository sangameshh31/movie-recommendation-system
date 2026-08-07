from frontend.streamlit_app import EXPLORE_CARDS


def test_explore_cards_have_recruiter_friendly_prompts():
    assert len(EXPLORE_CARDS) >= 4
    assert any("sci-fi" in card["label"].lower() for card in EXPLORE_CARDS)
    assert all(card["query"].strip() for card in EXPLORE_CARDS)
