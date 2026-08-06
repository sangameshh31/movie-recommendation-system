from frontend.streamlit_app import get_explore_cards


def test_get_explore_cards_have_recruiter_friendly_prompts():
    cards = get_explore_cards()

    assert len(cards) >= 4
    assert any(card["label"] == "Mind-bending sci-fi" for card in cards)
    assert all(card["query"].strip() for card in cards)
