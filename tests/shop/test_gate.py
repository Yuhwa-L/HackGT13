from shop.gate import ALLOWED, allowed_actions, can_checkout, effective_decision


def test_allowed_table():
    assert "checkout" in ALLOWED["trust"]
    assert "checkout" not in ALLOWED["caution"]
    assert ALLOWED["reject"] == {"ask_retake"}


def test_allowed_actions_matches_table():
    for d, acts in ALLOWED.items():
        assert allowed_actions(d) == acts


def test_caution_confirm_upgrades_to_trust():
    assert effective_decision("caution", "backpack", ["backpack", "purse"]) == "trust"
    assert effective_decision("caution", None, ["backpack", "purse"]) == "caution"
    assert effective_decision("reject", "backpack", ["backpack"]) == "reject"


def test_checkout_gate():
    assert can_checkout("trust", None, ["backpack"])
    assert not can_checkout("caution", None, ["backpack", "purse"])
    assert can_checkout("caution", "backpack", ["backpack", "purse"])
    assert not can_checkout("reject", "backpack", ["backpack"])
