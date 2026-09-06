from app.services.recommender import ACTION_RISK, MAX_RECOMMENDATIONS, sanitize


def test_risk_comes_from_the_action_not_the_model():
    """A model that could label a production rollback low-risk could talk its
    way past the policy engine. Risk is a property of the action type."""
    out = sanitize(
        [{"title": "Roll back", "action_type": "rollback_deployment", "risk": "low"}],
        is_default_branch=False,
    )

    assert out[0]["risk"] == "critical"


def test_unknown_action_types_degrade_to_manual():
    out = sanitize([{"title": "rm -rf /", "action_type": "delete_everything"}],
                   is_default_branch=False)

    assert out[0]["action_type"] == "manual"
    assert out[0]["risk"] == ACTION_RISK["manual"]


def test_default_branch_escalates_one_level():
    payload = [{"title": "Edit compose", "action_type": "edit_file"}]

    assert sanitize(payload, is_default_branch=False)[0]["risk"] == "medium"
    assert sanitize(payload, is_default_branch=True)[0]["risk"] == "high"


def test_low_risk_actions_do_not_escalate():
    """Escalating "go look at it" to high risk would only teach people to ignore
    the label."""
    payload = [{"title": "Investigate", "action_type": "investigate"}]

    assert sanitize(payload, is_default_branch=True)[0]["risk"] == "low"


def test_critical_cannot_escalate_past_critical():
    payload = [{"title": "Roll back", "action_type": "rollback_deployment"}]

    assert sanitize(payload, is_default_branch=True)[0]["risk"] == "critical"


def test_recommendations_are_capped():
    payload = [{"title": f"r{i}", "action_type": "investigate"} for i in range(20)]

    assert len(sanitize(payload, is_default_branch=False)) == MAX_RECOMMENDATIONS


def test_affected_files_are_bounded_and_confidence_clamped():
    out = sanitize(
        [{"title": "x", "action_type": "edit_file",
          "affected_files": [f"f{i}.php" for i in range(50)], "confidence": 7}],
        is_default_branch=False,
    )

    assert len(out[0]["affected_files"]) == 20
    assert out[0]["confidence"] == 1.0


def test_every_action_type_has_a_risk():
    """A new action type without a risk entry would silently become 'manual'."""
    assert set(ACTION_RISK) >= {
        "investigate", "retry_job", "retry_pipeline", "create_issue", "edit_file",
        "update_dependency", "create_merge_request", "manual", "update_config",
        "rollback_deployment",
    }


def test_schema_action_types_and_risk_table_stay_in_sync():
    """Drift here is silent: an action the schema allows but the risk table does
    not know becomes 'manual', quietly discarding what the model proposed."""
    from app.services.prompts import ANALYZE_SCHEMA

    schema_actions = set(
        ANALYZE_SCHEMA["properties"]["recommendations"]["items"]["properties"]["action_type"]["enum"]
    )

    assert schema_actions == set(ACTION_RISK)
