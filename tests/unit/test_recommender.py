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


class TestPatchValidation:
    """A hallucinated diff has exactly the same shape as a real one.

    It is the most damaging thing this system could hand a user, so the check is
    structural rather than a matter of trusting the model to have been careful.
    """

    VALID = (
        "--- a/src/AuthPage.tsx\n"
        "+++ b/src/AuthPage.tsx\n"
        "@@ -274,7 +274,7 @@\n"
        "-                    </button/>\n"
        "+                    </button>\n"
    )

    def test_a_patch_for_a_file_never_shown_is_discarded(self):
        from app.services.recommender import validate_patch

        # The model was shown AuthPage.tsx and wrote a diff for a config file it
        # only knows the name of. Confident, plausible, and invented.
        assert validate_patch(self.VALID, known_files={"docker-compose.yml"}) is None

    def test_a_patch_for_a_shown_file_survives(self):
        from app.services.recommender import validate_patch

        assert validate_patch(self.VALID, known_files={"src/AuthPage.tsx"}) == self.VALID

    def test_a_repository_relative_path_still_matches(self):
        from app.services.recommender import validate_patch

        # A diff header may carry a shorter path than the prompt did.
        assert validate_patch(
            self.VALID, known_files={"IntelliLearn-AI-frontend/src/AuthPage.tsx"}
        ) == self.VALID

    def test_prose_without_a_hunk_header_is_not_a_patch(self):
        from app.services.recommender import validate_patch

        assert validate_patch("change </button/> to </button>", known_files=None) is None

    def test_a_whole_file_rewrite_is_refused(self):
        from app.services.recommender import validate_patch

        huge = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n" + ("+line\n" * 5000)

        assert validate_patch(huge, known_files={"x.py"}) is None

    def test_non_strings_are_refused(self):
        from app.services.recommender import validate_patch

        for value in (None, "", "   ", 42, {"patch": "x"}):
            assert validate_patch(value, known_files=None) is None

    def test_sanitize_strips_an_out_of_scope_patch_but_keeps_the_advice(self):
        out = sanitize(
            [{
                "title": "Fix the closing tag",
                "action_type": "edit_file",
                "patch": self.VALID,
            }],
            is_default_branch=False,
            known_files={"something/else.ts"},
        )

        # The recommendation survives; only the untrustworthy diff is removed.
        assert out[0]["title"] == "Fix the closing tag"
        assert out[0]["patch"] is None
