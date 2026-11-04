from tgcc.contracts import Truth
from tgcc.runtime import snapshot_evidence


def snapshot(value=False, *, source="oracle", valid_nodes=True):
    names = ["food", "robot:ee:left", "robot:ee:right", "container"]
    nodes = [
        dict(
            entity_id=str(i),
            instance_id=name,
            timestamp=1.0,
            last_seen=1.0,
            confidence=1.0,
            validity=valid_nodes,
            pose=[[1]],
            source=source,
            attributes={},
        )
        for i, name in enumerate(names)
    ]
    edges = [
        dict(
            source_id="0",
            target_id=str(i),
            relation_type="held_by" if i < 3 else "inside",
            relative_attributes=dict(value=value),
            timestamp=1.0,
            confidence=1.0,
            validity=value is True,
            source=source,
        )
        for i in (1, 2, 3)
    ]
    return dict(entities=nodes, edges=edges)


def test_explicit_negative_oracle_relation_is_not_missing_evidence():
    result = snapshot_evidence(snapshot(), 1.0)
    assert result["held", "food", ""].truth(1.0) == Truth.FALSE
    assert result["inside", "food", "container"].truth(1.0) == Truth.FALSE


def test_missing_relation_and_invalid_entities_do_not_prove_false():
    for s in (
        snapshot(None),
        snapshot(valid_nodes=False),
        snapshot(source="unverified"),
    ):
        assert snapshot_evidence(s, 1.0)["held", "food", ""].truth(1.0) == Truth.UNKNOWN


def test_positive_and_explicit_observation_provenance_are_preserved():
    assert (
        snapshot_evidence(snapshot(True), 1.0)["held", "food", ""].truth(1.0)
        == Truth.TRUE
    )
    s = snapshot(source="observation")
    for edge in s["edges"]:
        edge["relative_attributes"]["evidence_valid"] = True
    assert snapshot_evidence(s, 1.0)["held", "food", ""].truth(1.0) == Truth.FALSE


def test_duplicate_one_hand_negatives_cannot_prove_both_hands_empty():
    s = snapshot()
    s["edges"] = [s["edges"][0], s["edges"][0]]
    assert snapshot_evidence(s, 1.0)["held", "food", ""].truth(1.0) == Truth.UNKNOWN
