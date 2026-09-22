import pandas as pd

from src.panel import crosswalk


def _rows(ncessch, years, name, x, y, seasch="1"):
    return [dict(ncessch=ncessch, year=t, school_name=name, x=x, y=y, seasch=seasch) for t in years]


def test_links_renumbered_school_and_ignores_distant_one(params):
    df = pd.DataFrame(
        _rows("130000100001", range(2000, 2005), "Oak Grove Elementary School", 0, 0, "A")
        + _rows("130000200009", range(2005, 2010), "Oak Grove ES", 50, 0, "B")
        # Same name, starts the right year, but 5 km away: a different school.
        + _rows("130000300003", range(2005, 2010), "Oak Grove Elementary", 5000, 0, "C")
        # Same place, different name and state ID: a new school in the old building.
        + _rows("130000400004", range(2005, 2010), "Riverside Middle", 10, 0, "D")
    )
    links, uid = crosswalk.build(df, params)
    assert len(links) == 1
    assert links.iloc[0][["old_ncessch", "new_ncessch"]].tolist() == ["130000100001", "130000200009"]
    m = dict(zip(uid["ncessch"], uid["school_uid"]))
    assert m["130000200009"] == "130000100001"
    assert m["130000300003"] == "130000300003"
    assert m["130000400004"] == "130000400004"


def test_state_id_links_renamed_school(params):
    df = pd.DataFrame(
        _rows("130000100001", range(2000, 2003), "Central High", 0, 0, "0101")
        + _rows("130000900001", range(2003, 2006), "Martin Luther King Jr STEM Academy", 20, 0, "0101")
    )
    links, uid = crosswalk.build(df, params)
    assert links.iloc[0]["match_basis"] == "state_id"
    assert uid["school_uid"].nunique() == 1


def test_norm_name_abbreviations():
    assert crosswalk.norm_name("Oak Grove Elem. Sch.") == crosswalk.norm_name("Oak Grove Elementary School")
