import logging

import pandas as pd
import trueskill

logger = logging.getLogger(__name__)


def fit_trueskill_large(pairs_df: pd.DataFrame) -> dict[str, float]:
    """
    Estimate TrueSkill ``mu`` ratings for each image from pairwise win/loss rows.

    Uses draw probability 0 and updates ratings in row order; image ids are normalized to stems
    (text before the first ``.``).

    :param pairs_df: DataFrame with columns ``left``, ``right``, ``choice`` (``"left"`` or ``"right"``).
    :return: Dict ``image_id_stem -> mu`` as floats.

    Example::
        In: fit_trueskill_large(pairs_df)
        Out: {"abc": 25.3, "def": 18.1, ...}
    """
    n_pairs = len(pairs_df)
    logger.info("TrueSkill: fitting on %d pairwise comparisons", n_pairs)
    # Pairwise urban prefs are win/loss only; ties are not modeled.
    trueskill_env = trueskill.TrueSkill(draw_probability=0)
    uniq_ids = pd.unique(pd.concat([pairs_df["left"], pairs_df["right"]]))
    ratings = {str(img_id).split(".")[0]: trueskill_env.create_rating() for img_id in uniq_ids}

    # Walk comparisons in file order; each row updates the two involved image ratings.
    for _, row in pairs_df.iterrows():
        l_id = str(row["left"]).split(".")[0]
        r_id = str(row["right"]).split(".")[0]
        if row["choice"] == "left":
            ratings[l_id], ratings[r_id] = trueskill.rate_1vs1(ratings[l_id], ratings[r_id], env=trueskill_env)
        else:
            # rate_1vs1 expects (winner, loser); swap ids when the right image wins.
            ratings[r_id], ratings[l_id] = trueskill.rate_1vs1(ratings[r_id], ratings[l_id], env=trueskill_env)
    out = {img_id: float(r.mu) for img_id, r in ratings.items()}
    logger.info("TrueSkill: finished with %d unique image ids", len(out))
    return out
