import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PlacePulse(Dataset):
    """
    PyTorch ``Dataset`` for pairwise image comparisons with multi-category TrueSkill targets.

    Resolves image files under ``images_root`` (optionally under ``final_photo_dataset``),
    filters pairs whose stems exist on disk, and returns tensors plus category index,
    preference sign, and auxiliary score vectors per image.

    Example::
        In: PlacePulse(train_df, "/data/images", transform, ts_maps)
        Out: dataset yielding (img_l, img_r, cat_idx, y_rank, l_aux, r_aux, l_id, r_id)
    """

    def __init__(
        self,
        pairs_df: pd.DataFrame,
        images_root: Path | str,
        transform: Callable[[Image.Image], torch.Tensor],
        ts_maps: Mapping[str, Mapping[str, float]],
    ) -> None:
        """
        Build index of image paths and filter pairs to rows with both images present.

        :param pairs_df: DataFrame with columns ``study_question``, ``left``, ``right``, ``choice``.
        :param images_root: Root directory to search for ``.jpg`` / ``.jpeg`` / ``.png`` files.
        :param transform: torchvision transform applied to both left and right PIL images.
        :param ts_maps: Per-category dicts ``image_id_stem -> TrueSkill mu`` (six categories).
        :return: None.

        Example::
            In: PlacePulse(df, Path("data/img"), val_tf, {"safer": {...}, ...})
            Out: PlacePulse instance ready for DataLoader
        """
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.img_root = Path(images_root)
        self.transform = transform
        self.ts_maps = ts_maps
        self.cat_to_idx = {
            "safer": 0,
            "wealthier": 1,
            "more beautiful": 2,
            "livelier": 3,
            "less depressing": 4,
            "less boring": 5,
        }

        # Prefer nested layout when present (e.g. some dataset bundles); else search the root.
        search_root = (
            self.img_root / "final_photo_dataset" if (self.img_root / "final_photo_dataset").exists() else self.img_root
        )
        # Stem -> path so CSV ids can include extensions; we match on basename without suffix.
        self.id_to_path = {f.stem: f for f in search_root.rglob("*.*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")}

        # Drop pairs if either image file is missing under search_root.
        pair_mask = self.pairs_df["left"].astype(str).str.split(".").str[0].isin(self.id_to_path) & self.pairs_df[
            "right"
        ].astype(str).str.split(".").str[0].isin(self.id_to_path)
        n_before = len(self.pairs_df)
        self.pairs_df = self.pairs_df[pair_mask].reset_index(drop=True)
        n_after = len(self.pairs_df)
        n_drop = n_before - n_after
        # One log line whether we dropped rows or not (helps debug empty datasets).
        if n_drop:
            logger.info(
                "PlacePulse: %d -> %d pairs after image path filter (%d dropped, missing files under %s)",
                n_before,
                n_after,
                n_drop,
                search_root,
            )
        else:
            logger.info("PlacePulse: %d pairs, all image paths resolved under %s", n_after, search_root)

    def __len__(self) -> int:
        """
        Return the number of pairwise samples after filtering.

        :return: Length of the internal filtered ``pairs_df``.

        Example::
            In: len(dataset)
            Out: 12000  # example count
        """
        return len(self.pairs_df)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, float, torch.Tensor, torch.Tensor, str, str]:
        """
        Load one pair, apply transforms, and attach category index and auxiliary scores.

        :param idx: Row index into the filtered ``pairs_df``.
        :return: Tuple ``(img_l, img_r, cat_idx, y_rank, l_aux, r_aux, l_id, r_id)`` — tensors and string ids.

        Example::
            In: dataset[0]
            Out: (tensor, tensor, 0, 1.0, tensor(...), tensor(...), "img1", "img2")
        """
        row = self.pairs_df.iloc[idx]
        l_id, r_id = str(row["left"]).split(".")[0], str(row["right"]).split(".")[0]
        cat = str(row["study_question"]).strip().lower()

        # Six global TrueSkill mus per image (NaN if that image never appeared in that category's votes).
        l_aux_scores = np.array([self.ts_maps[c].get(l_id, np.nan) for c in self.cat_to_idx.keys()])
        r_aux_scores = np.array([self.ts_maps[c].get(r_id, np.nan) for c in self.cat_to_idx.keys()])

        # Load RGB, apply the same transform to both sides, return tensors + ids for eval aggregation.
        img_l = Image.open(self.id_to_path[l_id]).convert("RGB")
        img_r = Image.open(self.id_to_path[r_id]).convert("RGB")

        return (
            self.transform(img_l),
            self.transform(img_r),
            self.cat_to_idx[cat],
            float(1.0 if row["choice"] == "left" else -1.0),
            torch.tensor(l_aux_scores, dtype=torch.float32),
            torch.tensor(r_aux_scores, dtype=torch.float32),
            l_id,
            r_id,
        )
