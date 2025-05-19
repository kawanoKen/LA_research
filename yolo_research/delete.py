from pathlib import Path
from typing import Iterable, Sequence

def remove_orphan_images(
    img_dir: str | Path,
    lbl_dir: str | Path,
    img_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".txt"),
    lbl_ext: str = ".txt",
    dry_run: bool = False,
) -> list[Path]:
    """
    指定フォルダ内でラベルファイルが存在しない画像ファイルを削除する。

    Parameters
    ----------
    img_dir : str | Path
        画像ファイルを含むディレクトリ（再帰的に探索）
    lbl_dir : str | Path
        ラベルファイルを含むディレクトリ（再帰的に探索）
    img_exts : Sequence[str], optional
        対象とする画像拡張子（"." を含む小文字で指定）
    lbl_ext : str, optional
        ラベルファイルの拡張子
    dry_run : bool, optional
        True の場合は削除せず、対象ファイルの一覧だけ返す

    Returns
    -------
    list[Path]
        削除（または削除予定）となった画像ファイルのパス一覧
    """
    img_dir = Path(img_dir)
    lbl_dir = Path(lbl_dir)

    # ラベルファイル（拡張子を除いた stem）の集合を構築
    label_stems: set[str] = {
        f.stem for f in lbl_dir.rglob(f"*{lbl_ext}") if f.is_file()
    }

    # orphan = 対応ラベルが存在しない画像
    orphan_imgs: list[Path] = [
        img
        for ext in img_exts
        for img in img_dir.rglob(f"*{ext}")
        if img.is_file() and img.stem not in label_stems
    ]

    for img in orphan_imgs:
        if dry_run:
            print("[DRY-RUN] would remove:", img)
        else:
            img.unlink()
            print("removed:", img)

    return orphan_imgs


# 本当に削除せず確認だけ
remove_orphan_images(
    img_dir="/work/kawano/LA/datasets/african-wildlife_noise/train/images",
    lbl_dir="/work/kawano/LA/datasets/african-wildlife_noise/train/labels",
    dry_run=True,
)

# 削除を実行
remove_orphan_images(
    img_dir="/work/kawano/LA/datasets/african-wildlife_noise/train/images",
    lbl_dir="/work/kawano/LA/datasets/african-wildlife_noise/train/labels",
)