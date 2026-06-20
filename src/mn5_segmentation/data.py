from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

from .utils import IMAGE_EXTENSIONS


class DatasetSetupError(RuntimeError):
    pass


def _inspect_zip(zip_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile as exc:
        raise DatasetSetupError(f"Invalid zip archive: {zip_path}") from exc

    images = {
        Path(name).stem: name
        for name in names
        if "/images/" in name and Path(name).suffix.lower() in IMAGE_EXTENSIONS
    }
    masks = {
        Path(name).stem: name
        for name in names
        if "/masks/" in name and Path(name).suffix.lower() in IMAGE_EXTENSIONS
    }
    return images, masks


def _inspect_directory(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    images_dir = root / "images"
    masks_dir = root / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise DatasetSetupError(f"Expected images/ and masks/ inside {root}")

    images = {
        path.stem: path
        for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    }
    masks = {
        path.stem: path
        for path in masks_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    }
    return images, masks


def find_dataset_source(data_root: str | Path) -> dict:
    data_root = Path(data_root)
    zip_candidates = [
        data_root / "kvasir-seg.zip",
        data_root / "Kvasir-SEG.zip",
        Path("kvasir-seg.zip"),
    ]

    if data_root.is_file() and zipfile.is_zipfile(data_root):
        images, masks = _inspect_zip(data_root)
        return {"kind": "zip", "path": str(data_root), "images": images, "masks": masks}

    if data_root.is_file():
        raise DatasetSetupError(f"{data_root} is a file, not a dataset folder or zip archive")

    if data_root.is_dir():
        candidates = [data_root / "Kvasir-SEG", data_root / "kvasir-seg"]
        candidates.extend(path for path in data_root.iterdir() if path.is_dir())
        for candidate in candidates:
            if (candidate / "images").is_dir() and (candidate / "masks").is_dir():
                images, masks = _inspect_directory(candidate)
                return {"kind": "directory", "path": str(candidate), "images": images, "masks": masks}

    zip_path = next((path for path in zip_candidates if path.exists()), None)
    if zip_path is None:
        expected = " or ".join(str(path) for path in zip_candidates)
        raise DatasetSetupError(f"Dataset not found. Place Kvasir-SEG at {expected}")

    images, masks = _inspect_zip(zip_path)
    return {"kind": "zip", "path": str(zip_path), "images": images, "masks": masks}


def list_pairs(source: dict) -> list[str]:
    keys = sorted(set(source["images"]) & set(source["masks"]))
    if not keys:
        raise DatasetSetupError("No matching image/mask pairs found")
    return keys


class KvasirSegDataset:
    def __init__(self, source: dict, keys: list[str], image_size: int = 256):
        self.source = source
        self.keys = keys
        self.image_size = image_size

    def __len__(self):
        return len(self.keys)

    def _load_pair(self, key: str):
        from PIL import Image

        if self.source["kind"] == "directory":
            image = Image.open(self.source["images"][key]).convert("RGB")
            mask = Image.open(self.source["masks"][key]).convert("L")
            return image, mask

        zip_path = Path(self.source["path"])
        with zipfile.ZipFile(zip_path) as archive:
            image = Image.open(BytesIO(archive.read(self.source["images"][key]))).convert("RGB")
            mask = Image.open(BytesIO(archive.read(self.source["masks"][key]))).convert("L")
        return image, mask

    def __getitem__(self, index: int):
        import numpy as np
        from PIL import Image
        import torch

        key = self.keys[index]
        image, mask = self._load_pair(key)
        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return image_tensor, mask_tensor, key


def load_image_tensor(image_path: str | Path, image_size: int = 256) -> torch.Tensor:
    import numpy as np
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
