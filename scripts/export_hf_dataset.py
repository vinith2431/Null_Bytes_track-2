
import csv
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "hf_prompt_injection"
DATASET_NAME = "neuralchemy/Prompt-injection-dataset"
CONFIG_NAME = "core"

FIELDS = [
    "text",
    "label",
    "category",
    "source",
    "severity",
    "group_id",
    "augmented",
    "tags",
]


def main():
    dataset = load_dataset(DATASET_NAME, CONFIG_NAME)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    expected_splits = {"train", "validation", "test"}
    if not expected_splits.issubset(dataset.keys()):
        raise ValueError(
            f"Missing expected splits. Found: {list(dataset.keys())}"
        )

    for split in ("train", "validation", "test"):
        rows = dataset[split]
        output_file = OUTPUT_DIR / f"{split}.csv"
        class_counts = {0: 0, 1: 0}

        with output_file.open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()

            for row in rows:
                text = row["text"]
                label = int(row["label"])

                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"Empty/invalid text in {split}"
                    )
                if label not in (0, 1):
                    raise ValueError(
                        f"Unexpected label {label} in {split}"
                    )

                output_row = {
                    field: row.get(field, "")
                    for field in FIELDS
                }
                output_row["label"] = label
                output_row["tags"] = json.dumps(
                    output_row["tags"], ensure_ascii=False
                )
                writer.writerow(output_row)
                class_counts[label] += 1

        print(
            f"{split}: {len(rows)} rows | "
            f"benign={class_counts[0]} | "
            f"suspicious={class_counts[1]}"
        )
        print(f"  Saved: {output_file}")

    print("\nExport complete. Original labels and splits preserved.")


if __name__ == "__main__":
    main()