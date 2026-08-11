"""Create a local, ordered archive of MIT 6.S191 student lab materials.

The MITDeepLearning/introtodeeplearning repository is licensed under MIT. This
script preserves source attribution and excludes solution notebooks, images, and
large data assets. It imports only student-facing notebooks, lab READMEs, and
support Python modules from official year branches (2018 through 2026).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path


REPOSITORY = "https://github.com/MITDeepLearning/introtodeeplearning.git"
SOURCE_URL = "https://github.com/MITDeepLearning/introtodeeplearning"
YEARS = ("2018", "2019", "2020", "2021", "2022_archive", "2022", "2023", "2024", "2025", "2026")
DESTINATION = Path(__file__).parent / "mit_6s191_labs"
TOPICS = (
    "01_foundations_and_tensorflow",
    "02_sequence_modeling_and_music_generation",
    "03_computer_vision_and_mnist",
    "04_responsible_ai_debiasing_and_face_detection",
    "05_reinforcement_learning",
    "06_llm_finetuning",
    "07_uncertainty_and_ai_for_science",
)
TOPIC_RULES = (
    ("llm_finetun", "06_llm_finetuning", "lab_01_llm_finetuning"),
    ("part1_introductioncapsa", "07_uncertainty_and_ai_for_science", "lab_01_capsa_introduction"),
    ("uncertainty", "07_uncertainty_and_ai_for_science", "lab_02_bias_and_uncertainty"),
    ("rl_pong", "05_reinforcement_learning", "lab_02_pong"),
    ("rl_selfdriving", "05_reinforcement_learning", "lab_03_self_driving"),
    ("lab3_rl", "05_reinforcement_learning", "lab_01_reinforcement_learning_foundations"),
    ("lab3/rl", "05_reinforcement_learning", "lab_01_reinforcement_learning_foundations"),
    ("lab3_old/rl", "05_reinforcement_learning", "lab_01_reinforcement_learning_foundations"),
    ("facedetection", "04_responsible_ai_debiasing_and_face_detection", "lab_02_face_detection"),
    ("debiasing", "04_responsible_ai_debiasing_and_face_detection", "lab_01_debiasing"),
    ("capsa", "07_uncertainty_and_ai_for_science", "lab_02_bias_and_uncertainty"),
    ("mnist", "03_computer_vision_and_mnist", "lab_01_mnist_classification"),
    ("lab2_part", "03_computer_vision_and_mnist", "lab_02_medical_imaging"),
    ("music_generation", "02_sequence_modeling_and_music_generation", "lab_01_music_generation"),
    ("lab1_part2", "02_sequence_modeling_and_music_generation", "lab_01_music_generation"),
    ("tensorflow", "01_foundations_and_tensorflow", "lab_01_neural_network_foundations"),
    ("part1_intro", "01_foundations_and_tensorflow", "lab_01_neural_network_foundations"),
    ("lab1_part1", "01_foundations_and_tensorflow", "lab_01_neural_network_foundations"),
)


def run(*arguments: str, cwd: Path | None = None) -> str:
    for attempt in range(2):
        try:
            return subprocess.run(arguments, cwd=cwd, check=True, capture_output=True, text=True).stdout
        except subprocess.CalledProcessError:
            if attempt == 1:
                raise
            time.sleep(1)
    raise RuntimeError("Unreachable")


def is_student_file(path: str) -> bool:
    parts = Path(path).parts
    filename = Path(path).name.lower()
    if "solutions" in parts or "img" in parts or "data" in parts or "solution" in filename:
        return False
    return path.endswith(".ipynb")


def display_year(branch: str) -> str:
    return "2022 archive" if branch == "2022_archive" else branch


def topic_and_lab_for_path(path: str) -> tuple[str, str]:
    normalized_path = path.lower()
    for marker, topic, lab in TOPIC_RULES:
        if marker in normalized_path:
            return topic, lab
    raise ValueError(f"No topic mapping exists for notebook: {path}")


def canonical_variant(variants: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    """Prefer the newest TensorFlow variant, then the newest available lab."""

    def priority(variant: tuple[str, str, str]) -> tuple[int, int]:
        branch, path, _ = variant
        year = 2022 if branch == "2022_archive" else int(branch)
        tensorflow = int("tf_" in path.lower() or "tensorflow" in path.lower())
        return tensorflow, year

    return max(variants, key=priority)


def write_lab_manifest(lab_path: Path, topic: str, lab: str, branch: str, path: str) -> None:
    rows = [
        f"# {lab}",
        "",
        f"Topic: `{topic}`",
        "",
        "This folder contains one canonical MIT 6.S191 assignment notebook. It was selected",
        "from the newest available official course variant, preferring TensorFlow/Keras when",
        "MIT provides both TensorFlow and PyTorch editions.",
        "",
        "| Local file | Official year branch | Original MIT path |",
        "|---|---|---|",
    ]
    local_name = f"assignment{Path(path).suffix}"
    rows.append(f"| [{local_name}]({local_name}) | {display_year(branch)} | `{path}` |")
    lab_path.joinpath("SOURCE.md").write_text("\n".join(rows) + "\n")


def write_catalog(entries: list[tuple[str, str, str, str]]) -> None:
    rows = [
        "# MIT 6.S191 Lab Archive (2018-2026)",
        "",
        "This directory contains student-facing notebooks, lab READMEs, and support Python files",
        "from the official [MIT 6.S191 repository](https://github.com/MITDeepLearning/introtodeeplearning).",
        "It deliberately excludes completed solution notebooks, images, and data assets. Use the official",
        "repository for full datasets and any updated instructions. Original materials are licensed by MIT",
        "DeepLearning under the repository's MIT License.",
        "",
        "## Study Order",
        "",
        "1. `01_foundations_and_tensorflow`",
        "2. `02_sequence_modeling_and_music_generation`",
        "3. `03_computer_vision_and_mnist`",
        "4. `04_responsible_ai_debiasing_and_face_detection`",
        "5. `05_reinforcement_learning`",
        "6. `06_llm_finetuning`",
        "7. `07_uncertainty_and_ai_for_science`",
        "",
        "Each folder contains one canonical assignment notebook selected from the newest available MIT course version, preferring TensorFlow/Keras when both TensorFlow and PyTorch editions exist. `SOURCE.md` records its official source.",
        "",
        "## Imported Assignments by Topic",
        "",
        "| Topic | Direct labs |",
        "|---|---|",
    ]
    for topic in TOPICS:
        labs = sorted({lab for item_topic, lab, _, _ in entries if item_topic == topic})
        links = "<br>".join(f"[{lab}]({topic}/{lab}/SOURCE.md)" for lab in labs)
        rows.append(f"| `{topic}` | {links or 'No student files found'} |")
    rows.extend(
        [
            "",
            "## Official Source",
            "",
            f"- Repository: {SOURCE_URL}",
            f"- Branches included: {', '.join(YEARS)}",
            "- To refresh the archive, run `python sync_mit_6s191_labs.py` from the workspace root.",
        ]
    )
    (DESTINATION / "README.md").write_text("\n".join(rows) + "\n")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mit-6s191-") as temporary_directory:
        repository_path = Path(temporary_directory) / "introtodeeplearning"
        run("git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(repository_path))

        if DESTINATION.exists():
            shutil.rmtree(DESTINATION)
        DESTINATION.mkdir()
        grouped_files: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)

        for branch in YEARS:
            paths = run("git", "ls-tree", "-r", "--name-only", f"origin/{branch}", cwd=repository_path).splitlines()
            selected_paths = [path for path in paths if is_student_file(path)]
            for path in selected_paths:
                contents = run("git", "show", f"origin/{branch}:{path}", cwd=repository_path)
                topic, lab = topic_and_lab_for_path(path)
                grouped_files[(topic, lab)].append((branch, path, contents))
            print(f"Imported {len(selected_paths):2d} student-facing files from {display_year(branch)}.")

        entries: list[tuple[str, str, str, str]] = []
        for (topic, lab), variants in grouped_files.items():
            lab_path = DESTINATION / topic / lab
            lab_path.mkdir(parents=True, exist_ok=True)
            branch, path, contents = canonical_variant(variants)
            target = lab_path / f"assignment{Path(path).suffix}"
            target.write_text(contents)
            entries.append((topic, lab, branch, path))
            write_lab_manifest(lab_path, topic, lab, branch, path)

        write_catalog(entries)
        print(f"Created {DESTINATION.relative_to(Path(__file__).parent)} with {len(entries)} source files.")


if __name__ == "__main__":
    main()