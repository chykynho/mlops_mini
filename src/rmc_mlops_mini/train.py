"""Treina uma reta e registra o caminho do modelo. Somente dados sintéticos."""

import argparse
import hashlib
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .plot import plot_run


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_info(root):
    """Fora de um Git, registrar a ausência em vez de inventar um commit."""
    def command(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        commit = command("rev-parse", "HEAD")
        status = command("status", "--porcelain")
        return {"commit": commit, "dirty": bool(status) if status is not None else None}
    except FileNotFoundError:
        return {"commit": None, "dirty": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Semente da divisão treino/teste")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction deve estar entre 0 e 1")

    # A semente dos dados fica fixa: mudar --seed altera apenas a divisão.
    rng = random.Random(7)
    rows = [
        {"id": i, "carga_sintetica": i, "cpu_sintetica": round(10 + 0.6 * i + rng.uniform(-5, 5), 4)}
        for i in range(1, 101)
    ]
    data_bytes = json.dumps(rows, sort_keys=True, indent=2).encode()
    shuffled = rows.copy()
    random.Random(args.seed).shuffle(shuffled)
    cut = int(len(rows) * args.train_fraction)
    if cut < 2 or cut == len(rows):
        parser.error("A divisão precisa de pelo menos 2 itens no treino e 1 no teste")
    train, test = shuffled[:cut], shuffled[cut:]

    # Regressão linear com intercepto: aprender a reta somente com o treino.
    mean_x = sum(r["carga_sintetica"] for r in train) / len(train)
    mean_y = sum(r["cpu_sintetica"] for r in train) / len(train)
    slope = sum((r["carga_sintetica"] - mean_x) * (r["cpu_sintetica"] - mean_y) for r in train) / sum(
        (r["carga_sintetica"] - mean_x) ** 2 for r in train
    )
    intercept = mean_y - slope * mean_x
    predictions = [intercept + slope * r["carga_sintetica"] for r in test]
    mae = sum(abs(r["cpu_sintetica"] - p) for r, p in zip(test, predictions)) / len(test)
    # Baseline: prever sempre a média aprendida no treino.
    baseline_mae = sum(abs(r["cpu_sintetica"] - mean_y) for r in test) / len(test)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = args.output / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "data.json").write_bytes(data_bytes)
    model_bytes = json.dumps({"slope": slope, "intercept": intercept}, indent=2).encode()
    (output / "model.json").write_bytes(model_bytes)
    source = Path(__file__).resolve()
    record = {
        "run_id": run_id,
        "data_type": "synthetic_demo_not_real_vm_metrics",
        "data_sha256": sha256(data_bytes),
        "training_code_sha256": sha256(source.read_bytes()),
        "plotting_code_sha256": sha256(source.with_name("plot.py").read_bytes()),
        "git": git_info(Path.cwd()),
        "python": platform.python_version(),
        "platform": platform.system(),
        "parameters": {"data_seed": 7, "split_seed": args.seed, "train_fraction": args.train_fraction},
        "split": {"train_ids": [r["id"] for r in train], "test_ids": [r["id"] for r in test]},
        "metrics": {"test_mae": mae, "baseline_test_mae": baseline_mae},
        "model_sha256": sha256(model_bytes),
        "deployment": "not_performed",
    }
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    chart = plot_run(output)
    print(f"Execução: {run_id}")
    print(f"MAE teste: {mae:.4f} | baseline: {baseline_mae:.4f}")
    print(f"Registro: {output / 'run.json'}")
    print(f"Gráfico: {chart}")


if __name__ == "__main__":
    main()
