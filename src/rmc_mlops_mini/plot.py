"""Desenha o modelo salvo e as amostras da execução, sem treinar novamente."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Salva PNG no WSL mesmo sem janela gráfica.
import matplotlib.pyplot as plt


def plot_run(folder):
    folder = Path(folder)
    rows = json.loads((folder / "data.json").read_text())
    model = json.loads((folder / "model.json").read_text())
    run = json.loads((folder / "run.json").read_text())
    train_ids = set(run["split"]["train_ids"])
    train = [r for r in rows if r["id"] in train_ids]
    test = [r for r in rows if r["id"] not in train_ids]
    def predict(x):
        return model["intercept"] + model["slope"] * x
    xs = [min(r["carga_sintetica"] for r in rows), max(r["carga_sintetica"] for r in rows)]

    with plt.rc_context({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
        fig.subplots_adjust(left=0.11, right=0.96, top=0.78, bottom=0.18)
        fig.suptitle("A reta que o modelo aprendeu", fontsize=25, fontweight="bold", y=0.96)
        fig.text(0.5, 0.90, "Demonstração com 100 registros sintéticos • Não são métricas reais de VMs", ha="center", fontsize=13)
        equation = f"CPU prevista = {model['intercept']:.3f} + {model['slope']:.3f} × carga"
        fig.text(0.5, 0.85, equation, ha="center", fontsize=16, color="#0b7275")
        ax.scatter([r["carga_sintetica"] for r in train], [r["cpu_sintetica"] for r in train],
                   color="#617c96", s=37, alpha=0.7, label=f"Treino ({len(train)} pontos)")
        ax.scatter([r["carga_sintetica"] for r in test], [r["cpu_sintetica"] for r in test],
                   color="#db7928", marker="^", s=75, label=f"Teste ({len(test)} pontos)", zorder=3)
        ax.plot(xs, [predict(x) for x in xs], color="#0b7275", linewidth=3, label="Reta aprendida no treino")
        point = max(test, key=lambda r: abs(r["cpu_sintetica"] - predict(r["carga_sintetica"])))
        x = point["carga_sintetica"]
        ax.plot([x, x], [point["cpu_sintetica"], predict(x)], color="#8d2839", linewidth=2.5,
                linestyle="--", label="Erro de um ponto de teste", zorder=4)
        ax.set_xlabel("Carga sintética (unidades fictícias)", labelpad=12)
        ax.set_ylabel("CPU sintética (escala fictícia)", labelpad=12)
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=11, frameon=True)
        fig.text(0.11, 0.09, f"Semente da divisão: {run['parameters']['split_seed']}  |  MAE no teste: {run['metrics']['test_mae']:.4f}", fontsize=14)
        fig.text(0.11, 0.045, "RMC AI Labs  •  Predict. Optimize. Evolve.", fontsize=12, color="#5b6470")
        path = folder / "regressao.png"
        fig.savefig(path)
        plt.close(fig)
    return path
