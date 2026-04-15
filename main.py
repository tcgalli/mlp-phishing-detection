"""
============================================================
  MLP Phishing Detector — Experimento Completo
  Disciplina: Redes Neurais Artificiais
============================================================
  Dataset  : Phishing Websites (UCI ML Repository)
  Modelo   : MLPClassifier (scikit-learn)
  Parâmetros variados:
    1. Quantidade de Neurônios por camada
    2. Taxa de Aprendizado Inicial
    3. Momentum
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os, time, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# CONFIGURAÇÃO GERAL
# ─────────────────────────────────────────────
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
VAL_SIZE      = 0.10          # validação extraída do treino
OUTPUT_DIR    = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo visual
PALETTE = {
    "bg":       "#0D1117",
    "surface":  "#161B22",
    "border":   "#30363D",
    "accent1":  "#58A6FF",
    "accent2":  "#3FB950",
    "accent3":  "#FF7B72",
    "accent4":  "#D2A8FF",
    "text":     "#E6EDF3",
    "subtext":  "#8B949E",
}
plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["subtext"],
    "ytick.color":       PALETTE["subtext"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["border"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "font.size":         10,
})

# ─────────────────────────────────────────────
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
def load_data():
    """
    Carrega o dataset Phishing Websites (UCI ML Repository ID=327).
    Tenta primeiro via ucimlrepo; fallback para 'phishing.csv' local.
    Dataset: 11055 amostras, 30 atributos binários/ternários.
    Classes: -1 (phishing), 0 (suspeito), 1 (legítimo)
             → binarizado: 1 = legítimo, 0 = phishing/suspeito
    """
    print("📦 Carregando dataset...")
    csv_path = os.path.join(os.path.dirname(__file__), "phishing.csv")

    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=327)
        X = ds.data.features
        y = ds.data.targets.squeeze()
        print(f"   ✔ UCI ML Repository  |  {X.shape[0]} amostras, {X.shape[1]} atributos")
    except Exception:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            target_col = "Result"
            X = df.drop(columns=[target_col])
            y = df[target_col]
            print(f"   ✔ CSV local          |  {X.shape[0]} amostras, {X.shape[1]} atributos")
        else:
            raise RuntimeError(
                "Dataset não encontrado.\n"
                "Baixe em: https://archive.ics.uci.edu/dataset/327/phishing+websites\n"
                "e salve como 'phishing.csv' no mesmo diretório."
            )

    # Binarização: 1 → legítimo (1), -1/0 → phishing (0)
    y = (y == 1).astype(int)
    print(f"   Classes  : phishing={( y==0 ).sum()}  legítimo={( y==1 ).sum()}")
    return X, y


def preprocess(X, y):
    """Divisão estratificada treino/val/teste + normalização."""
    # Treino+val  /  Teste
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # Treino  /  Validação
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"\n📐 Divisão dos dados:")
    print(f"   Treino     : {len(X_train):>5} amostras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validação  : {len(X_val):>5} amostras ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Teste      : {len(X_test):>5} amostras ({len(X_test)/len(X)*100:.1f}%)")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ─────────────────────────────────────────────
# 2. GRADE DE HIPERPARÂMETROS  (3³ = 27)
# ─────────────────────────────────────────────
PARAM_GRID = {
    "neurons":       [32, 64, 128],      # neurônios por camada oculta
    "learning_rate": [0.001, 0.01, 0.1], # taxa de aprendizado inicial
    "momentum":      [0.5, 0.8, 0.9],   # momentum (SGD)
}

FIXED = {
    "hidden_layer_sizes_fn": lambda n: (n, n),   # 2 camadas ocultas fixas
    "max_iter":              500,
    "activation":           "relu",
    "solver":               "sgd",
    "early_stopping":        True,
    "validation_fraction":   0.1,
    "n_iter_no_change":      20,
    "random_state":          RANDOM_STATE,
}


def build_model(neurons, lr, momentum):
    return MLPClassifier(
        hidden_layer_sizes=FIXED["hidden_layer_sizes_fn"](neurons),
        activation=FIXED["activation"],
        solver=FIXED["solver"],
        learning_rate_init=lr,
        momentum=momentum,
        max_iter=FIXED["max_iter"],
        early_stopping=FIXED["early_stopping"],
        validation_fraction=FIXED["validation_fraction"],
        n_iter_no_change=FIXED["n_iter_no_change"],
        random_state=FIXED["random_state"],
        verbose=False,
    )


def metrics(y_true, y_pred, prefix=""):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


# ─────────────────────────────────────────────
# 3. BUSCA EM GRADE (27 EXPERIMENTOS)
# ─────────────────────────────────────────────
def run_grid_search(X_train, X_val, X_test, y_train, y_val, y_test):
    combos = list(itertools.product(
        PARAM_GRID["neurons"],
        PARAM_GRID["learning_rate"],
        PARAM_GRID["momentum"],
    ))
    assert len(combos) == 27, "Deve haver exatamente 27 combinações."

    records     = []
    loss_curves = {}

    print(f"\n🔬 Grid Search — {len(combos)} experimentos\n")
    print(f"{'#':>3}  {'Neurons':>7}  {'LR':>7}  {'Mom':>5}  {'Acc':>6}  {'F1':>6}  {'Iter':>5}")
    print("─" * 56)

    for idx, (n, lr, mom) in enumerate(combos, 1):
        model = build_model(n, lr, mom)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred_val  = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        m_val  = metrics(y_val,  y_pred_val)
        m_test = metrics(y_test, y_pred_test)

        records.append({
            "exp":      idx,
            "neurons":  n,
            "lr":       lr,
            "momentum": mom,
            "n_iter":   model.n_iter_,
            "time_s":   round(elapsed, 2),
            # val
            "val_acc":  m_val["acc"],
            "val_prec": m_val["prec"],
            "val_rec":  m_val["rec"],
            "val_f1":   m_val["f1"],
            # test
            "test_acc":  m_test["acc"],
            "test_prec": m_test["prec"],
            "test_rec":  m_test["rec"],
            "test_f1":   m_test["f1"],
        })

        loss_curves[idx] = {
            "loss":        model.loss_curve_,
            "val_loss":    model.validation_scores_,   # accuracy on val
            "params":      (n, lr, mom),
        }

        print(f"{idx:>3}  {n:>7}  {lr:>7.3f}  {mom:>5.2f}  "
              f"{m_val['acc']:>6.4f}  {m_val['f1']:>6.4f}  {model.n_iter_:>5}")

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/grid_results.csv", index=False)
    print(f"\n✅ Resultados salvos em '{OUTPUT_DIR}/grid_results.csv'")
    return df, loss_curves


# ─────────────────────────────────────────────
# 4. TESTE DE ROBUSTEZ (5 seeds)
# ─────────────────────────────────────────────
def run_robustness(best_row, X_train, X_test, y_train, y_test, n_runs=5):
    n   = int(best_row["neurons"])
    lr  = float(best_row["lr"])
    mom = float(best_row["momentum"])

    print(f"\n🧪 Teste de Robustez  |  neurons={n}, lr={lr}, momentum={mom}")
    print(f"   {n_runs} execuções com seeds aleatórias distintas\n")
    print(f"{'Run':>4}  {'Acc':>8}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}")
    print("─" * 48)

    records = []
    for run in range(1, n_runs + 1):
        seed = run * 13           # seeds distintas: 13, 26, 39, 52, 65
        mdl = MLPClassifier(
            hidden_layer_sizes=FIXED["hidden_layer_sizes_fn"](n),
            activation=FIXED["activation"],
            solver=FIXED["solver"],
            learning_rate_init=lr,
            momentum=mom,
            max_iter=FIXED["max_iter"],
            early_stopping=FIXED["early_stopping"],
            validation_fraction=FIXED["validation_fraction"],
            n_iter_no_change=FIXED["n_iter_no_change"],
            random_state=seed,
            verbose=False,
        )
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        m = metrics(y_test, y_pred)
        records.append({**m, "seed": seed, "n_iter": mdl.n_iter_})
        print(f"{run:>4}  {m['acc']:>8.4f}  {m['prec']:>8.4f}  {m['rec']:>8.4f}  {m['f1']:>8.4f}")

    df_rob = pd.DataFrame(records)
    print("\n" + "─" * 48)
    for col in ["acc", "prec", "rec", "f1"]:
        mu = df_rob[col].mean()
        sd = df_rob[col].std()
        label = {"acc":"Acurácia","prec":"Precisão","rec":"Recall","f1":"F1-Score"}[col]
        print(f"   {label:<10}: {mu:.4f} ± {sd:.4f}")

    df_rob.to_csv(f"{OUTPUT_DIR}/robustness_results.csv", index=False)
    return df_rob


# ─────────────────────────────────────────────
# 5. VISUALIZAÇÕES
# ─────────────────────────────────────────────

def _set_spine(ax):
    for s in ax.spines.values():
        s.set_edgecolor(PALETTE["border"])


def plot_loss_curves(loss_curves, df_grid, top_n=9):
    """Plota as curvas de perda dos top-N experimentos por val_f1."""
    top_ids = df_grid.nlargest(top_n, "val_f1")["exp"].tolist()
    cols = 3
    rows = top_n // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Curvas de Convergência — Top-9 por F1 (Validação)",
                 color=PALETTE["text"], fontsize=14, fontweight="bold", y=1.01)

    for ax, exp_id in zip(axes.flat, top_ids):
        curve = loss_curves[exp_id]
        n, lr, mom = curve["params"]
        loss = curve["loss"]
        epochs = range(1, len(loss) + 1)

        ax.plot(epochs, loss, color=PALETTE["accent1"], lw=1.8, label="Train Loss")
        ax.set_title(f"Exp #{exp_id}  n={n} lr={lr} mom={mom}",
                     color=PALETTE["subtext"], fontsize=9)
        ax.set_xlabel("Época"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(True); _set_spine(ax)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/loss_curves_top9.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"   📈 {path}")


def plot_grid_heatmaps(df_grid):
    """Heatmaps: F1 médio para cada par de parâmetros."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Busca em Grade — F1 Médio (Validação)",
                 color=PALETTE["text"], fontsize=14, fontweight="bold")

    pairs = [
        ("neurons", "lr",       "Neurônios × Taxa Aprendizado"),
        ("neurons", "momentum", "Neurônios × Momentum"),
        ("lr",      "momentum", "Taxa Aprendizado × Momentum"),
    ]
    cmap = sns.color_palette("mako", as_cmap=True)

    for ax, (row_p, col_p, title) in zip(axes, pairs):
        pivot = df_grid.groupby([row_p, col_p])["val_f1"].mean().unstack()
        sns.heatmap(
            pivot, ax=ax, cmap=cmap, annot=True, fmt=".4f",
            linewidths=0.5, linecolor=PALETTE["border"],
            annot_kws={"size": 9, "color": PALETTE["text"]},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, color=PALETTE["text"], fontsize=11)
        ax.set_facecolor(PALETTE["surface"])
        _set_spine(ax)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/grid_heatmaps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"   🗺  {path}")


def plot_robustness(df_rob):
    """Box + strip plot das métricas de robustez."""
    metrics_cols = ["acc", "prec", "rec", "f1"]
    labels       = ["Acurácia", "Precisão", "Recall", "F1-Score"]
    colors       = [PALETTE["accent1"], PALETTE["accent2"],
                    PALETTE["accent3"], PALETTE["accent4"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    positions = np.arange(len(metrics_cols))
    for i, (col, lbl, clr) in enumerate(zip(metrics_cols, labels, colors)):
        vals = df_rob[col].values
        bp = ax.boxplot(
            vals, positions=[i], widths=0.35,
            patch_artist=True, notch=False,
            boxprops    =dict(facecolor=clr + "44", color=clr, lw=1.5),
            medianprops =dict(color=clr, lw=2.5),
            whiskerprops=dict(color=clr, lw=1.2, linestyle="--"),
            capprops    =dict(color=clr, lw=1.5),
            flierprops  =dict(marker="o", color=clr, alpha=0.5),
        )
        # jitter strip
        jitter = np.random.default_rng(0).uniform(-0.05, 0.05, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=clr, s=50, zorder=5, alpha=0.85)
        mu, sd = vals.mean(), vals.std()
        ax.text(i, vals.min() - 0.003, f"μ={mu:.4f}\nσ={sd:.4f}",
                ha="center", va="top", color=clr, fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Teste de Robustez — 5 Seeds Distintas (Melhor Config.)",
                 color=PALETTE["text"], fontsize=12, fontweight="bold")
    ax.grid(True, axis="y"); _set_spine(ax)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/robustness_boxplot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"   📦 {path}")


def plot_confusion_matrix(y_test, y_pred, title="Melhor Modelo — Matriz de Confusão"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Phishing", "Legítimo"],
        yticklabels=["Phishing", "Legítimo"],
        linewidths=1, linecolor=PALETTE["border"],
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    ax.set_title(title, color=PALETTE["text"], fontsize=12, fontweight="bold")
    _set_spine(ax)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"   🟦 {path}")


def plot_top27_bar(df_grid):
    """Bar chart de F1 dos 27 experimentos ordenados."""
    df_sorted = df_grid.sort_values("val_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    bars = ax.barh(
        [f"#{r['exp']} n={r['neurons']} lr={r['lr']} m={r['momentum']}"
         for _, r in df_sorted.iterrows()],
        df_sorted["val_f1"],
        color=[PALETTE["accent2"] if v == df_sorted["val_f1"].max()
               else PALETTE["accent1"] + "99"
               for v in df_sorted["val_f1"]],
        edgecolor=PALETTE["border"], height=0.7,
    )
    ax.axvline(df_sorted["val_f1"].max(), color=PALETTE["accent2"],
               lw=1.5, linestyle="--", alpha=0.7)
    ax.set_xlabel("F1-Score (Validação)")
    ax.set_title("27 Experimentos — F1 por Configuração",
                 color=PALETTE["text"], fontsize=13, fontweight="bold")
    ax.grid(True, axis="x"); _set_spine(ax)

    # Label no melhor
    best_val = df_sorted["val_f1"].max()
    ax.text(best_val + 0.0002, len(df_sorted) - 0.8,
            f"  ★ {best_val:.4f}", color=PALETTE["accent2"],
            fontsize=9, va="center")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/all27_f1_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"   📊 {path}")


def plot_best_loss_full(best_model_loss, best_row):
    """Curva de perda completa do melhor modelo."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    epochs = range(1, len(best_model_loss) + 1)
    ax.plot(epochs, best_model_loss, color=PALETTE["accent1"], lw=2, label="Train Loss")
    ax.fill_between(epochs, best_model_loss, alpha=0.15, color=PALETTE["accent1"])

    ax.set_xlabel("Época"); ax.set_ylabel("Loss (Cross-Entropy)")
    n, lr, mom = int(best_row['neurons']), float(best_row['lr']), float(best_row['momentum'])
    ax.set_title(f"Convergência — Melhor Modelo  (n={n}, lr={lr}, mom={mom})",
                 color=PALETTE["text"], fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True); _set_spine(ax)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/best_model_loss_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"   📉 {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    sep = "=" * 60
    print(sep)
    print("  MLP PHISHING DETECTOR — Experimento Completo")
    print(sep)

    # 1. Dados
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess(X, y)

    # 2. Grid Search
    df_grid, loss_curves = run_grid_search(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # 3. Melhor configuração (por val_f1)
    best_row = df_grid.loc[df_grid["val_f1"].idxmax()]
    print(f"\n🏆 Melhor configuração:")
    print(f"   Exp #{int(best_row['exp'])}  |  "
          f"neurons={int(best_row['neurons'])}  "
          f"lr={float(best_row['lr'])}  "
          f"momentum={float(best_row['momentum'])}")
    print(f"   Val  → acc={best_row['val_acc']:.4f}  f1={best_row['val_f1']:.4f}")
    print(f"   Test → acc={best_row['test_acc']:.4f}  f1={best_row['test_f1']:.4f}")

    # Retreinar melhor modelo para obter predições finais
    best_model = build_model(
        int(best_row["neurons"]), float(best_row["lr"]), float(best_row["momentum"])
    )
    best_model.fit(X_train, y_train)
    y_pred_final = best_model.predict(X_test)

    print("\n📋 Relatório de Classificação (Teste):")
    print(classification_report(y_test, y_pred_final,
                                 target_names=["Phishing", "Legítimo"]))

    # 4. Robustez
    df_rob = run_robustness(best_row, X_train, X_test, y_train, y_test)

    # 5. Gráficos
    print(f"\n🎨 Gerando visualizações em '{OUTPUT_DIR}/'...")
    plot_loss_curves(loss_curves, df_grid, top_n=9)
    plot_grid_heatmaps(df_grid)
    plot_robustness(df_rob)
    plot_confusion_matrix(y_test, y_pred_final)
    plot_top27_bar(df_grid)
    plot_best_loss_full(best_model.loss_curve_, best_row)

    print(f"\n{sep}")
    print("  ✅ Experimento finalizado!")
    print(f"  📁 Todos os resultados estão em: ./{OUTPUT_DIR}/")
    print(sep)


if __name__ == "__main__":
    main()