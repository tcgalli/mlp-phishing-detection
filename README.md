# 🛡️ MLP Phishing Detector

Projeto de implementação e análise experimental de uma **Rede Neural Multilayer Perceptron (MLP)** para detecção de sites de phishing.

> Disciplina: Redes Neurais Artificiais

---

## 📌 Visão Geral

O objetivo é treinar uma rede neural capaz de classificar se uma URL/site é **legítima** ou um **ataque de phishing**, com base em 30 atributos técnicos extraídos da URL e da página (comprimento da URL, presença de HTTPS, uso de IP, subdomínios, etc.).

**Dataset:** [Phishing Websites — UCI ML Repository (ID: 327)](https://archive.ics.uci.edu/dataset/327/phishing+websites)  
**Amostras:** 11.055 | **Atributos:** 30 | **Classes:** Phishing / Legítimo

---

## 🔬 Metodologia Experimental

### Parâmetros variados (3 de 5 exigidos)

| Parâmetro | Valores testados |
|---|---|
| Neurônios por camada | 32 · 64 · 128 |
| Taxa de aprendizado inicial | 0.001 · 0.01 · 0.1 |
| Momentum | 0.5 · 0.8 · 0.9 |

**Fixados:** 2 camadas ocultas · solver=SGD · ativação=ReLU · máx. 500 épocas · early stopping

### Etapas

1. **Busca em Grade** — 3³ = **27 treinamentos** com todas as combinações
2. **Análise de Convergência** — curvas de perda para cada experimento
3. **Seleção do melhor modelo** — maior F1-Score na validação
4. **Teste de Robustez** — 5 execuções com seeds distintas (μ ± σ)

---

## 🏆 Melhor Configuração Encontrada

| Parâmetro | Valor |
|---|---|
| Neurônios | 128 |
| Taxa de Aprendizado | 0.1 |
| Momentum | 0.5 |

**Resultados no conjunto de teste:**

| Métrica | Valor |
|---|---|
| Acurácia | 0.8358 ± 0.0031 |
| Precisão | 0.8463 ± 0.0112 |
| Recall | 0.8605 ± 0.0165 |
| F1-Score | 0.8532 ± 0.0039 |

---

## 🛠️ Tecnologias

- **Python 3.x**
- **Scikit-Learn** — motor da MLP (`MLPClassifier`)
- **Pandas & NumPy** — manipulação de dados
- **Matplotlib & Seaborn** — visualizações
- **ucimlrepo** — acesso automático ao dataset UCI

---

## 🚀 Como Executar

**1. Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/mlp-phishing-detection.git
cd mlp-phishing-detection
```

**2. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**3. (Opcional) Obtenha o dataset:**

O script tenta baixar automaticamente via `ucimlrepo`. Se não funcionar, baixe manualmente em:
https://archive.ics.uci.edu/dataset/327/phishing+websites

Salve o arquivo como `phishing.csv` na mesma pasta do `main.py`.

**4. Execute:**
```bash
python main.py
```

---

## 📊 Saídas Geradas (`results/`)

| Arquivo | Descrição |
|---|---|
| `grid_results.csv` | Métricas dos 27 experimentos |
| `robustness_results.csv` | Resultados das 5 seeds |
| `all27_f1_bar.png` | Ranking de F1 dos 27 experimentos |
| `grid_heatmaps.png` | Heatmaps de F1 por par de parâmetros |
| `loss_curves_top9.png` | Curvas de convergência (top-9) |
| `best_model_loss_curve.png` | Curva completa do melhor modelo |
| `robustness_boxplot.png` | Distribuição das métricas (5 seeds) |
| `confusion_matrix.png` | Matriz de confusão do melhor modelo |

---

## 📁 Estrutura do Projeto

```
mlp-phishing-detection/
├── main.py              ← experimento completo
├── requirements.txt
├── phishing.csv         ← dataset (adicionar manualmente se necessário)
└── results/
    ├── grid_results.csv
    ├── robustness_results.csv
    └── *.png            ← todos os gráficos
```

---

## 📋 Métricas de Avaliação

- **Acurácia** — percentual geral de acertos
- **Precisão** — dos sites classificados como legítimos, quantos realmente são (evita falsos positivos)
- **Recall** — dos sites realmente phishing, quantos foram detectados (evita falsos negativos)
- **F1-Score** — média harmônica de precisão e recall (métrica principal do ranking)
- **Matriz de Confusão** — visão completa dos erros por classe
