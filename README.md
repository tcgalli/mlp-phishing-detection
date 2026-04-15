# MLP Phishing Detector Analysis 🛡️

Projeto de implementação e análise experimental de uma Rede Neural do tipo **Multilayer Perceptron (MLP)** para a detecção de sites de phishing. Este repositório contém o pipeline completo de treinamento, busca em grade (Grid Search) e testes de robustez.

## 📌 Visão Geral

O objetivo deste projeto é treinar uma rede neural capaz de identificar se uma URL/site é legítima ou um ataque de phishing baseado em 30 atributos técnicos (comprimento da URL, presença de prefixos/sufixos, uso de SSL, etc.).

## 🔬 Metodologia Experimental

Seguindo o rigor científico, o projeto realiza:

1.  **Busca em Grade (Grid Search)**:
    -   Avaliação de **27 configurações** diferentes de hiperparâmetros.
    -   Variação de: Quantidade de Neurônios, Taxa de Aprendizado e Momentum.
2.  **Análise de Convergência**:
    -   Geração de curvas de perda (Loss Curves) para cada experimento.
3.  **Teste de Robustez**:
    -   Execução de 5 rodadas com diferentes inicializações de pesos para a melhor configuração encontrada.
    -   Cálculo de média e desvio padrão.

## 🛠️ Tecnologias Utilizadas

-   **Python 3.x**
-   **Scikit-Learn** (Motor da MLP)
-   **Pandas & NumPy** (Tratamento de dados)
-   **Matplotlib & Seaborn** (Visualização de dados/gráficos)

## 📂 Estrutura do Projeto

```text
├── data/               # Dataset (original e processado)
├── results/            # Gráficos de convergência e tabelas
├── src/                # Código fonte
│   ├── data_loader.py  # Preparação dos dados
│   ├── training.py     # Loop de experimentos
│   └── utils.py        # Funções auxiliares de plotagem
└── main.py             # Script principal para execução
```

## 🚀 Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/mlp-phishing-detection.git
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o experimento:
   ```bash
   python main.py
   ```

## 📊 Resultados e Métricas

As métricas utilizadas para avaliação foram:
-   **Acurácia**
-   **Precisão** (Foco principal para evitar falsos positivos)
-   **F1-Score**
-   **Matriz de Confusão**

---
*Este projeto foi desenvolvido para fins acadêmicos na disciplina de Redes Neurais.*
