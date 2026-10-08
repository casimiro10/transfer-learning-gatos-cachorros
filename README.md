# 🐱🐶 Transfer Learning com TensorFlow — Gatos vs. Cachorros

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/casimiro10/transfer-learning-gatos-cachorros/blob/main/transfer_learning_gatos_cachorros.ipynb)

Projeto do desafio **Transfer Learning em Deep Learning** da [DIO](https://www.dio.me/).
Classificador de imagens de **gatos e cachorros** usando Python, TensorFlow e a rede pré-treinada **MobileNetV2**, executado no Google Colab.

**Resultado: 98,41% de acurácia no conjunto de teste.**

---

## 📌 O que é Transfer Learning?

Treinar uma rede neural de visão computacional do zero exige muitos dados e muito processamento.
No **Transfer Learning**, reaproveitamos uma rede que já foi treinada em milhões de imagens (o **ImageNet**) e adaptamos apenas a parte final dela ao nosso problema.

É como contratar um profissional que já tem experiência: em vez de ensinar tudo do zero, só ensinamos a tarefa específica. Aqui, a MobileNetV2 já sabe reconhecer bordas, texturas e formas; nós ensinamos apenas a decidir **gato ou cachorro**.

---

## 🗂️ Dataset

- **Nome:** `cats_vs_dogs`
- **Fonte:** [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)
- **Classes:** `cat` e `dog`
- **Carregamento:** automático via `tensorflow_datasets` (sem download manual)

| Conjunto  | Proporção | Finalidade                               |
|-----------|-----------|------------------------------------------|
| Treino    | 80%       | Ajustar os pesos do modelo               |
| Validação | 10%       | Acompanhar o desempenho durante o treino |
| Teste     | 10%       | Avaliação final em imagens nunca vistas  |

---

## 🛠️ Tecnologias

Python 3 · TensorFlow/Keras · TensorFlow Datasets · Matplotlib · NumPy · Google Colab (GPU T4)

---

## 🧠 Passo a passo do notebook

1. **Instalação de dependência** (`importlib_resources`), necessária no Colab com Python 3.13.
2. **Imports e verificação de GPU.**
3. **Carregamento do dataset** e divisão em treino, validação e teste.
4. **Visualização** de amostras para conferir os dados.
5. **Pré-processamento:** redimensionamento para 160x160, embaralhamento, lotes de 32 e `prefetch`.
6. **Montagem do modelo:**
   - *Data augmentation* (espelhamento e rotação leve);
   - **MobileNetV2** pré-treinada (ImageNet), sem a camada final e **congelada**;
   - `GlobalAveragePooling2D` + `Dropout(0.2)` + `Dense(1)` como nova camada de saída.
7. **Compilação e treino** (Adam, `lr=1e-4`, `BinaryCrossentropy`, 10 épocas).
8. **Gráficos** de acurácia e perda (treino vs. validação).
9. **Fine-tuning:** descongelamento das camadas finais da base e re-treino por 5 épocas com `lr=1e-5`.
10. **Avaliação final** no conjunto de teste.
11. **Previsões visuais** em imagens reais (verde = acerto, vermelho = erro).

---

## 📊 Resultados

| Métrica                              | Valor                  |
|--------------------------------------|------------------------|
| Acurácia no teste                    | **98,41%**             |
| Base pré-treinada                    | MobileNetV2 (ImageNet) |
| Parâmetros treináveis (fase inicial) | 1.281                  |
| Épocas                               | 10 + 5 (fine-tuning)   |

Na fase inicial, apenas **1.281 parâmetros** foram treinados (contra ~2,2 milhões da base congelada), o que torna o treino rápido e eficiente mesmo com poucos recursos.

<!-- Para mostrar prints, salve-os na pasta images/ e descomente:
![Gráficos de treino](images/graficos.png)
![Previsões](images/previsoes.png)
-->

---

## ▶️ Como executar

1. Clique no botão **Open in Colab** no topo deste README (ou envie o `.ipynb` ao [Google Colab](https://colab.research.google.com/)).
2. Ative a GPU: **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)**.
3. Execute as células em ordem (**Ambiente de execução → Executar tudo**). O dataset é baixado automaticamente.

Execução local (opcional):

```bash
pip install tensorflow tensorflow-datasets matplotlib importlib_resources
jupyter notebook
```

---

## 📚 Aprendizados

- Redes pré-treinadas carregam um "conhecimento visual" que pode ser reaproveitado em problemas novos.
- Congelar a base e depois aplicar **fine-tuning** é uma estratégia estável e eficiente.
- Separar treino, validação e teste é essencial para medir o desempenho real.
- Data augmentation e dropout ajudam a evitar *overfitting*.
- Resolver problemas de ambiente (como a dependência no Python 3.13) faz parte do dia a dia em ML.

---

## 👤 Autor

**João Raphael Casimiro de Almeida**

- GitHub: [@casimiro10](https://github.com/casimiro10)
- LinkedIn: [João Raphael Casimiro de Almeida](https://www.linkedin.com/in/joão-raphael-casimiro-de-almeida-63681a23b)
