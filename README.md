# 🔬 Miniprojeto – Inteligência Computacional Aplicada à Análise de TDS

---

## 📚 Artigo Científico Escolhido

- **Título:** Machine learning-based parameter identification from Thermal Desorption Spectroscopy spectra  
- **Autores:** N. Marrani et al.  
- **Periódico:** International Journal of Hydrogen Energy  
- **Ano de publicação:** 2025  
- **DOI:** https://doi.org/10.1016/j.ijhydene.2025.150874  

---

## Resumo do Artigo

O artigo propõe o uso de técnicas de Machine Learning, especificamente Redes Neurais Artificiais (ANNs), para identificar parâmetros de aprisionamento de hidrogênio em materiais metálicos a partir de espectros de Thermal Desorption Spectroscopy (TDS).
O modelo desenvolvido é composto por:
- Uma rede de classificação → número de traps  
- Uma rede de regressão → energias e densidades  
Os dados utilizados são sintéticos, gerados por simulação numérica (FEM), e posteriormente aplicados a dados experimentais.
---
## Objetivo do Projeto
Reproduzir e adaptar a abordagem do artigo utilizando:
- Simulação simplificada de TDS
- Geração de dados sintéticos
- Treinamento de uma rede neural para prever energias de traps
---
## Metodologia
### Simulação Física (TDS)
- Modelo 1D simplificado
- Evolução temporal da concentração de hidrogênio
- Dependência com temperatura (Arrhenius)
- Consideração de múltiplos traps
---
### Geração de Dataset
- Dados sintéticos gerados via simulação
- Número de amostras: até **2000**
- Cada amostra contém:
  - Curva TDS (64 pontos)
  - Energias dos traps (até 4)
---
### Pré-processamento
- Log transform no fluxo
- Normalização (média 0, desvio padrão 1)
- Normalização da saída (energia / 120000)
---
## Modelo de Machine Learning
- Tipo: Rede Neural Artificial (ANN)
- Arquitetura:
Entrada (64)

Dense (128) + ReLU

Dropout (0.2)

Dense (128) + ReLU

Dropout (0.2)

Dense (64)

Dense (32)

Saída (4 energias)

- Função de perda: MSE  
- Otimizador: Adam  
---
## Validação
- Divisão:
  - 80% treino
  - 20% teste
- Validação cruzada:
  - K-Fold (k = 5)
---
## Hiperparâmetros
- Epochs: 60  
- Batch size: 32  
- Dropout: 0.2  
 Escolha feita empiricamente (não foi utilizado Grid Search)
---
## Técnicas Aplicadas

### Regularização
- Dropout nas camadas ocultas

### Data Augmentation
- Adição de ruído gaussiano nas curvas TDS

### Normalização
- Entrada e saída normalizadas

---

## Desbalanceamento de Dados

- Não aplicável  
- Dataset sintético controlado

---

## Métricas de Avaliação

- MSE (Mean Squared Error)  
- MAE (Mean Absolute Error)  
---
## Modificações em Relação ao Artigo

Este projeto difere do artigo original nos seguintes pontos:

- Uso de modelo físico simplificado (não FEM)
- Utilização de apenas uma rede (regressão)
- Inclusão de:
  - Data augmentation
  - Validação cruzada (K-Fold)
  - Regularização com Dropout
- Treinamento e avaliação apenas em dados sintéticos
---
## Implementação

- Linguagem: Python  
- Bibliotecas:
  - NumPy
  - Matplotlib
  - TensorFlow / Keras
  - Scikit-learn  

---
## Estrutura do Projeto
main.py # Simulação TDS
generate_dataset.py # Geração do dataset
train_model.py # Treinamento da ANN
evaluate_model.py # Avaliação do modelo


---

## Integrantes e Contribuições
| Nome              | Contribuição |
|-------------------|-------------|
| Igor Gabriel      | Evaluate + Train |
| Breno             | k-fold |
| Henrique Azevedo  | Main + Dataset |
| João Marcos       | Pesquisa |
| João Coutinho     | Slides |
---
## Limitações

- Modelo físico simplificado
- Dados sintéticos
- Não reproduz perfeitamente múltiplos picos TDS reais
- Aproximação da cinética de traps
---
## Resultados

- O modelo foi capaz de aprender padrões nas curvas TDS
- Apresentou erro elevado devido à complexidade do problema
- Demonstra viabilidade da abordagem
---
## Conclusão
O projeto demonstra a aplicação de redes neurais para análise de espectros TDS, integrando física e aprendizado de máquina.

Mesmo com simplificações, o modelo mostra potencial para prever parâmetros físicos relevantes a partir de dados simulados.
---
## Referências
- Artigo base  
- Documentação TensorFlow / Keras  
- Materiais de apoio  
- Trabalhos sobre difusão de hidrogênio  