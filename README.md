#  Miniprojeto – Inteligência Computacional Aplicada à Análise de TDS

##  Artigo Científico Escolhido

- **Título:** Machine learning-based parameter identification from Thermal Desorption Spectroscopy spectra  
- **Autores:** N. Marrani et al.  
- **Periódico:** International Journal of Hydrogen Energy  
- **Ano de publicação:** 2025  
- **DOI:** https://doi.org/10.1016/j.ijhydene.2025.150874  

---

##  Resumo do Artigo

O artigo propõe o uso de técnicas de Machine Learning, especificamente Redes Neurais Artificiais (ANNs), para identificar parâmetros de aprisionamento de hidrogênio em materiais metálicos a partir de espectros de Thermal Desorption Spectroscopy (TDS).

O modelo desenvolvido é composto por duas redes neurais:
- Uma rede de classificação para determinar o número de tipos de armadilhas (traps)
- Uma rede de regressão para prever as energias de ligação e densidades dessas armadilhas

Os modelos são treinados exclusivamente com dados sintéticos gerados por simulação numérica (método dos elementos finitos) e posteriormente aplicados a dados experimentais reais.

---

##  Objetivo do Artigo

Desenvolver um modelo baseado em Machine Learning capaz de:
- Identificar automaticamente o número de traps em um material
- Estimar suas energias de ligação
- Determinar suas densidades

sem a necessidade de estimativas iniciais ou métodos iterativos tradicionais.

---

##  Contribuições Principais

- Uso de redes neurais para análise de espectros TDS  
- Eliminação da necessidade de ajuste manual de parâmetros  
- Redução do custo computacional comparado a métodos tradicionais  
- Possibilidade de reutilização dos dados de treinamento  

---

##  Técnica de Inteligência Computacional Utilizada

- Redes Neurais Artificiais (ANNs)
- Aprendizado supervisionado


##  Integrantes e Contribuições

| Nome              | Contribuição |
|-------------------|-------------|
| Igor Gabriel      | Implementação do modelo e análise dos resultados |
| Breno             | Geração de dados e simulação |
| Henrique Azevedo  | Treinamento das redes neurais |
| João Marcos       | Avaliação e métricas |
| João Coutinho     | Documentação e slides |

---

## 📚 Referências
- Artigo base do projeto  
- Documentação Keras  
- Trabalhos relacionados a TDS  
- Métodos clássicos de difusão de hidrogênio  

---

##  Apresentação

- Tempo mínimo individual: 10 minutos  
- Conteúdo:
  - Fundamentação teórica  
  - Metodologia  
  - Resultados  
  - Comparação com métodos tradicionais  

---

##  Conclusão

O uso de redes neurais mostrou-se eficiente para analisar espectros TDS, permitindo prever parâmetros complexos com maior rapidez e precisão em comparação com métodos tradicionais.
