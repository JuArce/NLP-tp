# TPE NLP

82.18 - Procesamiento del Lenguaje Natural Segundo Cuat. 2023

## Integrantes

- [Julián Arce](https://github.com/JuArce)
- [Sebastián Itokazu](https://github.com/sebitokazu)
- [Gian Luca Pecile](https://github.com/glpecile)
- [Valentín Ratti](https://github.com/valenratti)

## Entregas

En el directorio [`docs`](/docs/) se encuentran las entregas del trabajo práctico.

A continuación se listan los links a los archivos:

- [1° Entrega](/docs/Informe_1°_Entrega_NLP.pdf)
- [2° Entrega](/docs/Slides_2°_Entrega_NLP.pdf)

## Corpus

- Datasets inicial de [scripts](https://www.kaggle.com/datasets/e14349f732b3f35aa1bcb5fe68961b4a79a757bc5c84fe678acd0ffa69018c72).
- Este fue modificado para tomar sólo los diálogos “head_type” donde se tiene un “speaker/title”. Ej:

```json
{
"head_type":"speaker/title",
"Head_text": {
  "speaker/title":"WEST"
},
"text":"Three cases in two years? Who was  she handling, the Rosenbergs?”
},
```

## Análisis

### Exploratorio

En el directorio [`output`](/output) se encuentran los archivos relacionados al análisis exploratorio. Estos incluten

- En el mismo se realiza un cloud of words para cada guionista con una selección reducida de todos los guiones.
- Por otro lado también se realizó un análisis de la media de largo de dialogos por guionista.

### Modelos

Los modelos usados son los siguientes:

- [TF-IDF](/src/tfidf.py)
- BERT
  - [no fine-tuning](/src/bert-no-fine-tuning.py)
  - [fine-tuning](/src/bert-fine-tuning.py)

### Métricas

Las métricas usadas son las siguientes:

- [Accuracy](/src/metrics.py)
- [F1](/src/metrics.py)
- [Precision](/src/metrics.py)
- [Recall](/src/metrics.py)

------
