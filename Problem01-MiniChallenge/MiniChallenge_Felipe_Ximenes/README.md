

Neste trabalho foi desenvolvido uma rede de aprendizado profundo (deep learning) para o minichallenge proposto na disciplina Redes Neurais Profundas e Aplicações (Deep learning), ministrado no CBPF em 2018.2.

# Projeto

As informações referente ao projeto podem ser obtidas no seguinte [link](https://bitbucket.org/kognitalab/images_mini_challange/src/master/).

# Configurações

- Foi utlizado o framework Keras para a elaboração do projeto.
- A configuração do ambiente pode ser encontrada no arquivo `requirements.txt`.
- O modelo da rede pode ser obtido neste [link](https://drive.google.com/open?id=1Iyi1jb2hh2z46fN7c0E0yZUcw4bWzgou).

## Dataset

O dataset fornecido possui as seguintes características:

| Categorias  |      Quantidade de fotos      |  Porcentagem (%) |
|-------------|:-----------------------------:|-----------------:|
| Graduation  |  6992                         | 33,32            |
| Meeting     |    7000                       |   33,5           |
| Picnic      |  6995                         |    33,33         |

#### Observações

1. Podemos identificar que o dataset está bem dimensionado em relação a quantidade de dados por categoria.
2. As imagens possuem uma resolução alta e não possuem uma resolução uniforme, desta forma será neceessário realizar o redimencionamento delas.


#### Pré-processamento do dataset


1. Redimensionar todas as images para a resolução 128 x 128
2. Converter as imagens para array.
3. Associar o label disponibilizado no arquivo csv com cada imagem do dataset.
4. Rescalonar a itensidade de pixel das imagens para que fiquem no intervalo [0,1].
5. Particionar o dataset em treinamento e teste, utilizando a seguinte métrica ( 75% para treinamento e 25% para teste).
6. Converter os labels de inteiros para vetores.

#### Contruir o Dataset

```sh
python buildDataset.py
```

## Rede

A rede utilizada neste prjeto foi a [ResNet](https://arxiv.org/pdf/1512.03385.pdf). A rede foi implementada no arquivo /dnn/resnet.pt.

#### Treinar a rede

```sh
python train_dnn_scratch.py
```

Abaixo segue o gráfico do treinamento obtido.

![Training](https://github.com/cdebom/DeepLearningProblemsSolutions/blob/master/Problem01-MiniChallenge/MiniChallenge_Felipe_Ximenes/assets/train.png)

#### Avaliar rede

```sh
python evaluate.py
```

### Gerar ROC curve

```sh
python roc_curve.py
```

Abaixo segue a ROC obtida após o treinamento.

![ROC](https://github.com/cdebom/DeepLearningProblemsSolutions/blob/master/Problem01-MiniChallenge/MiniChallenge_Felipe_Ximenes/assets/roc.png)

