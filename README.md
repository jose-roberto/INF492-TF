# Classificação de espécies sobre o dataset iNaturalist

## Definição do tema

- Escolhido: Classificação de espécies sobre o dataset iNaturalist
- Segunda opção: Segmentação de instância de toras sobre o dataset TimberVision 

## Rascunho da entrega parcial

### Parte 1

É necessário apresentar em detalhes o problema a ser abordado, a motivação para abordar esse problema e uma aplicação prática do mesmo.

#### Dataset escolhido: iNaturalist
- Realizar descrição mais detalhada

#### Detalhes do problema a ser abordado:
- Explicar o que é classificação
- Detalhar complexidade do problema de classificação de espécies: basicamente a enorme variabilidade de formas, cores, etc na composição do dado

#### Motivação para abordar esse problema:

Inicialmente iríamos abordar um tema relacionado à IC do José, mas não havia dado disponível, assim mudamos para uma temática que já era do nosso interesse:
Visão Computacional aplicada na área da biologia. Pesquisando dentro deste escopo encontramos com facilidade um dataset super robusto.

#### Aplicação prática do mesmo:

O problema de classificação de espécies pode ser aplicado em diversos ramos dentro da área biológica, como, por exemplo: pesquisa ecológica,
monitoramento de biodiversidade, conservação de animais, conservação de vida selvagem.

### Parte 2

Plano de execução dos experimentos: o que será treinado, testado, comparado, onde será executado, contra quem será comparado, etc.

#### 1 - Escolher duas ou mais redes para treino, teste e comparação:

A comparação será feita entre a ResNet50 incializada com os pesos da ImageNet e a EfficientNet também inicializada com os pesos da ImageNet.

A ideia é realizar a comparação entre duas ou mais redes, aplicando pelo menos dois dos seguintes conceitos: transfer learning, extração de features ou
classificação direta.

#### 2 - Onde será executado:

Iremos utilizar máquinas do GeoDEF e do MaVILab. (detalhar especificação das máquinas)

## Referências e/ou links úteis:
- [iNaturalist - paperswithcode](https://paperswithcode.com/paper/the-inaturalist-species-classification-and)
- [iNaturalist on GitHub](https://github.com/visipedia/inat_comp/tree/master/2021)
- [iNaturalist on PyTorch](https://pytorch.org/vision/stable/generated/torchvision.datasets.INaturalist.html)
- [Deep CNNs for large scale species classification - paperswithcode](https://paperswithcode.com/paper/deep-cnns-for-large-scale-species)
- [A gentle introduction to computer vision-based specimen classification in ecological datasets](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2656.14042)
- [EfficientNet](https://www.geeksforgeeks.org/efficientnet-architecture/)
- [Benchmarking Representation Learning for Natural World Image Collections](https://arxiv.org/pdf/2103.16483)

