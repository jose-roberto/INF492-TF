# Classificação de espécies sobre o dataset iNaturalist

## Entrega parcial

### Motivação para abordar esse problema:
Interesse em aplicar Visão Computacional no campo da biologia.

### Aplicação prática:
O problema de classificação de espécies pode ser aplicado em diversos ramos dentro da área biológica, como, por exemplo: pesquisa ecológica,
monitoramento de biodiversidade, conservação de animais, conservação de vida selvagem.

### Detalhamento do dataset
Características do dado:
- Grande variabilidade de formas e cores
- Imagens com mais de uma espécie
- Espécies parecidas
- Espaço de cor RGB e extensão .jpg
- Resoluções diferentes
- Número de classes: 10.000 espécies
- Amostras por classe: 50
- Quantidade total de imagens: 500.000

### Plano de execução dos experimentos

- O dataset já possui um split em treino, validação e teste. Será nosso ponto de partida.

- Arquiteturas escolhidas:
  EfficientNet
  YOLO11

- Onde será executado?

  GeoDEF - pcB:
  - Intel i7-12700 2.10 GHz
  - NVIDIA T1000 4GB
  - 64GB RAM
  - Windows

  MaVILab - proc2:
  - Intel i7-12700K 2.10 GHz
  - NVIDIA GeForce RTX 4090 24GB
  - 126GB RAM
  - Ubuntu

- Comparação com o benchmark de Van Horn et al. (ResNet50)

## Referências e/ou links úteis:
- [iNaturalist - paperswithcode](https://paperswithcode.com/paper/the-inaturalist-species-classification-and)
- [iNaturalist on GitHub](https://github.com/visipedia/inat_comp/tree/master/2021)
- [iNaturalist on PyTorch](https://pytorch.org/vision/stable/generated/torchvision.datasets.INaturalist.html)
- [Deep CNNs for large scale species classification - paperswithcode](https://paperswithcode.com/paper/deep-cnns-for-large-scale-species)
- [A gentle introduction to computer vision-based specimen classification in ecological datasets](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2656.14042)
- [EfficientNet](https://www.geeksforgeeks.org/efficientnet-architecture/)
- [Benchmarking Representation Learning for Natural World Image Collections](https://arxiv.org/pdf/2103.16483)
- [iNat Challenge 2021 - FGVC8 - Discussion](https://www.kaggle.com/competitions/inaturalist-2021/discussion/242521)
- [Yolo v11 - Classification](https://docs.ultralytics.com/tasks/classify/)
