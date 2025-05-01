| **Tarefa**                     | **Conda**                                  | **Micromamba**                             |
|--------------------------------|--------------------------------------------|--------------------------------------------|
| Criar ambiente                 | `conda create -n <env> [pacotes...]`       | `micromamba create -n <env> [pacotes...]`  |
| Criar ambiente via YAML    | `conda env create -f environment.yml`      | `micromamba env create -f environment.yml` |
| Instalar pacote                | `conda install -c conda-forge <pacote>`                   | `micromamba install -c conda-forge <pacote>`              |
| Atualizar pacote               | `conda update <pacote>`                    | `micromamba update <pacote>`               |
| Listar pacotes                 | `conda list`                               | `micromamba list`                          |
| Remover pacote                 | `conda remove <pacote>`                    | `micromamba remove <pacote>`               |
| Listar ambientes               | `conda env list` / `conda info --envs`     | `micromamba env list`                      |
| Ativar ambiente                | `conda activate <env>`                     | `micromamba activate <env>`                |
| Desativar ambiente             | `conda deactivate`                         | `micromamba deactivate`                    |
| Buscar pacote                  | `conda search <termo>`                     | `micromamba search <termo>`                |
| Limpar cache                   | `conda clean --all`                        | `micromamba clean --all`                   |
| Informações do gerenciador     | `conda info`                               | `micromamba info`                          |
| Configurar canais              | `conda config --add channels <canal>`      | `micromamba config --add channels <canal>` |
