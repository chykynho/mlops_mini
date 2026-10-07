# RMC AI Labs — MLOps em um exemplo pequeno

**Motivação:** a métrica mudou. Foram os dados, o código ou a configuração?

Este exemplo aprende uma reta entre carga e CPU **inteiramente sintéticas**. Em cada treino, salva os dados, o modelo e um registro da execução. O objetivo é mostrar rastreabilidade de experimentos. Os dados foram gerados para fins didáticos e não representam medições de infraestrutura.

## Preparação no WSL2 Ubuntu + VS Code

Extraia o projeto, abra o diretório no VS Code conectado ao WSL e execute cada comando separadamente. O pacote e suas dependências são declarados exclusivamente no pyproject.toml. O treinamento usa a biblioteca padrão, e o gráfico usa Matplotlib; setuptools é a ferramenta de empacotamento.

```bash
cd ~/PythonProjects/rmc-mlops-mini
conda create -n mlops-mini python=3.12
conda activate mlops-mini
python --version
python -m pip install .
python -c "import sys, rmc_mlops_mini; print(sys.executable); print(rmc_mlops_mini.__file__)"
```

No VS Code, selecione o interpretador do ambiente `mlops-mini`. Este exemplo é um script, sem notebook: não há kernel a selecionar. Se criar um notebook posteriormente, selecione o kernel desse mesmo ambiente e declare qualquer dependência adicional no pyproject.toml.

## Dois experimentos

```bash
python -m rmc_mlops_mini.train --seed 42
python -m rmc_mlops_mini.train --seed 21
```

Cada comando imprime o caminho de `run.json`. Abra os dois arquivos pelo VS Code.

| Registro | O que comparar |
| --- | --- |
| data_sha256 | Os dados permanecem iguais |
| training_code_sha256 | O script permanece igual |
| parameters.split_seed | A configuração mudou |
| split | As amostras de treino/teste mudaram |
| metrics | Compare MAE do modelo e da baseline no teste |
| model_sha256 | Identifica o artefato produzido |
| git | Commit e indicação de mudanças locais |

O MAE é a média do erro absoluto: quanto menor, melhor nesse mesmo problema e conjunto de teste. Comparar métricas em divisões diferentes não demonstra que um modelo é superior. Aqui a mudança da divisão é justamente a demonstração.

O código aprende os coeficientes e a baseline exclusivamente no treino; avalia ambos no teste. Não treina com o teste. A fórmula é uma regressão linear de mínimos quadrados com intercepto, adequada à relação linear inventada para a demonstração.

## Registro da execução

```python
"data_sha256": sha256(data_bytes),
"training_code_sha256": sha256(source.read_bytes()),
"parameters": {"data_seed": 7, "split_seed": args.seed, "train_fraction": args.train_fraction},
"metrics": {"test_mae": mae, "baseline_test_mae": baseline_mae},
"model_sha256": sha256(model_bytes),
```

Esses campos fazem parte do dicionário `record` em `train.py`. Cada pasta de execução contém `data.json` (dados), `model.json` (coeficientes da reta), `run.json` (metadados do experimento) e `regressao.png` (gráfico).

## Veja a reta aprendida

Abra o arquivo `regressao.png` gerado na pasta de cada execução. Círculos azuis representam os registros de treino; triângulos laranja representam o teste. A linha verde é a previsão do modelo salvo. Um segmento vertical tracejado destaca o erro de um ponto de teste: a distância entre observado e previsto.

A função em `plot.py` lê os três JSONs salvos e desenha o modelo sem treiná-lo novamente. A figura mostra o que foi aprendido; `run.json` registra como esse modelo foi produzido. O script de gráfico também tem seu hash registrado. As retas de duas divisões podem ficar muito próximas visualmente, mesmo com coeficientes e métricas diferentes.

## Como funciona o train.py

1. Lê os argumentos da linha de comando: semente da divisão, fração de treino e diretório de saída.
2. Gera 100 registros sintéticos com `cpu = 10 + 0.6 * carga + ruído`. A semente 7 fixa os dados. O código de treino não recebe os coeficientes 10 e 0.6: precisa estimá-los a partir das amostras.
3. Embaralha uma cópia dos dados com a semente de `--seed`. Com a fração padrão 0.8, separa 80 registros para treino e 20 para teste.
4. Aprende a inclinação (`slope`) e o intercepto (`intercept`) da reta por mínimos quadrados. Calcula ambos exclusivamente no treino.
5. Prevê a CPU sintética no teste e calcula o MAE. Também avalia uma baseline que sempre prevê a média da CPU do treino.
6. Cria uma pasta com data UTC e identificador aleatório, salva os registros e o gráfico e imprime seus caminhos.

### A reta aprendida

O modelo prevê `y = intercept + slope * x`. A inclinação é calculada dividindo a soma dos produtos dos desvios de x e y pela soma dos quadrados dos desvios de x. O intercepto é `mean_y - slope * mean_x`. Essa solução minimiza a soma dos erros quadráticos no treino; o MAE é utilizado somente para avaliar o teste.

Por exemplo, se a CPU sintética observada fosse 40 e a previsão fosse 43, o erro absoluto seria 3. O MAE é a média desses erros nas 20 amostras do teste, em pontos da escala de CPU sintética.

### Sementes e hashes

`data_seed = 7` controla a geração dos dados. `split_seed = --seed` controla apenas a divisão. Por isso, mudar `--seed` mantém o hash dos dados e pode mudar os coeficientes, o modelo e as métricas.

SHA-256 produz uma impressão digital dos bytes de um arquivo. Serve para identificar alterações; não criptografa nem anonimiza o conteúdo. O hash do script é o do código instalado que executou o treino. Depois de alterar o código-fonte, reinstale com `python -m pip install .` antes de executar novamente.

## Colocar no Git

Na pasta do projeto, crie o primeiro commit antes de treinar para que o registro consiga capturar uma versão do código. Os resultados locais em `runs/` ficam ignorados; `examples/` contém duas execuções sintéticas de referência.

```bash
git init
git add .
git commit -m "Adiciona exemplo de rastreabilidade em MLOps com dados sintéticos"
git branch -M main
```

Crie um repositório vazio no GitHub, copie sua URL HTTPS e substitua `URL_DO_REPOSITORIO` no comando seguinte.

```bash
git remote add origin URL_DO_REPOSITORIO
git push -u origin main
```

Execute o treino a partir da raiz do repositório: o programa consulta o Git do diretório atual. Sem commit, `git.commit` é null. Com alterações não commitadas, `git.dirty` é true: o commit sozinho não representa todos os arquivos. O hash registrado cobre apenas o script de treino, não todo o projeto. Dados, código e parâmetros ajudam na rastreabilidade, mas não garantem reprodução idêntica em qualquer ambiente. O registro inclui versão de Python; um ambiente integralmente bloqueado e monitoramento são extensões futuras.

## Limites

Não há DVC, MLflow, deploy, monitoramento, CI/CD ou dados reais. A previsão sintética não serve para rightsizing ou decisões de capacidade. Dependências de build não estão congeladas; não se promete reprodução bit a bit. Os registros em `examples/` foram produzidos com Python 3.12.14, sem commit Git, e incluem essa informação em `run.json`.

José Francisco Alves dos Santos — RMC AI Labs

Predict. Optimize. Evolve.
