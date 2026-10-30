# Lab 04 — Data Drift x Concept Drift: medir o impacto no desempenho

**RMC AI Labs | Série MLOps | Publicação de 28/10/2026**

Este laboratório continua o [Lab 03 — Data Drift](../data_drift/README.md), publicado no repositório https://github.com/chykynho/mlops_mini. Não modifica o Lab 02 nem o Lab 03.

## Objetivo e conceitos

No Lab 03 medimos a mudança na distribuição das entradas usando PSI sem conhecer a qualidade das previsões. Agora avaliamos **PSI + MAE** quando os valores reais dos alvos estão disponíveis. São 3 cenários de laboratório **100% sintéticos**. Não são métricas reais de servidores.

- **Controle:** dados comparáveis à referência, relação CPU → memória preservada.
- **Data Drift:** distribuição de CPU muda, mas a relação com a memória permanece.
- **Concept Drift:** mesmas entradas do cenário anterior, mas a relação com o alvo muda em 12 pontos.

O mesmo preditor permanece fixo. Não há retreinamento, MLflow, acesso a produção ou execução em cloud.

## Pré-requisitos

- Git, Conda/Miniconda, Python 3.12, VS Code com extensões Python e Jupyter.
- macOS, Linux ou WSL2 Ubuntu; no Windows nativo, use PowerShell e adapte apenas o comando de ativação Conda.
- Repositório `chykynho/mlops_mini` já clonado ou clonado conforme abaixo.

## Instalação — macOS e Linux/WSL2

Caso ainda não tenha o projeto, execute **uma vez**:

```bash
git clone https://github.com/chykynho/mlops_mini.git
cd mlops_mini
```

Se já estiver clonado, entre no diretório existente; não faça clone novamente. Antes de atualizar, confira se há mudanças locais:

```bash
git status
git pull --ff-only origin main
```

Execute o `git pull` somente se não houver alterações conflitantes. Para entrar no Lab 04:

```bash
cd posts/model_drift
```

Se `mlops-mini` já existir, use-o. Caso contrário, crie-o uma única vez:

```bash
conda env list
conda create -n mlops-mini python=3.12 -y
```

Ative o ambiente e instale as dependências declaradas **neste laboratório**, e não no `pyproject.toml` da raiz:

```bash
conda activate mlops-mini
python -m pip install .
python -m pip check
python -m ipykernel install --user --name mlops-mini --display-name "Python 3.12 (MLOps Mini)"
```

**Atenção:** não execute `conda create` se o ambiente já existir. O `pip install .` deve ser executado dentro de `posts/model_drift` para instalar apenas as dependências necessárias ao Lab 04.

## Windows nativo (PowerShell com Conda instalado)

```powershell
conda activate mlops-mini
cd posts/model_drift
python -m pip install .
python -m ipykernel install --user --name mlops-mini --display-name "Python 3.12 (MLOps Mini)"
```

## Executar no VS Code e Jupyter

Abra o VS Code na pasta `posts/model_drift` (comando `code .` se habilitado). Abra `Laboratorio_04_Data_Drift_Concept_Drift.ipynb`, selecione o kernel **Python 3.12 (MLOps Mini)**, use **Restart Kernel → Run All**. Para usar somente scripts:

```bash
python executar_lab04.py
```

Este comando cria `figuras/distribuicoes.png`, `figuras/psi_mae_comparacao.png` e `figuras/resultados.json`.

## Método

- Semente fixa: `20261028`; 400 observações em cada janela sintética.
- Variável de entrada: CPU simulada (%), com referência de média 49 e desvio 8; janela deslocada de média 70 e desvio 8.
- Preditor fixo: `memoria_prevista = 15 + 0.7 × cpu`.
- Valores observados sintéticos: predição + ruído normal (desvio 2); no cenário de Concept Drift soma-se deslocamento de 12.
- PSI: proporções em 10 faixas construídas exclusivamente com a referência, ajustadas com epsilon.
- MAE: média dos erros absolutos entre rótulos observados e previsões.

## Resultados validados localmente

| Cenário | PSI | MAE | Alerta entrada | Alerta desempenho |
|---|---:|---:|---|---|
| Controle | 0.063 | 1.595 | Não | Não |
| Data Drift | 9.299 | 1.614 | Sim | Não |
| Concept Drift | 9.299 | 11.946 | Sim | Sim |

*Limiar didático de PSI: 0,25; limiar didático de MAE: 5. Esses números não constituem políticas gerais de produção.*

O **Data Drift** modifica a distribuição, mas não necessariamente eleva o erro. No **Concept Drift**, a relação entre entrada e rótulo muda e o MAE sobe, embora o PSI seja igual ao cenário de Data Drift. Não use apenas o PSI para determinar retreinamento.

## Como interpretar as figuras

- `figuras/distribuicoes.png`: histogramas de CPU sintética para referência, controle e janela alterada.
- `figuras/psi_mae_comparacao.png`: compara quantitativamente mudança estatística e degradação de previsão.
- `figuras/resultados.json`: evidência reproduzível, semente, limites e resultados completos.

## Solução de problemas

Se `ModuleNotFoundError` aparecer, confirme `conda activate mlops-mini`, `python -m pip install .` a partir desta pasta e selecione o mesmo kernel no VS Code. Se `figuras/` não surgir, confirme que a pasta atual no terminal ou notebook é `posts/model_drift`. `git pull` não deve ser usado para descartar mudanças locais do Lab 02.

## Publicar (após validar)

Na **raiz do repositório**, confira as alterações antes do commit:

```bash
git status
git add posts/model_drift
git diff --cached --name-only
git diff --cached --check
git commit -m "feat: adiciona lab 04 de data drift e concept drift"
git push origin main
```

Não inclua alterações do Lab 02 nem do Lab 03 sem necessidade. Os gráficos deste laboratório são derivados de dados sintéticos e podem ser publicados.

## Continuidade

**Lab 05:** métricas em janelas móveis, segmentação por VMs e desenho de uma política de alerta auditável — antes de qualquer retreinamento.

RMC AI Labs — Predict. Optimize. Evolve.
