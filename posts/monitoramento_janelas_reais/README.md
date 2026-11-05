# Lab 05 — Extensão com histórico real anonimizado (CPU e memória)

**RMC AI Labs | Complemento do Lab 05 | 04/11/2026**

Esta extensão **não modifica** os Labs 02–04 nem substitui a versão sintética do Lab 05. Ela aplica PSI, MAE e monitoramento por VM aos CSVs reais já presentes em `posts/monitoramento_dados/dados/`, com nomes de VMs e clusters fictícios. Nenhuma anomalia é injetada. Não inventamos resultados: execute para descobrir o que os dados mostram.

## Onde colocar

Extraia o pacote **na raiz do clone** `mlops_mini`, preservando diretórios existentes. A pasta `posts/monitoramento_janelas_reais/` será acrescentada e os dados serão lidos da pasta `posts/monitoramento_dados/dados/` já versionada no GitHub.

## Preparação — WSL Ubuntu / Linux / macOS

1. Abra o terminal e entre no **seu** clone (no WSL conhecido):

```bash
cd ~/PythonProjects/LAB_POSTS/mlops/rmc-mlops-mini
```

No Mac, se seu clone estiver em outro diretório, use o caminho real dele, como `~/PythonProjects/mlops_mini`.

2. Confira o repositório oficial antes de atualizar:

```bash
git remote -v
git status
```

3. Com o diretório limpo e sem commits exclusivos locais, você pode atualizar:

```bash
git pull --ff-only origin main
```

4. Reutilize o ambiente existente:

```bash
conda env list
conda activate mlops-mini
python --version
```

5. Se o ambiente não existir, crie-o uma única vez:

```bash
conda create -n mlops-mini python=3.12 -y
conda activate mlops-mini
```

6. Entre na pasta desta extensão e instale as dependências declaradas **neste** pyproject:

```bash
cd posts/monitoramento_janelas_reais
python -m pip install .
python -m pip check
python -m ipykernel install --user --name mlops-mini --display-name "Python 3.12 (MLOps Mini)"
```

7. Confira os dados existentes, sem copiá-los:

```bash
ls -lh ../monitoramento_dados/dados/cpu.csv
ls -lh ../monitoramento_dados/dados/mem.csv
```

8. Execute o script ou o notebook (o notebook chama o mesmo script, evitando lógica duplicada):

```bash
python executar_lab05_real.py
```

Abra `Laboratorio_05_Monitoramento_Real.ipynb` no VS Code, selecione **Python 3.12 (MLOps Mini)** e use **Restart Kernel → Run All**. O notebook exibe os resultados calculados e as figuras.

## Conceitos e cortes

- Variáveis: CPU e memória **do dia anterior**, prevendo CPU observada do dia seguinte; só pares com intervalo exato de um dia.
- Treino: até 30/04/2026; referência de monitoramento: 01/05 a 14/05/2026; avaliação: 15/05 a 10/06/2026 (se essas datas permanecerem disponíveis).
- Mesmo modelo `LinearRegression` em todas as janelas, ajustado **somente** no treino.
- PSI: usa 10 faixas de quantis da referência (ou menos se empates). Compara a distribuição dos **lags** em janelas globais de 7 dias com a referência fixa; usa suavização de 0,5 contagem por faixa.
- MAE: erro de previsão de CPU em pontos percentuais. `MAE7_GLOBAL` usa a média em janela dos últimos 7 dias; `MAE7_VM` usa até 7 observações recentes daquela VM, exigindo ao menos 5.
- Alerta por VM: `MAE7_VM` acima do percentil 95 dos erros móveis do período de referência, por **3 dias consecutivos observados**. Trata-se de uma regra *exploratória*, não um SLA de produção.
- O PSI e o MAE não demonstram, sozinhos, a causa do drift, nem justificam retreinamento automático.

## Evidências geradas

- `evidencias_reais/resultados.json` — contagens e resumo, gerado após execução.
- `evidencias_reais/metricas_janelas_7d.csv` — PSI CPU/MEM e MAE agregado por janela.
- `evidencias_reais/metricas_vm_dia.csv` — erros e decisões por VM e dia.
- `evidencias_reais/psi_janelas_reais.png` — PSI temporal.
- `evidencias_reais/mae_janelas_reais.png` — MAE temporal.
- `evidencias_reais/ranking_mae_vms.png` — ranking de erros por VM.

**Não publique números do histórico real antes de executar o notebook e conferir `resultados.json`.** Os exemplos sintéticos continuam documentados no Lab 04 e na versão demonstrativa do Lab 05.

## Publicação no GitHub, somente após validar

Na raiz do clone:

```bash
git status
git add posts/monitoramento_janelas_reais/README.md
git add posts/monitoramento_janelas_reais/pyproject.toml
git add posts/monitoramento_janelas_reais/executar_lab05_real.py
git add posts/monitoramento_janelas_reais/Laboratorio_05_Monitoramento_Real.ipynb
git diff --cached --name-only
git diff --cached --check
git commit -m "feat: adiciona extensao real ao lab 05 de monitoramento"
git push origin main
```

Os CSVs de evidência **não são adicionados automaticamente**: revise-os antes de publicar. Os históricos `cpu.csv` e `mem.csv` já estão no repositório, não precisam ser duplicados.

## Solução de problemas

- `FileNotFoundError`: execute dentro do clone completo; verifique `posts/monitoramento_dados/dados/`.
- `ModuleNotFoundError`: ative `mlops-mini`, execute `python -m pip install .` nesta pasta e selecione o kernel correto no VS Code.
- Erro de datas / amostras insuficientes: consulte os intervalos reais nos CSVs, sem mudar silenciosamente a divisão temporal.
- Alterações não relacionadas em `git status`: não as inclua no commit do Lab 05.

**Validação disponível:** execução estrutural com CSV fixture no ambiente de criação do pacote. A execução final contra os dados reais do GitHub depende do clone local do usuário e não deve ser presumida.
