# Lab 03 — Monitoramento de Data Drift (21/10/2026)

**RMC AI Labs — Série Posts de MLOps**  
Repositório: https://github.com/chykynho/mlops_mini

## O que este laboratório demonstra

Este é o **terceiro laboratório** da série. Ele demonstra como detectar alteração na distribuição dos dados de entrada com o **Population Stability Index (PSI)**, mantendo uma mesma regra demonstrativa de previsão. O notebook compara uma referência, uma janela de controle e uma janela alterada; desenha as distribuições e verifica a estabilidade da regra por SHA-256.

**Natureza dos dados:** as três janelas de CPU (%) são **sintéticas**, geradas pelo notebook com semente `42`. Este Lab 03 **não lê os CSVs reais** do laboratório anterior e **não usa um modelo treinado**: a regra `15 + 0,7 × CPU` é didática. Portanto, o hash verifica apenas a string da fórmula, não um artefato de produção. Não confunda PSI elevado com perda comprovada de acurácia.

## Onde cada laboratório fica

```text
mlops_mini/
├── README.md                              # Projeto de rastreabilidade original (não substituir)
├── pyproject.toml                         # Dependências do exemplo original (não substituir)
├── posts/
│   ├── monitoramento_dados/               # Lab 02: dados de servidores e falha de escala (não alterar)
│   │   ├── README.md
│   │   ├── Laboratorio_Monitoramento_Dados_Modelo.ipynb
│   │   ├── dados/
│   │   └── pyproject.toml
│   └── data_drift/                        # Lab 03: este laboratório
│       ├── README.md
│       ├── pyproject.toml
│       └── Laboratorio_03_Monitoramento_Data_Drift.ipynb
└── src/rmc_mlops_mini/
```

O Lab 02 (15/10) mediu o efeito de **falha de escala simulada** sobre o MAE usando medições de servidores com nomes fictícios. O Lab 03 (21/10) trata de **mudança de distribuição**, sem rótulos para calcular MAE. São experimentos diferentes; arquivos, resultados e documentações permanecem separados.

## Pré-requisitos

- Git; VS Code com extensões **Python** e **Jupyter**.
- Miniconda/Conda com Python 3.12. Ambiente recomendado: `mlops-mini`.
- Windows: usar WSL2 Ubuntu e executar os comandos no terminal Ubuntu; Linux/macOS: usar o terminal Bash ou compatível.
- Internet na instalação inicial das dependências.

## Preparação inicial — só se ainda não clonou o projeto

Abra um terminal. **Se já clonou, ignore a clonagem e passe à seção seguinte.**

```bash
mkdir -p ~/PythonProjects
cd ~/PythonProjects
git clone https://github.com/chykynho/mlops_mini.git
cd mlops_mini
```

## Usar o ambiente Conda que já existe

Execute na raiz do clone. Não recrie o ambiente se ele já existir:

```bash
cd ~/PythonProjects/mlops_mini
conda env list
conda activate mlops-mini
python --version
```

Somente quando `mlops-mini` **não aparecer** no resultado de `conda env list`, crie e ative:

```bash
conda create -n mlops-mini python=3.12 -y
conda activate mlops-mini
```

## Instalar dependências específicas do Lab 03

O projeto usa exclusivamente `pyproject.toml` (nada de `requirements.txt`). O `pyproject.toml` **desta pasta** tem as bibliotecas usadas por este notebook. Execute os comandos abaixo:

```bash
cd ~/PythonProjects/mlops_mini/posts/data_drift
python -m pip install .
python -m pip check
python -m ipykernel install --user --name mlops-mini --display-name "Python 3.12 (MLOps Mini)"
python -c "import numpy, pandas, matplotlib, ipykernel; print('Dependências do Lab 03 disponíveis')"
```

Se aparecer `No module named ipykernel`, volte à etapa `python -m pip install .` e verifique se houve erro. A versão exibida por `python -m ipykernel --version` pode refletir a versão do IPython; para conferir o pacote instalado utilize `python -m pip show ipykernel`.

## Abrir e executar no VS Code

Abra a raiz do projeto:

```bash
cd ~/PythonProjects/mlops_mini
code .
```

Se `code` não estiver disponível, use **File > Open Folder** no VS Code e selecione `~/PythonProjects/mlops_mini`.

1. Navegue até `posts/data_drift/Laboratorio_03_Monitoramento_Data_Drift.ipynb`.
2. Em **Select Kernel**, selecione **Python 3.12 (MLOps Mini)**.
3. Clique em **Restart & Run All** (ou execute cada célula em ordem).
4. Verifique as três distribuições, os dois valores de PSI e a mensagem `PASSOU` na última célula de código.

Não é preciso copiar dados de `monitoramento_dados/dados/` nem executar o Lab 02 antes deste notebook.

## Como o experimento funciona

1. **Geração:** CPU de referência (`n=240`), controle (`n=120`) e amostra alterada (`n=120`), com a mesma semente para reprodução.
2. **Regra fixa:** `memoria_prevista = clip(15 + 0.7 * cpu_pct, 0, 100)`; serve somente como ilustração, não é um modelo treinado.
3. **PSI:** quantis extraídos apenas da referência definem faixas. Calcula-se a diferença relativa das frequências entre referência e cada janela.
4. **Visualização:** três histogramas de CPU sobrepostos.
5. **Validação:** o notebook verifica que a string da regra não foi alterada e que o PSI da janela alterada supera o da janela de controle.

Fórmula do PSI:

```text
PSI = Σ (p_atual_i - p_referencia_i) × ln(p_atual_i / p_referencia_i)
```

Quando alguma faixa tem frequência zero, o notebook usa um valor pequeno (`epsilon`) para permitir o logaritmo. Isso pode tornar o PSI grande quando as distribuições se afastam muito; não adote thresholds universais sem calibração.

## Resultados esperados e interpretação

Com a versão atual do notebook e a semente 42, o controle tende a apresentar PSI menor que a janela alterada. O notebook imprime os **valores exatos obtidos na execução**. As médias de CPU também serão distintas, sendo a amostra alterada gerada com média nominal superior à da referência.

**Interpretação correta:** um PSI maior é evidência de deslocamento na distribuição das entradas deste cenário sintético. **Não** demonstra perda de desempenho, redução da disponibilidade, falha em produção ou necessidade automática de retreinamento. Para medir MAE, seria necessário obter valores reais de resposta e comparar previsões com observações.

## Problemas comuns

- `No module named pandas` ou `matplotlib`: verifique se o kernel escolhido é `mlops-mini` e execute `python -m pip install .` **dentro de `posts/data_drift`**.
- `conda: command not found`: inicialize o Conda no terminal correspondente ou abra um shell em que o Miniconda esteja disponível.
- Kernel não aparece: execute `python -m ipykernel install --user --name mlops-mini --display-name "Python 3.12 (MLOps Mini)"` com o ambiente ativado e reinicie o VS Code.
- `code: command not found`: use **File > Open Folder** no VS Code.
- Resultados diferentes: verifique se a semente é `42`, se as células foram executadas em ordem e se o notebook foi modificado.

## Publicar no GitHub — sem alterar os laboratórios anteriores

Na **raiz** do repositório, inspecione as mudanças e adicione apenas o Lab 03:

```bash
cd ~/PythonProjects/mlops_mini
git status
git add posts/data_drift/README.md posts/data_drift/pyproject.toml posts/data_drift/Laboratorio_03_Monitoramento_Data_Drift.ipynb
git diff --cached --stat
git commit -m "feat: adiciona laboratorio 03 de monitoramento de data drift"
git push origin main
```

Não adicione credenciais, dados privados nem saídas temporárias. Se o notebook gerar saídas de execução, revise-as antes do commit. O comando `git pull --ff-only origin main` pode ser usado para atualizar o clone **antes de começar**, desde que não haja alterações locais pendentes.

## Próximo experimento

Avaliação de **desempenho preditivo** com rótulos/observações disponíveis: medir MAE ao longo de janelas, distinguir drift de degradação e estabelecer critérios auditáveis para investigar ou retreinar.
