# Monitoramento de dados e desempenho

RMC AI Labs — José Francisco Alves dos Santos

Previsão diária de CPU com histórico real de servidores e nomes fictícios. O laboratório mantém o modelo fixo e avalia o efeito de uma falha de escala na entrada.

## Dados

Fonte: `RMC_Recursos_vialactea.xlsx`. Os arquivos `dados/cpu.csv` e `dados/mem.csv` correspondem às abas `HIST_CPU` e `HIST_MEM`, com 21.199 registros cada, 241 VMs e datas entre 12/03/2026 e 10/06/2026. Os nomes fictícios da planilha foram preservados. `dados/fonte.json` registra a origem e o hash da planilha.

A preparação mantém somente pares completos de CPU e memória com observação exatamente um dia antes. Não substitui ausências por zero. Após esse filtro, restam séries de 240 VMs. As variáveis são CPU e memória do dia anterior, e o alvo é a CPU observada no dia seguinte.

## Instalação

No repositório `mlops_mini`, coloque esta pasta em `posts/monitoramento_dados`. Reutilize o ambiente Conda `mlops-mini` com Python 3.12. A partir da raiz do repositório:

```bash
conda activate mlops-mini
python --version
cd posts/monitoramento_dados
python -m pip install .
python -m pip check
python -m ipykernel install --user --name mlops-mini --display-name "Python 3.12 (mlops-mini)"
```

Para uma primeira instalação, crie o ambiente antes de executar os comandos acima:

```bash
conda create -n mlops-mini python=3.12 -y
```

## Execução

Abra `Laboratorio_Monitoramento_Dados_Modelo.ipynb` no VS Code. Selecione o kernel **Python 3.12 (mlops-mini)**, reinicie o kernel e execute todas as células. Execute a partir da pasta do notebook para localizar `dados/`. Os gráficos e resultados são gravados em `evidencias/`.

## Método

- Treino: observações anteriores a 01/05/2026.
- Referência: 01/05/2026 a 14/05/2026.
- Avaliação: 15/05/2026 a 10/06/2026.
- Modelo: regressão linear, ajustada somente no treino.
- Falha simulada: dividir as duas entradas por 100 em uma cópia dos dados.
- Restauração: repetir as previsões com as entradas originais.

As três condições usam os mesmos 6.318 pares de avaliação. O notebook verifica que os parâmetros do modelo permanecem inalterados. O MAE é medido em pontos percentuais de CPU. O resumo por VM permite examinar diferenças que a média geral pode ocultar. O experimento também calcula a previsão por persistência como referência simples.

## Resultados

| Condição | Observações | MAE de CPU (pp) |
| --- | ---: | ---: |
| Dados originais | 6.318 | 1,0273 |
| Falha de escala simulada | 6.318 | 9,4808 |
| Entradas restauradas | 6.318 | 1,0273 |

O MAE da referência temporal é 0,9238 pp. A previsão simples por persistência tem MAE de 0,9663 pp na avaliação, inferior ao da regressão neste recorte. A escolha do modelo precisa considerar essa comparação.

## Interpretação

A falha de escala é uma intervenção controlada, não um incidente observado na origem. As medições originais permanecem preservadas. Valores em escala incorreta podem continuar dentro de 0 a 100. Por isso, conferir apenas os limites numéricos não substitui validar unidade, contrato e distribuição.

O indicador de entrada mede linhas com alguma variável fora dos percentis 1 e 99 do treino. O limite de 20% e o limite de erro de três vezes o MAE de referência são didáticos. Um sinal de mudança nas entradas não comprova, sozinho, perda de desempenho. Para medir o erro são necessários os valores observados.

O estudo não demonstra mudança natural de conceito, correção de produção, retreinamento ou disponibilidade de serviço. Restaurar a entrada representa apenas a reversão da transformação no experimento. A comparação mede desempenho nos servidores presentes no histórico.

## Arquivos

- `dados/`: medições e registro da fonte.
- `evidencias/resultados.json`: resultados agregados.
- `evidencias/erro_por_vm.csv`: MAE por servidor e quantidade de observações.
- `evidencias/monitoramento.png`: gráfico calculado pelo notebook.
- `pyproject.toml`: dependências do laboratório.

Repositório: https://github.com/chykynho/mlops_mini
