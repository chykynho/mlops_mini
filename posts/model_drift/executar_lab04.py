from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

SEED = 20261028
rng = np.random.default_rng(SEED)
N = 400
REFERENCIA_MEDIA, REFERENCIA_DESVIO = 49.0, 8.0

# Dados inteiramente sintéticos, rótulos disponíveis após observação.
referencia_cpu = np.clip(rng.normal(49, 8, N), 0, 100)
controle_cpu = np.clip(rng.normal(49, 8, N), 0, 100)
drift_cpu = np.clip(rng.normal(70, 8, N), 0, 100)

# Preditor previamente fixado — não é retreinado nesta experiência.
def prever_memoria(cpu):
    return 15 + .7 * np.asarray(cpu)

def observar_memoria(cpu, rng, mudou_relacao=False):
    y = prever_memoria(cpu) + rng.normal(0, 2, len(cpu))
    if mudou_relacao:
        y = y + 12.0  # mudança conceitual explícita e controlada
    return y

controle_y = observar_memoria(controle_cpu, rng)
drift_y = observar_memoria(drift_cpu, rng)
conceito_y = observar_memoria(drift_cpu, rng, mudou_relacao=True)

# Limites de percentis calculados apenas sobre dados de referência.
limites = np.unique(np.quantile(referencia_cpu, np.linspace(0, 1, 11)))
limites[0], limites[-1] = -np.inf, np.inf

def psi(base, atual, limites, epsilon=1e-6):
    p = np.histogram(base, bins=limites)[0] / len(base)
    q = np.histogram(atual, bins=limites)[0] / len(atual)
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    return float(np.sum((q-p)*np.log(q/p)))

def mae(y_real, y_previsto):
    return float(np.mean(np.abs(np.asarray(y_real)-np.asarray(y_previsto))))

resultados = {
    'controle': {'psi': psi(referencia_cpu,controle_cpu,limites), 'mae': mae(controle_y,prever_memoria(controle_cpu))},
    'data_drift': {'psi': psi(referencia_cpu,drift_cpu,limites), 'mae': mae(drift_y,prever_memoria(drift_cpu))},
    'concept_drift': {'psi': psi(referencia_cpu,drift_cpu,limites), 'mae': mae(conceito_y,prever_memoria(drift_cpu))}
}
# Limites pedagógicos, NÃO políticas de produção.
LIMIAR_PSI, LIMIAR_MAE = .25, 5.0
for v in resultados.values():
    v['alerta_entrada'] = v['psi'] > LIMIAR_PSI
    v['alerta_desempenho'] = v['mae'] > LIMIAR_MAE

assert resultados['data_drift']['psi'] > resultados['controle']['psi']
assert resultados['data_drift']['mae'] < LIMIAR_MAE
assert resultados['concept_drift']['mae'] > LIMIAR_MAE
assert resultados['data_drift']['psi'] == resultados['concept_drift']['psi']
print(json.dumps(resultados,indent=2,ensure_ascii=False))

# Artefatos gráficos e saída JSON na pasta do próprio laboratório.
saida=Path('figuras');saida.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(10,5.8))
ax.hist(referencia_cpu,bins=np.arange(0,101,5),alpha=.5,density=True,label='Referência')
ax.hist(controle_cpu,bins=np.arange(0,101,5),alpha=.45,density=True,label='Controle')
ax.hist(drift_cpu,bins=np.arange(0,101,5),alpha=.45,density=True,label='Data Drift / Concept Drift')
ax.set(xlabel='CPU simulada (%)',ylabel='Densidade',title='Distribuições de entrada — cenário sintético');ax.legend();fig.tight_layout()
fig.savefig(saida/'distribuicoes.png',dpi=160);plt.close(fig)
nomes=['Controle','Data Drift','Concept Drift']
fig,axes=plt.subplots(1,2,figsize=(11,5))
for ax,chave,titulo in [(axes[0],'psi','PSI das entradas'),(axes[1],'mae','MAE após obtenção do alvo')]:
    valores=[resultados[x][chave] for x in resultados]
    bars=ax.bar(nomes,valores,color=['#168bcc','#00cdd7','#ea8f47'])
    for b,v in zip(bars,valores):ax.text(b.get_x()+b.get_width()/2,v+max(valores)*.03,f'{v:.2f}',ha='center')
    ax.set_title(titulo);ax.set_ylim(0,max(valores)*1.22)
    ax.tick_params(axis='x',labelrotation=15)
fig.tight_layout();fig.savefig(saida/'psi_mae_comparacao.png',dpi=160);plt.close(fig)
(saida/'resultados.json').write_text(json.dumps({'natureza':'dados sintéticos', 'seed':SEED,'amostras_por_janela':N,'criterios_didaticos':{'PSI':LIMIAR_PSI,'MAE':LIMIAR_MAE},'resultados':resultados},indent=2,ensure_ascii=False),encoding='utf-8')
print('Evidências criadas em figuras/')
