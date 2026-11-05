"""Lab 05 complementar: observabilidade temporal de histórico REAL anonimizado.
Não altera a fonte nem injeta mudanças; executa somente quando CSVs existem no clone.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

LAB=Path(__file__).resolve().parent
REPO=LAB.parents[1] # posts/monitoramento_janelas_reais -> raiz mlops_mini
FONTE=REPO/'posts'/'monitoramento_dados'/'dados'
OUT=LAB/'evidencias_reais'
OUT.mkdir(parents=True,exist_ok=True)
for name in ('cpu.csv','mem.csv','fonte.json'):
    if not (FONTE/name).is_file():
        raise FileNotFoundError(f'Arquivo ausente: {FONTE/name}. Execute dentro do clone completo de mlops_mini.')
meta=json.loads((FONTE/'fonte.json').read_text(encoding='utf-8'))
print('Fonte declarada:',meta.get('fonte'))

keys=['cluster','vm','date']
def ler(tipo):
    df=pd.read_csv(FONTE/f'{tipo}.csv',parse_dates=['date'])
    missing=set(keys+['used_pct'])-set(df.columns)
    if missing: raise ValueError(f'{tipo}: colunas ausentes: {missing}')
    if df.duplicated(keys).any():raise ValueError(f'{tipo}: chaves duplicadas')
    if not df['used_pct'].dropna().between(0,100).all():raise ValueError(f'{tipo}: medição fora de 0..100')
    return df[keys+['used_pct']]
cpu=ler('cpu'); mem=ler('mem')
dados=cpu.merge(mem,on=keys,how='inner',validate='one_to_one',suffixes=('_cpu','_mem'))
dados=dados.sort_values(['cluster','vm','date']).copy()
g=dados.groupby(['cluster','vm'],sort=False)
dados['cpu_lag']=g['used_pct_cpu'].shift(1)
dados['mem_lag']=g['used_pct_mem'].shift(1)
dados['date_lag']=g['date'].shift(1)
pares=dados.loc[(dados.date-dados.date_lag).dt.days.eq(1)].dropna(subset=['cpu_lag','mem_lag','used_pct_cpu']).copy()
train=pares[pares.date<pd.Timestamp('2026-05-01')].copy()
ref=pares[(pares.date>=pd.Timestamp('2026-05-01'))&(pares.date<pd.Timestamp('2026-05-15'))].copy()
val=pares[pares.date>=pd.Timestamp('2026-05-15')].copy()
if min(len(train),len(ref),len(val))<20:raise ValueError('Amostras insuficientes para as três janelas temporais; confira o histórico.')
feat=['cpu_lag','mem_lag']
model=LinearRegression().fit(train[feat],train.used_pct_cpu)
ref['pred']=model.predict(ref[feat]);ref['erro']=abs(ref.used_pct_cpu-ref.pred)
val['pred']=model.predict(val[feat]);val['erro']=abs(val.used_pct_cpu-val.pred)

# PSI: referências e bordas calculadas uma vez, sem acesso aos dados futuros.
def make_bins(a):
    b=np.unique(np.quantile(np.asarray(a,dtype=float),np.linspace(0,1,11)))
    if len(b)<3:raise ValueError('Distribuição constante: PSI por quantis não definido.')
    b[0]=-np.inf;b[-1]=np.inf
    return b
bins={f:make_bins(ref[f].to_numpy()) for f in feat}
def calc_psi(reference,current,edges,alpha=.5):
    a=np.histogram(reference,bins=edges)[0].astype(float)
    b=np.histogram(current,bins=edges)[0].astype(float)
    p=(a+alpha)/(a.sum()+alpha*len(a))
    q=(b+alpha)/(b.sum()+alpha*len(b))
    return float(((q-p)*np.log(q/p)).sum())

# Referência de monitoramento para limiar: 95o percentil do MAE por VM em janelas de 7 observações no período de referência.
ref=ref.sort_values(['cluster','vm','date'])
ref['mae7']=ref.groupby(['cluster','vm']).erro.transform(lambda s:s.rolling(7,min_periods=5).mean())
maes_ref=ref['mae7'].dropna()
if len(maes_ref)<10:raise ValueError('Poucas observações de referência para calibrar MAE por VM.')
limiar_mae=float(maes_ref.quantile(.95))

# Calcular somente para janelas de avaliação realmente existentes.
val=val.sort_values(['cluster','vm','date']).copy()
calendario=sorted(val.date.unique())
rows=[]
for date in calendario:
    start=pd.Timestamp(date)-pd.Timedelta(days=6)
    window=val[(val.date>=start)&(val.date<=date)]
    if len(window)<30: continue
    rows.append({'date':str(pd.Timestamp(date).date()),'n':int(len(window)),
                 'PSI_CPU':calc_psi(ref.cpu_lag,window.cpu_lag,bins['cpu_lag']),
                 'PSI_MEM':calc_psi(ref.mem_lag,window.mem_lag,bins['mem_lag']),
                 'MAE7_GLOBAL':float(window.erro.mean())})
agg=pd.DataFrame(rows)
if agg.empty:raise ValueError('Sem janelas de monitoramento suficientes')

# Dias consecutivos de ALERTA POR VM: usa janelas móveis de 7 observações (mínimo 5) e continuidade do calendário.
val['MAE7_VM']=val.groupby(['cluster','vm']).erro.transform(lambda s:s.rolling(7,min_periods=5).mean())
val['alerta_mae']=val['MAE7_VM'].gt(limiar_mae)&val['MAE7_VM'].notna()
results=[]
for (cluster,vm),g in val.groupby(['cluster','vm'],sort=True):
    streak=0;previous=None
    for x in g.sort_values('date').itertuples():
        is_contiguous=previous is not None and (x.date-previous).days==1
        streak=(streak+1 if is_contiguous else 1) if x.alerta_mae else 0
        results.append({'cluster':cluster,'vm':vm,'date':str(x.date.date()),'cpu_observada':float(x.used_pct_cpu),
                        'pred_cpu':float(x.pred),'erro_abs':float(x.erro),'MAE7_VM':None if pd.isna(x.MAE7_VM) else float(x.MAE7_VM),
                        'alerta_mae':bool(x.alerta_mae),'dias_consecutivos':streak,
                        'decisao':'INVESTIGAR_PERSISTENTE' if streak>=3 else ('OBSERVAR' if x.alerta_mae else 'NORMAL')})
        previous=x.date
vmday=pd.DataFrame(results)
last_date=vmday.date.max();latest=vmday[vmday.date==last_date]
summary={'fonte':meta.get('fonte'),'tipo_dados':'medições históricas reais, identificadores fictícios',
         'periodo_treino':'antes de 2026-05-01','periodo_referencia':'2026-05-01 a 2026-05-14',
         'periodo_avaliacao':'a partir de 2026-05-15','n_treino':len(train),'n_referencia':len(ref),'n_avaliacao':len(val),
         'n_vms_avaliadas':int(val.vm.nunique()),'data_final':last_date,
         'mae_ref_pp':round(float(ref.erro.mean()),4),'limiar_mae_vm_pp_95p':round(limiar_mae,4),
         'janela_dias':7,'persistencia_dias':3,
         'vms_investigar_data_final':int((latest.decisao=='INVESTIGAR_PERSISTENTE').sum()),
         'PSI_CPU_ultima_janela':round(float(agg.PSI_CPU.iloc[-1]),4),
         'PSI_MEM_ultima_janela':round(float(agg.PSI_MEM.iloc[-1]),4),
         'MAE7_GLOBAL_ultima_janela':round(float(agg.MAE7_GLOBAL.iloc[-1]),4),
         'advertencia':'PSI nao prova concept drift. MAE maior sozinho nao prova mudanca de P(Y|X). Limiares exploratorios.'}
agg.to_csv(OUT/'metricas_janelas_7d.csv',index=False)
vmday.to_csv(OUT/'metricas_vm_dia.csv',index=False)
(OUT/'resultados.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

plt.rcParams.update({'font.size':10})
fig,ax=plt.subplots(figsize=(11,5));ax.plot(agg.date,agg.PSI_CPU,label='PSI CPU');ax.plot(agg.date,agg.PSI_MEM,label='PSI MEM');ax.set_title('PSI — histórico real anonimizado (janelas 7 dias)');ax.set_ylabel('PSI');ax.legend();ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=65);ax.set_xticks(agg.date[::max(1,len(agg)//8)]);fig.tight_layout();fig.savefig(OUT/'psi_janelas_reais.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5));ax.plot(agg.date,agg.MAE7_GLOBAL,label='MAE agregado');ax.axhline(ref.erro.mean(),linestyle='--',label='MAE referência');ax.set_title('Erro de previsão de CPU — histórico real');ax.set_ylabel('MAE (pontos percentuais de CPU)');ax.legend();ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=65);ax.set_xticks(agg.date[::max(1,len(agg)//8)]);fig.tight_layout();fig.savefig(OUT/'mae_janelas_reais.png',dpi=150);plt.close(fig)
# Top 20 VM por MAE7 médio; não atribui causa a nenhuma delas.
rank=vmday.groupby('vm').MAE7_VM.mean().dropna().nlargest(20)
fig,ax=plt.subplots(figsize=(11,7));rank.sort_values().plot.barh(ax=ax);ax.set(xlabel='MAE 7 observações (pp de CPU)',title='VMs com maiores erros médios na avaliação');fig.tight_layout();fig.savefig(OUT/'ranking_mae_vms.png',dpi=150);plt.close(fig)
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('Evidências:',OUT)
