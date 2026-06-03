# Relatório - AV1: Apoio à Decisão para Campanhas Educativas no Trânsito

**Curso:** Sistemas de Informação  
**Disciplina:** Sistemas de Apoio à Decisão  
**Modelo:** Árvore de Decisão (Entropia / Ganho de Informação)  
**Dados:** Sinistros de Trânsito PRF 2025 - Portal de Dados Abertos do Governo Federal

---

## 1. Contexto e Objetivo

O planejamento de campanhas educativas de trânsito é historicamente baseado em experiências práticas e estatísticas gerais agregadas. Esse modelo de tomada de decisão deixa lacunas importantes:

1. Não identifica quais fatores específicos mais contribuem para a gravidade dos sinistros;
2. Não permite priorizar o público-alvo ou o tipo de intervenção de forma orientada por dados.

Este trabalho adota uma abordagem de aprendizado de máquina supervisionado para responder à seguinte **questão gerencial**:

> *Quais características dos sinistros estão mais associadas à ocorrência de vítimas e devem ser priorizadas como foco de campanhas educativas e ações preventivas?*

O modelo desenvolvido tem três objetivos práticos para o gestor público:

1. Identificar atributos críticos associados à gravidade dos sinistros
2. Transformar dados históricos em conhecimento aplicável à política pública
3. Apoiar decisões estratégicas de prevenção com base em evidências

---

## 2. Base de Dados

### Fonte

**Polícia Rodoviária Federal (PRF)** - Portal de Dados Abertos do Governo Federal  
Conjunto: *Sinistros de Trânsito Agrupados por Ocorrência*  
Período: ano **2025**  
Acesso: https://dados.gov.br/dados/conjuntos-dados/sinistros-de-transito-agrupados-por-ocorrencia

### Características do Dataset

| Atributo | Valor |
|---|---|
| Registros totais | 72.529 |
| Colunas | 30 |
| Formato | CSV, separador `;`, encoding `latin-1` |
| Registros utilizados | 72.528 (1 removido por ausência de classificação) |

### Variável Alvo

A coluna `classificacao_acidente` possui três categorias:

| Categoria | Registros | % |
|---|---|---|
| Com Vítimas Feridas | 56.181 | 77,46% |
| Com Vítimas Fatais | 5.209 | 7,18% |
| Sem Vítimas | 11.138 | 15,36% |

Para o modelo, essa variável foi **binarizada**:

- **1 - Com Vítimas**: agrupa "Com Vítimas Feridas" e "Com Vítimas Fatais" → 61.390 registros (84,64%)
- **0 - Sem Vítimas**: mantém "Sem Vítimas" → 11.138 registros (15,36%)

### Features Selecionadas

Foram utilizadas exclusivamente **colunas categóricas sem data leakage**, ou seja, sem colunas que são consequência direta do resultado do sinistro (`mortos`, `feridos_leves`, `feridos_graves`). Usar essas colunas tornaria o modelo inútil na prática, pois elas só existem depois que o sinistro acontece.

| Feature | Categorias | Descrição |
|---|---|---|
| `causa_acidente` | 69 | Causa principal do sinistro |
| `tipo_acidente` | 17 | Tipo (colisão traseira, tombamento, atropelamento…) |
| `fase_dia` | 4 | Amanhecer / Pleno dia / Anoitecer / Plena noite |
| `condicao_metereologica` | 9 | Chuva, céu claro, nublado, garoa… |
| `tipo_pista` | 3 | Simples, dupla ou múltipla |
| `tracado_via` | 605 | Reta, curva, declive, aclive e combinações |
| `uso_solo` | 2 | Área urbana (Sim) ou rural (Não) |
| `sentido_via` | 3 | Crescente / Decrescente / Não informado |
| `dia_semana` | 7 | Dia da semana |
| `uf` | 27 | Unidade Federativa |

---

## 3. Fundamentação Teórica - Árvore de Decisão

A Árvore de Decisão é um modelo de aprendizado supervisionado que utiliza a estratégia de **dividir para conquistar**: um problema complexo é decomposto em sub-problemas mais simples, recursivamente, até que cada subconjunto seja suficientemente homogêneo (GAMA, 2004).

A estrutura da árvore é composta por:
- **Nós de decisão**: contêm um teste sobre um atributo
- **Ramos**: correspondem aos possíveis valores do atributo testado
- **Folhas**: associadas a uma classe, no nosso caso, *Com Vítimas* ou *Sem Vítimas*
- **Percurso raiz → folha**: equivale a uma regra de classificação do tipo SE-ENTÃO

### Critério de Divisão: Entropia e Ganho de Informação

O algoritmo implementado usa **Entropia** como medida de impureza de um conjunto de dados:

$$Entropia(S) = -\sum_{i=1}^{c} p_i \log_2 p_i$$

Onde $p_i$ é a proporção de amostras da classe $i$ no conjunto $S$.

- Entropia **0**: conjunto completamente puro (todos os registros são da mesma classe)
- Entropia **1**: conjunto completamente heterogêneo (classes igualmente distribuídas)

O **Ganho de Informação** mede a redução de entropia ao particionar os dados pelo atributo $A$:

$$IG(S, A) = Entropia(S) - \sum_{x \in P(A)} \frac{|S_x|}{|S|} \cdot Entropia(S_x)$$

A cada nó, o algoritmo calcula o ganho de informação para todos os atributos disponíveis e seleciona aquele com o **maior ganho** para realizar a divisão. Esse processo é equivalente ao algoritmo **ID3** e ao **C4.5** (QUINLAN, 1993; MITCHELL, 1997).

### Controle de Overfitting

*Overfitting* (sobreajuste) ocorre quando a árvore se ajusta excessivamente aos dados de treinamento, capturando ruídos e exceções que não representam o comportamento geral, resultando em baixo desempenho em dados novos (BREIMAN et al., 1984).

Para prevenir overfitting, foram definidos dois hiperparâmetros de poda preventiva (pré-podagem):

| Hiperparâmetro | Valor | Justificativa |
|---|---|---|
| `max_depth` | 7 | Limita a profundidade, evitando regras excessivamente específicas |
| `min_samples_leaf` | 50 | Exige ao menos 50 amostras por folha, garantindo relevância estatística |

---

## 4. Implementação

### Ferramentas Utilizadas

| Ferramenta | Versão | Finalidade |
|---|---|---|
| Python | 3.11 | Linguagem principal |
| pandas | ≥ 2.0 | Manipulação e limpeza dos dados |
| scikit-learn | ≥ 1.3 | Modelo, codificação e métricas |
| matplotlib / seaborn | ≥ 3.7 / 0.12 | Visualizações |

### Pipeline de Desenvolvimento

```
CSV (PRF 2025)
    │
    ▼
[1] Carregamento e inspeção
    │   └─ pandas.read_csv (sep=';', encoding='latin-1')
    │
    ▼
[2] Limpeza
    │   └─ Remoção de 1 registro sem classificação
    │
    ▼
[3] Engenharia da variável alvo
    │   └─ Binarização: Com Vítimas=1 / Sem Vítimas=0
    │
    ▼
[4] Codificação das features
    │   └─ One-Hot Encoding por coluna categórica (1 binária por categoria)
    │
    ▼
[5] Divisão treino/teste
    │   └─ 80% treino / 20% teste (estratificado)
    │
    ▼
[6] Treinamento
    │   └─ DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_leaf=50)
    │
    ▼
[7] Avaliação
    │   └─ Acurácia + Matriz de Confusão + Classification Report
    │
    ▼
[8] Interpretação
        └─ feature_importances_ + export_text + plot_tree
```

### Divisão Treino / Teste

A divisão foi feita com **estratificação**, garantindo que a proporção entre as classes (84,6% Com Vítimas / 15,4% Sem Vítimas) seja mantida tanto no conjunto de treino quanto no de teste, evitando viés na avaliação.

| Conjunto | Registros |
|---|---|
| Treino (80%) | 58.022 |
| Teste (20%) | 14.506 |

---

## 5. Treinamento e Resultado da Árvore

O modelo foi treinado com o seguinte comando:

```python
modelo = DecisionTreeClassifier(
    criterion='entropy',       # Ganho de Informação (ID3/C4.5)
    max_depth=7,               # profundidade máxima
    min_samples_leaf=50,       # folhas estatisticamente relevantes
    random_state=42            # reprodutibilidade
)
modelo.fit(X_train, y_train)
```

### Estrutura da Árvore Resultante

| Métrica | Valor |
|---|---|
| Total de nós | 85 |
| Profundidade real | 7 |
| Folhas | 43 |

A árvore seleciona uma categoria de `tipo_acidente` como **nó raiz** (`tipo_acidente_Incêndio`). A importância agregada (Seção 7) confirma que `tipo_acidente` e `causa_acidente` são, juntos, os atributos que mais discriminam sinistros com e sem vítimas — ou seja, a **dinâmica do evento e o comportamento do condutor** pesam mais que as condições da via ou a meteorologia.

---

## 6. Avaliação do Modelo

### Acurácia

| Conjunto | Acurácia |
|---|---|
| Treino | **87,08%** |
| Teste | **86,76%** |
| Diferença | 0,31% |

A diferença de apenas 0,31 ponto percentual entre treino e teste indica **ausência de overfitting**: o modelo generaliza bem para dados novos.

### Matriz de Confusão

|  | Previsto: Sem Vítimas | Previsto: Com Vítimas |
|---|---|---|
| **Real: Sem Vítimas** | 325 (VN) | 1.903 (FP) |
| **Real: Com Vítimas** | 17 (FN) | 12.261 (VP) |

**Interpretação:**

- O modelo identifica corretamente **12.261 de 12.278** sinistros com vítimas (recall de 99,9%) - alta sensibilidade para a classe crítica
- O principal erro são os **Falsos Positivos**: 1.903 sinistros sem vítimas classificados como com vítimas (o modelo "peca pelo excesso de cautela", o que é aceitável em contexto de segurança pública)
- Apenas **17 sinistros com vítimas** foram incorretamente classificados como sem vítimas - erro de alta gravidade praticamente eliminado
- Ressalva: a base é desbalanceada (84,6% com vítimas), então a acurácia é puxada pela classe majoritária; o **recall de apenas 15% na classe "Sem Vítimas"** mostra que o modelo tem dificuldade de identificar sinistros sem vítimas. O valor gerencial está na priorização por importância de atributos, não na predição individual

### Relatório de Classificação

| Classe | Precisão | Recall | F1-Score | Suporte |
|---|---|---|---|---|
| Sem Vítimas | 0,95 | 0,15 | 0,25 | 2.228 |
| Com Vítimas | 0,87 | 1,00 | 0,93 | 12.278 |
| **Média ponderada** | **0,88** | **0,87** | **0,82** | **14.506** |

O F1-Score de **0,93** para a classe "Com Vítimas" confirma a boa performance do modelo para o caso de maior interesse gerencial.

---

## 7. Atributos Relevantes - Análise de Importância

A importância de cada feature é calculada como a **redução total de entropia ponderada pelo número de amostras** que passam pelos nós onde aquele atributo foi usado para divisão. Como as features foram codificadas com One-Hot, a importância nativa do modelo é **por categoria** (ex.: `tipo_acidente_Colisão frontal`); os valores abaixo são **reagregados por feature original** (soma das colunas de cada feature).

| Rank | Feature | Importância | % Acumulado |
|---|---|---|---|
| 1 | `tipo_acidente` | 78,47% | 78,47% |
| 2 | `causa_acidente` | 15,40% | 93,87% |
| 3 | `sentido_via` | 2,63% | 96,50% |
| 4 | `fase_dia` | 1,70% | 98,20% |
| 5 | `uf` | 1,22% | 99,42% |
| 6 | `uso_solo` | 0,25% | 99,67% |
| 7 | `condicao_metereologica` | 0,20% | 99,87% |
| 8 | `tipo_pista` | 0,11% | 99,98% |
| 9 | `tracado_via` | 0,02% | 100,00% |
| 10 | `dia_semana` | 0,00% | 100,00% |

### Interpretação

Os dois primeiros atributos - `tipo_acidente` e `causa_acidente` - respondem por **93,87% do poder explicativo** do modelo. Isso revela que a gravidade do sinistro é determinada principalmente por **fatores comportamentais e dinâmicos do evento** (que tipo de colisão ocorre e o que a causou), e não pelas condições da via ou meteorologia. Essa descoberta contraria a percepção intuitiva de que fatores externos (chuva, pista ruim) seriam os principais vilões.

### Nota metodológica - escolha do encoding

As 10 features são **nominais** (sem ordem natural). A versão inicial usava `LabelEncoder`, que atribui um inteiro arbitrário a cada categoria. Em uma árvore de decisão, essa ordem falsa permite cortes em faixas de códigos sem sentido e, sobretudo, **infla a importância de features de alta cardinalidade** (viés conhecido). A troca para `OneHotEncoder` mantém a acurácia praticamente idêntica (86,58% → 86,76%), mas corrige o ranking de importância — que é o produto que fundamenta as campanhas. O efeito ficou evidente em `tracado_via`: com label encoding aparecia com 0,64% de importância; com one-hot cai para 0,02%, revelando que era um artefato de cardinalidade.

> 🚩 **Alerta de qualidade de dado:** `tracado_via` apresenta **605 categorias**, pois a base PRF combina múltiplos traçados em um mesmo campo (ex.: `Aclive;Curva;Em Obras`). Recomenda-se, em trabalhos futuros, **decompor** esse campo em colunas binárias por traçado (curva, aclive, em obras…) para uma análise mais limpa.

---

## 8. Proposta de Campanhas Educativas

Com base nos atributos identificados como mais relevantes, propõem-se cinco campanhas prioritárias. A identificação dos fatores pelos dados permite **otimizar a alocação de recursos**, evitando campanhas genéricas de baixo impacto.

---

### Campanha 1 - "Distância Segura Salva Vidas"
**Feature:** `tipo_acidente` (importância: 78,47%)

Os tipos de acidente mais frequentes com vítimas são: *colisão traseira*, *colisão frontal*, *saída de leito carroçável*, *tombamento* e *atropelamento de pedestre*. A colisão traseira, em particular, está diretamente relacionada à falta de distância segura e atenção do condutor ao veículo à frente.

| Item | Detalhe |
|---|---|
| **Foco** | Prevenção de colisões traseiras, frontais, saídas de pista e atropelamentos |
| **Ações** | Painéis de mensagem variável indicando distância segura; fiscalização com câmeras inteligentes de ultrapassagem; treinamento específico para motoristas de veículos pesados |
| **Público-alvo** | Motoristas de caminhões, ônibus e veículos de carga |

---

### Campanha 2 - "Dirija com Atenção"
**Feature:** `causa_acidente` (importância: 15,40%)

As causas de maior incidência identificadas no dataset incluem: *ausência de reação do condutor*, *reação tardia ou ineficiente*, *velocidade incompatível* e *ingestão de álcool*. Todas são causas comportamentais e, portanto, passíveis de intervenção educativa.

| Item | Detalhe |
|---|---|
| **Foco** | Comportamento do condutor: distração, fadiga, álcool, velocidade |
| **Ações** | Blitze educativas nas rodovias com maior incidência; parceria com autoescolas; alertas contextuais em apps de navegação (Waze, Google Maps); campanha nas redes sociais com dados reais de sinistros |
| **Público-alvo** | Condutores em geral, especialmente jovens de 18 a 30 anos |

---

### Campanha 3 - "Sentido Correto - Vida Garantida"
**Feature:** `sentido_via` (importância: 2,63%)

O sentido da via aparece associado a sinistros mais graves, especialmente em casos de contramão e ultrapassagens indevidas em rodovias de pista simples, onde o risco de colisão frontal é elevado.

| Item | Detalhe |
|---|---|
| **Foco** | Contramão e ultrapassagens indevidas |
| **Ações** | Fiscalização com câmeras de monitoramento inteligente; educação sobre regras de preferência e sinalização de sentido |
| **Público-alvo** | Condutores em rodovias de pista simples |

---

### Campanha 4 - "Noite Exige Mais Cuidado"
**Feature:** `fase_dia` (importância: 1,70%)

Sinistros ocorridos em *plena noite* e *anoitecer* apresentam maior proporção de vítimas, associados à redução da visibilidade e ao aumento da fadiga do condutor.

| Item | Detalhe |
|---|---|
| **Foco** | Plena noite e madrugada |
| **Ações** | Programas de descanso obrigatório em postos homologados; alertas luminosos em pontos críticos noturnos; campanha de revisão do sistema de iluminação veicular |
| **Público-alvo** | Caminhoneiros, motoristas de aplicativo e viajantes noturnos |

---

### Campanha 5 - Campanhas Regionalizadas por Estado
**Feature:** `uf` (importância: 1,22%)

A variação por estado indica que o perfil dos sinistros difere significativamente entre as UFs, influenciado por características da malha viária, frota, clima e perfil socioeconômico local.

| Item | Detalhe |
|---|---|
| **Foco** | Estados com maior concentração de sinistros com vítimas |
| **Ações** | Parcerias entre SENATRAN e DESTRANs estaduais; campanhas com linguagem e contexto adaptados ao perfil regional |
| **Público-alvo** | Órgãos gestores estaduais de trânsito e condutores por UF |

---

## 9. Conclusão

O modelo de Árvore de Decisão treinado sobre **72.528 registros reais** da PRF alcançou **86,76% de acurácia** no conjunto de teste, sem overfitting, e revelou um resultado de alto valor para a política pública:

> **93,87% do poder explicativo da gravidade dos sinistros está concentrado em dois atributos: o tipo do acidente e a causa do acidente**, ambos de natureza predominantemente comportamental.

Isso significa que **investir em educação e fiscalização do comportamento do condutor tem maior retorno esperado** do que focar exclusivamente em melhorias de infraestrutura, sinalização ou condições climáticas, embora esses fatores também sejam relevantes.

O modelo gera regras interpretáveis que permitem ao gestor público:
- Priorizar o tipo de campanha com base em evidências
- Segmentar o público-alvo por perfil de risco real
- Monitorar a evolução dos padrões ao retreinar o modelo com dados futuros

---

## Referências

- BREIMAN, L. et al. *Classification and Regression Trees (CART)*. Wadsworth, 1984.
- GAMA, J. *Árvore de Decisão*. PUC-Rio - Certificação Digital Nº 0024879/CA, 2004.
- MITCHELL, T. *Machine Learning*. McGraw-Hill, 1997.
- QUINLAN, J. R. *C4.5: Programs for Machine Learning*. Morgan Kaufmann, 1993.
- POLÍCIA RODOVIÁRIA FEDERAL. *Sinistros de Trânsito - Portal de Dados Abertos*. Disponível em: https://dados.gov.br/dados/conjuntos-dados/sinistros-de-transito-agrupados-por-ocorrencia
