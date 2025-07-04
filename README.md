# 🎯 Projeto de Análise Estatística da Mega-Sena

Este projeto realiza análise estatística completa dos resultados históricos da Mega-Sena e gera predições baseadas em diferentes métodos estatísticos e machine learning.

## 🚀 Funcionalidades

### 📊 Análises Estatísticas Implementadas:

1. **Análise de Frequência**
   - Números mais e menos sorteados
   - Distribuição de frequências
   - Análise por faixas de números

2. **Análise de Padrões**
   - Distribuição da soma das dezenas
   - Padrões de números pares e ímpares
   - Análise de amplitude (diferença entre maior e menor número)
   - Identificação de sequências numéricas

3. **Análise Temporal**
   - Evolução dos padrões ao longo dos anos
   - Sazonalidade por meses
   - Análise por dias da semana

4. **Análise de Correlações**
   - Correlações entre diferentes variáveis estatísticas
   - Identificação de relações entre padrões

5. **Predições Inteligentes**
   - **Método Estatístico**: Baseado em frequências históricas e tendências recentes
   - **Método Machine Learning**: Usando Random Forest para predição
   - **Método Combinado**: Combinação dos dois métodos anteriores

## 🛠️ Instalação e Uso

### Pré-requisitos
- Python 3.8+
- Conexão com internet (para buscar dados da API)

### Dependências Principais
```
pandas>=1.3.0          # Manipulação de dados
numpy>=1.21.0          # Computação numérica
scikit-learn>=1.0.0    # Machine Learning (Random Forest)
requests>=2.25.0       # Requisições HTTP para API
```

### Instalação
```bash
# Clone o repositório
git clone https://github.com/andersonlpd/Projecao_Loteria.git

# Navegue para o diretório
cd Projecao_Loteria

# Instale as dependências
pip install -r requirements.txt
```

### Uso
```bash
# Execute o menu interativo (recomendado)
python menu.py

# Ou execute diretamente o analisador principal
python megasena_analyzer.py
```

### ⚙️ Configurações Avançadas

Para usuários avançados, é possível ajustar os parâmetros do Random Forest editando o arquivo `megasena_analyzer.py`:

```python
# Parâmetros do Random Forest (linha ~870)
model = RandomForestRegressor(
    n_estimators=100,      # Número de árvores (padrão: 100)
    max_depth=10,          # Profundidade máxima (padrão: 10)
    min_samples_split=5,   # Amostras mínimas para dividir (padrão: 5)
    min_samples_leaf=2,    # Amostras mínimas por folha (padrão: 2)
    random_state=42        # Seed para reprodutibilidade
)
```

## 📁 Estrutura do Projeto

```
Projecao_Loteria/
├── megasena_analyzer.py      # Análise principal e predições
├── menu.py                   # Interface interativa
├── requirements.txt          # Dependências do projeto
└── README.md                 # Esta documentação
```

## 🔍 Detalhes das Análises

### Análise de Frequência
- **Números Quentes**: Mais sorteados historicamente
- **Números Frios**: Menos sorteados historicamente
- **Distribuição Equilibrada**: Análise se há tendências significativas

### Análise de Padrões
- **Soma Ideal**: Faixa de soma mais comum (geralmente entre 150-200)
- **Proporção Pares/Ímpares**: Tendência de 3 pares e 3 ímpares
- **Amplitude**: Diferença entre o maior e menor número do sorteio
- **Sequências**: Identificação de números consecutivos

### Métodos de Predição

#### 1. Método Estatístico
- Análise de frequência histórica (70% do peso)
- Tendência dos últimos 20 sorteios (30% do peso)
- Aplicação de restrições para jogo equilibrado

#### 2. Método Machine Learning

O método de Machine Learning utiliza o algoritmo **Random Forest Regressor** para identificar padrões complexos nos dados históricos da Mega-Sena. Este é um dos métodos mais avançados do projeto.

##### 🤖 Algoritmo Random Forest - Conceitos Fundamentais

**Random Forest** é um algoritmo de ensemble que combina múltiplas árvores de decisão para criar predições mais robustas e precisas:

- **Ensemble Learning**: Combina predições de 100 árvores de decisão independentes
- **Bagging**: Cada árvore é treinada com uma amostra aleatória dos dados
- **Feature Randomness**: Cada divisão nas árvores considera apenas um subconjunto aleatório das features
- **Resistência a Overfitting**: A combinação de múltiplas árvores reduz o risco de sobreajuste

##### 🔧 Arquitetura do Modelo

**1. Janela Temporal Adaptativa**
O modelo usa uma janela temporal que se adapta automaticamente ao volume de dados disponíveis:
- **≥100 sorteios**: Janela de 15 sorteios (90 features)
- **≥50 sorteios**: Janela de 10 sorteios (60 features)
- **≥20 sorteios**: Janela de 5 sorteios (30 features)
- **<20 sorteios**: Janela de 3 sorteios ou 1/4 do total (18+ features)

**2. Estrutura de Features**
- **Input**: Sequência de N sorteios anteriores (6 números × N sorteios)
- **Output**: Predição para as 6 posições do próximo sorteio
- **Exemplo**: Se janela = 10, cada amostra tem 60 features (10 sorteios × 6 números)

**3. Múltiplos Modelos Especializados**
O sistema treina 6 modelos Random Forest separados:
- **Modelo 1**: Prediz a 1ª dezena do próximo sorteio
- **Modelo 2**: Prediz a 2ª dezena do próximo sorteio
- **...**: (e assim por diante)
- **Modelo 6**: Prediz a 6ª dezena do próximo sorteio

##### 🎯 Processo de Treinamento

**Fase 1: Preparação dos Dados**
```
Sorteio 1: [05, 12, 23, 34, 45, 56] ─┐
Sorteio 2: [03, 15, 28, 39, 41, 58] ─┤ Features (Janela)
Sorteio 3: [07, 19, 25, 33, 47, 52] ─┘
                    ↓
Sorteio 4: [02, 18, 29, 35, 44, 59] ← Target (Predição)
```

**Fase 2: Configuração do Random Forest**
- **n_estimators**: 100 árvores de decisão
- **max_depth**: 10 níveis máximos por árvore
- **min_samples_split**: 5 amostras mínimas para dividir um nó
- **min_samples_leaf**: 2 amostras mínimas por folha
- **random_state**: 42 (para reprodutibilidade)

**Fase 3: Treinamento Multi-Output**
- Cada um dos 6 modelos é treinado independentemente
- Divisão treino/teste: 80%/20%
- Validação com métrica R² (coeficiente de determinação)

##### 📊 Processo de Predição

**Passo 1: Coleta das Features**
- Extrai os últimos N sorteios (conforme janela adaptativa)
- Converte em vetor de features: [num1, num2, ..., num6N]

**Passo 2: Predição Individual**
- Cada modelo especializado prediz sua posição específica
- Resultado: 6 números (podem ter duplicatas)

**Passo 3: Pós-processamento**
- **Validação de Range**: Garante números entre 1-60
- **Remoção de Duplicatas**: Elimina números repetidos
- **Completamento**: Adiciona números aleatórios se necessário
- **Ordenação**: Retorna os 6 números ordenados

##### 🧠 Vantagens do Random Forest

1. **Captura Padrões Complexos**: Identifica relações não-lineares entre sorteios
2. **Robustez**: Menos sensível a outliers e ruídos nos dados
3. **Não Requer Pré-processamento**: Funciona bem com dados numéricos brutos
4. **Interpretabilidade**: Permite análise da importância das features
5. **Versatilidade**: Adapta-se a diferentes volumes de dados históricos

##### 📈 Métricas de Avaliação

O sistema avalia a qualidade do modelo através de:
- **Score R²**: Mede a qualidade do ajuste (0 a 1, onde 1 é perfeito)
- **Validação Estatística**: Compara predições com médias históricas
- **Análise de Amplitude**: Verifica se predições estão em faixas realistas
- **Distribuição Pares/Ímpares**: Valida equilíbrio das predições

##### ⚠️ Limitações e Considerações

1. **Aleatoriedade Fundamental**: Loterias são eventos aleatórios por natureza
2. **Overfitting**: Mesmo com Random Forest, pode ocorrer sobreajuste
3. **Dados Insuficientes**: Requer pelo menos 20 sorteios para treinamento confiável
4. **Falsa Correlação**: Padrões podem ser coincidências estatísticas

O modelo serve como ferramenta de análise estatística avançada, mas não garante acertos nas predições.

#### 3. Método Combinado
- Combina os dois métodos anteriores
- Elimina duplicatas e otimiza o resultado

### 🎯 Comparação dos Métodos de Predição

| Aspecto | Método Estatístico | Machine Learning | Método Combinado |
|---------|-------------------|------------------|------------------|
| **Base** | Frequências históricas | Padrões sequenciais | Fusão inteligente |
| **Estabilidade** | ✅ Alta | ⚠️ Média | ✅ Alta |
| **Adaptabilidade** | ⚠️ Média | ✅ Alta | ✅ Alta |
| **Interpretabilidade** | ✅ Alta | ❌ Baixa | ✅ Média |
| **Dados Necessários** | ≥10 sorteios | ≥20 sorteios | ≥20 sorteios |
| **Complexidade** | Baixa | Alta | Média |

### 📊 Exemplo de Saída do Machine Learning

```
🤖 RESULTADO DO MACHINE LEARNING:
Preparando dados de treino...
Total de sorteios disponíveis: 2756
Janela temporal adaptativa: 15 sorteios
Features por amostra: 90 números

Dataset criado:
  Features shape: (2741, 90)
  Targets shape: (2741, 6)
  Amostras de treino: 2741
  Aproveitamento dos dados: 99.5%

Treinando 6 modelos (um para cada posição):
  Modelo 1/6: Score R² no teste: 0.1245
  Modelo 2/6: Score R² no teste: 0.1156
  ...
  Modelo 6/6: Score R² no teste: 0.1089

Predição ML: [08, 15, 27, 34, 49, 58]
Soma: 191
Pares: 4
Amplitude: 50
✅ Predição dentro de faixa estatística razoável
```

## 📊 Menu Interativo

O arquivo `menu.py` oferece uma interface amigável com as seguintes opções:

1. 🚀 **Predição Rápida** - Recomendação imediata para o próximo sorteio
2. 📊 **Análise de Frequência** - Números mais e menos sorteados
3. 🔍 **Análise de Padrões** - Padrões estatísticos identificados
4. ⏰ **Análise Temporal** - Evolução dos padrões ao longo do tempo
5. 📈 **Relatório Completo** - Análise completa com todas as estatísticas
6. ❓ **Sobre as Análises** - Informações sobre os métodos utilizados

## 📊 Interpretação dos Resultados

### Sugestões Estatísticas Fornecidas:
1. **Faixa de Soma Recomendada**: Baseada na média ± desvio padrão histórico
2. **Quantidade de Pares**: Recomendação baseada na distribuição histórica
3. **Números Quentes e Frios**: Sugestão de mix equilibrado
4. **Amplitude Recomendada**: Faixa ideal de distribuição dos números

### Exemplo de Saída:
```
📊 SUGESTÕES PARA PRÓXIMO JOGO:
• Soma recomendada: 165 a 195
• Quantidade de pares recomendada: 3
• Considere incluir números 'quentes': [10, 53, 33, 23, 42]
• Considere incluir números 'frios': [13, 26, 46]
• Amplitude recomendada: 45 ± 10
```

## ⚠️ Avisos Importantes

1. **Este projeto é para fins educacionais e estatísticos**
2. **Jogos de loteria são baseados em sorteios aleatórios**
3. **Não há garantia de acerto nas predições**
4. **Jogue com responsabilidade**

## 🔧 Personalização

### Adicionar Novas Análises:
O código é modular e permite fácil extensão com novas análises estatísticas.

### 🚀 Performance e Otimização

#### Tempo de Execução (aproximado)
- **Método Estatístico**: ~2-5 segundos
- **Machine Learning**: ~10-30 segundos (dependendo do histórico)
- **Método Combinado**: ~15-35 segundos

#### Fatores que Afetam a Performance
1. **Volume de Dados**: Mais sorteios = mais tempo de treinamento
2. **Janela Temporal**: Janelas maiores = mais features = mais processamento
3. **Número de Árvores**: Mais árvores = maior precisão = mais tempo
4. **Hardware**: CPU multi-core acelera o Random Forest

#### Otimizações Implementadas
- **Janela Adaptativa**: Evita processamento desnecessário com poucos dados
- **Validação Eficiente**: Divisão treino/teste otimizada
- **Paralelização**: Random Forest utiliza múltiplos cores automaticamente
- **Caching**: Evita recálculos desnecessários

### 🎛️ Ajustes Avançados de Machine Learning

#### Modificar Janela Temporal
```python
# Em megasena_analyzer.py (linha ~800)
if total_sorteios >= 100:
    janela_temporal = 20  # Aumentar para mais contexto
elif total_sorteios >= 50:
    janela_temporal = 15  # Padrão médio
# ...
```

#### Ajustar Parâmetros do Random Forest
```python
# Configuração mais conservadora (mais rápida)
model = RandomForestRegressor(
    n_estimators=50,       # Menos árvores
    max_depth=5,           # Menos profundidade
    min_samples_split=10   # Mais amostras por divisão
)

# Configuração mais agressiva (mais lenta, potencialmente mais precisa)
model = RandomForestRegressor(
    n_estimators=200,      # Mais árvores
    max_depth=15,          # Mais profundidade
    min_samples_split=2    # Menos amostras por divisão
)
```

## 📈 API Utilizada

O projeto utiliza a API pública: `https://loteriascaixa-api.herokuapp.com/api/megasena`

Esta API fornece:
- Histórico completo de sorteios
- Dados de premiação
- Informações de acumulação
- Datas e locais dos sorteios

Este projeto é mantido no seguinte projeto do GitHub [guto-alves/loterias-api](https://github.com/guto-alves/loterias-api). Apoiem, pois é muito importante!

## 🤝 Contribuições

Sugestões de melhorias são bem-vindas! Especialmente:

### 📊 Análises Estatísticas
- Novos métodos de análise estatística
- Métricas de validação adicionais
- Análises de distribuição avançadas

### 🤖 Machine Learning
- Algoritmos alternativos (XGBoost, Neural Networks, SVM)
- Técnicas de ensemble mais sofisticadas
- Otimização de hiperparâmetros automatizada
- Validação cruzada temporal
- Análise de importância das features

### 🔧 Otimizações
- Melhorias na performance
- Paralelização de processamento
- Caching inteligente de resultados
- Redução de uso de memória

### 🎨 Interface
- Melhorias na interface do usuário
- Visualizações gráficas dos resultados
- Dashboard interativo
- Exportação de relatórios

### 📝 Ideias para Implementação Futura
1. **Análise de Séries Temporais**: ARIMA, LSTM para capturar padrões temporais
2. **Clustering**: Agrupar sorteios similares para identificar padrões
3. **Análise de Componentes Principais**: Reduzir dimensionalidade das features
4. **Validação Backtesting**: Testar predições em dados históricos
5. **Métricas de Confiança**: Intervalos de confiança para predições
6. **API REST**: Disponibilizar predições via API web

## 📝 Licença

Este projeto é de código aberto para fins educacionais.

---