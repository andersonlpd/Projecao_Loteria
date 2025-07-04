#!/usr/bin/env python3
"""
Análise Estatística da Mega-Sena
=====================================

Este módulo realiza análise estatística completa dos resultados históricos da Mega-Sena
e gera predições baseadas em diferentes métodos:

1. Análise Estatística Clássica: Baseada em frequências e tendências
2. Machine Learning: Utiliza Random Forest para predição de padrões
3. Análise Temporal: Identifica sazonalidades e evoluções temporais

Autor: Sistema de Análise Estatística
Data: Junho 2025
Versão: 2.0 (Simplificada)
"""

# Importações para manipulação de dados e requests HTTP
import requests          # Para fazer requisições à API da Mega-Sena
import pandas as pd      # Manipulação e análise de dados estruturados
import numpy as np       # Operações matemáticas e arrays

# Importações para análise estatística
from collections import Counter    # Contagem de frequências
from datetime import datetime, timedelta  # Manipulação de datas
from scipy import stats          # Funções estatísticas avançadas

# Importações para Machine Learning
from sklearn.ensemble import RandomForestRegressor  # Algoritmo de floresta aleatória
from sklearn.model_selection import train_test_split  # Divisão de dados treino/teste

# Suprimir warnings para saída mais limpa
import warnings
warnings.filterwarnings('ignore')

class MegaSenaAnalyzer:
    """
    Classe principal para análise estatística da Mega-Sena
    
    Esta classe encapsula todas as funcionalidades de análise:
    - Coleta de dados via API
    - Processamento e limpeza dos dados
    - Análises estatísticas diversas
    - Predições usando métodos estatísticos e ML
    
    Atributos:
        data (list): Dados brutos obtidos da API
        df (pandas.DataFrame): Dados processados em formato tabular
        api_url (str): URL da API da Mega-Sena
    """
    
    def __init__(self):
        """
        Inicializa o analisador com configurações padrão
        
        Define os atributos iniciais como None e configura a URL da API.
        A URL utilizada é de uma API pública que fornece dados históricos
        completos dos sorteios da Mega-Sena.
        """
        self.data = None        # Armazenará os dados brutos da API
        self.df = None          # Armazenará o DataFrame processado
        # API pública com histórico completo da Mega-Sena
        self.api_url = "https://loteriascaixa-api.herokuapp.com/api/megasena"
        
    def fetch_data(self):
        """
        Busca dados históricos da Mega-Sena via API
        
        Realiza uma requisição HTTP GET para a API pública da Mega-Sena
        e armazena os dados brutos para processamento posterior.
        
        A API retorna uma lista de dicionários, onde cada dicionário
        representa um sorteio com as seguintes informações principais:
        - Número do concurso
        - Data do sorteio
        - Números sorteados
        - Informações de premiação
        - Status de acumulação
        
        Returns:
            bool: True se os dados foram obtidos com sucesso, False caso contrário
            
        Raises:
            requests.RequestException: Problemas de conectividade ou API
            ValueError: Problemas na decodificação JSON
        """
        try:
            print("Buscando dados da API...")
            # Faz a requisição HTTP para a API
            response = requests.get(self.api_url)
            # Verifica se a requisição foi bem-sucedida (status 200)
            response.raise_for_status()
            # Converte a resposta JSON para estrutura Python
            self.data = response.json()
            print(f"Dados obtidos: {len(self.data)} sorteios")
            return True
        except Exception as e:
            # Captura qualquer erro e informa ao usuário
            print(f"Erro ao buscar dados: {e}")
            return False
    
    def process_data(self):
        """
        Processa os dados brutos da API para análise estatística
        
        Transforma os dados JSON em um DataFrame do pandas, calculando
        diversas métricas estatísticas para cada sorteio:
        
        Métricas Calculadas:
        - Soma: Soma de todas as 6 dezenas do sorteio
        - Média: Média aritmética das dezenas
        - Mediana: Valor central das dezenas ordenadas
        - Amplitude: Diferença entre maior e menor dezena
        - Pares: Quantidade de números pares no sorteio
        - Ímpares: Quantidade de números ímpares no sorteio
        
        Estas métricas são fundamentais para identificar padrões
        estatísticos nos sorteios históricos.
        
        Returns:
            bool: True se processamento foi bem-sucedido, False caso contrário
        """
        if not self.data:
            print("Nenhum dado disponível. Execute fetch_data() primeiro.")
            return False
            
        # Lista para armazenar os dados processados
        processed_data = []
        
        # Processa cada sorteio individualmente
        for sorteio in self.data:
            # Cria um dicionário com informações básicas do sorteio
            row = {
                'concurso': sorteio['concurso'],  # Número do concurso
                # Converte string de data para objeto datetime do pandas
                'data': pd.to_datetime(sorteio['data'], format='%d/%m/%Y'),
                'acumulou': sorteio['acumulou']   # Boolean: se acumulou ou não
            }
            
            # Converte as dezenas de string para inteiros
            # A API retorna as dezenas como strings com zero à esquerda
            dezenas = [int(d) for d in sorteio['dezenas']]
            
            # Adiciona cada dezena como uma coluna separada
            # Isso facilita análises posicionais posteriores
            for i, dezena in enumerate(dezenas, 1):
                row[f'dezena_{i}'] = dezena
            
            # === CÁLCULO DE MÉTRICAS ESTATÍSTICAS ===
            
            # Soma total das 6 dezenas (importante indicador estatístico)
            row['soma'] = sum(dezenas)
            
            # Média aritmética das dezenas
            row['media'] = np.mean(dezenas)
            
            # Mediana (valor central quando ordenados)
            row['mediana'] = np.median(dezenas)
            
            # Amplitude: diferença entre maior e menor número
            # Indica a "espalhamento" dos números no sorteio
            row['amplitude'] = max(dezenas) - min(dezenas)
            
            # Contagem de números pares
            # Padrão típico: 3 pares e 3 ímpares
            row['pares'] = sum(1 for d in dezenas if d % 2 == 0)
            
            # Contagem de números ímpares (complementar aos pares)
            row['impares'] = 6 - row['pares']
            
            # Adiciona o registro processado à lista
            processed_data.append(row)
        
        # Converte lista de dicionários em DataFrame do pandas
        self.df = pd.DataFrame(processed_data)
        
        # Ordena por data para análises temporais
        # Reset_index remove o índice antigo e cria um novo sequencial
        self.df = self.df.sort_values('data').reset_index(drop=True)
        
        print(f"Dados processados: {len(self.df)} registros")
        return True
    
    def analyze_frequency(self):
        """
        Análise de frequência dos números da Mega-Sena
        
        Realiza análise estatística da frequência com que cada número
        (1 a 60) aparece nos sorteios históricos. Esta análise é fundamental
        para identificar:
        
        - Números "quentes": Mais sorteados que a média
        - Números "frios": Menos sorteados que a média  
        - Distribuição de probabilidades empíricas
        - Desvios da distribuição uniforme esperada
        
        Em uma distribuição perfeitamente aleatória, todos os números
        deveriam ter frequência similar. Desvios podem indicar:
        1. Coincidências estatísticas normais
        2. Possíveis vieses no sistema de sorteio
        3. Padrões temporais de longo prazo
        
        Returns:
            pandas.DataFrame: DataFrame com colunas 'numero' e 'frequencia'
                             ordenado por frequência decrescente
        """
        print("\n=== ANÁLISE DE FREQUÊNCIA ===")
        
        # === COLETA DE TODOS OS NÚMEROS SORTEADOS ===
        # Cria uma lista única com todos os números de todos os sorteios
        all_numbers = []
        for _, row in self.df.iterrows():
            # Percorre as 6 dezenas de cada sorteio
            for i in range(1, 7):
                all_numbers.append(row[f'dezena_{i}'])
        
        # === CONTAGEM DE FREQUÊNCIAS ===
        # Counter é uma subclasse de dict especializada em contagem
        frequency = Counter(all_numbers)
        
        # Converte para DataFrame para facilitar análises
        freq_df = pd.DataFrame(list(frequency.items()), columns=['numero', 'frequencia'])
        # Ordena por frequência decrescente (mais sorteados primeiro)
        freq_df = freq_df.sort_values('frequencia', ascending=False)
        
        # === RELATÓRIO DE FREQUÊNCIAS ===
        print("Top 10 números mais sorteados:")
        print(freq_df.head(10).to_string(index=False))
        
        print("\nTop 10 números menos sorteados:")
        print(freq_df.tail(10).to_string(index=False))
        
        # === ANÁLISE ESTATÍSTICA DAS FREQUÊNCIAS ===
        # Calcula estatísticas descritivas das frequências
        media_freq = freq_df['frequencia'].mean()
        desvio_freq = freq_df['frequencia'].std()
        
        print(f"\nEstatísticas de frequência:")
        print(f"Média: {media_freq:.2f}")
        print(f"Desvio padrão: {desvio_freq:.2f}")
        
        # Coeficiente de variação: mede dispersão relativa
        # Valores baixos indicam distribuição mais uniforme
        coef_variacao = (desvio_freq/media_freq) * 100
        print(f"Coeficiente de variação: {coef_variacao:.2f}%")
        
        # Interpretação do coeficiente de variação
        if coef_variacao < 10:
            print("→ Distribuição relativamente uniforme")
        elif coef_variacao < 20:
            print("→ Distribuição moderadamente dispersa")
        else:
            print("→ Distribuição altamente dispersa")
        
        return freq_df
    
    def analyze_patterns(self):
        """
        Análise de padrões estatísticos nos sorteios
        
        Examina diversos padrões matemáticos e estatísticos que podem
        ocorrer nos sorteios da Mega-Sena:
        
        1. ANÁLISE DA SOMA:
           - Soma das 6 dezenas de cada sorteio
           - Faixa típica: 150-200 (distribuição aproximadamente normal)
           - Valores extremos são raros mas possíveis
        
        2. ANÁLISE PARES/ÍMPARES:
           - Distribuição entre números pares e ímpares
           - Padrão mais comum: 3 pares + 3 ímpares
           - Extremos (6 pares ou 6 ímpares) são muito raros
        
        3. ANÁLISE DE AMPLITUDE:
           - Diferença entre maior e menor número do sorteio
           - Indica o "espalhamento" dos números
           - Valores muito baixos ou altos são menos comuns
        
        4. ANÁLISE DE SEQUÊNCIAS:
           - Identifica números consecutivos no mesmo sorteio
           - Ex: 15, 16 ou 30, 31, 32
        
        Returns:
            dict: Dicionário com estatísticas de cada padrão analisado
        """
        print("\n=== ANÁLISE DE PADRÕES ===")
        
        # === ANÁLISE DA SOMA DAS DEZENAS ===
        # A soma segue aproximadamente uma distribuição normal
        # Faixa típica: 150-200, com média ~183
        soma_stats = self.df['soma'].describe()
        print("Estatísticas da soma das dezenas:")
        print(soma_stats)
        
        # Interpretação da soma
        soma_media = soma_stats['mean']
        if 170 <= soma_media <= 190:
            print("→ Soma média dentro do padrão esperado")
        else:
            print("→ Soma média atípica - verificar dados")
        
        # === ANÁLISE DE PARES E ÍMPARES ===
        # Padrão esperado: distribuição próxima de 3 pares e 3 ímpares
        pares_stats = self.df['pares'].describe()
        print(f"\nDistribuição de números pares:")
        print(pares_stats)
        
        # Cálculo de probabilidades de cada configuração
        total_sorteios = len(self.df)
        print(f"\nDistribuição pares/ímpares:")
        for i in range(7):  # 0 a 6 pares
            count = len(self.df[self.df['pares'] == i])
            percentual = (count / total_sorteios) * 100
            print(f"  {i} pares: {count:4d} sorteios ({percentual:5.2f}%)")
        
        # === ANÁLISE DE AMPLITUDE ===
        # Amplitude = diferença entre maior e menor número
        # Valores típicos: 35-50
        amplitude_stats = self.df['amplitude'].describe()
        print(f"\nEstatísticas da amplitude:")
        print(amplitude_stats)
        
        # === ANÁLISE DE SEQUÊNCIAS NUMÉRICAS ===
        # Identifica números consecutivos (ex: 15,16 ou 30,31,32)
        sequencias = self.analyze_sequences()
        print(f"\nSequências consecutivas encontradas: {len(sequencias)}")
        
        # Análise estatística das sequências
        if sequencias:
            tamanhos = [len(seq) for seq in sequencias]
            print(f"  Tamanho médio das sequências: {np.mean(tamanhos):.1f}")
            print(f"  Maior sequência: {max(tamanhos)} números")
            
            # Exemplo de algumas sequências
            print("  Exemplos de sequências:")
            for i, seq in enumerate(sequencias[:5]):  # Mostra as 5 primeiras
                print(f"    {seq}")
        
        return {
            'soma_stats': soma_stats,
            'pares_stats': pares_stats,
            'amplitude_stats': amplitude_stats,
            'sequencias': sequencias
        }
    
    def analyze_sequences(self):
        """
        Identifica sequências de números consecutivos nos sorteios
        
        Uma sequência é definida como 2 ou mais números consecutivos
        que aparecem no mesmo sorteio. Por exemplo:
        - [15, 16] é uma sequência de tamanho 2
        - [30, 31, 32] é uma sequência de tamanho 3
        
        Sequências são relativamente comuns na Mega-Sena, contrariando
        a intuição popular de que são raras. Matematicamente, a probabilidade
        de NÃO ter sequências é menor do que se imagina.
        
        Returns:
            list: Lista de listas, onde cada sublista é uma sequência encontrada
        """
        sequencias = []
        
        # Processa cada sorteio individualmente
        for _, row in self.df.iterrows():
            # Ordena as dezenas do sorteio para verificar consecutividade
            dezenas = sorted([row[f'dezena_{i}'] for i in range(1, 7)])
            
            # === ALGORITMO DE DETECÇÃO DE SEQUÊNCIAS ===
            # Inicia com o primeiro número
            seq_atual = [dezenas[0]]
            
            # Percorre os números restantes
            for i in range(1, len(dezenas)):
                # Verifica se o número atual é consecutivo ao anterior
                if dezenas[i] == dezenas[i-1] + 1:
                    # Adiciona à sequência atual
                    seq_atual.append(dezenas[i])
                else:
                    # Sequência quebrou - salva se tem 2+ números
                    if len(seq_atual) >= 2:
                        sequencias.append(seq_atual.copy())
                    # Inicia nova sequência
                    seq_atual = [dezenas[i]]
            
            # Verifica a última sequência após o loop
            if len(seq_atual) >= 2:
                sequencias.append(seq_atual.copy())
        
        return sequencias
    
    def analyze_temporal_patterns(self):
        """
        Análise de padrões temporais nos sorteios
        
        Examina como os padrões estatísticos evoluem ao longo do tempo,
        buscando identificar:
        
        1. TENDÊNCIAS ANUAIS:
           - Mudanças na soma média ao longo dos anos
           - Evolução na proporção pares/ímpares
           - Variações na taxa de acumulação
        
        2. SAZONALIDADE MENSAL:
           - Possíveis variações por mês do ano
           - Análise se determinados meses têm características especiais
        
        3. PADRÕES SEMANAIS:
           - Análise por dia da semana (segunda, terça, etc.)
           - Verificação de consistência temporal
        
        Esta análise é importante para detectar:
        - Mudanças no sistema de sorteio ao longo do tempo
        - Possíveis sazonalidades (improvável mas verificável)
        - Consistência temporal dos padrões estatísticos
        
        Returns:
            tuple: (estatísticas_por_ano, estatísticas_por_mês)
        """
        print("\n=== ANÁLISE TEMPORAL ===")
        
        # === CRIAÇÃO DE VARIÁVEIS TEMPORAIS ===
        # Extrai componentes temporais das datas
        self.df['ano'] = self.df['data'].dt.year        # Ano (ex: 2025)
        self.df['mes'] = self.df['data'].dt.month       # Mês (1-12)
        self.df['dia_semana'] = self.df['data'].dt.dayofweek  # Dia semana (0=Segunda)
        
        # === ANÁLISE POR ANO ===
        # Agrupa dados por ano e calcula estatísticas
        por_ano = self.df.groupby('ano').agg({
            'soma': ['mean', 'std'],     # Média e desvio padrão da soma
            'pares': 'mean',             # Média de números pares
            'acumulou': 'mean'           # Taxa de acumulação
        }).round(2)
        
        print("Estatísticas por ano:")
        print(por_ano)
        
        # Interpretação das tendências anuais
        somas_anuais = self.df.groupby('ano')['soma'].mean()
        if len(somas_anuais) > 1:
            tendencia = "crescente" if somas_anuais.iloc[-1] > somas_anuais.iloc[0] else "decrescente"
            print(f"→ Tendência da soma ao longo dos anos: {tendencia}")
        
        # === ANÁLISE POR MÊS ===
        # Verifica se há sazonalidade mensal
        por_mes = self.df.groupby('mes').agg({
            'soma': 'mean',
            'pares': 'mean',
            'acumulou': 'mean'
        }).round(2)
        
        print("\nEstatísticas por mês:")
        print(por_mes)
        
        # Análise de variabilidade mensal
        cv_mensal = (por_mes['soma'].std() / por_mes['soma'].mean()) * 100
        print(f"→ Coeficiente de variação mensal da soma: {cv_mensal:.2f}%")
        if cv_mensal < 5:
            print("→ Baixa variabilidade mensal (padrão esperado)")
        else:
            print("→ Alta variabilidade mensal (investigar)")
        
        return por_ano, por_mes
    
    def analyze_correlations(self):
        """
        Análise de correlações entre variáveis estatísticas
        
        Calcula a matriz de correlação de Pearson entre diferentes
        métricas estatísticas dos sorteios para identificar:
        
        1. CORRELAÇÕES POSITIVAS FORTES (r > 0.7):
           - Variáveis que tendem a crescer juntas
           - Ex: soma e amplitude geralmente correlacionadas
        
        2. CORRELAÇÕES NEGATIVAS FORTES (r < -0.7):
           - Variáveis inversamente relacionadas
           - Ex: pares e ímpares (correlação perfeita = -1)
        
        3. CORRELAÇÕES FRACAS (|r| < 0.3):
           - Variáveis independentes ou pouco relacionadas
           - Indica aleatoriedade saudável
        
        Correlações muito fortes podem indicar:
        - Dependências matemáticas naturais
        - Possíveis vieses no sistema
        - Padrões não-aleatórios
        
        Returns:
            pandas.DataFrame: Matriz de correlação entre as variáveis
        """
        print("\n=== ANÁLISE DE CORRELAÇÕES ===")
        
        # === SELEÇÃO DE VARIÁVEIS NUMÉRICAS ===
        # Escolhe variáveis estatísticas principais para correlação
        numeric_cols = ['soma', 'media', 'mediana', 'amplitude', 'pares', 'impares']
        
        # Calcula matriz de correlação de Pearson
        corr_matrix = self.df[numeric_cols].corr()
        
        print("Matriz de correlação:")
        print(corr_matrix.round(3))
        
        # === INTERPRETAÇÃO DAS CORRELAÇÕES ===
        print("\nInterpretação das correlações principais:")
        
        # Correlação entre pares e ímpares (deve ser -1.0)
        corr_pares_impares = corr_matrix.loc['pares', 'impares']
        print(f"  Pares vs Ímpares: {corr_pares_impares:.3f} (esperado: -1.000)")
        
        # Correlação entre soma e amplitude
        corr_soma_amplitude = corr_matrix.loc['soma', 'amplitude']
        print(f"  Soma vs Amplitude: {corr_soma_amplitude:.3f}")
        if corr_soma_amplitude > 0.5:
            print("    → Correlação forte: somas maiores tendem a ter maior amplitude")
        
        # Correlação entre média e mediana
        corr_media_mediana = corr_matrix.loc['media', 'mediana']
        print(f"  Média vs Mediana: {corr_media_mediana:.3f}")
        if corr_media_mediana > 0.9:
            print("    → Correlação muito forte (esperado para dados similares)")
        
        return corr_matrix
    
    def generate_prediction(self, method='statistical'):
        """
        Gera predição para o próximo sorteio usando diferentes métodos
        
        Este é o método principal de predição que oferece três abordagens:
        
        1. STATISTICAL: Análise estatística clássica
           - Baseado em frequências históricas
           - Considera tendências recentes
           - Aplica restrições de equilibrio
        
        2. MACHINE_LEARNING: Algoritmo de aprendizado de máquina
           - Utiliza Random Forest Regressor
           - Analisa padrões sequenciais
           - Prediz cada posição individualmente
        
        3. COMBINED: Combinação dos métodos anteriores
           - Fusão inteligente dos resultados
           - Elimina duplicatas
           - Otimiza a predição final
        
        Args:
            method (str): Método de predição ('statistical', 'machine_learning', 'combined')
        
        Returns:
            list: Lista com 6 números preditos para o próximo sorteio
        """
        print(f"\n=== PREDIÇÃO ({method.upper()}) ===")
        
        # Direciona para o método específico baseado no parâmetro
        if method == 'statistical':
            return self._statistical_prediction()
        elif method == 'machine_learning':
            return self._ml_prediction()
        else:
            return self._combined_prediction()
    
    def _statistical_prediction(self):
        """
        Predição baseada em análise estatística clássica
        
        Este método utiliza princípios de estatística descritiva e teoria
        das probabilidades para gerar uma predição. O algoritmo combina:
        
        1. ANÁLISE DE FREQUÊNCIA HISTÓRICA (70% do peso):
           - Calcula a frequência relativa de cada número (1-60)
           - Números mais sorteados historicamente têm maior peso
           - Baseado na Lei dos Grandes Números
        
        2. ANÁLISE DE TENDÊNCIA RECENTE (30% do peso):
           - Considera apenas os últimos 20 sorteios
           - Captura possíveis mudanças recentes no padrão
           - Equilibra estabilidade histórica com adaptabilidade
        
        3. APLICAÇÃO DE RESTRIÇÕES:
           - Garante distribuição equilibrada pares/ímpares
           - Evita configurações extremamente raras
           - Otimiza a jogabilidade prática
        
        FUNDAMENTAÇÃO MATEMÁTICA:
        - Probabilidade combinada = 0.7 × P_histórica + 0.3 × P_recente
        - P_histórica = frequência_número / total_números_sorteados
        - P_recente = freq_recente / (20_sorteios × 6_números)
        
        Returns:
            list: Lista com 6 números preditos ordenados
        """
        
        # === COLETA DE DADOS HISTÓRICOS ===
        # Extrai todos os números de todos os sorteios históricos
        all_numbers = []
        for _, row in self.df.iterrows():
            for i in range(1, 7):
                all_numbers.append(row[f'dezena_{i}'])
        
        # Conta a frequência de cada número (1-60)
        frequency = Counter(all_numbers)
        
        # === CÁLCULO DE PROBABILIDADES HISTÓRICAS ===
        # Converte frequências absolutas em probabilidades relativas
        total_sorteios = len(all_numbers)  # Total de números sorteados na história
        probabilities = {num: freq/total_sorteios for num, freq in frequency.items()}
        
        print(f"Base histórica: {len(self.df)} sorteios, {total_sorteios} números")
        
        # === ANÁLISE DE TENDÊNCIA RECENTE ===
        # Considera apenas os últimos 20 sorteios para capturar tendências
        recent_numbers = []
        num_recent_draws = min(20, len(self.df))  # Usa no máximo 20 ou todos disponíveis
        
        for _, row in self.df.tail(num_recent_draws).iterrows():
            for i in range(1, 7):
                recent_numbers.append(row[f'dezena_{i}'])
        
        recent_frequency = Counter(recent_numbers)
        print(f"Análise recente: últimos {num_recent_draws} sorteios")
        
        # === COMBINAÇÃO DE PROBABILIDADES ===
        # Combina análise histórica (estabilidade) com recente (adaptabilidade)
        combined_scores = {}
        peso_historico = 0.7    # 70% para dados históricos
        peso_recente = 0.3      # 30% para tendência recente
        
        for num in range(1, 61):  # Todos os números possíveis (1-60)
            # Probabilidade histórica (pode ser 0 se número nunca saiu)
            hist_score = probabilities.get(num, 0)
            
            # Probabilidade recente (pode ser 0 se não saiu recentemente)
            recent_score = recent_frequency.get(num, 0) / len(recent_numbers) if recent_numbers else 0
            
            # Score combinado usando média ponderada
            combined_scores[num] = hist_score * peso_historico + recent_score * peso_recente
        
        # === ORDENAÇÃO POR PROBABILIDADE ===
        # Ordena números por score decrescente (mais prováveis primeiro)
        sorted_numbers = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Debug: mostra os top 10 números mais prováveis
        print("Top 10 números mais prováveis:")
        for i, (num, score) in enumerate(sorted_numbers[:10]):
            hist_freq = frequency.get(num, 0)
            recent_freq = recent_frequency.get(num, 0)
            print(f"  {i+1:2d}. Número {num:2d}: score={score:.4f} (hist:{hist_freq}, rec:{recent_freq})")
        
        # === APLICAÇÃO DE RESTRIÇÕES ===
        # Aplica regras para um jogo mais equilibrado e realista
        prediction = self._apply_constraints(sorted_numbers)
        
        # === RELATÓRIO FINAL ===
        print(f"\nPredição estatística: {sorted(prediction)}")
        print(f"Soma: {sum(prediction)}")
        print(f"Pares: {sum(1 for n in prediction if n % 2 == 0)}")
        print(f"Amplitude: {max(prediction) - min(prediction)}")
        
        return prediction
    
    def _apply_constraints(self, sorted_numbers):
        """
        Aplica restrições estatísticas para um jogo mais equilibrado
        
        Este método recebe a lista de números ordenados por probabilidade
        e aplica filtros baseados em padrões estatísticos históricos para
        gerar um jogo mais realista e equilibrado.
        
        RESTRIÇÕES APLICADAS:
        
        1. EQUILÍBRIO PARES/ÍMPARES:
           - Máximo 4 pares ou 4 ímpares por jogo
           - Evita extremos (6 pares ou 6 ímpares) que são muito raros
           - Baseado na distribuição histórica observada
        
        2. FALLBACK INTELIGENTE:
           - Se as restrições impedirem seleção de 6 números
           - Completa com próximos números da lista ordenada
           - Garante sempre 6 números na predição final
        
        FUNDAMENTAÇÃO ESTATÍSTICA:
        - Configuração 6-0 ou 0-6 (pares/ímpares): ~1.6% dos casos
        - Configurações 4-2 ou 2-4: ~31% dos cases
        - Configuração 3-3: ~31% dos casos
        - Restrição visa configurações com >60% de probabilidade histórica
        
        Args:
            sorted_numbers (list): Lista de tuplas (número, score) ordenada por probabilidade
        
        Returns:
            list: Lista com exatamente 6 números que atendem às restrições
        """
        prediction = []
        
        # === CONTADORES DE RESTRIÇÕES ===
        pares_count = 0      # Contador de números pares selecionados
        impares_count = 0    # Contador de números ímpares selecionados
        max_pares = 4        # Máximo de pares permitidos
        max_impares = 4      # Máximo de ímpares permitidos
        
        print(f"Aplicando restrições: máx {max_pares} pares, máx {max_impares} ímpares")
        
        # === PRIMEIRA PASSADA: SELEÇÃO COM RESTRIÇÕES ===
        for num, score in sorted_numbers:
            # Para se já temos 6 números
            if len(prediction) >= 6:
                break
                
            # Verifica se o número é par ou ímpar
            if num % 2 == 0:  # Número par
                if pares_count < max_pares:
                    prediction.append(num)
                    pares_count += 1
                    print(f"  Selecionado par: {num} (score: {score:.4f})")
                # Se já tem muitos pares, pula este número
                
            else:  # Número ímpar
                if impares_count < max_impares:
                    prediction.append(num)
                    impares_count += 1
                    print(f"  Selecionado ímpar: {num} (score: {score:.4f})")
                # Se já tem muitos ímpares, pula este número
        
        # === SEGUNDA PASSADA: COMPLETAR SE NECESSÁRIO ===
        # Se as restrições foram muito rígidas e não conseguimos 6 números
        if len(prediction) < 6:
            print(f"Apenas {len(prediction)} números selecionados. Completando...")
            
            for num, score in sorted_numbers:
                # Adiciona números que ainda não estão na predição
                if num not in prediction:
                    prediction.append(num)
                    tipo = "par" if num % 2 == 0 else "ímpar"
                    print(f"  Completando com {tipo}: {num} (score: {score:.4f})")
                    
                    # Para quando tiver exatamente 6 números
                    if len(prediction) >= 6:
                        break
        
        # === VERIFICAÇÃO FINAL ===
        final_pares = sum(1 for n in prediction if n % 2 == 0)
        final_impares = 6 - final_pares
        print(f"Resultado final: {final_pares} pares, {final_impares} ímpares")
        
        return prediction
    
    def _ml_prediction(self):
        """
        Predição usando Machine Learning - Random Forest Regressor
        
        ============================================================================
        CONCEITO FUNDAMENTAL:
        ============================================================================
        
        Este método utiliza algoritmos de aprendizado de máquina para identificar
        padrões complexos nos dados históricos que podem não ser capturados por
        análise estatística tradicional.
        
        ALGORITMO ESCOLHIDO: Random Forest Regressor
        - Ensemble de múltiplas árvores de decisão
        - Resistente a overfitting
        - Capaz de capturar relações não-lineares
        - Boa performance com dados numéricos sequenciais
        
        ============================================================================
        ARQUITETURA DO MODELO:
        ============================================================================
        
        1. JANELA TEMPORAL ADAPTATIVA (Lookback Window):
           - Usa uma janela que se adapta ao total de sorteios disponíveis
           - Muitos dados (≥100): 15 sorteios como features
           - Dados médios (≥50): 10 sorteios como features  
           - Poucos dados (≥20): 5 sorteios como features
           - Muito poucos dados (<20): 3 sorteios ou 1/4 do total
           - Cada sorteio tem 6 números = janela × 6 features por amostra
           - Janela deslizante ao longo de todo o histórico disponível
        
        2. MÚLTIPLOS MODELOS:
           - Treina 6 modelos separados (um para cada posição)
           - Cada modelo prediz uma das 6 dezenas do próximo sorteio
           - Abordagem Multi-Output individual
        
        3. DADOS DE TREINO:
           - Features: janela_temporal sorteios anteriores [janela × 6 números]
           - Target: Próximo sorteio [6 números]
           - Exemplo: Sorteios 1-janela → Sorteio (janela+1), etc.
           - Utiliza TODOS os sorteios disponíveis para treinamento
        
        ============================================================================
        PROCESSO DE TREINAMENTO:
        ============================================================================
        """
        
        print("Iniciando predição com Machine Learning...")
        print("Algoritmo: Random Forest Regressor")
        
        # === PREPARAÇÃO DOS DADOS ===
        # Estruturas para armazenar features (X) e targets (y)
        features = []  # Lista de arrays com features (janela_temporal sorteios × 6 números)
        targets = []   # Lista de arrays com targets (6 números do próximo sorteio)
        
        print(f"Preparando dados de treino...")
        print(f"Total de sorteios disponíveis: {len(self.df)}")
        
        # === CÁLCULO DA JANELA TEMPORAL ADAPTATIVA ===
        # Usa uma janela que se adapta ao número total de sorteios disponíveis
        total_sorteios = len(self.df)
        
        if total_sorteios >= 100:
            # Para muitos dados: usa janela de 15 sorteios para capturar mais padrões
            janela_temporal = 15
        elif total_sorteios >= 50:
            # Para dados médios: usa janela de 10 sorteios (padrão original)
            janela_temporal = 10
        elif total_sorteios >= 20:
            # Para poucos dados: usa janela de 5 sorteios
            janela_temporal = 5
        else:
            # Para muito poucos dados: usa janela mínima de 3 sorteios
            janela_temporal = max(3, total_sorteios // 4)
        
        print(f"Janela temporal adaptativa: {janela_temporal} sorteios")
        print(f"Features por amostra: {janela_temporal * 6} números")
        
        # === CRIAÇÃO DO DATASET DE TREINAMENTO ===
        # Começa no sorteio janela_temporal porque precisa dos sorteios anteriores
        samples_created = 0
        for i in range(janela_temporal, len(self.df)):
            
            # === CONSTRUÇÃO DAS FEATURES (X) ===
            # Para cada amostra, usa os sorteios anteriores como features
            feature_row = []
            
            # Loop pelos sorteios anteriores ao sorteio i
            for j in range(janela_temporal):
                row_idx = i - janela_temporal + j  # Índice do sorteio j dentro da janela
                
                # Adiciona as 6 dezenas deste sorteio às features
                for k in range(1, 7):
                    numero = self.df.iloc[row_idx][f'dezena_{k}']
                    feature_row.append(numero)
            
            # feature_row agora tem (janela_temporal × 6) números
            features.append(feature_row)
            samples_created += 1
            
            # === CONSTRUÇÃO DO TARGET (y) ===
            # O target é o sorteio i (que vem após a janela temporal das features)
            target_row = [self.df.iloc[i][f'dezena_{k}'] for k in range(1, 7)]
            targets.append(target_row)
        
        # Converte listas em arrays NumPy para uso no scikit-learn
        features = np.array(features)  # Shape: (n_samples, janela_temporal * 6)
        targets = np.array(targets)    # Shape: (n_samples, 6)
        
        print(f"Dataset criado:")
        print(f"  Features shape: {features.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Amostras de treino: {samples_created}")
        print(f"  Aproveitamento dos dados: {(samples_created/total_sorteios)*100:.1f}%")
        
        # Verifica se temos dados suficientes para treinar
        if samples_created < 10:
            print("⚠️  AVISO: Poucos dados para treinamento confiável")
            print("   Recomenda-se pelo menos 50 sorteios históricos")
        
        # === TREINAMENTO DE MÚLTIPLOS MODELOS ===
        # Treina um modelo separado para cada uma das 6 posições
        predictions = []
        
        print(f"\nTreinando {6} modelos (um para cada posição):")
        
        for pos in range(6):
            print(f"  Modelo {pos+1}/6: Predizendo dezena na posição {pos+1}...")
            
            # === DIVISÃO TREINO/TESTE ===
            # Divide dados em treino (80%) e teste (20%)
            X_train, X_test, y_train, y_test = train_test_split(
                features,           # Features: 10 sorteios anteriores
                targets[:, pos],    # Target: posição específica do próximo sorteio
                test_size=0.2,      # 20% para teste
                random_state=42     # Seed para reprodutibilidade
            )
            
            # === CONFIGURAÇÃO DO MODELO ===
            # Random Forest com 100 árvores
            model = RandomForestRegressor(
                n_estimators=100,   # 100 árvores de decisão
                random_state=42,    # Seed para reprodutibilidade
                max_depth=10,       # Profundidade máxima das árvores
                min_samples_split=5, # Mínimo de amostras para dividir nó
                min_samples_leaf=2   # Mínimo de amostras por folha
            )
            
            # === TREINAMENTO ===
            model.fit(X_train, y_train)
            
            # === AVALIAÇÃO DO MODELO ===
            # Calcula score no conjunto de teste
            score = model.score(X_test, y_test)
            print(f"    Score R² no teste: {score:.4f}")
            
            # === PREDIÇÃO PARA O PRÓXIMO SORTEIO ===
            # Usa os últimos janela_temporal sorteios como features para predição (consistente com o treinamento)
            last_features = []
            
            # Coleta os últimos janela_temporal sorteios
            for j in range(janela_temporal):
                row_idx = len(self.df) - janela_temporal + j  # Últimos janela_temporal sorteios
                for k in range(1, 7):
                    last_features.append(self.df.iloc[row_idx][f'dezena_{k}'])
            
            # Faz a predição
            pred_raw = model.predict([last_features])[0]
            
            # === PÓS-PROCESSAMENTO DA PREDIÇÃO ===
            # Garante que a predição está no range válido (1-60)
            pred_final = max(1, min(60, round(pred_raw)))
            predictions.append(pred_final)
            
            print(f"    Predição bruta: {pred_raw:.2f} → Final: {pred_final}")
        
        # === TRATAMENTO DE DUPLICATAS ===
        # Remove números duplicados (problema comum em ML)
        unique_predictions = list(set(predictions))
        
        print(f"\nPredições iniciais: {predictions}")
        print(f"Predições únicas: {unique_predictions}")
        
        # === COMPLETAR PREDIÇÃO SE NECESSÁRIO ===
        # Se temos menos de 6 números únicos, completa aleatoriamente
        while len(unique_predictions) < 6:
            # Gera número aleatório que não esteja na lista
            candidate = np.random.randint(1, 61)
            if candidate not in unique_predictions:
                unique_predictions.append(candidate)
                print(f"Adicionado número aleatório: {candidate}")
        
        # Pega apenas os 6 primeiros números e ordena
        ml_prediction = sorted(unique_predictions[:6])
        
        # === RELATÓRIO FINAL ===
        print(f"\n🤖 RESULTADO DO MACHINE LEARNING:")
        print(f"Predição ML: {ml_prediction}")
        print(f"Soma: {sum(ml_prediction)}")
        print(f"Pares: {sum(1 for n in ml_prediction if n % 2 == 0)}")
        print(f"Amplitude: {max(ml_prediction) - min(ml_prediction)}")
        
        # === ANÁLISE DE CONFIANÇA ===
        # Verifica se a predição está dentro de faixas estatísticas razoáveis
        soma_pred = sum(ml_prediction)
        soma_media_hist = self.df['soma'].mean()
        soma_std_hist = self.df['soma'].std()
        
        if abs(soma_pred - soma_media_hist) <= 2 * soma_std_hist:
            print("✅ Predição dentro de faixa estatística razoável")
        else:
            print("⚠️  Predição fora da faixa estatística típica")
        
        return ml_prediction
    
    def _combined_prediction(self):
        """
        Predição combinando métodos estatístico e machine learning
        
        Esta função representa a síntese inteligente dos dois métodos principais
        de predição, combinando as vantagens de cada abordagem:
        
        MÉTODO ESTATÍSTICO:
        ✅ Vantagens: Estável, interpretável, baseado em frequências históricas
        ❌ Limitações: Não captura padrões complexos ou não-lineares
        
        MÉTODO MACHINE LEARNING:
        ✅ Vantagens: Detecta padrões complexos, adapta-se a mudanças
        ❌ Limitações: Pode ser instável, menos interpretável
        
        ESTRATÉGIA DE COMBINAÇÃO:
        1. Executa ambos os métodos independentemente
        2. Combina os resultados em um conjunto único
        3. Remove duplicatas naturalmente
        4. Seleciona os 6 melhores números
        5. Completa aleatoriamente se necessário (raro)
        
        Esta abordagem ensemble geralmente produz resultados mais robustos
        do que qualquer método individual.
        
        Returns:
            list: Lista com 6 números combinados dos dois métodos
        """
        print("🔀 COMBINANDO MÉTODOS ESTATÍSTICO E MACHINE LEARNING")
        print("-" * 60)
        
        # === EXECUÇÃO DOS MÉTODOS INDIVIDUAIS ===
        print("1️⃣ Executando predição estatística...")
        stat_pred = self._statistical_prediction()
        
        print("\n2️⃣ Executando predição com machine learning...")
        ml_pred = self._ml_prediction()
        
        # === COMBINAÇÃO INTELIGENTE ===
        print(f"\n3️⃣ Combinando resultados...")
        print(f"Predição estatística: {sorted(stat_pred)}")
        print(f"Predição ML:          {sorted(ml_pred)}")
        
        # Combina as duas listas e remove duplicatas automaticamente com set()
        combined = list(set(stat_pred + ml_pred))
        print(f"Números únicos combinados: {sorted(combined)} ({len(combined)} números)")
        
        # === SELEÇÃO FINAL ===
        if len(combined) >= 6:
            # Temos números suficientes - pega os 6 primeiros
            combined_prediction = sorted(combined[:6])
            print(f"✅ Selecionados os primeiros 6 números")
        else:
            # Caso raro: menos de 6 números únicos
            print(f"⚠️  Apenas {len(combined)} números únicos. Completando...")
            
            # Completa com números aleatórios que não estejam na lista
            while len(combined) < 6:
                candidate = np.random.randint(1, 61)
                if candidate not in combined:
                    combined.append(candidate)
                    print(f"   Adicionado aleatoriamente: {candidate}")
            
            combined_prediction = sorted(combined)
        
        # === RELATÓRIO FINAL DA COMBINAÇÃO ===
        print(f"\n🎯 RESULTADO DA COMBINAÇÃO:")
        print(f"Predição combinada: {combined_prediction}")
        print(f"Soma: {sum(combined_prediction)}")
        print(f"Pares: {sum(1 for n in combined_prediction if n % 2 == 0)}")
        print(f"Amplitude: {max(combined_prediction) - min(combined_prediction)}")
        
        # Análise de sobreposição entre métodos
        overlap = len(set(stat_pred) & set(ml_pred))
        print(f"Números em comum entre métodos: {overlap}/6")
        if overlap >= 3:
            print("✅ Alta concordância entre métodos")
        elif overlap >= 1:
            print("⚠️  Concordância moderada entre métodos")
        else:
            print("❌ Baixa concordância entre métodos")
        
        return combined_prediction
    
    def generate_report(self):
        """
        Gera relatório completo de análise da Mega-Sena
        
        Este método orquestra todas as análises disponíveis e gera um
        relatório abrangente com:
        
        1. ANÁLISES ESTATÍSTICAS:
           - Frequência dos números
           - Padrões nos sorteios
           - Análise temporal
           - Correlações entre variáveis
        
        2. PREDIÇÕES:
           - Método estatístico
           - Método machine learning  
           - Método combinado
        
        3. SUGESTÕES PRÁTICAS:
           - Recomendações baseadas em dados
           - Faixas estatísticas ideais
           - Números quentes e frios
        
        Returns:
            dict: Dicionário com todos os resultados das análises
        """
        print("=" * 60)
        print("RELATÓRIO COMPLETO DE ANÁLISE DA MEGA-SENA")
        print("=" * 60)
        print(f"Data da análise: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"Total de sorteios analisados: {len(self.df)}")
        print(f"Período: {self.df['data'].min().strftime('%d/%m/%Y')} a {self.df['data'].max().strftime('%d/%m/%Y')}")
        
        # === EXECUÇÃO DAS ANÁLISES ===
        print(f"\n🔍 Executando análises estatísticas...")
        
        freq_df = self.analyze_frequency()
        patterns = self.analyze_patterns()
        temporal = self.analyze_temporal_patterns()
        correlations = self.analyze_correlations()
        
        # === GERAÇÃO DE PREDIÇÕES ===
        print("\n" + "=" * 60)
        print("PREDIÇÕES PARA O PRÓXIMO SORTEIO")
        print("=" * 60)
        
        # Executa os três métodos de predição
        stat_pred = self.generate_prediction('statistical')
        ml_pred = self.generate_prediction('machine_learning')
        combined_pred = self.generate_prediction('combined')
        
        # === SUGESTÕES E RECOMENDAÇÕES ===
        print("\n" + "=" * 60)
        print("SUGESTÕES ESTATÍSTICAS")
        print("=" * 60)
        
        self.print_suggestions(freq_df, patterns)
        
        # === COMPILAÇÃO DOS RESULTADOS ===
        results = {
            'frequency': freq_df,
            'patterns': patterns,
            'temporal': temporal,
            'correlations': correlations,
            'predictions': {
                'statistical': stat_pred,
                'machine_learning': ml_pred,
                'combined': combined_pred
            }
        }
        
        return results
    
    def print_suggestions(self, freq_df, patterns):
        """
        Imprime sugestões práticas baseadas na análise estatística
        
        Gera recomendações acionáveis para auxiliar na escolha de números,
        baseadas em padrões estatísticos identificados nos dados históricos.
        
        Args:
            freq_df (DataFrame): DataFrame com frequências dos números
            patterns (dict): Dicionário com padrões estatísticos
        """
        print("\n📊 SUGESTÕES PARA PRÓXIMO JOGO:")
        print("-" * 40)
        
        # === SUGESTÃO DE SOMA ===
        # Recomenda faixa de soma baseada na distribuição histórica
        soma_media = patterns['soma_stats']['mean']
        soma_std = patterns['soma_stats']['std']
        soma_min = soma_media - soma_std
        soma_max = soma_media + soma_std
        
        print(f"• Soma recomendada: {soma_min:.0f} a {soma_max:.0f}")
        print(f"  (Média histórica: {soma_media:.1f} ± {soma_std:.1f})")
        
        # === SUGESTÃO DE PARES/ÍMPARES ===
        # Recomenda quantidade de pares baseada na média histórica
        pares_media = patterns['pares_stats']['mean']
        print(f"• Quantidade de pares recomendada: {pares_media:.0f}")
        
        # Distribuição histórica detalhada
        pares_mode = patterns['pares_stats'].get('50%', pares_media)  # Mediana
        print(f"  (Mais comum: {pares_mode:.0f} pares)")
        
        # === NÚMEROS QUENTES E FRIOS ===
        # Identifica números com frequências extremas
        numeros_quentes = freq_df.head(10)['numero'].tolist()
        numeros_frios = freq_df.tail(10)['numero'].tolist()
        
        print(f"• Considere incluir números 'quentes': {numeros_quentes[:5]}")
        print(f"  (Mais sorteados historicamente)")
        print(f"• Considere incluir números 'frios': {numeros_frios[:3]}")
        print(f"  (Menos sorteados - podem estar 'devendo')")
        
        # === SUGESTÃO DE AMPLITUDE ===
        # Recomenda faixa de amplitude baseada na distribuição
        amplitude_media = patterns['amplitude_stats']['mean']
        amplitude_std = patterns['amplitude_stats']['std']
        
        print(f"• Amplitude recomendada: {amplitude_media:.0f} ± {amplitude_std:.0f}")
        print(f"  (Diferença entre maior e menor número)")
        
        # === ANÁLISE DE SEQUÊNCIAS ===
        # Estatísticas sobre números consecutivos
        num_sequencias = len(patterns['sequencias'])
        total_sorteios = len(self.df) if hasattr(self, 'df') else 1
        prob_sequencia = (num_sequencias / total_sorteios) * 100
        
        print(f"• Probabilidade de sequência: {prob_sequencia:.1f}%")
        if prob_sequencia > 30:
            print("  (Considere incluir números consecutivos)")
        else:
            print("  (Sequências são relativamente raras)")
        
        # === DICAS FINAIS ===
        print(f"\n💡 DICAS ADICIONAIS:")
        print(f"• Evite padrões óbvios (1,2,3,4,5,6 ou todos pares)")
        print(f"• Misture números de diferentes dezenas")
        print(f"• Considere usar tanto números baixos (<30) quanto altos (>30)")
        print(f"• Lembre-se: cada sorteio é independente!")

def main():
    """
    Função principal do programa
    
    Executa o fluxo completo de análise:
    1. Cria instância do analisador
    2. Busca dados da API
    3. Processa os dados
    4. Gera relatório completo
    5. Exibe resultados
    """
    print("🎯 ANALISADOR ESTATÍSTICO DA MEGA-SENA")
    print("=" * 50)
    print("Iniciando análise completa...")
    
    # Instancia o analisador
    analyzer = MegaSenaAnalyzer()
    
    # === COLETA DE DADOS ===
    print("\n📡 Fase 1: Coleta de dados")
    if not analyzer.fetch_data():
        print("❌ Falha na coleta de dados. Encerrando.")
        return
    
    # === PROCESSAMENTO ===
    print("\n⚙️ Fase 2: Processamento dos dados")
    if not analyzer.process_data():
        print("❌ Falha no processamento. Encerrando.")
        return
    
    # === ANÁLISE COMPLETA ===
    print("\n📊 Fase 3: Análise estatística completa")
    results = analyzer.generate_report()
    
    # === FINALIZAÇÃO ===
    print("\n" + "=" * 60)
    print("🎯 ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    print("✅ Todas as análises foram executadas")
    print("✅ Predições geradas com múltiplos métodos")
    print("✅ Sugestões baseadas em dados históricos")
    print("\n💡 Use as informações acima como referência para suas escolhas!")
    print("⚠️  Lembre-se: jogos de azar não têm garantias!")

if __name__ == "__main__":
    main()
