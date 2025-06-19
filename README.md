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

### Instalação
```bash
# Clone o repositório
git clone https://github.com/andersonlpd/Projecao_Loteria.git

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
- Utiliza Random Forest Regressor
- Treina com padrões dos últimos 10 sorteios
- Prediz cada posição individualmente

#### 3. Método Combinado
- Combina os dois métodos anteriores
- Elimina duplicatas e otimiza o resultado

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
- Novos métodos de análise estatística
- Algoritmos de machine learning alternativos
- Otimizações de performance
- Melhorias na interface do usuário

## 📝 Licença

Este projeto é de código aberto para fins educacionais.

---