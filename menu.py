#!/usr/bin/env python3
"""
Menu interativo para análise da Mega-Sena
"""

from megasena_analyzer import MegaSenaAnalyzer
import sys

def show_menu():
    """Exibe o menu de opções"""
    print("\n" + "="*60)
    print("🎯 ANALISADOR DE MEGA-SENA - MENU PRINCIPAL")
    print("="*60)
    print("1. 🚀 Predição Rápida (Recomendação para hoje)")
    print("2. 📊 Análise de Frequência")
    print("3. 🔍 Análise de Padrões")
    print("4. ⏰ Análise Temporal")
    print("5. 📈 Relatório Completo")
    print("6. ❓ Sobre as Análises")
    print("0. 🚪 Sair")
    print("="*60)

def quick_prediction(analyzer):
    """Predição rápida"""
    print("\n🚀 PREDIÇÃO RÁPIDA PARA HOJE")
    print("="*40)
    
    # Gerar predição combinada
    prediction = analyzer.generate_prediction('combined')
    
    print(f"\n🎯 NÚMEROS RECOMENDADOS: {sorted(prediction)}")
    print(f"📊 Soma: {sum(prediction)}")
    print(f"🎲 Pares: {sum(1 for n in prediction if n % 2 == 0)}")
    print(f"📐 Amplitude: {max(prediction) - min(prediction)}")
    print("\n💡 Dica: Esta é uma sugestão baseada em análise estatística!")

def frequency_analysis(analyzer):
    """Análise de frequência"""
    print("\n📊 ANÁLISE DE FREQUÊNCIA")
    print("="*40)
    
    freq_df = analyzer.analyze_frequency()
    print(f"\n📈 ESTATÍSTICAS GERAIS:")
    print(f"Total de sorteios analisados: {len(analyzer.df)}")
    print(f"Média de frequência: {freq_df['frequencia'].mean():.1f}")
    print(f"Desvio padrão: {freq_df['frequencia'].std():.1f}")

def pattern_analysis(analyzer):
    """Análise de padrões"""
    print("\n🔍 ANÁLISE DE PADRÕES")
    print("="*40)
    
    patterns = analyzer.analyze_patterns()
    print(f"\n📊 PADRÕES IDENTIFICADOS:")
    print(f"Soma média: {patterns['soma_stats']['mean']:.1f}")
    print(f"Amplitude média: {patterns['amplitude_stats']['mean']:.1f}")
    print(f"Pares médios: {patterns['pares_stats']['mean']:.1f}")
    print(f"Sequências encontradas: {len(patterns['sequencias'])}")

def temporal_analysis(analyzer):
    """Análise temporal"""
    print("\n⏰ ANÁLISE TEMPORAL")
    print("="*40)
    
    analyzer.analyze_temporal_patterns()

def complete_report(analyzer):
    """Relatório completo"""
    print("\n📈 GERANDO RELATÓRIO COMPLETO...")
    print("="*40)
    
    results = analyzer.generate_report()
    print("\n✅ Relatório gerado com sucesso!")

def show_about():
    """Informações sobre as análises"""
    print("\n❓ SOBRE AS ANÁLISES ESTATÍSTICAS")
    print("="*50)
    print("""
🔍 TIPOS DE ANÁLISES DISPONÍVEIS:

1. ANÁLISE DE FREQUÊNCIA:
   • Identifica números mais e menos sorteados
   • Calcula médias e desvios padrão
   • Sugere números "quentes" e "frios"

2. ANÁLISE DE PADRÕES:
   • Estuda soma das dezenas
   • Analisa proporção pares/ímpares
   • Identifica sequências numéricas
   • Calcula amplitude dos sorteios

3. ANÁLISE TEMPORAL:
   • Evolução dos padrões ao longo dos anos
   • Sazonalidade por meses
   • Tendências recentes

4. MÉTODOS DE PREDIÇÃO:
   • Estatístico: Baseado em frequências
   • Machine Learning: Random Forest
   • Combinado: Fusão dos métodos

⚠️  IMPORTANTE:
• Resultados são para fins educacionais
• Não garantem acertos futuros
• Jogue com responsabilidade
    """)

def main():
    """Função principal do menu"""
    print("🎯 Iniciando Analisador de Mega-Sena...")
    
    # Inicializar analisador
    analyzer = MegaSenaAnalyzer()
    
    print("📡 Carregando dados...")
    if not analyzer.fetch_data():
        print("❌ Erro ao carregar dados. Verifique sua conexão.")
        return
    
    if not analyzer.process_data():
        print("❌ Erro ao processar dados.")
        return
    
    print("✅ Dados carregados com sucesso!")
    
    # Loop do menu
    while True:
        show_menu()
        
        try:
            choice = input("\n🎯 Escolha uma opção (0-6): ").strip()
            
            if choice == '0':
                print("\n👋 Obrigado por usar o Analisador de Mega-Sena!")
                break
            elif choice == '1':
                quick_prediction(analyzer)
            elif choice == '2':
                frequency_analysis(analyzer)
            elif choice == '3':
                pattern_analysis(analyzer)
            elif choice == '4':
                temporal_analysis(analyzer)
            elif choice == '5':
                complete_report(analyzer)
            elif choice == '6':
                show_about()
            else:
                print("❌ Opção inválida! Escolha um número de 0 a 6.")
            
            input("\n⏸️  Pressione ENTER para continuar...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Saindo do programa...")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            input("\n⏸️  Pressione ENTER para continuar...")

if __name__ == "__main__":
    main()
