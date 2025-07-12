import json
from collections import defaultdict
from pyabsa import ATEPCCheckpointManager
import matplotlib.pyplot as plt
import spacy

# Cargar modelo de lenguaje para NLP avanzado
nlp = spacy.load("en_core_web_lg")

def analyze_competitor_weaknesses():
    # 1. Cargar y filtrar reseñas negativas (1-3 estrellas)
    with open('input.json', 'r', encoding='utf-8') as file:
        reviews = json.load(file)
    
    negative_reviews = [r for r in reviews if 1 <= r["overall"] < 5]
    texts = [r["reviewText"] for r in negative_reviews]

    if not texts:
        print("⚠️ No hay reseñas negativas para analizar")
        return

    # 2. Extraer aspectos y contexto
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
        checkpoint="english",
        auto_device=True
    )
    
    results = aspect_extractor.extract_aspect(
        inference_source=texts,
        pred_sentiment=True,
        print_result=False
    )

    # 3. Procesamiento avanzado
    weakness_analysis = defaultdict(lambda: {
        "mentions": 0,
        "adjectives": defaultdict(int),
        "examples": []
    })

    for review, result in zip(negative_reviews, results):
        doc = nlp(review["reviewText"])
        
        for aspect, sentiment in zip(result['aspect'], result['sentiment']):
            if sentiment == 'Negative':
                aspect_lower = aspect.lower()
                weakness_analysis[aspect_lower]["mentions"] += 1
                
                # Extraer adjetivos cercanos al aspecto
                for token in doc:
                    if aspect.lower() in token.text.lower() and token.head.pos_ == "ADJ":
                        adj = token.head.text.lower()
                        weakness_analysis[aspect_lower]["adjectives"][adj] += 1
                
                # Guardar ejemplos representativos (máx. 3)
                if len(weakness_analysis[aspect_lower]["examples"]) < 3:
                    weakness_analysis[aspect_lower]["examples"].append(review["reviewText"])

    # 4. Generar recomendaciones basadas en el análisis
    recommendations = []
    for aspect, data in sorted(weakness_analysis.items(), key=lambda x: x[1]["mentions"], reverse=True)[:5]:
        top_adjectives = sorted(data["adjectives"].items(), key=lambda x: x[1], reverse=True)[:3]
        adjectives_str = ", ".join([f"{adj} ({count})" for adj, count in top_adjectives])
        
        recommendation = {
            "aspect": aspect,
            "mentions": data["mentions"],
            "main_issues": adjectives_str,
            "recommendation": generate_recommendation(aspect, top_adjectives),
            "example_reviews": data["examples"]
        }
        recommendations.append(recommendation)

    # 5. Guardar y mostrar resultados
    save_results(recommendations, len(negative_reviews))
    plot_results(recommendations)

def generate_recommendation(aspect, adjectives):
    """Genera recomendaciones personalizadas basadas en aspectos y adjetivos"""
    suggestions = {
        "battery": "Mejorar capacidad de batería con componentes de mayor densidad energética",
        "customer service": "Implementar capacitación en servicio al cliente y reducir tiempos de respuesta",
        "shipping": "Optimizar logística con socios locales y ofrecer seguimiento en tiempo real",
        "price": "Crear paquetes de valor agregado o programas de fidelización",
        "quality": "Auditar proveedores e implementar controles de calidad más estrictos"
    }
    
    base_suggestion = suggestions.get(aspect, f"Mejorar {aspect} mediante benchmarking competitivo")
    
    # Personalizar según adjetivos
    if "expensive" in [a[0] for a in adjectives]:
        base_suggestion += " y revisar estructura de precios"
    if "slow" in [a[0] for a in adjectives]:
        base_suggestion += " con procesos más ágiles"
    
    return base_suggestion

def save_results(data, total_reviews):
    """Guarda resultados en formato JSON estructurado"""
    output = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "total_negative_reviews": total_reviews,
        "weaknesses": data
    }
    
    with open("competitor_weaknesses_detailed.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

def plot_results(data):
    """Genera visualización interactiva de resultados"""
    aspects = [d["aspect"] for d in data]
    mentions = [d["mentions"] for d in data]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(aspects, mentions, color=['#ff6b6b', '#ffa36b', '#ffdf6b', '#a3ff6b', '#6bffa3'])
    
    plt.title('Top 5 Debilidades Competitivas con Contexto', pad=20)
    plt.xlabel('Número de Menciones')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Añadir etiquetas con adjetivos
    for i, bar in enumerate(bars):
        adj_text = data[i]["main_issues"].split(", ")[0]
        plt.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2, 
                adj_text, ha='right', va='center', color='white', weight='bold')
    
    plt.tight_layout()
    plt.savefig('weaknesses_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    from datetime import datetime
    analyze_competitor_weaknesses()