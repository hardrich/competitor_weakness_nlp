import json
from collections import defaultdict
from pyabsa import ATEPCCheckpointManager
import matplotlib.pyplot as plt

def analyze_competitor_weaknesses():
  # 1. Cargar reviews
  with open('input.json', 'r', encoding='utf-8') as file:
      reviews = json.load(file)
  
  texts = [review["reviewText"] for review in reviews if review["overall"] < 5]  # Solo reviews negativas

  # 2. Extraer aspectos con sentimiento
  aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
    checkpoint="english", 
    auto_device=True
  )
  
  results = aspect_extractor.extract_aspect(
    inference_source=texts,
    pred_sentiment=True,
    print_result=False
  )

  # 3. Procesar resultados para encontrar debilidades principales
  weakness_counter = defaultdict(int)
  
  for result in results:
    for aspect, sentiment in zip(result['aspect'], result['sentiment']):
      if sentiment == 'Negative':
        weakness_counter[aspect.lower()] += 1
  
  # 4. Ordenar y mostrar las 5 principales debilidades
  top_weaknesses = sorted(weakness_counter.items(), key=lambda x: x[1], reverse=True)[:5]
  
  print("🔍 Principales debilidades encontradas:")
  for weakness, count in top_weaknesses:
    print(f"- {weakness.capitalize()}: {count} menciones")
  
  # 5. Visualización (opcional)
  plt.figure(figsize=(10, 5))
  plt.bar([w[0] for w in top_weaknesses], [w[1] for w in top_weaknesses])
  plt.title("Top 5 Debilidades de Competidores")
  plt.ylabel("Número de menciones")
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.savefig('weaknesses_analysis.png')
  plt.show()

  # 6. Guardar resultados estructurados
  output = {
    "total_reviews_analizadas": len(texts),
    "top_weaknesses": [{"aspect": w[0], "mentions": w[1]} for w in top_weaknesses],
    "raw_data": results
  }
  
  with open("weaknesses_report.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
  analyze_competitor_weaknesses()
