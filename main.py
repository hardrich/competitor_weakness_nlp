import json
from pyabsa import ATEPCCheckpointManager

# Cargar reviews
with open('input.json', 'r', encoding='utf-8') as file:
    reviews = json.load(file)

texts = [review["reviewText"] for review in reviews]

# Cargar modelo preentrenado
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint="english", auto_device=True)
#ATEPC(model="english", auto_device=True)

# Ejecutar análisis
results = aspect_extractor.extract_aspect(
    inference_source=texts,
    pred_sentiment=True,
    print_result=True,
    save_result=True
)

# Guardar en archivo
with open("output_absa.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("✅ ¡Análisis completado! Revisa el archivo output_absa.json")
