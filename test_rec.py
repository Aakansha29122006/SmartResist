import app
import json

app.load_model_artifacts()

res = app.recommend_antibiotics("aac", 0, 1000)
print(json.dumps(res, indent=2))
