# searchmark.py
import pickle

import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse


class retclass:
    namemark: str
    probavalue: float


app = FastAPI()


def get_AI_mark(goodsname) -> retclass:
    new_text1 = [goodsname]
    new_text = [x.lower() for x in new_text1]
    X_new = vectorizer.transform(new_text)

    # Определение кластера для нового текста
    pred_cluster = kmeans.predict(X_new)

    # Определение ключевого слова для нового текста
    pred_label = clf.predict(X_new)
    pred_proba = clf.predict_proba(X_new)
    i = 0
    for i in range(len(new_text)):
        #   print("Predicted cluster:", pred_cluster[i])
        #   print("Predicted key word:", pred_label[i])
        sortindex = pred_proba[i].argsort()
        max_index = sortindex[-1]
        pred_proba_label = clf.classes_[max_index]
        pred_proba_value = pred_proba[i][max_index] * 100
    #   print("MAX Probability of key word '{}': {:.2f}%".format(pred_proba_label, pred_proba[i][max_index] * 100))
    return {"namemark": pred_proba_label, "probavalue": pred_proba_value}

@app.get("/search/{goodsname}")
async def search_mark(goodsname):
    retmark = get_AI_mark(goodsname)
    jsonmark = jsonable_encoder(retmark)
    return JSONResponse(content = jsonmark)

kmeans = pickle.load(open("save_kmean_nom_model.pkl", "rb"))
clf = pickle.load(open("save_clf_nom_model.pkl", "rb"))
vectorizer = pickle.load(open("save_vectorizer_nom_model.pkl", "rb"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)