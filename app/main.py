from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import create_graph
from create_graph import load
import os


app = FastAPI()


class Item(BaseModel):
    name: str = "sample"
    pos_scores: List[float] = load(path="pos_sample.txt")
    neg_scores: List[float] = load(path="neg_sample.txt")


@app.post("/")
def get_graph(item: Item):
    create_graph.main(
        name=item.name,
        pos_scores=item.pos_scores,
        neg_scores=item.neg_scores,
    )

    host_pwd = os.environ["HOST_PWD"]
    det_path = os.path.join(host_pwd, "result", f"{item.name}_DET.png")
    dist_path = os.path.join(host_pwd, "result", f"{item.name}_Distribution.png")

    return {
        "det_png": det_path,
        "dist_png": dist_path,
    }
