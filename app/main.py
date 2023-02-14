from fastapi import FastAPI, HTTPException
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


@app.post("/create_graphs")
def create_2_graphs(item: Item):
    create_graph.main(
        name=item.name,
        pos_scores=item.pos_scores,
        neg_scores=item.neg_scores,
    )

    # Server에서 테스트할 때
    host_pwd = os.environ["HOST_PWD"]
    det_path = os.path.join(host_pwd, "result", f"{item.name}_DET.png")
    dist_path = os.path.join(host_pwd, "result", f"{item.name}_Distribution.png")

    # Local에서 테스트할 때
    # det_path = os.path.join(os.getcwd(), "result", f"{item.name}_DET.png")
    # dist_path = os.path.join(os.getcwd(), "result", f"{item.name}_Distribution.png")

    return {
        "det_png": det_path,
        "dist_png": dist_path,
    }


@app.get("/show_list")
def show_graph_list():
    path = os.path.join(os.getcwd(), "result")
    graphs = os.listdir(path)

    return {
        "graphs": graphs
    }


@app.get("/show_det")
def show_det_curves_graph(name: str = "sample"):
    det_path = os.path.join(os.getcwd(), "result", f"{name}_DET.png")

    if not os.path.isfile(det_path):
        raise HTTPException(status_code=444, detail=f"그려진 그래프가 없습니다. 그래프를 먼저 만드세요. (Path: {det_path})")

    return FileResponse(det_path)


@app.get("/show_dist")
def show_distribution_graph(name: str = "sample"):
    dist_path = os.path.join(os.getcwd(), "result", f"{name}_Distribution.png")

    if not os.path.isfile(dist_path):
        raise HTTPException(status_code=444, detail=f"그려진 그래프가 없습니다. 그래프를 먼저 만드세요. (Path: {dist_path})")

    return FileResponse(dist_path)
