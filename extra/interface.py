import os
import json
import shutil
import time
import traceback
import subprocess
import asyncio
import uuid
import aiofiles
import uvicorn

import aux_interface

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, WebSocket
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Set
from pathlib import Path
from file_read_backwards import FileReadBackwards

from utils import get_matches, ROOT, convert_file_to_utf8, get_creation_time, get_loss, get_step, save_task_info, encode_base64

app = FastAPI()
clients: Dict[str, Set[WebSocket]] = {}


class ExperimentBody(BaseModel):
    exp_name: str = Field(..., description="task name for getting the finetuning model")


class DatasetBody(BaseModel):
    dataset_name: str = Field(..., description="dir name of the dataset")


class DatasetNameBody(BaseModel):
    dataset_name: str = Field(..., description="old dir name of the dataset")
    dataset_new_name: str = Field(..., description="new dir name of the dataset")


class DatasetFileBody(BaseModel):
    dataset_name: str = Field(..., description="dir name of the dataset")
    file: List[str] = Field(..., description="delete files in dataset")


class DatasetTxtBody(BaseModel):
    dataset_name: str = Field(..., description="dir name of the dataset")
    id: str = Field(..., description="txt name in dataset")
    text: str = Field(..., description="text content")


class EditSamplesBody(BaseModel):
    data: List[dict] = Field(..., description="qwen_image_edit samples, k is a prompt, v is a list of image path")


class LossBody(BaseModel):
    exp_name: str = Field(..., description="task name for getting the finetuning model")
    since_step: int = Field(..., description="data after the since step")
    limit: int = Field(2000, description="the max length of the loss data")


class ConfigBody(BaseModel):
    job_type: str = Field(..., description="model name for the task")
    exp_name: str = Field(..., description="task name for getting the finetuning model")
    target_type: str = Field(..., description="training method in [lora, lokr]")
    concept_sentence: str = Field(..., description="trigger word")
    steps: int = Field(..., description="total steps of the training stage")
    lr: float = Field(..., description="learning rate")
    rank: int = Field(..., description="lora rank")
    factor: int = Field(..., description="lokr factor")
    dataset_name: str = Field(..., description="dataset dir name")
    control_data: List[str] = Field(default_factory=list, description="control dataset dir name", max_length=3)
    samples: List[Union[str, dict]] = Field(default_factory=list, description="sample prompts")
    low_vram: bool = Field(True, description="whether to use low vram mode")


@app.get("/v1/health", tags=["health"])
async def health():
    return {"code": 0, "msg": "success!", "data": {}}


@app.get("/v1/get_dataset", tags=["dataset"])
def get_dataset_list():
    dataset_abstract = []
    dir_list = os.listdir(os.path.join(ROOT, "datasets"))
    for each in dir_list:
        dataset_dir = os.path.join(ROOT, f"datasets/{each}")
        count = 0
        random_image = ""
        flag = True
        if os.path.isdir(dataset_dir):
            for tmp in os.listdir(dataset_dir):
                if tmp.split(".")[-1].lower() in ["jpg", "jpeg", "bmp", "png", "tif", "tiff", "webp"]:
                    count += 1
                    if flag:
                        # random_image = encode_base64(os.path.join(dataset_dir, tmp))
                        random_image = os.path.join(dataset_dir, tmp)
                        flag = False

            create_time, time_stamp = get_creation_time(dataset_dir)
            dataset_abstract.append({"name": each, "size": count, "create_time": create_time, "image_url": random_image, "time_stamp": time_stamp})

    if dataset_abstract:
        dataset_abstract = sorted(
            dataset_abstract,
            key=lambda x: x["time_stamp"],
            reverse=True
        )

    if len(dataset_abstract) > 0:
        return {"code": 0, "msg": "success!", "data": dataset_abstract}
    else:
        return {"code": -1, "msg": "dataset is not found!", "data": dataset_abstract}


@app.post("/v1/get_dataset_info", tags=["dataset"])
def get_dataset_info(body: DatasetBody):
    dataset_dir = os.path.join(ROOT, f"datasets/{body.dataset_name}")
    if not os.path.exists(dataset_dir):
        return {"code": -1, "msg": "dataset does not exist!", "data": []}

    txt_map = {}
    img_map = {}
    for file in os.listdir(dataset_dir):
        name, ext = os.path.splitext(file)
        ext = ext.lower()

        path = os.path.join(dataset_dir, file)

        if ext == ".txt":
            txt_map[name] = convert_file_to_utf8(path)

        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
            img_map[name] = file

        else:
            print("warning: invalid data format in dataset!")

    out = []
    for k, v in img_map.items():
        text = txt_map.get(k, None)
        tmp = {"id": k, "image_url": os.path.join(dataset_dir, v), "text": text}
        out.append(tmp)

    return {"code": 0, "msg": "success", "data": out}


@app.post("/v1/edit_txt", tags=["dataset"])
def modify_txt(body: DatasetTxtBody):
    dataset_dir = os.path.join(ROOT, f"datasets/{body.dataset_name}")
    if not os.path.exists(dataset_dir):
        return {"code": -1, "msg": "dataset_name is not found!", "data": ""}

    txt_name = body.id + ".txt"
    txt_file = os.path.join(dataset_dir, txt_name)
    if os.path.exists(txt_file):
        with open(txt_file, "w") as f:
            f.write(body.text)
        return {"code": 0, "msg": "success!", "data": body.text}

    return {"code": -1, "msg": f"{txt_file} is not found!", "data": ""}


@app.get("/v1/get_image", tags=["dataset"])
def get_image(image_url: str):
    if not os.path.isfile(image_url):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        image_url,
        media_type="image/*",
        filename=os.path.basename(image_url)
    )


@app.post("/v1/rename_dataset", tags=["dataset"])
def modify_dataset_name(body: DatasetNameBody):
    dataset_dir = os.path.join(ROOT, f"datasets/{body.dataset_name}")
    if not os.path.exists(dataset_dir):
        return {"code": -1, "msg": "dataset_name is not found!", "data": body.dataset_new_name}

    new_dataset_dir = os.path.join(ROOT, f"datasets/{body.dataset_new_name}")
    if os.path.exists(new_dataset_dir):
        return {"code": -2, "msg": "new dataset_name is already exists!", "data": body.dataset_new_name}

    os.rename(dataset_dir, new_dataset_dir)
    return {"code": 0, "msg": "success!", "data": body.dataset_new_name}


@app.post("/v1/delete_dataset", tags=["dataset"])
def delete_dataset(body: DatasetBody):
    dataset_dir = os.path.join(ROOT, f"datasets/{body.dataset_name}")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        return {"code": 0, "msg": "success!", "data": body.dataset_name}

    return {"code": -1, "msg": "dataset_name is not found!", "data": body.dataset_name}


@app.post("/v1/delete_dataset_file", tags=["dataset"])
def delete_dataset_file(body: DatasetFileBody):
    dataset_dir = os.path.join(ROOT, f"datasets/{body.dataset_name}")
    if os.path.exists(dataset_dir):
        for each in body.file:
            image_path = os.path.join(dataset_dir, each)
            name, _ = os.path.splitext(each)
            txt_path = os.path.join(dataset_dir, name + ".txt")
            if os.path.exists(image_path):
                os.remove(image_path)

            if os.path.exists(txt_path):
                os.remove(txt_path)

        return {"code": 0, "msg": "success!", "data": body.dataset_name}

    return {"code": -1, "msg": "dataset file is not found!", "data": body.dataset_name}


def get_task_state(exp_name: str):
    tmp_data_list = aux_interface.get_all()
    for each in tmp_data_list:
        if list(each.keys())[0] == exp_name:
            return "running"

    return "exited"


async def run_in_background(log_file_path: str, args: list):
    process = await asyncio.create_subprocess_exec(
        "python3", "-u", os.path.join(ROOT, "run_job.py"), *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )

    process_id = str(process.pid)
    aux_interface.update_data(args[3], process_id)
    print(f"Training Task PID: {process_id}")

    try:
        async with aiofiles.open(log_file_path, "a") as log_file:
            while True:
                chunk = await process.stdout.read(1024)
                if not chunk:
                    break

                decoded = chunk.decode("utf-8", errors="ignore").replace('\r', '\n')
                print(decoded, end="")
                await log_file.write(decoded)
                await log_file.flush()

                for ws in list(clients.get(args[3], set())):
                    try:
                        await ws.send_text(decoded)
                    except Exception as e:
                        clients[args[3]].discard(ws)
                        print(str(e))

        await process.wait()

    except Exception as e:
        print(str(e))
        traceback.print_exc()

    for ws in list(clients.get(args[3], set())):
        try:
            await ws.send_text("[EOF]")
        except Exception as e:
            print(str(e))

    print(f"Finsh Training Task: {args[3]}")
    aux_interface.pop_data(args[3], process_id)
    clients.pop(args[3], None)
    return log_file_path


@app.post("/v1/train", tags=["task"])
async def train(body: ConfigBody):
    aux_interface.add_data(body.exp_name, "")

    log_dir = os.path.join(ROOT, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    exp_dir = os.path.join(ROOT, f"output/{body.exp_name}")
    if os.path.exists(exp_dir):
        aux_interface.pop_data(body.exp_name, "")
        return {"code": -1, "msg": "experiment name is already exist, please modify your experiment name", "data": ""}

    else:
        os.makedirs(exp_dir)

    _ = save_task_info(body.exp_name, body.job_type, body.steps, exp_dir)
    log_file_path = os.path.join(ROOT, f"logs/{body.exp_name}_training.log")
    args = [
        "--job_type", body.job_type,
        "--exp_name", body.exp_name,
        "--target_type", body.target_type,
        "--concept_sentence", body.concept_sentence,
        "--steps", str(body.steps),
        "--lr", str(body.lr),
        "--rank", str(body.rank),
        "--factor", str(body.factor),
        "--dataset_folder", os.path.join(ROOT, f"datasets/{body.dataset_name}"),
        "--low_vram", str(body.low_vram),
        "--samples", str(body.samples)
    ]

    if body.job_type in ["qwen_image_edit_2509", "qwen_image_edit_2511"]:
        control_folders = [os.path.join(ROOT, f"datasets/{i}") for i in body.control_data]
        extra_params = ["--control_folders", str(control_folders)]
        args.extend(extra_params)

    asyncio.create_task(run_in_background(log_file_path, args))

    return {"code": 0, "msg": "success!", "data": body.exp_name}


@app.post("/v1/if_completed", tags=["task"])
async def if_train_completed(body: ExperimentBody):
    tmp_data_list = aux_interface.get_all()
    for each in tmp_data_list:
        if list(each.keys())[0] == body.exp_name:
            return {"code": 0, "msg": "success!", "data": "running"}

    return {"code": 0, "msg": "success!", "data": "completed"}


def get_task_info(exp_name: str):
    json_path = os.path.join(ROOT, f"output/{exp_name}/.task_info.json")
    if not os.path.exists(json_path):
        print("Warning: json file does not exists! please wait!")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    create_time = json_data["create_time"]
    total_steps = json_data["total_steps"]
    base_model = json_data["base_model"]

    if_completed = True if get_task_state(exp_name) == "exited" else False
    db_path = os.path.join(ROOT, f"output/{exp_name}/loss_log.db")
    if if_completed:
        progress = 100
        current_step = 0
        if os.path.exists(db_path):
            current_step = get_step(db_path)
            progress = round(current_step / total_steps * 100)

    else:
        if os.path.exists(db_path):
            current_step = get_step(db_path)
        else:
            current_step = 0

        progress = round(current_step / total_steps * 100)
        count = 0
        while current_step < 0:
            current_step = get_step(db_path)
            count += 1
            if count > 5:
                break

        if current_step < 0:
            print("database connect error!")
            return None

    data = {
        "current_time": create_time,
        "base_model": base_model,
        "if_completed": if_completed,
        "train_progress": progress,
        "current_step": current_step,
        "total_steps": total_steps
    }

    return data


@app.post("/v1/get_task_info", tags=["task"])
def get_task_detail(body: ExperimentBody):
    out = get_task_info(body.exp_name)
    if out is not None:
        return {"code": 0, "msg": "success!", "data": out}
    else:
        return {"code": -1, "msg": "database connect error *** or *** exp_name does not exists!", "data": {}}


@app.get("/v1/get_training_tasks", tags=["task"])
async def get_training_task():
    task_list = aux_interface.get_all()
    if len(task_list) > 0:
        return {"code": 0, "msg": "success!", "data": task_list}

    return {"code": 0, "msg": "no running tasks in queue", "data": []}


@app.get("/v1/get_all_tasks", tags=["task"])
def get_all_tasks():
    all_task_info = []
    task_folders = os.path.join(ROOT, "output")
    tasks = []

    for name in os.listdir(task_folders):
        if name.startswith("."):
            continue

        full_path = os.path.join(task_folders, name)
        if os.path.isdir(full_path):
            tasks.append({
                "name": name,
                "ctime": os.path.getctime(full_path)
            })
    tasks.sort(key=lambda x: x["ctime"], reverse=True)

    tasks_name = [t["name"] for t in tasks]
    for each in tasks_name:
        tmp_info = get_task_info(each)
        count = 0
        while tmp_info is None:
            tmp_info = get_task_info(each)
            time.sleep(0.2)
            count += 1
            if count > 5:
                break

        tmp_info["exp_name"] = each
        all_task_info.append(tmp_info)

    return {"code": 0, "msg": "success!", "data": all_task_info}


@app.post("/v1/stop_train", tags=["task"])
async def stop_train_delay(body: ExperimentBody):
    await asyncio.sleep(3)
    tmp_data_list = aux_interface.get_all()
    for each in tmp_data_list:
        exp_name = list(each.keys())[0]
        if exp_name == body.exp_name:
            subprocess.run(["kill", "-9", each[exp_name]])
            aux_interface.pop_data(body.exp_name, each[exp_name])
            print(f"Stop the Training Task: {body.exp_name}")
            return {"code": 0, "msg": "success!", "data": {exp_name: each[exp_name]}}
    return {"code": -1, "msg": "exp_name does not exist!", "data": {body.exp_name: "no pid found"}}


@app.post("/v1/get_samples", tags=["task"])
def get_samples(body: ExperimentBody):
    samples_dir = os.path.join(ROOT, f"output/{body.exp_name}/samples")
    if not os.path.exists(samples_dir):
        return {"code": 0, "msg": "samples is not generated!", "data": []}

    image_list = []
    for each in os.listdir(samples_dir):
        name, ext = os.path.splitext(each)
        ext = ext.lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
            step = int(name.split("_")[-2])
            image_url = os.path.join(samples_dir, each)
            image_list.append({"step": step, "image_url": image_url})

    image_list.sort(key=lambda x: x["step"])
    return {"code": 0, "msg": "success!", "data": image_list}


@app.post("/v1/delete_task", tags=["task"])
def remove_dir(body: ExperimentBody):
    exp_dir = os.path.join(ROOT, f"output/{body.exp_name}")
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
        return {"code": 0, "msg": "success!", "data": body.exp_name}

    return {"code": -1, "msg": "exp_name is not found!", "data": body.exp_name}


@app.post("/v1/get_time", tags=["tools"])
def get_train_time(body: ExperimentBody):
    log_path = os.path.join(ROOT, f"logs/{body.exp_name}_training.log")
    progress, times = None, None
    if not os.path.exists(log_path):
        return {"code": -1, "msg": "log file is not found!", "data": {"progress": "", "time": ""}}

    with FileReadBackwards(log_path, encoding="utf-8") as f:
        for line in f:
            if ("lr:" in line) and ("loss:" in line):
                progress, times = get_matches(line)
                break

    if progress is not None and times is not None:
        return {"code": 0, "msg": "success!", "data": {"progress": progress, "time": times}}
    return {"code": -1, "msg": "time and progress is not reachable!", "data": {"progress": "", "time": ""}}


@app.post("/v1/upload_files/", tags=["file"])
async def upload_files(files: List[UploadFile] = File(...), dataset_dir_name: str = Form(...)):
    error_files = []
    dataset_dir = os.path.join(ROOT, f"datasets/{dataset_dir_name}")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for file in files:
        file_path = os.path.join(dataset_dir, file.filename)
        try:
            async with aiofiles.open(file_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):  # 每次读取 1024 KB
                    await f.write(chunk)
        except Exception as e:
            print(str(e))
            error_files.append(file.filename)

    if len(error_files) == 0:
        return {"code": 0, "msg": "success!", "data": dataset_dir_name}
    else:
        return {"code": -1, "msg": f"Upload Error: failed to upload {error_files}!", "data": dataset_dir_name}


@app.post("/v1/upload_image/", tags=["file"])
async def upload_image(file: UploadFile = File(...)):
    data_dir = os.path.join(ROOT, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_name = file.filename.lower()
    if not file_name.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", "tif", "tiff")):
        raise HTTPException(status_code=400, detail="Only jpg/jpeg/png/webp/bmp/tif/tiff images are allowed")

    suffix = Path(file.filename).suffix
    final_name = f"{uuid.uuid4().hex}{suffix}"
    file_path = os.path.join(data_dir, final_name)
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                await f.write(chunk)
    except Exception as e:
        print(str(e))
        return {"code": -1, "msg": "failed to upload file!", "data": {"image_path": "", "image_base64": ""}}

    image_base64 = encode_base64(file_path)

    return {"code": 0, "msg": "success!", "data": {"image_path": file_path, "image_base64": image_base64}}


@app.post("/v1/format_samples", tags=["tools"])
async def format_samples_edit(body: EditSamplesBody):
    samples = []
    for item in body.data:
        k = list(item.keys())[0]
        v = list(item.values())[0]
        tmp = {"prompt": k}
        for ind, each in enumerate(v):
            if ind > 2:
                break

            tmp["ctrl_img{}".format(ind + 1)] = each

        samples.append(tmp)

    return {"code": 0, "msg": "success!", "data": samples}


@app.get("/v1/get_models", tags=["tools"])
async def get_model_list():
    model_list = [
        "z_image",
        "z_image_turbo",
        "z_image_de_turbo",
        "qwen_image_2512",
        "qwen_image_edit_2511",
        "qwen_image_edit_2509",
        "qwen_image"
    ]
    return {"code": 0, "msg": "success!", "data": model_list}


@app.get("/v1/get_checkpoints", tags=["tools"])
def get_checkpoints_list(exp_name: str):
    checkpoint_list = []
    out_size = ""
    dir_path = os.path.join(ROOT, f"output/{exp_name}")
    if os.path.exists(dir_path):
        for each in os.listdir(dir_path):
            if each.split(".")[-1] == "safetensors":
                file_path = os.path.join(dir_path, each)
                create_time, time_stamp = get_creation_time(file_path)
                size = os.path.getsize(file_path)
                for unit in ["KB", "MB", "GB"]:
                    size /= 1024
                    if size < 1024:
                        out_size = f"{size:.1f}{unit}"
                        break

                checkpoint_list.append({"exp_name": exp_name, "ckpt_name": each, "create_time": create_time, "time_stamp": time_stamp, "size": out_size})

        checkpoint_list.sort(key=lambda x: x["time_stamp"], reverse=True)
        return {"code": 0, "msg": "success!", "data": checkpoint_list}

    else:
        return {"code": 0, "msg": "checkpoints is not found!", "data": []}


@app.get("/v1/download", tags=["file"])
def download(exp_name: str, checkpoint_name: str):
    checkpoint_path = os.path.join(ROOT, f"output/{exp_name}/{checkpoint_name}")

    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=404, detail="checkpoint file not found!")

    file_size = os.path.getsize(checkpoint_path)

    def iter_file(chunk_size: int = 1024 * 1024):
        """按块读取文件，每次 1MB"""
        with open(checkpoint_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    headers = {
        "Content-Disposition": f"attachment; filename={checkpoint_name}",
        "Content-Length": str(file_size),
    }

    return StreamingResponse(
        iter_file(),
        media_type="application/octet-stream",
        headers=headers
    )


@app.websocket("/ws/train/{exp_name}")
async def websocket_logs(ws: WebSocket, exp_name: str):
    await ws.accept()

    clients.setdefault(exp_name, set()).add(ws)

    log_path = os.path.join(ROOT, f"logs/{exp_name}_training.log")
    if os.path.exists(log_path):
        async with aiofiles.open(log_path, "r") as f:
            while chunk := await f.read(4096):
                await ws.send_text(chunk)

    if get_task_state(exp_name) == "exited":
        await ws.send_text("[EOF]")
        await ws.close()
        return

    try:
        while True:
            await asyncio.sleep(30)
    finally:
        clients.get(exp_name, set()).discard(ws)


@app.post("/v1/get_current_step", tags=["tools"])
def get_current_step(body: ExperimentBody):
    db_path = os.path.join(ROOT, f"output/{body.exp_name}/loss_log.db")

    if os.path.exists(db_path):
        current_step = get_step(db_path)
        if current_step > 0:
            return {"code": 0, "msg": "success!", "data": {"step": current_step}}

    return {"code": 0, "msg": "loading model or exp_name does not exist!", "data": {"step": -1}}


@app.post("/v1/get_loss", tags=["tools"])
def get_loss_state(body: LossBody):
    db_path = os.path.join(ROOT, f"output/{body.exp_name}/loss_log.db")
    if os.path.exists(db_path):
        loss_list = get_loss(db_path, since_step=body.since_step, limit=body.limit)
        if len(loss_list) > 0:
            return {"code": 0, "msg": "success!", "data": loss_list}

    return {"code": 0, "msg": "loading model or exp_name does not exist!", "data": []}


if __name__ == '__main__':
    uvicorn.run("interface:app", host="0.0.0.0", port=8000)
