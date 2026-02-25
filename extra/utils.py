import re
import os
import time
import copy
import json
import base64
import chardet
import threading
import sqlite3

from datetime import datetime


current_file_path = os.path.abspath(__file__)
ROOT = current_file_path.split("utils")[0]


class DataList(object):
    def __init__(self):
        self.data = []
        self.mutex = threading.Lock()

    def put(self, tmp):
        with self.mutex:
            self.data.append(tmp)
            return copy.deepcopy(self.data)

    def get(self):
        with self.mutex:
            return copy.deepcopy(self.data)

    def pop(self, tmp):
        with self.mutex:
            if tmp in self.data:
                self.data.remove(tmp)
                return True
            else:
                return False

    def get_by_index(self, ind):
        with self.mutex:
            return copy.deepcopy(self.data[ind])

    def update(self, ind, tmp):
        with self.mutex:
            self.data[ind] = tmp
            return copy.deepcopy(self.data)

    def delete(self, tmp):
        with self.mutex:
            if tmp in self.data:
                self.data.remove(tmp)
            return copy.deepcopy(self.data)


def convert_file_to_utf8(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()

    # 自动检测编码
    detect = chardet.detect(raw_data)
    encoding = detect["encoding"]

    if encoding is None:
        print(f"无法检测编码：{file_path}")
        return ""

    try:
        text = raw_data.decode(encoding)
    except Exception as e:
        print(f"解码失败：{e}")
        return

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def get_matches(text: str):
    result = re.findall(r"\[(.*?)\]", text)[0]
    progress = re.findall(r"\d+/\d+", text)[0]
    times = result.split(" ")[0].strip(",")
    return progress, times


def get_task_progress(db_path: str, exp_name: str):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    cursor = conn.cursor()
    cursor.execute("PRAGMA query_only = ON;")
    cursor.execute("PRAGMA busy_timeout = 5000;")
    try:
        rows1 = cursor.execute(
            """
            SELECT MAX(step) AS current_step
            FROM steps;
            """,
        ).fetchall()
        current_step = rows1[0][0]
        eta_time = "00:00:<?"

        if current_step > 1:
            pass
    except sqlite3.OperationalError as e:
        print(f"[SQLite error] {e}")
        return []
    finally:
        conn.close()


def get_creation_time(path: str):
    stat_info = os.stat(path)
    if hasattr(stat_info, "st_birthtime"):
        creation_time = stat_info.st_birthtime
    else:
        creation_time = stat_info.st_mtime

    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(creation_time)), datetime.fromtimestamp(creation_time)


def encode_base64(file):
    with open(file, 'rb') as f:
        img_data = f.read()
        base64_data = base64.b64encode(img_data)
        return str(base64_data, 'utf-8')


def save_task_info(exp_name: str, model: str, total_steps: int, save_dir: str):
    model_dict = {
        "qwen_image_edit_2511": "Qwen-Image-Edit-2511",
        "qwen_image_edit_2509": "Qwen-Image-Edit-2509",
        "qwen_image": "Qwen-Image",
        "qwen_image_2512": "Qwen-Image-2512",
        "z_image_turbo": "Z-Image-Turbo",
        "z_image_de_turbo": "Z-Image-De-Turbo",
        "z_image": "Z-Image"
    }

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "exp_name": exp_name,
        "create_time": current_time,
        "base_model": model_dict.get(model, "unknown"),
        "total_steps": total_steps
    }
    json_file = os.path.join(save_dir, ".task_info.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return data


def get_loss(db_path: str, since_step: int = 100, limit: int = 2000):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    cursor = conn.cursor()
    cursor.execute("PRAGMA query_only = ON;")
    cursor.execute("PRAGMA busy_timeout = 5000;")
    try:
        rows = cursor.execute(
            """
            SELECT step, value_real
            FROM metrics
            WHERE key = 'loss/loss'
              AND step > ?
            ORDER BY step
            LIMIT ?
            """,
            (since_step, limit)
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"[SQLite error] {e}")
        return []
    finally:
        conn.close()

    loss = []
    for each in rows:
        loss.append({"step": each[0], "loss": each[1]})

    return loss


def get_step(db_path: str):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    cursor = conn.cursor()
    cursor.execute("PRAGMA query_only = ON;")
    cursor.execute("PRAGMA busy_timeout = 5000;")
    try:
        rows = cursor.execute(
            """
            SELECT MAX(step)
            FROM metrics
            """,
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"[SQLite error] {e}")
        return -1
    finally:
        conn.close()

    if rows:
        if rows[0][0]:
            return int(rows[0][0])

    return 0


data_list = DataList()


if __name__ == '__main__':
    # a = get_creation_time("/home/cai/project/ai-toolkit/datasets/1764400575875/bg.png")
    b = get_step("/home/cai/project/ai-toolkit/output/loss_log.db")
