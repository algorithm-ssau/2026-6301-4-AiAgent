import shutil
from datetime import datetime
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO

ROOT = Path(__file__).parent  # папка проекта, независимо от положения курсора в терминале

if __name__ == "__main__":
    # Скачать датасет с Roboflow
    # Зарегистрироваться на roboflow.com, найти датасет (alcohol detection / beer bottle)
    # и вставить api_key, workspace, project, version ниже
    rf = Roboflow(api_key="OBVmpImmwIEhLApDNlkk")
    project = rf.workspace("adonantonin").project("alcohol-iaeeq")
    dataset = project.version(4).download("yolo26", location=str(ROOT / "dataset"))

    model = YOLO("yolo26n.pt")

    model.train(
        data=str(ROOT / "dataset" / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,      # можно больше если хватает памяти GPU/CPU
        project=str(ROOT / "runs"),
        name="alcohol",
    )

    # Экспорт в ONNX и кладем в Models/
    exported = YOLO(model.trainer.best).export(format="onnx", opset=12)
    answer = input("Сохранить как основную модель (best.onnx)? [y/N]: ").strip().lower()
    archive = ROOT / "Models" / "archive"
    archive.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    if answer == "y":
        # Заменяем старую модель новой. Помещаем старую в архив
        dest = ROOT / "Models" / "best.onnx"
        if dest.exists():
            shutil.move(str(dest), str(archive / f"best_{timestamp}.onnx"))
        shutil.copy(str(exported), str(dest))
        print("Модель сохранена в Models/best.onnx")
    else:
        archive_path = archive / f"model_{timestamp}.onnx"
        shutil.copy(str(exported), str(archive_path))
        print(f"Модель сохранена в Models/archive/model_{timestamp}.onnx")