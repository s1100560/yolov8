# app.py
import os
import warnings
import traceback
from flask import Flask, request, jsonify
import torch

# Attempt to import ultralytics YOLO (if installed)
try:
    from ultralytics import YOLO
    import ultralytics
except Exception as e:
    YOLO = None
    ultralytics = None
    print(f"Warning: ultralytics import failed: {e}")

# ===========================
# Helper: add many safe globals
# ===========================
def _add_safe_globals_for_ultralytics():
    added = []
    try:
        # Always allow torch.nn.Sequential
        try:
            from torch.nn.modules.container import Sequential
            torch.serialization.add_safe_globals([Sequential])
            added.append("torch.nn.modules.container.Sequential")
        except Exception:
            pass

        # Try to collect classes from ultralytics.nn.modules and ultralytics.nn.tasks
        if ultralytics is not None:
            modules_to_scan = []
            try:
                import ultralytics.nn.modules as u_modules
                modules_to_scan.append(u_modules)
            except Exception:
                pass
            # try conv submodule
            try:
                import ultralytics.nn.modules.conv as u_conv
                modules_to_scan.append(u_conv)
            except Exception:
                pass
            # try tasks
            try:
                import ultralytics.nn.tasks as u_tasks
                modules_to_scan.append(u_tasks)
            except Exception:
                pass

            # collect classes
            classes = []
            for mod in modules_to_scan:
                for name in dir(mod):
                    try:
                        attr = getattr(mod, name)
                        if isinstance(attr, type):
                            classes.append(attr)
                    except Exception:
                        continue
            # Deduplicate
            classes = list({c: None for c in classes}.keys())
            if classes:
                torch.serialization.add_safe_globals(classes)
                added += [cls.__module__ + "." + cls.__name__ for cls in classes]
    except Exception as e:
        print(f"add_safe_globals error: {e}\n{traceback.format_exc()}")
    return added

# Call once at import time to try to allow ultralytics classes
_added = _add_safe_globals_for_ultralytics()
if _added:
    print("Added safe globals for:", _added[:10], ("...(+%d more)" % (len(_added)-10) if len(_added)>10 else ""))

# Flask app
app = Flask(__name__)

# ===========================
# Load model robustly
# ===========================
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "freshness_fruit_and_vegetables.pt"))
_model = None
_model_load_error = None

def _try_load_model(path):
    """
    Try to load the model robustly:
    1) Try YOLO(path) after safe-globals are added
    2) If fails, try torch.load(path, weights_only=True) (safe)
    3) If still fails, fallback to torch.load(path, weights_only=False) (unsafe, but often works)
    """
    global _model, _model_load_error
    # 1) prefer YOLO(...) if available
    if YOLO is not None:
        try:
            print("Attempting to load with YOLO(...)")
            m = YOLO(path)
            print("Loaded via YOLO(...) successfully.")
            return m, None
        except Exception as e:
            print(f"YOLO(...) failed: {e}")
    # 2) try torch.load with weights_only=True (safe deserialization)
    try:
        print("Attempting torch.load(..., weights_only=True)")
        obj = torch.load(path, map_location="cpu", weights_only=True)
        # if this returns something that looks like a YOLO model object, try wrap/use it
        # If ultralytics provided YOLO, try to call YOLO with path anyway (some cases succeed)
        print("torch.load(weights_only=True) returned object of type:", type(obj))
        # If ultralytics YOLO class exists and the file path worked earlier maybe YOLO() will now work - try again
        if YOLO is not None:
            try:
                m = YOLO(path)
                print("Second attempt YOLO(path) succeeded after weights_only=True.")
                return m, None
            except Exception as e:
                print("Second YOLO(path) still failed:", e)
        # If the loaded obj itself is an nn.Module, return it (but note predict logic expects Ultralitytics YOLO)
        if isinstance(obj, torch.nn.Module):
            print("Loaded object is an nn.Module; returning it (Note: predict may not match YOLO API).")
            return obj, None
    except Exception as e:
        print("torch.load(..., weights_only=True) failed:", e)

    # 3) Fallback to unsafe load (weights_only=False)
    try:
        print("FALLBACK: Attempting torch.load(..., weights_only=False) — UNSAFE, do only if you trust the checkpoint")
        obj2 = torch.load(path, map_location="cpu", weights_only=False)
        print("torch.load(weights_only=False) returned:", type(obj2))
        # if ultralytics available, again try YOLO(path)
        if YOLO is not None:
            try:
                m = YOLO(path)
                print("YOLO(path) succeeded after unsafe torch.load fallback.")
                return m, None
            except Exception as e:
                print("YOLO(path) still failed after unsafe fallback:", e)
        if isinstance(obj2, torch.nn.Module):
            return obj2, None
        # else just return the raw object
        return obj2, None
    except Exception as e:
        print("Unsafe torch.load fallback failed:", e)
        return None, e

# Try to load at startup
try:
    print("載入模型從:", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型檔案不存在: {MODEL_PATH}")
    _model, err = _try_load_model(MODEL_PATH)
    if _model is None and err is not None:
        raise err
    print("✅ 模型載入完成")
except Exception as e:
    _model = None
    _model_load_error = str(e)
    print("❌ 模型載入失敗:", _model_load_error)
    print(traceback.format_exc())

# ========== API endpoints ==========
@app.route("/")
def home():
    if _model is None:
        return "❌ YOLOv8 API 啟動成功，但模型載入失敗。請查看日誌。", 500
    return "✅ YOLOv8 API is running on Render!"

@app.route("/test")
def test():
    return jsonify({"message": "API 測試成功", "model_loaded": _model is not None, "model_load_error": _model_load_error})

@app.route("/health")
def health():
    if _model is None:
        return jsonify({"status": "error", "message": "模型未載入", "error": _model_load_error}), 500
    return jsonify({"status": "healthy", "message": "服務正常"})

@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"error": "模型未正確載入", "detail": _model_load_error}), 500

    # Expect file upload form-data "file"
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save tmp
    tmp_path = "temp_upload.jpg"
    try:
        file.save(tmp_path)
        # If the loaded object is Ultralitytics YOLO-like, use same call
        if YOLO is not None and isinstance(_model, (YOLO.__class__, type(ultralytics)) ) is False:
            # NOTE: YOLO.__class__ check above is not robust; instead we'll try to call in a try block
            pass

        # First try to use ultralytics style call
        try:
            # If model is an Ultralitytics YOLO object, model(tmp_path) works
            results = _model(tmp_path)
            detections = []
            if results and len(results) > 0 and hasattr(results[0], "boxes"):
                for r in results[0].boxes:
                    cls_idx = int(r.cls) if hasattr(r, "cls") else None
                    label = _model.names[cls_idx] if (hasattr(_model, "names") and cls_idx is not None) else str(cls_idx)
                    conf = float(r.conf) if hasattr(r, "conf") else None
                    bbox = r.xyxy[0].tolist() if hasattr(r, "xyxy") else None
                    detections.append({"class": label, "confidence": conf, "bbox": bbox})
            else:
                # fallback: no boxes attribute
                detections = []
            # cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return jsonify({"detections": detections})
        except Exception as e:
            print("Ultralytics-style inference failed:", e)
            # try generic nn.Module forward (may not accept filepath)
            try:
                img = None
                # attempt to open with PIL and convert to tensor if required
                from PIL import Image
                import numpy as np
                im = Image.open(tmp_path).convert("RGB")
                arr = np.asarray(im) / 255.0
                # create 4D tensor [1,C,H,W]
                import torch as _torch
                tensor = _torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
                out = _model(tensor)  # best-effort
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return jsonify({"raw_output_type": str(type(out)), "message": "Generic nn.Module run; output returned raw."})
            except Exception as e2:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return jsonify({"error": "Inference failed", "trace": str(e), "trace2": str(e2)}), 500
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"error": f"處理失敗: {str(e)}"}), 500

# Entrypoint for gunicorn (no app.run here)






