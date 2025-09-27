# app.py
import os
import io
import importlib
import traceback
from flask import Flask, request, jsonify
import torch
import warnings

warnings.filterwarnings("ignore")

# Try import ultralytics if available
try:
    import ultralytics
    from ultralytics import YOLO
except Exception as e:
    ultralytics = None
    YOLO = None
    print("ultralytics import failed:", e)

app = Flask(__name__)

# ---------- Utility: collect candidate classes and add to torch safe globals ----------
def collect_and_add_safe_globals():
    """
    Collect common classes used by ultralytics checkpoints (and some torch classes)
    and add them to torch.serialization.add_safe_globals(...) so torch.load(..., weights_only=True)
    can accept them where appropriate.
    """
    classes = []
    # Add a few torch builtins
    try:
        from torch.nn.modules.container import Sequential
        classes.append(Sequential)
    except Exception:
        pass
    try:
        from torch.nn.modules.conv import Conv2d
        classes.append(Conv2d)
    except Exception:
        pass

    # If ultralytics is present, try to import several submodules and collect class objects
    if ultralytics is not None:
        module_names = [
            "ultralytics.nn.modules",
            "ultralytics.nn.modules.conv",
            "ultralytics.nn.modules.common",
            "ultralytics.nn.tasks",
            "ultralytics.nn.models",
            "ultralytics.nn"
        ]
        for mn in module_names:
            try:
                mod = importlib.import_module(mn)
            except Exception:
                continue
            for name in dir(mod):
                try:
                    attr = getattr(mod, name)
                    if isinstance(attr, type):
                        classes.append(attr)
                except Exception:
                    continue

        # also try explicit small list known to appear in many checkpoints
        try:
            from ultralytics.nn.tasks import DetectionModel
            classes.append(DetectionModel)
        except Exception:
            pass

    # Deduplicate by (module,name)
    uniq = {}
    final = []
    for c in classes:
        key = (getattr(c, "__module__", None), getattr(c, "__name__", None))
        if key not in uniq and key[0] is not None:
            uniq[key] = True
            final.append(c)

    # Add to torch safe globals if any
    added_names = []
    if final:
        try:
            torch.serialization.add_safe_globals(final)
            added_names = [f"{c.__module__}.{c.__name__}" for c in final]
        except Exception as e:
            print("add_safe_globals error:", e, traceback.format_exc())
    return added_names

# Run the collection once at import/start
_safe_added = collect_and_add_safe_globals()
print("Safe-globals added (count):", len(_safe_added))
if _safe_added:
    print("Sample safe-globals:", _safe_added[:10])

# ---------- Model load logic (robust with fallbacks) ----------
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "freshness_fruit_and_vegetables.pt"))
_model = None
_model_load_trace = ""

def try_load_model(path):
    global _model, _model_load_trace
    _model = None
    _model_load_trace = ""
    if not os.path.exists(path):
        _model_load_trace = f"MODEL PATH NOT FOUND: {path}"
        print(_model_load_trace)
        return

    # 1) Preferred: use ultralytics.YOLO if available (let it handle checkpoint)
    if YOLO is not None:
        try:
            print("Attempting YOLO(path) ...")
            _model = YOLO(path)
            _model_load_trace = "Loaded via ultralytics.YOLO(path)"
            print(_model_load_trace)
            return
        except Exception as e:
            _model_load_trace = f"YOLO(path) failed: {e}"
            print(_model_load_trace)
            print(traceback.format_exc())

    # 2) Try torch.load(weights_only=True) — safe attempt
    try:
        print("Attempting torch.load(..., weights_only=True) ...")
        obj = torch.load(path, map_location="cpu", weights_only=True)
        _model_load_trace = f"torch.load(weights_only=True) returned {type(obj)}"
        print(_model_load_trace)
        # If object is an nn.Module, use it
        if isinstance(obj, torch.nn.Module):
            _model = obj
            print("Using loaded nn.Module as model")
            return
        # If ultralytics available, try YOLO again (some versions handle after weights-only load)
        if YOLO is not None:
            try:
                _model = YOLO(path)
                _model_load_trace += " | YOLO(path) succeeded after weights_only=True"
                print("YOLO(path) succeeded after safe load")
                return
            except Exception as e:
                _model_load_trace += f" | YOLO still failed: {e}"
                print("YOLO still failed after weights_only=True", e)
    except Exception as e:
        print("torch.load(weights_only=True) failed:", e)

    # 3) Fallback: unsafe torch.load(weights_only=False)
    try:
        print("FALLBACK: torch.load(..., weights_only=False) — UNSAFE (only for trusted checkpoints).")
        obj2 = torch.load(path, map_location="cpu", weights_only=False)
        _model_load_trace += f" | torch.load(weights_only=False) returned {type(obj2)}"
        print("torch.load(weights_only=False) returned:", type(obj2))
        # If ultralytics available, try YOLO again
        if YOLO is not None:
            try:
                _model = YOLO(path)
                _model_load_trace += " | YOLO(path) succeeded after unsafe load"
                print("YOLO(path) succeeded after unsafe torch.load fallback")
                return
            except Exception as e:
                _model_load_trace += f" | YOLO still failed after unsafe load: {e}"
                print("YOLO still failed after unsafe load", e)
        # If the object itself is an nn.Module, use it
        if isinstance(obj2, torch.nn.Module):
            _model = obj2
            print("Using loaded nn.Module from unsafe fallback")
            return
        # else we'll store obj2 in trace for debugging
        _model_load_trace += f" | final raw type: {type(obj2)}"
    except Exception as e:
        _model_load_trace += f" | unsafe torch.load failed: {e}"
        print("Unsafe torch.load fallback failed:", e, traceback.format_exc())

# Try to load now
print("載入模型從:", MODEL_PATH)
try_load_model(MODEL_PATH)
print("MODEL:", type(_model), "trace:", _model_load_trace)

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    status = {"message": "YOLOv8 API on Render", "model_loaded": _model is not None, "model_type": str(type(_model)), "model_load_trace": _model_load_trace}
    return jsonify(status)

@app.route("/health", methods=["GET"])
def health():
    if _model is None:
        return jsonify({"status": "error", "model_loaded": False, "trace": _model_load_trace}), 500
    return jsonify({"status": "ok", "model_loaded": True})

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"ok": True, "model_loaded": _model is not None, "model_type": str(type(_model)), "trace": _model_load_trace})

@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"error": "模型尚未載入", "trace": _model_load_trace}), 500

    if "file" not in request.files:
        return jsonify({"error": "請上傳圖片 (form-data key=file)"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "無檔名"}), 400

    # Save temporary
    tmp_path = os.path.join("/tmp", f.filename)
    f.save(tmp_path)

    try:
        # Preferred: if ultralytics YOLO object, call its API
        if YOLO is not None and hasattr(_model, "predict"):
            # ultralytics accepts PIL image, path, or numpy
            results = _model.predict(tmp_path)  # returns list of Results
            predictions = []
            for r in results:
                # r.boxes is list-like
                for box in getattr(r, "boxes", []):
                    try:
                        cls_idx = int(box.cls[0].item()) if hasattr(box, "cls") else None
                    except Exception:
                        try:
                            cls_idx = int(box.cls)
                        except:
                            cls_idx = None
                    conf = float(box.conf[0].item()) if hasattr(box, "conf") else (float(box.conf) if hasattr(box, "conf") else None)
                    label = None
                    if hasattr(_model, "names") and cls_idx is not None:
                        label = _model.names.get(cls_idx) if isinstance(_model.names, dict) else _model.names[cls_idx]
                    predictions.append({
                        "class": label if label is not None else str(cls_idx),
                        "confidence": conf,
                        "bbox": box.xyxy[0].tolist() if hasattr(box, "xyxy") else None
                    })
            return jsonify({"predictions": predictions})
        else:
            # Generic nn.Module inference: try to run best-effort (image -> tensor)
            from PIL import Image
            import numpy as np
            import torch as _torch
            img = Image.open(tmp_path).convert("RGB")
            arr = (np.array(img).astype("float32") / 255.0)[..., ::-1]  # BGR maybe required by some models
            tensor = _torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            with _torch.no_grad():
                out = _model(tensor)
            # return basic info
            return jsonify({"raw_output_type": str(type(out)), "note": "Generic model output returned (format depends on your model)."})
    except Exception as e:
        return jsonify({"error": "inference failed", "trace": str(e), "model_trace": _model_load_trace}), 500
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# Entrypoint - Gunicorn uses "app"
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)




