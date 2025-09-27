# app.py (robust dynamic safe-globals loader for YOLO .pt)
import os
import re
import io
import importlib
import traceback
import warnings
from flask import Flask, request, jsonify
import torch

warnings.filterwarnings("ignore")

# try import ultralytics (YOLO)
try:
    from ultralytics import YOLO
    import ultralytics
except Exception as e:
    print("ultralytics import failed:", e)
    YOLO = None
    ultralytics = None

app = Flask(__name__)

# --------------- helper: add safe global from a full dotted path ---------------
def add_safe_global_from_fullname(fullname):
    """
    fullname example: 'torch.nn.modules.batchnorm.BatchNorm2d'
    Tries to import module and get attribute, then add to torch safe globals.
    Returns (True, message) if added, else (False, message).
    """
    try:
        parts = fullname.split(".")
        if len(parts) < 2:
            return False, f"bad fullname {fullname}"
        module_name = ".".join(parts[:-1])
        class_name = parts[-1]
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        torch.serialization.add_safe_globals([cls])
        return True, f"added {module_name}.{class_name}"
    except Exception as e:
        return False, f"failed add {fullname}: {e}"

# --------------- attempt iterative YOLO load with dynamic safe-globals ---------------
def iterative_yolo_load(path, max_rounds=10):
    """
    Try to load YOLO(path). If fails due to Unsupported global, extract those names
    and add them to torch safe globals, then retry. Repeat up to max_rounds.
    Returns (model_or_none, trace_string).
    """
    trace = []
    added = set()

    for attempt in range(1, max_rounds + 1):
        trace.append(f"Attempt {attempt}: trying YOLO(path)...")
        print(trace[-1])
        try:
            if YOLO is None:
                raise RuntimeError("ultralytics.YOLO not available in environment.")
            model = YOLO(path)
            trace.append(f"SUCCESS on attempt {attempt}: YOLO(path) loaded.")
            print(trace[-1])
            return model, "\n".join(trace)
        except Exception as e:
            tb = traceback.format_exc()
            msg = str(e)
            trace.append(f"YOLO(path) failed: {msg}")
            print(trace[-1])
            print(tb)

            # find GLOBAL names in the message, common pattern:
            # "Unsupported global: GLOBAL torch.nn.modules.batchnorm.BatchNorm2d was not an allowed..."
            # We'll capture sequences of letters/digits/_. that look like module paths
            matches = re.findall(r"GLOBAL\s+([A-Za-z0-9_.]+)", msg)
            # also search in the full traceback text
            if not matches:
                matches = re.findall(r"GLOBAL\s+([A-Za-z0-9_.]+)", tb)

            if not matches:
                trace.append("No GLOBAL pattern found in error; cannot auto-add safe globals.")
                print(trace[-1])
                break

            any_added_this_round = False
            for fullname in matches:
                if fullname in added:
                    trace.append(f"{fullname} already attempted.")
                    continue
                ok, info = add_safe_global_from_fullname(fullname)
                trace.append(f"Attempt add {fullname}: {ok} - {info}")
                print(trace[-1])
                if ok:
                    any_added_this_round = True
                    added.add(fullname)

            if not any_added_this_round:
                trace.append("No new safe-globals were added this round; stopping retries.")
                print(trace[-1])
                break
            else:
                trace.append(f"Added {len(added)} safe-globals so far; will retry YOLO(path).")
                print(trace[-1])
                # loop will retry

    # after attempts, also provide a fallback info: try torch.load to inspect type
    try:
        trace.append("Final fallback: try torch.load(weights_only=True) to inspect checkpoint.")
        print(trace[-1])
        ck = torch.load(path, map_location="cpu", weights_only=True)
        trace.append(f"torch.load(weights_only=True) returned type {type(ck)}")
        print(trace[-1])
    except Exception as e:
        trace.append(f"torch.load(weights_only=True) failed: {e}")
        print(trace[-1])
        # try unsafe to inspect raw type
        try:
            ck2 = torch.load(path, map_location="cpu", weights_only=False)
            trace.append(f"torch.load(weights_only=False) returned type {type(ck2)}")
            print(trace[-1])
        except Exception as e2:
            trace.append(f"unsafe torch.load also failed: {e2}")
            print(trace[-1])

    return None, "\n".join(trace)

# --------------- load model at startup ---------------
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "freshness_fruit_and_vegetables.pt"))
print("MODEL_PATH:", MODEL_PATH)
_model = None
_model_trace = ""

if not os.path.exists(MODEL_PATH):
    _model_trace = f"MODEL PATH NOT FOUND: {MODEL_PATH}"
    print(_model_trace)
else:
    try:
        _model, _model_trace = iterative_yolo_load(MODEL_PATH, max_rounds=12)
    except Exception as e:
        _model = None
        _model_trace = f"iterative loader exception: {e}\n{traceback.format_exc()}"
    print("Model load result type:", type(_model))
    print("Trace:\n", _model_trace)

# --------------- Flask endpoints ---------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "YOLOv8 API (robust loader)",
        "model_loaded": _model is not None,
        "model_type": str(type(_model)),
        "trace_summary": _model_trace[:2000]  # limit length returned
    })

@app.route("/health", methods=["GET"])
def health():
    if _model is None:
        return jsonify({"status": "error", "model_loaded": False, "trace": _model_trace}), 500
    return jsonify({"status": "ok", "model_loaded": True})

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"ok": True, "model_loaded": _model is not None, "model_type": str(type(_model)), "trace": _model_trace})

@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"error": "model not loaded", "trace": _model_trace}), 500
    if "file" not in request.files:
        return jsonify({"error": "please upload file as form-data key 'file'"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    tmp = os.path.join("/tmp", f.filename)
    f.save(tmp)

    try:
        # prefer ultralytics API if available
        if YOLO is not None and hasattr(_model, "predict"):
            results = _model.predict(tmp)
            preds = []
            for r in results:
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
                        try:
                            label = _model.names[cls_idx]
                        except Exception:
                            try:
                                label = _model.names.get(cls_idx)
                            except:
                                label = str(cls_idx)
                    preds.append({
                        "class": label if label is not None else str(cls_idx),
                        "confidence": conf,
                        "bbox": box.xyxy[0].tolist() if hasattr(box, "xyxy") else None
                    })
            return jsonify({"predictions": preds})
        else:
            # generic fallback (best-effort)
            from PIL import Image
            import numpy as np
            import torch as _torch
            img = Image.open(tmp).convert("RGB")
            arr = (np.array(img).astype("float32") / 255.0)[..., ::-1]
            tensor = _torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
            with _torch.no_grad():
                out = _model(tensor)
            return jsonify({"raw_type": str(type(out))})
    except Exception as e:
        return jsonify({"error": "inference failed", "trace": str(e), "model_trace": _model_trace}), 500
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)




