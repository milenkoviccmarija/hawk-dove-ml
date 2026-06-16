import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import pandas as pd
from joblib import load

from preprocess import FEATURE_COLUMNS

HOST = "127.0.0.1"
PORT = 8000
MAX_PORT_ATTEMPTS = 10

# Putanje su prilagođene unifikovanom Pipeline objektu
DEPLOYMENT_DIR = Path("deployment")
PIPELINE_PATH = DEPLOYMENT_DIR / "best_pipeline.joblib"
FALLBACK_PIPELINE_PATH = Path("models/best_pipeline.joblib")

FIELD_CONFIG = {
    "V": {
        "label": "Vrednost resursa (V)",
        "min": 0,
        "max": 50,
        "step": 0.1,
        "value": 12.0,
    },
    "C": {
        "label": "Cena konflikta (C)",
        "min": 0,
        "max": 100,
        "step": 0.1,
        "value": 18.0,
    },
    "initial_hawk": {
        "label": "Pocetni udeo hawk",
        "min": 0,
        "max": 1,
        "step": 0.01,
        "value": 0.45,
    },
    "iterations": {
        "label": "Broj iteracija simulacije",
        "min": 1,
        "max": 1000,
        "step": 1,
        "value": 120,
    },
    "learning_rate": {
        "label": "Stopa ucenja (learning rate)",
        "min": 0.001,
        "max": 0.5,
        "step": 0.001,
        "value": 0.08,
    },
    "mutation_rate": {
        "label": "Stopa mutacije (mutation rate)",
        "min": 0,
        "max": 0.1,
        "step": 0.001,
        "value": 0.01,
    },
    "environment_volatility": {
        "label": "Volatilnost okruzenja",
        "min": 0,
        "max": 0.5,
        "step": 0.001,
        "value": 0.05,
    },
    "population_size": {
        "label": "Velicina populacije",
        "min": 10,
        "max": 5000,
        "step": 1,
        "value": 500,
    },
}


def predict_final_hawk(values):
    # Provera i učitavanje unifikovanog cevovoda (Pipeline)
    if PIPELINE_PATH.exists():
        pipeline = load(PIPELINE_PATH)
    elif FALLBACK_PIPELINE_PATH.exists():
        pipeline = load(FALLBACK_PIPELINE_PATH)
    else:
        raise FileNotFoundError(
            "Nije pronadjen istrenirani ML cevovod (best_pipeline.joblib). Pokrenite train.py."
        )

    # Pravimo DataFrame sa originalnim nazivima kolona iz prosleđenih vrednosti
    scenario_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    
    # Pipeline u sebi ima ugrađen StandardScaler, tako da mu prosleđujemo SIROVE podatke.
    # On automatski radi transformaciju i prosleđuje model.predict() koraku.
    prediction = pipeline.predict(scenario_df)[0]
    prediction = max(0, min(1, prediction))
    
    return prediction


def parse_prediction_form(body):
    parsed = parse_qs(body)
    values = {}
    for field in FEATURE_COLUMNS:
        if field not in parsed:
            raise ValueError(f"Nedostaje obavezno polje: {field}")
        val_str = parsed[field][0]
        if FIELD_CONFIG[field]["step"] == 1:
            values[field] = int(val_str)
        else:
            values[field] = float(val_str)
    return values


def render_page(values=None, prediction=None, error=None):
    if values is None:
        values = {k: v["value"] for k, v in FIELD_CONFIG.items()}

    form_html = ""
    for field, config in FIELD_CONFIG.items():
        current_val = values.get(field, config["value"])
        form_html += f"""
        <div class="form-group">
            <label for="{field}">{html.escape(config["label"])}</label>
            <input type="number" id="{field}" name="{field}" 
                   min="{config["min"]}" max="{config["max"]}" 
                   step="{config["step"]}" value="{current_val}" required>
        </div>
        """

    result_html = ""
    if prediction is not None:
        dominant = "Hawk (Jastreb)" if prediction > 0.5 else "Dove (Golub)"
        result_html = f"""
        <div class="result-box">
            <h3>Rezultat Predikcije</h3>
            <p><strong>Predikovani udeo Hawk strategije:</strong> {prediction:.4f}</p>
            <p><strong>Predikovani udeo Dove strategije:</strong> {1 - prediction:.4f}</p>
            <p><strong>Dominantna strategija u populaciji:</strong> <span class="badge">{dominant}</span></p>
        </div>
        """

    error_html = f'<div class="error-box">{html.escape(error)}</div>' if error else ""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Hawk-Dove ML Prediktor</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f4f6f9; color: #333; margin: 0; padding: 40px 20px; }}
        .container {{ max-width: 650px; background: #fff; margin: 0 auto; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        h1 {{ text-align: center; color: #1e293b; margin-top: 0; margin-bottom: 25px; }}
        .form-group {{ margin-bottom: 18px; display: flex; flex-direction: column; }}
        label {{ font-weight: 600; margin-bottom: 6px; color: #475569; font-size: 14px; }}
        input {{ padding: 10px; border: 1px solid #cbd5e1; border-radius: 5px; font-size: 15px; }}
        input:focus {{ border-color: #3b82f6; outline: none; box-shadow: 0 0 0 3px rgba(59,130,246,0.15); }}
        button {{ width: 100%; background: #2563eb; color: #fff; border: none; padding: 12px; font-size: 16px; font-weight: 600; border-radius: 5px; cursor: pointer; margin-top: 10px; transition: background 0.2s; }}
        button:hover {{ background: #1d4ed8; }}
        .result-box {{ background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; padding: 20px; border-radius: 6px; margin-top: 25px; }}
        .result-box h3 {{ margin-top: 0; color: #14532d; }}
        .badge {{ background: #166534; color: #fff; padding: 3px 8px; border-radius: 4px; font-size: 14px; font-weight: 600; }}
        .error-box {{ background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; padding: 15px; border-radius: 6px; margin-bottom: 20px; font-weight: 500; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Hawk-Dove ML Prediktor</h1>
        {error_html}
        <form method="POST" action="/predict">
            {form_html}
            <button type="submit">Izvrši predikciju</button>
        </form>
        {result_html}
    </div>
</body>
</html>
"""


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    def server_bind(self):
        import socket
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


class AppHandler(BaseHTTPRequestHandler):
    def send_html(self, html_content, status=200):
        payload = html_content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/", "/predict"):
            self.send_html(render_page())
            return

        if self.path == "/api/config":
            payload = json.dumps(FIELD_CONFIG).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_html(render_page(error="Stranica nije pronađena."), status=404)

    def do_POST(self):
        if self.path != "/predict":
            self.send_html(render_page(error="Nepoznata akcija."), status=404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            values = parse_prediction_form(body)
            prediction = predict_final_hawk(values)
            self.send_html(render_page(values=values, prediction=prediction))
        except Exception as exc:
            self.send_html(render_page(error=str(exc)), status=400)

    def log_message(self, format, *args):
        return


def main():
    server = None
    selected_port = None

    for port in range(PORT, PORT + MAX_PORT_ATTEMPTS):
        try:
            server = ReusableThreadingHTTPServer((HOST, port), AppHandler)
            selected_port = port
            break
        except OSError:
            continue

    if server is None or selected_port is None:
        raise OSError(
            f"Nije pronađen slobodan port od {PORT} do {PORT + MAX_PORT_ATTEMPTS - 1}."
        )

    print(f"Hawk-Dove ML server uspešno pokrenut na http://{HOST}:{selected_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nZaustavljanje servera...")
        server.server_close()


if __name__ == "__main__":
    main()