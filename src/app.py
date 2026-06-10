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

DEPLOYMENT_DIR = Path("deployment")
MODEL_PATH = DEPLOYMENT_DIR / "best_model.joblib"
SCALER_PATH = DEPLOYMENT_DIR / "standard_scaler.joblib"
FALLBACK_MODEL_PATH = Path("models/best_model.joblib")
FALLBACK_SCALER_PATH = Path("models/standard_scaler.joblib")

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
        "label": "Broj iteracija",
        "min": 10,
        "max": 1000,
        "step": 1,
        "value": 120,
    },
    "learning_rate": {
        "label": "Learning rate",
        "min": 0,
        "max": 1,
        "step": 0.01,
        "value": 0.08,
    },
    "mutation_rate": {
        "label": "Mutation rate",
        "min": 0,
        "max": 0.5,
        "step": 0.001,
        "value": 0.01,
    },
    "environment_volatility": {
        "label": "Volatilnost okruzenja",
        "min": 0,
        "max": 1,
        "step": 0.01,
        "value": 0.05,
    },
    "population_size": {
        "label": "Velicina populacije",
        "min": 10,
        "max": 10000,
        "step": 1,
        "value": 500,
    },
}


def resolve_artifact(primary_path, fallback_path):
    if primary_path.exists():
        return primary_path
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(
        f"Nedostaje {primary_path}. Pokreni src/export_model.py ili src/train.py."
    )


def load_artifacts():
    model = load(resolve_artifact(MODEL_PATH, FALLBACK_MODEL_PATH))
    scaler = load(resolve_artifact(SCALER_PATH, FALLBACK_SCALER_PATH))
    return model, scaler


def predict_final_hawk(values):
    model, scaler = load_artifacts()
    scenario_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    scenario_scaled = scaler.transform(scenario_df)
    scenario_scaled = pd.DataFrame(scenario_scaled, columns=FEATURE_COLUMNS)
    prediction = float(model.predict(scenario_scaled)[0])
    return max(0.0, min(1.0, prediction))


def parse_prediction_form(body):
    parsed = parse_qs(body)
    values = {}

    for column in FEATURE_COLUMNS:
        raw_value = parsed.get(column, [""])[0]
        if raw_value == "":
            raise ValueError(f"Polje {column} je obavezno.")
        values[column] = float(raw_value)

    return values


def render_fields(values):
    field_html = []

    for column in FEATURE_COLUMNS:
        config = FIELD_CONFIG[column]
        value = values.get(column, config["value"])
        field_html.append(
            f"""
            <label class="field">
                <span>{html.escape(config["label"])}</span>
                <input
                    name="{column}"
                    type="number"
                    min="{config["min"]}"
                    max="{config["max"]}"
                    step="{config["step"]}"
                    value="{html.escape(str(value))}"
                    required
                >
            </label>
            """
        )

    return "\n".join(field_html)


def render_result(prediction):
    if prediction is None:
        return ""

    final_dove = 1 - prediction
    dominant_strategy = "hawk" if prediction > 0.5 else "dove"
    strategy_class = "hawk" if dominant_strategy == "hawk" else "dove"

    return f"""
    <section class="result">
        <div>
            <span class="eyebrow">Predikcija</span>
            <strong>{prediction:.4f}</strong>
            <small>final_hawk</small>
        </div>
        <div>
            <span class="eyebrow">Final dove</span>
            <strong>{final_dove:.4f}</strong>
            <small>1 - final_hawk</small>
        </div>
        <div>
            <span class="eyebrow">Dominantno</span>
            <strong class="{strategy_class}">{dominant_strategy}</strong>
            <small>prag je 0.5</small>
        </div>
    </section>
    """


def render_page(values=None, prediction=None, error=None):
    if values is None:
        values = {column: FIELD_CONFIG[column]["value"] for column in FEATURE_COLUMNS}

    error_html = ""
    if error:
        error_html = f'<p class="error">{html.escape(error)}</p>'

    return f"""<!doctype html>
<html lang="sr">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Hawk-Dove AI UI</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f6f7f9;
            --panel: #ffffff;
            --text: #1d2430;
            --muted: #667085;
            --line: #d7dce2;
            --accent: #126d57;
            --accent-dark: #0d4f40;
            --danger: #b42318;
            --hawk: #9f2d20;
            --dove: #295e9e;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            min-height: 100vh;
            background: var(--bg);
            color: var(--text);
            font-family: Arial, Helvetica, sans-serif;
        }}

        main {{
            width: min(1080px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 32px 0;
        }}

        header {{
            margin-bottom: 24px;
        }}

        h1 {{
            margin: 0 0 8px;
            font-size: clamp(28px, 4vw, 44px);
            font-weight: 760;
            letter-spacing: 0;
        }}

        .subtitle {{
            margin: 0;
            max-width: 760px;
            color: var(--muted);
            line-height: 1.55;
        }}

        .workspace {{
            display: grid;
            grid-template-columns: minmax(0, 1.15fr) minmax(300px, 0.85fr);
            gap: 20px;
            align-items: start;
        }}

        form, .info {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 20px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }}

        .field {{
            display: grid;
            gap: 7px;
            min-width: 0;
        }}

        .field span {{
            color: var(--muted);
            font-size: 14px;
        }}

        input {{
            width: 100%;
            min-height: 44px;
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 9px 10px;
            color: var(--text);
            font-size: 16px;
        }}

        input:focus {{
            outline: 2px solid rgba(18, 109, 87, 0.22);
            border-color: var(--accent);
        }}

        button {{
            width: 100%;
            min-height: 46px;
            margin-top: 18px;
            border: 0;
            border-radius: 6px;
            background: var(--accent);
            color: white;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
        }}

        button:hover {{
            background: var(--accent-dark);
        }}

        .result {{
            display: grid;
            gap: 12px;
        }}

        .result div, .info-item {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px;
        }}

        .eyebrow {{
            display: block;
            margin-bottom: 8px;
            color: var(--muted);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0;
        }}

        strong {{
            display: block;
            font-size: 34px;
            line-height: 1.1;
        }}

        small {{
            display: block;
            margin-top: 8px;
            color: var(--muted);
        }}

        .hawk {{
            color: var(--hawk);
        }}

        .dove {{
            color: var(--dove);
        }}

        .info {{
            margin-top: 12px;
        }}

        .info h2 {{
            margin: 0 0 12px;
            font-size: 20px;
        }}

        .info ul {{
            margin: 0;
            padding-left: 18px;
            color: var(--muted);
            line-height: 1.6;
        }}

        .error {{
            margin: 0 0 14px;
            color: var(--danger);
            font-weight: 700;
        }}

        @media (max-width: 820px) {{
            .workspace {{
                grid-template-columns: 1fr;
            }}

            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <main>
        <header>
            <h1>Hawk-Dove AI UI</h1>
            <p class="subtitle">
                Unesi parametre simulacije i dobij predikciju finalnog udela hawk strategije.
            </p>
        </header>

        <section class="workspace">
            <form method="post" action="/predict">
                {error_html}
                <div class="grid">
                    {render_fields(values)}
                </div>
                <button type="submit">Predvidi final_hawk</button>
            </form>

            <aside>
                {render_result(prediction)}
                <section class="info">
                    <h2>Deployment paket</h2>
                    <ul>
                        <li>Model: deployment/best_model.joblib</li>
                        <li>Scaler: deployment/standard_scaler.joblib</li>
                        <li>Metadata: deployment/model_metadata.json</li>
                    </ul>
                </section>
            </aside>
        </section>
    </main>
</body>
</html>"""


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class AppHandler(BaseHTTPRequestHandler):
    def send_empty(self, status=200):
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_html(self, content, status=200):
        encoded = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_HEAD(self):
        if self.path in {"/", "/health"}:
            self.send_empty()
            return

        self.send_empty(status=404)

    def do_GET(self):
        if self.path == "/":
            self.send_html(render_page())
            return

        if self.path == "/health":
            payload = json.dumps({"status": "ok"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_html(render_page(error="Stranica nije pronadjena."), status=404)

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
            f"Nije pronadjen slobodan port od {PORT} do {PORT + MAX_PORT_ATTEMPTS - 1}."
        )

    print(f"Hawk-Dove AI UI je pokrenut: http://{HOST}:{selected_port}")
    print("Za prekid servera pritisni Ctrl+C.")
    server.serve_forever()


if __name__ == "__main__":
    main()
