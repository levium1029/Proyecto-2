import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import base64
import os
import json


def serve_confusion_image(path, width_pct=50):
    if not os.path.exists(path):
        return html.Div(f"Archivo no encontrado: {path}", style={"color": "red"})
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return html.Img(src=f"data:image/png;base64,{encoded}", style={"width": f"{width_pct}%", "height": "auto"})

# Cargar métricas desde archivos JSON
with open("metrics_mate.json", "r") as f:
    metrics_mate = json.load(f)

with open("metrics_mate.json", "r") as f:
    metrics_ingles = json.load(f)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)

app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(label="2. Matriz Confusión Matemáticas", tab_id="tab2"),
        dbc.Tab(label="3. Matriz Confusión Inglés", tab_id="tab3"),
        dbc.Tab(label="5. Métricas Modelos", tab_id="tab5"),
    ], id="tabs", active_tab="tab2"),
    html.Div(id="tab-content", className="p-4")
], fluid=True)


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab(tab):
    if tab == "tab2":
        return html.Div([
            html.H3("Matriz de Confusión Matemáticas"),
            serve_confusion_image("confusion_matrix.png")
        ])
    elif tab == "tab3":
        return html.Div([
            html.H3("Matriz de Confusión Inglés"),
            serve_confusion_image("confusion_matrix.png")
        ])
    elif tab == "tab5":
        return html.Div([
            html.H3("Métricas de Desempeño"),
            dbc.Row([
                dbc.Col([
                    html.H4("Modelo Matemáticas"),
                    html.Ul([
                        html.Li(f"Accuracy: {metrics_mate['accuracy']:.2f}"),
                        html.Li(f"Precision: {metrics_mate['precision']:.2f}"),
                        html.Li(f"Recall: {metrics_mate['recall']:.2f}"),
                        html.Li(f"F1-score: {metrics_mate['f1']:.2f}"),
                    ])
                ], width=6),
                dbc.Col([
                    html.H4("Modelo Inglés"),
                    html.Ul([
                        html.Li(f"Accuracy: {metrics_ingles['accuracy']:.2f}"),
                        html.Li(f"Precision: {metrics_ingles['precision']:.2f}"),
                        html.Li(f"Recall: {metrics_ingles['recall']:.2f}"),
                        html.Li(f"F1-score: {metrics_ingles['f1']:.2f}"),
                    ])
                ], width=6),
            ])
        ])
    else:
        return html.Div("Selecciona una pestaña válida.")

if __name__ == "__main__":
    app.run(debug=True)