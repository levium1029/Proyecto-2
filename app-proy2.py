import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import base64
from tensorflow.keras.models import load_model
import os
import json
import pickle
import pandas as pd
import numpy as np

model_mate = load_model('Model_mate.keras')
with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

variables_seleccionadas = ["nivelingles", "automovil", "internet", "computador", "periodo",
                          "estrato", "lavadora", "sexo", "edupadre", "edumadre"]

options_dict = {}
for i, var in enumerate(variables_seleccionadas):
    options_dict[var] = one_hot_encoder.categories_[i].tolist()

def map_class_to_label(clase):
    return f"Nivel {clase+1}"

def generate_inputs_two_columns():
    vars_col1 = variables_seleccionadas[:5]
    vars_col2 = variables_seleccionadas[5:]
    
    col1 = []
    for var in vars_col1:
        options = [{"label": opt, "value": opt} for opt in options_dict[var]]
        col1.append(
            html.Div([
                html.Label(var.title()),
                dcc.Dropdown(
                    id=f"input-{var}",
                    options=options,
                    value=options[0]["value"],
                    clearable=False,
                    style={"width": "90%"}
                ),
                html.Br()
            ])
        )
        
    col2 = []
    for var in vars_col2:
        options = [{"label": opt, "value": opt} for opt in options_dict[var]]
        col2.append(
            html.Div([
                html.Label(var.title()),
                dcc.Dropdown(
                    id=f"input-{var}",
                    options=options,
                    value=options[0]["value"],
                    clearable=False,
                    style={"width": "90%"}
                ),
                html.Br()
            ])
        )
    
    return dbc.Row([
        dbc.Col(col1, width=6),
        dbc.Col(col2, width=6)
    ])

def transform_inputs_to_vector(input_dict):
    df_input = pd.DataFrame([input_dict])
    X_encoded = one_hot_encoder.transform(df_input)
    return X_encoded

def serve_confusion_image(path, width_pct=50):
    if not os.path.exists(path):
        return html.Div(f"Archivo no encontrado: {path}", style={"color": "red"})
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return html.Img(src=f"data:image/png;base64,{encoded}", style={"width": f"{width_pct}%", "height": "auto"})

with open("metrics_mate.json", "r") as f:
    metrics_mate = json.load(f)
with open("metrics_mate.json", "r") as f:
    metrics_ingles = json.load(f)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)

app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(label="2. Matriz Confusión Matemáticas", tab_id="tab2"),
        dbc.Tab(label="3. Matriz Confusión Inglés", tab_id="tab3"),
        dbc.Tab(label="4. Predicción Personalizada", tab_id="tab4"),
        dbc.Tab(label="5. Métricas Modelos", tab_id="tab5"),
    ], id="tabs", active_tab="tab4"),
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
    elif tab == "tab4":
        return html.Div([
            html.H3("Predicción Personalizada"),
            html.P("Selecciona las características del estudiante para predecir su nivel en matemáticas."),
            generate_inputs_two_columns(),
            dbc.Button("Predecir", id="btn-prediccion", color="primary", className="mt-3"),
            html.Div(id="output-prediccion", className="mt-4")
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

@app.callback(
    Output("output-prediccion", "children"),
    Input("btn-prediccion", "n_clicks"),
    [State(f"input-{var}", "value") for var in variables_seleccionadas]
)
def run_prediction(n_clicks, *values):
    if not n_clicks:
        return ""
    input_dict = dict(zip(variables_seleccionadas, values))
    try:
        X_vec = transform_inputs_to_vector(input_dict)
        pred_prob = model_mate.predict(X_vec)
        pred_class = np.argmax(pred_prob, axis=1)[0]
        categoria = map_class_to_label(pred_class)
        return html.Div([
            html.P(f"Predicción nivel Matemáticas: {categoria}")
        ], style={"fontSize": "22px", "fontWeight": "bold"})
    except Exception as e:
        return html.Div(f"Error en la predicción: {e}", style={"color": "red"})

if __name__ == "__main__":
    app.run(debug=True)