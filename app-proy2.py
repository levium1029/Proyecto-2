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

df_Mate = pd.read_csv('SaberMate.csv')
df_Ingles = pd.read_csv('SaberIngles.csv')

model_mate = load_model('model_mate.keras')
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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

import dash_bootstrap_components as dbc
from dash import html

app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="MEN - Analítica Educativa Saber 11",
        color="#003366",
        dark=True,
        fixed="top",
        style={"fontWeight": "bold", "fontSize": "20px"}
    ),

    # Espacio para que los tabs no queden debajo del Navbar fijo
    html.Div(style={"height": "60px"}),

    # Tabs visibles justo debajo del Navbar
    dbc.Tabs([
        dbc.Tab(label="1. Mapas por Departamento", tab_id="tab1"),
        dbc.Tab(label="2. Pregunta de negocio 1", tab_id="tab2"),
        dbc.Tab(label="3. Pregunta de negocio 2", tab_id="tab3"),
        dbc.Tab(label="4. Predicción Personalizada", tab_id="tab4"),
        dbc.Tab(label="5. Métricas y ROC", tab_id="tab5"),
    ], id="tabs", active_tab="tab1"),

    # Contenido dinámico según pestaña seleccionada
    html.Div(id="tab-content", className="p-4",
             style={"backgroundColor": "#f8f9fa",
                    "minHeight": "80vh",
                    "borderRadius": "10px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                    "marginTop": "20px"}),

    # Footer
    html.Footer(
        "© 2025 Ministerio de Educación Nacional de Colombia - Proyecto de Analítica Educativa",
        style={"textAlign": "center", "color": "#6c757d", "padding": "10px 0", "marginTop": "30px"}
    )

], fluid=True)


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab(tab):
    if tab == "tab1":
        with open("mapa_mate_colombia.html", "r", encoding="utf-8") as f_mate, \
            open("mapa_ingles_colombia.html", "r", encoding="utf-8") as f_ingles:
            mapa_html_mate = f_mate.read()
            mapa_html_ingles = f_ingles.read()

        return html.Div([
            html.H2("Mapa por Departamentos", style={
                "textAlign": "center",
                "marginBottom": "30px",
                "fontWeight": "bold"
            }),

            dbc.Row([
                dbc.Col([
                    html.H4("Nivel promedio Matemáticas", style={
                        "textAlign": "center",
                        "marginBottom": "10px"
                    }),
                    html.Iframe(
                        srcDoc=mapa_html_mate,
                        style={"width": "100%", "height": "90vh", "border": "none"}
                    )
                ], width=6),

                dbc.Col([
                    html.H4("Nivel promedio Inglés", style={
                        "textAlign": "center",
                        "marginBottom": "10px"
                    }),
                    html.Iframe(
                        srcDoc=mapa_html_ingles,
                        style={"width": "100%", "height": "90vh", "border": "none"}
                    )
                ], width=6, style={"height": "90vh"}),
            ],  style={"height": "90vh"})
        ],  style={"height": "90vh"})
    
    elif tab == "tab2":
        return html.Div([
            html.H2("Pregunta de negocio 1:", style={"fontWeight": "bold"}),
            html.P("¿Se puede predecir el resultado categorizado en matemáticas de acuerdo con el contexto personal y familiar del estudiante?",
                    style={"fontStyle": "italic", "fontSize": "18px", "marginBottom": "20px"}
            ),
            html.Div(
                    serve_confusion_image("confusion_matrix.png", width_pct=40),
                    style={"display": "flex", "justifyContent": "center", "marginBottom": "5px"}
            ),
            html.Div("Datos en %", style={"textAlign": "center", "fontStyle": "italic", "marginTop": "5px"}),
            html.Br(),
            html.Div([
                html.H4("Respuesta:"),
                html.P(
                    "Con base en la matriz de confusión, podemos observar que el modelo presenta una alta capacidad predictiva en las categorías más frecuentes, "
                    "con una tasa adecuada de verdaderos positivos y bajas tasas de error en las predicciones. Esto indica que sí es posible predecir el resultado "
                    "categorizado en matemáticas a partir del contexto personal y familiar del estudiante, aunque existen áreas de mejora en categorías menos representadas.",
                    style={"fontSize": "20px"}  # Tamaño más grande
                )
            ], style={"marginTop": "30px"})
        ], style={"padding": "20px"})
    
    elif tab == "tab3":
        return html.Div([
            html.H2("Pregunta de negocio 2:", style={"fontWeight": "bold"}),
            html.P(
                "¿Se puede predecir el resultado categorizado en inglés de acuerdo con el contexto escolar?",
                style={"fontStyle": "italic", "fontSize": "18px", "marginBottom": "20px"}
            ),
            html.Div(
                serve_confusion_image("confusion_matrix.png", width_pct=40),  # Cambia al archivo correcto
                style={"display": "flex", "justifyContent": "center", "marginBottom": "5px"}
            ),    
            html.Div("Datos en %", style={"textAlign": "center", "fontStyle": "italic", "marginTop": "5px"}),
            html.Br(),
            html.Div([
                html.H4("Respuesta:"),
                html.P(
                    "Según la matriz de confusión, el modelo demuestra una capacidad adecuada para predecir el nivel de inglés basado en el contexto "
                    "personal y familiar del estudiante, aunque hay oportunidades para mejorar la precisión en ciertas categorías.",
                    style={"fontSize": "20px"}
                )
            ], style={"marginTop": "30px"})
        ], style={"padding": "20px"})

    
    elif tab == "tab4":
        return html.Div([
            html.H3("Predicción Personalizada", style = {"fontWeight": "bold"}),
            html.P("Selecciona las características del estudiante para predecir su nivel en matemáticas."),
            generate_inputs_two_columns(),
            dbc.Button("Predecir", id="btn-prediccion", color="primary", className="mt-3"),
            html.Div(id="output-prediccion", className="mt-4")
        ])
    
    elif tab == "tab5":
        import base64

        with open("roc_mate.png", "rb") as image_file:
            encoded_roc_mate = base64.b64encode(image_file.read()).decode()
        
        with open("roc_mate.png", "rb") as image_file:
            encoded_roc_ingles = base64.b64encode(image_file.read()).decode()

        return html.Div([
            html.H2(
                "Métricas de Desempeño",
                style={"textAlign": "center", "fontWeight": "bold", "marginBottom": "30px"}
            ),

            dbc.Row([
                dbc.Col([
                    html.H4("Modelo Matemáticas", style={"fontWeight": "bold", "marginBottom": "15px"}),
                    html.Img(
                        src=f"data:image/png;base64,{encoded_roc_mate}",
                        style={"width": "100%", "height": "600px", "marginBottom": "20px"}
                    ),
                    html.Ul([
                        html.Li([html.B("Accuracy: "), f"{metrics_mate['accuracy']:.2f}"], style={"fontSize": "18px"}),
                        html.Li([html.B("Precision: "), f"{metrics_mate['precision']:.2f}"], style={"fontSize": "18px"}),
                        html.Li([html.B("Recall: "), f"{metrics_mate['recall']:.2f}"], style={"fontSize": "18px"}),
                        html.Li([html.B("F1-score: "), f"{metrics_mate['f1']:.2f}"], style={"fontSize": "18px"}),
                    ])
                ], width=6),

                dbc.Col([
                    html.H4("Modelo Inglés", style={"fontWeight": "bold", "marginBottom": "15px"}),
                    html.Img(
                        src=f"data:image/png;base64,{encoded_roc_ingles}",
                        style={"width": "100%", "height": "600px", "marginBottom": "20px"}
                    ),
                    html.Ul([
                        html.Li([html.B("Accuracy: "), f"{metrics_ingles['accuracy']:.2f}"], style={"fontSize": "18px"}),
                        html.Li([html.B("Precision: "), f"{metrics_ingles['precision']:.2f}"], style={"fontSize": "18px"}),
                        html.Li([html.B("Recall: "), f"{metrics_ingles['recall']:.2f}"], style={"fontSize": "18px"}),
                        html.Li([html.B("F1-score: "), f"{metrics_ingles['f1']:.2f}"], style={"fontSize": "18px"}),
                    ])
                ], width=6),
            ])
        ], style={"padding": "20px"})
    
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
        print("Input transformado:", X_vec)
        pred_prob = model_mate.predict(X_vec)
        print("Probabilidades:", pred_prob)
        pred_class = np.argmax(pred_prob, axis=1)[0]
        categoria = map_class_to_label(pred_class)
        return html.Div([
            html.P(f"Predicción nivel Matemáticas: {categoria}")
        ], style={"fontSize": "22px", "fontWeight": "bold"})
    except Exception as e:
        return html.Div(f"Error en la predicción: {e}", style={"color": "red"})


if __name__ == "__main__":
    app.run(debug=True)