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

estilo_titulo = {
    "color": "#003366",
    "fontWeight": "bold",
    "fontSize": "2.2rem",
    "textAlign": "center",
    "marginBottom": "28px",
    "marginTop": "8px",
    "letterSpacing": "1px"
}
estilo_subtitulo = {
    "color": "#A13E3A",
    "fontWeight": "bold",
    "fontSize": "1.25rem",
    "textAlign": "center",
    "marginBottom": "14px",
}
estilo_card = {
    "backgroundColor": "white",
    "borderRadius": "14px",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.07)",
    "padding": "28px",
    "margin": "0 auto",
    "marginBottom": "28px",
    "maxWidth": "900px"
}
estilo_texto = {
    "fontSize": "19px",
    "color": "#2C2C2C",
    "marginBottom": "16px",
}
estilo_boton = {
    "backgroundColor": "#A13E3A",
    "border": "none",
    "fontWeight": "bold",
    "fontSize": "18px"
}

def tabla_niveles_mate():
    niveles = []
    step = 100 / 6
    for i in range(6):
        lower = i * step
        upper = (i + 1) * step
        if i == 0:
            niveles.append(["Nivel 1", f"{lower:.2f} – {upper:.2f}"])
        else:
            niveles.append([f"Nivel {i+1}", f"{lower:.2f} – {upper:.2f}"])
    return dbc.Table(
        [html.Thead(html.Tr([html.Th("Nivel"), html.Th("Rango puntaje")], style={"backgroundColor": "#F6E9E8"}))] +
        [html.Tbody([html.Tr([html.Td(nivel, style={"fontWeight": "bold"}), html.Td(rango)]) for nivel, rango in niveles])],
        bordered=True,
        style={"marginBottom": "0", "marginTop": "10px", "backgroundColor": "#fff", "width": "100%"}
    )

def tabla_niveles_ingles():
    niveles = [
        ("A-", "Menos de 43"),
        ("A1", "43 – 52"),
        ("A2", "53 – 62"),
        ("B1", "63 – 82"),
        ("B+", "83 o más")
    ]
    return dbc.Table(
        [html.Thead(html.Tr([html.Th("Nivel"), html.Th("Rango puntaje")], style={"backgroundColor": "#F6E9E8"}))] +
        [html.Tbody([html.Tr([html.Td(nivel, style={"fontWeight": "bold"}), html.Td(rango)]) for nivel, rango in niveles])],
        bordered=True,
        style={"marginBottom": "0", "marginTop": "10px", "backgroundColor": "#fff", "width": "100%"}
    )

df_Mate = pd.read_csv('SaberMate.csv')
df_Ingles = pd.read_csv('SaberIngles.csv')

model_mate = load_model('modelo_matematicas_entrenado.keras')
model_ingles = load_model('modelo_ingles_entrenado.keras')

import base64

logo_path = "Logo_Ministerio_de_Educación_de_Colombia_2022-2026.png"
with open(logo_path, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()


with open('one_hot_encoder_mate.pkl', 'rb') as f:
    one_hot_encoder_mate = pickle.load(f)
with open('one_hot_encoder_ingles.pkl', 'rb') as f:
    one_hot_encoder_ingles = pickle.load(f)

variables_seleccionadas = list(one_hot_encoder_mate.feature_names_in_)
variables_ingles = list(one_hot_encoder_ingles.feature_names_in_)

options_dict = {}
for i, var in enumerate(variables_seleccionadas):
    options_dict[var] = one_hot_encoder_mate.categories_[i].tolist()

options_dict_ingles = {}
for i, var in enumerate(variables_ingles):
    options_dict_ingles[var] = one_hot_encoder_ingles.categories_[i].tolist()

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

def transform_inputs_to_vector_mate(input_dict):
    df_input = pd.DataFrame([input_dict])
    X_encoded = one_hot_encoder_mate.transform(df_input)
    return X_encoded

def transform_inputs_to_vector_english(input_dict):
    df_input = pd.DataFrame([input_dict])
    X_encoded = one_hot_encoder_ingles.transform(df_input)
    return X_encoded

def generate_inputs_english_two_columns():
    mid = len(variables_ingles) // 2
    vars_col1 = variables_ingles[:mid]
    vars_col2 = variables_ingles[mid:]

    col1 = []
    for var in vars_col1:
        options = [{"label": str(opt), "value": opt} for opt in options_dict_ingles[var]]
        col1.append(
            html.Div([
                html.Label(var.title()),
                dcc.Dropdown(
                    id=f"input-eng-{var}",
                    options=options,
                    value=options[0]["value"] if options else None,
                    clearable=False,
                    style={"width": "90%"}
                ),
                html.Br()
            ])
        )
    col2 = []
    for var in vars_col2:
        options = [{"label": str(opt), "value": opt} for opt in options_dict_ingles[var]]
        col2.append(
            html.Div([
                html.Label(var.title()),
                dcc.Dropdown(
                    id=f"input-eng-{var}",
                    options=options,
                    value=options[0]["value"] if options else None,
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

def serve_confusion_image(path, width_pct=50):
    if not os.path.exists(path):
        return html.Div(f"Archivo no encontrado: {path}", style={"color": "red"})
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return html.Img(src=f"data:image/png;base64,{encoded}", style={"width": f"{width_pct}%", "height": "auto"})

with open("metrics_mate.json", "r") as f:
    metrics_mate = json.load(f)
with open("metrics_english.json", "r") as f:
    metrics_ingles = json.load(f)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.Img(
                        src=f"data:image/png;base64,{logo_base64}",
                        style={
                            "height": "48px",
                            "paddingRight": "14px",
                            "backgroundColor": "white",
                            "borderRadius": "8px"
                        }
                    ),
                    width="auto",
                    align="center"
                ),
                dbc.Col(
                    html.Span([
                        html.Span("       MEN - Analítica Educativa Saber 11 Año 2013", style={
                            "color": "#003366", "fontWeight": "bold", "fontSize": "22px", "fontFamily": "Arial"
                        }),
                    ]),
                    align="center"
                ),
            ], align="center", className="g-0"),
        ], fluid=True),
        color="light",     # Fondo blanco en la Navbar
        dark=False,        # Texto oscuro
        fixed="top",
        style={
            "padding": "0px",
            "borderBottom": "4px solid #A13E3A",  # Línea roja institucional debajo
            "boxShadow": "0 2px 8px rgba(0,0,0,0.07)"
        }
    ),

    html.Div(style={"height": "60px"}),

    dbc.Tabs([
        dbc.Tab(label="1. Mapas por Departamento", tab_id="tab1"),
        dbc.Tab(label="2. Pregunta de negocio 1", tab_id="tab2"),
        dbc.Tab(label="3. Pregunta de negocio 2", tab_id="tab3"),
        dbc.Tab(label="4. Predicción Matemáticas", tab_id="tab4"),
        dbc.Tab(label="5. Predicción Inglés", tab_id="tab5"),
        dbc.Tab(label="6. Métricas y ROC", tab_id="tab6"),
    ], id="tabs", active_tab="tab1", style={
        "backgroundColor": "#F5F5F5",
        "borderBottom": "3px solid #A13E3A"
    }),

    html.Div(
        id="tab-content",
        className="p-4",
        style={
            "backgroundColor": "#F5F5F5",
            "minHeight": "80vh",
            "borderRadius": "10px",
            "boxShadow": "0 4px 8px rgba(0,0,0,0.08)",
            "marginTop": "20px"
        }
    ),

    html.Footer(
        "© 2025 Ministerio de Educación Nacional de Colombia - Proyecto de Analítica Educativa",
        style={
            "textAlign": "center",
            "color": "#A13E3A",
            "padding": "10px 0",
            "marginTop": "30px",
            "fontWeight": "bold",
            "backgroundColor": "#F5F5F5"
        }
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
            html.H2("Mapa por Departamentos", style=estilo_titulo),
            dbc.Row([
                dbc.Col([
                    html.H4("Nivel promedio Matemáticas", style=estilo_subtitulo),
                    html.Div(
                        html.Iframe(
                            srcDoc=mapa_html_mate,
                            style={"width": "100%", "height": "65vh", "border": "none"}
                        ),
                        style=estilo_card
                    )
                ], width=6),
                dbc.Col([
                    html.H4("Nivel promedio Inglés", style=estilo_subtitulo),
                    html.Div(
                        html.Iframe(
                            srcDoc=mapa_html_ingles,
                            style={"width": "100%", "height": "65vh", "border": "none"}
                        ),
                        style=estilo_card
                    )
                ], width=6),
            ], style={"justifyContent": "center"})
        ])
    
    elif tab == "tab2":
        return html.Div([
            html.H2("Pregunta de negocio 1:", style=estilo_titulo),
            html.P(
                "¿Se puede predecir el resultado categorizado en matemáticas de acuerdo con el contexto personal y familiar del estudiante?",
                style={**estilo_texto, "textAlign": "center", "fontStyle": "italic"}
            ),
            dbc.Row([
                dbc.Col(
                    html.Div(
                        serve_confusion_image("confusion_matrix_mate.png", width_pct=98),
                        style=estilo_card
                    ),
                    width=8
                ),
                dbc.Col(
                    html.Div([
                        html.H5("Equivalencia de Niveles", style=estilo_subtitulo),
                        tabla_niveles_mate(),
                        html.P(
                            "El puntaje Saber 11 de Matemáticas se divide en 6 niveles iguales entre 0 y 100.",
                            style={**estilo_texto, "fontSize": "17px", "color": "#555"}
                        )
                    ], style=estilo_card),
                    width=4
                )
            ], align="center", style={"marginBottom": "18px"}),
            html.Div([
                html.H4("Respuesta:", style=estilo_subtitulo),
                html.P(
                    "Con base en la matriz de confusión, se puede observar que el modelo predice con alta precisión para los niveles 0, 2, 4 y 5. "
                    "Sin embargo, en los niveles 1 y 3 se presentan confusiones o errores de clasificación, lo que se debe al problema de desbalanceo de clases.",
                    style=estilo_texto
                ),
                html.P(
                    "Esto indica que sí es posible predecir el resultado en matemáticas con base en el contexto personal y familiar del estudiante.",
                    style={**estilo_texto, "fontWeight": "bold"}
                )
            ], style=estilo_card)
        ])

    elif tab == "tab3":
        return html.Div([
            html.H2("Pregunta de negocio 2:", style=estilo_titulo),
            html.P(
                "¿Se puede predecir el resultado categorizado en inglés de acuerdo con el contexto escolar?",
                style={**estilo_texto, "textAlign": "center", "fontStyle": "italic"}
            ),
            dbc.Row([
                dbc.Col(
                    html.Div(
                        serve_confusion_image("confusion_matrix_ingles.png", width_pct=98),
                        style=estilo_card
                    ),
                    width=8
                ),
                dbc.Col(
                    html.Div([
                        html.H5("Equivalencia de Niveles", style=estilo_subtitulo),
                        tabla_niveles_ingles(),
                        html.P(
                            "El puntaje Saber 11 de Inglés se agrupa en niveles oficiales del MCER.",
                            style={**estilo_texto, "fontSize": "17px", "color": "#555"}
                        )
                    ], style=estilo_card),
                    width=4
                )
            ], align="center", style={"marginBottom": "18px"}),
            html.Div([
                html.H4("Respuesta:", style=estilo_subtitulo),
                html.P([
                    "Según la matriz de confusión, el modelo predice de manera efectiva los niveles 0 y 4, no obstante, presenta confusiones en los niveles 1, 2 y 3. ",
                    "Esto se debe al problema mencionado anteriormente de ",
                    html.B("clases desbalanceadas", style={"color": "#A13E3A"}),
                    "."
                ], style=estilo_texto),
                html.P(
                    "En conclusión, sí es posible predecir el resultado categórico en inglés de acuerdo con el contexto escolar.",
                    style={**estilo_texto, "fontWeight": "bold"}
                )
            ], style=estilo_card)
        ])

    elif tab == "tab4":
        return html.Div([
            html.H3("Predicción Personalizada Matemáticas", style=estilo_titulo),
            html.P("Selecciona las características del estudiante para predecir su nivel en matemáticas.",
                style={**estilo_texto, "textAlign": "center"}
            ),
            html.Div(generate_inputs_two_columns(), style={"marginBottom": "20px"}),
            dbc.Button("Predecir", id="btn-prediccion", color="danger", className="mt-3", style=estilo_boton),
            html.Div(id="output-prediccion", className="mt-4")
        ], style=estilo_card)
    
    elif tab == "tab5":
        return html.Div([
            html.H3("Predicción Personalizada Inglés", style=estilo_titulo),
            html.P("Selecciona las características del estudiante para predecir su nivel en inglés.",
                style={**estilo_texto, "textAlign": "center"}
            ),
            html.Div(generate_inputs_english_two_columns(), style={"marginBottom": "20px"}),
            dbc.Button("Predecir", id="btn-prediccion-ingles", color="danger", className="mt-3", style=estilo_boton),
            html.Div(id="output-prediccion-ingles", className="mt-4")
        ], style=estilo_card)
    
    elif tab == "tab6":
        import base64
        with open("roc_mate.png", "rb") as image_file:
            encoded_roc_mate = base64.b64encode(image_file.read()).decode()
        with open("roc_ingles.png", "rb") as image_file:
            encoded_roc_ingles = base64.b64encode(image_file.read()).decode()
        return html.Div([
            html.H2("Métricas de Desempeño", style=estilo_titulo),
            dbc.Row([
                dbc.Col([
                    html.H4("Modelo Matemáticas", style=estilo_subtitulo),
                    html.Img(
                        src=f"data:image/png;base64,{encoded_roc_mate}",
                        style={"width": "100%", "height": "350px", "marginBottom": "18px"}
                    ),
                    html.Ul([
                        html.Li([html.B("Accuracy: "), f"{metrics_mate['accuracy']:.2f}"], style=estilo_texto),
                        html.Li([html.B("Precision: "), f"{metrics_mate['precision']:.2f}"], style=estilo_texto),
                        html.Li([html.B("Recall: "), f"{metrics_mate['recall']:.2f}"], style=estilo_texto),
                        html.Li([html.B("F1-score: "), f"{metrics_mate['f1']:.2f}"], style=estilo_texto),
                    ], style={"listStyle": "none", "paddingLeft": 0}),
                ], width=6),
                dbc.Col([
                    html.H4("Modelo Inglés", style=estilo_subtitulo),
                    html.Img(
                        src=f"data:image/png;base64,{encoded_roc_ingles}",
                        style={"width": "100%", "height": "350px", "marginBottom": "18px"}
                    ),
                    html.Ul([
                        html.Li([html.B("Accuracy: "), f"{metrics_ingles['accuracy']:.2f}"], style=estilo_texto),
                        html.Li([html.B("Precision: "), f"{metrics_ingles['precision']:.2f}"], style=estilo_texto),
                        html.Li([html.B("Recall: "), f"{metrics_ingles['recall']:.2f}"], style=estilo_texto),
                        html.Li([html.B("F1-score: "), f"{metrics_ingles['f1']:.2f}"], style=estilo_texto),
                    ], style={"listStyle": "none", "paddingLeft": 0}),
                ], width=6),
            ], style={"justifyContent": "center"}),
        ], style=estilo_card)
    else:
        return html.Div("Selecciona una pestaña válida.", style=estilo_card)

# CALLBACK para matemáticas
@app.callback(
    Output("output-prediccion", "children"),
    Input("btn-prediccion", "n_clicks"),
    [State(f"input-{var}", "value") for var in variables_seleccionadas]
)
def run_prediction_mate(n_clicks, *values):
    if not n_clicks:
        return ""
    input_dict = dict(zip(variables_seleccionadas, values))
    try:
        X_vec = transform_inputs_to_vector_mate(input_dict)
        pred_prob = model_mate.predict(X_vec)
        pred_class = np.argmax(pred_prob, axis=1)[0]
        categoria = map_class_to_label(pred_class)
        return html.Div([
            html.P(f"Predicción nivel Matemáticas: {categoria}")
        ], style={"fontSize": "22px", "fontWeight": "bold"})
    except Exception as e:
        return html.Div(f"Error en la predicción: {e}", style={"color": "red"})

# CALLBACK para inglés
@app.callback(
    Output("output-prediccion-ingles", "children"),
    Input("btn-prediccion-ingles", "n_clicks"),
    [State(f"input-eng-{var}", "value") for var in variables_ingles]
)
def run_prediction_ingles(n_clicks, *values):
    if not n_clicks:
        return ""
    input_dict = dict(zip(variables_ingles, values))
    try:
        X_vec = transform_inputs_to_vector_english(input_dict)
        pred_prob = model_ingles.predict(X_vec)
        pred_class = np.argmax(pred_prob, axis=1)[0]
        categoria = map_class_to_label(pred_class)
        return html.Div([
            html.P(f"Predicción nivel Inglés: {categoria}")
        ], style={"fontSize": "22px", "fontWeight": "bold"})
    except Exception as e:
        return html.Div(f"Error en la predicción: {e}", style={"color": "red"})

if __name__ == "__main__":
    app.run(debug=True)