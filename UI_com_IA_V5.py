import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.express as px
from datetime import datetime, timedelta
import locale
import os
import base64
import io

# Carregar os dados
file_path = r"Construtoras_classificado1_atualizado_V2.xlsx"
original_data = pd.read_excel(file_path)

data = original_data.copy()

# Configurar locale para português
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')
    except:
        print("Aviso: Não foi possível configurar o locale para português. Algumas datas podem não ser interpretadas corretamente.")

# Função para converter datas
def convert_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except:
        try:
            if isinstance(date_str, str):
                dia, mes, ano = date_str.split('-')
                meses = {
                    'jan': '01', 'fev': '02', 'mar': '03', 'abr': '04',
                    'mai': '05', 'jun': '06', 'jul': '07', 'ago': '08',
                    'set': '09', 'out': '10', 'nov': '11', 'dez': '12'
                }
                mes = meses.get(mes.lower(), '01')
                data_str = f"{ano}-{mes}-{dia}"
                return pd.to_datetime(data_str, errors='coerce')
        except:
            pass
    return pd.NaT

# Converter a coluna de data usando a função personalizada
data["Data de coleta"] = data["Data de coleta"].apply(convert_date)

# Remover registros com datas inválidas
data = data[data["Data de coleta"].notna()]

# Função para filtrar dados com base no tipo de compartimento
def filtrar_dados_por_compartimento(tipo_compartimento):
    filtered_data = data if tipo_compartimento == "Todos" else data[data["Tipo de Compartimento"] == tipo_compartimento]
    global multi_model
    multi_model = treinar_modelo(filtered_data)
    return filtered_data

# Inicialmente, filtrar por "Todos" como padrão
data_compartimento = data.copy()

# Garantir que as colunas essenciais não tenham valores nulos
data_compartimento = data_compartimento.dropna(subset=["Equipamento", "Data de coleta", "Modos de falha"])

# Ordenar os dados por equipamento e data
data_compartimento = data_compartimento.sort_values(by=["Equipamento", "Data de coleta"])

# Separar os modos de falha em múltiplos rótulos
data_compartimento["Modos de falha"] = data_compartimento["Modos de falha"].apply(
    lambda x: [mode.strip() for mode in str(x).split('.') if mode.strip()]
)

# Criar colunas binárias para cada modo de falha
unique_modes_split = sorted(set([mode for sublist in data_compartimento["Modos de falha"] for mode in sublist]))
for mode in unique_modes_split:
    data_compartimento[f"Mode_{mode}"] = data_compartimento["Modos de falha"].apply(
        lambda x: 1 if isinstance(x, list) and mode in x else 0
    )

# Selecionar características
selected_features = [
    "Viscosidade a 40ºC [cSt]",
    "Ponto de fulgor [ºC]",
    "TBN [mg KOH/g]",
    "Insolúveis [%]",
    "Al [ppm]",
    "Cr [ppm]",
    "Cu [ppm]",
    "Fe [ppm]",
    "Si [ppm]"
]

def treinar_modelo(data):
    X = data[selected_features]
    y = data[["Mode_" + mode for mode in unique_modes_split]]
    
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    clf.fit(X, y)

    print(f"Modelo treinado com sucesso para o compartimento {data['Tipo de Compartimento'].iloc[0]}")
    
    return clf

multi_model = treinar_modelo(data_compartimento)

def interpretar_metrica(valor, tipo):
    """Gera uma explicação com base no valor da métrica."""
    if tipo == "Acurácia":
        if valor > 0.90:
            return "O modelo tem um excelente desempenho geral, acertando a grande maioria das previsões."
        elif valor > 0.75:
            return "O modelo tem um bom desempenho, mas pode haver algumas previsões incorretas."
        else:
            return "O modelo pode não estar confiável, pois há muitas previsões erradas."

    elif tipo == "Precisão":
        if valor > 0.90:
            return "Quando o modelo indica uma falha, ele está quase sempre certo, o que significa poucos alarmes falsos."
        elif valor > 0.75:
            return "O modelo geralmente acerta quando alerta uma falha, mas pode gerar alguns alarmes falsos."
        else:
            return "O modelo pode estar gerando muitos alertas falsos, o que pode causar confusão."

    elif tipo == "Recall":
        if valor > 0.90:
            return "O modelo identifica quase todas as falhas reais, sendo muito eficiente em detectar problemas."
        elif valor > 0.75:
            return "O modelo consegue encontrar a maioria das falhas, mas algumas podem passar despercebidas."
        else:
            return "O modelo pode estar deixando muitas falhas reais passarem sem serem detectadas."

    elif tipo == "F1-Score":
        if valor > 0.90:
            return "O modelo tem um equilíbrio excelente entre detecção de falhas e precisão, minimizando erros."
        elif valor > 0.75:
            return "O modelo equilibra bem a detecção e a precisão, mas há espaço para melhorias."
        else:
            return "O modelo pode estar errando bastante ou deixando falhas passarem despercebidas."

    return "Sem interpretação disponível."

def calcular_metricas(model, X_test, y_test):
    """Calcula as métricas do modelo e retorna um dicionário com explicações dinâmicas."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return {
        "Acurácia": {
            "valor": f"{accuracy:.2%}",
            "explicação": interpretar_metrica(accuracy, "Acurácia")
        },
        "Precisão": {
            "valor": f"{precision:.2%}",
            "explicação": interpretar_metrica(precision, "Precisão")
        },
        "Recall": {
            "valor": f"{recall:.2%}",
            "explicação": interpretar_metrica(recall, "Recall")
        },
        "F1-Score": {
            "valor": f"{f1:.2%}",
            "explicação": interpretar_metrica(f1, "F1-Score")
        }
    }

# Inicializar o Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Lista de features para o popup
features = [
    'SAE MAX', 'SAE', 'NAS', 'Ag [ppm]', 'Al [ppm]', 'B [ppm]', 'Ba [ppm]', 'Be [ppm]', 'Ca [ppm]', 'Cl [ppm]',
    'Cr [ppm]', 'Cu [ppm]', 'Fe [ppm]', 'K [ppm]', 'Mg [ppm]', 'Mn [ppm]', 'Mo [ppm]', 'Na [ppm]', 'Ni [ppm]', 'P [ppm]',
    'Pb [ppm]', 'S [ppm]', 'Si [ppm]', 'Sn [ppm]', 'Ti [ppm]', 'V [ppm]', 'Zn [ppm]', 'Microscopia industrial/automotivo',
    'Conc. de Minério de Ferro', 'Conc. esfoliação', 'Conc. abrasão', 'Conc. fadiga', 'Conc. arrastamento',
    'Conc. desgaste severo', 'Conc. óxidos escuros', 'Conc. óxidos vermelhos', 'Conc. ligas de cobre',
    'Conc. ligas de alumínio', 'Conc. areia', 'Conc. fibras', 'Conc. borracha', 'Conc. borra', 'Conc. cont. amorfos',
    'Alterações Físico-Químicas', 'Avaliação do óleo', 'Desgaste', 'Avaliação do equipamento', 'Contaminações',
    'Avaliação de contaminações', 'Modos de falha', 'Recomendações', 'Condição da amostra',
    'Compartimento / Circuito / Medidor', 'Horas Totais do Equipamento', 'Horas de Uso do Óleo', ' ', ' .1', ' .2'
]

# Layout do popup com inputs (corrigindo a geração dos IDs)
popup_inputs = []
for feature in features:
    feature_id = f"input-{feature.lower().strip().replace(' ', '_').replace('[', '').replace(']', '').replace('/', '_').replace('.', '').replace(',', '').replace('%', '').replace('(', '').replace(')', '')}"
    popup_inputs.extend([
        html.Label(feature),html.Br(),
        dcc.Input(id=feature_id, type="text", placeholder=f"Digite {feature}..."),
        html.Br(), html.Br()
    ])


# Layout do Dashboard
app.layout = html.Div([
    # Cabeçalho
    html.H1(
        "Sistema de Análise Preditiva de Modos de Falha",
        style={
            "textAlign": "center",
            "color": "#2c3e50",
            "padding": "20px",
            "backgroundColor": "#ecf0f1",
            "borderRadius": "10px",
            "marginBottom": "20px"
        }
    ),

    # Painel de Filtros
    html.Div([
        # Filtros da primeira linha
        html.Div([
            # Cliente
            html.Div([
                html.Label("Cliente:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="cliente-dropdown",
                    options=[{"label": "Todos", "value": "Todos"}] + 
                            [{"label": str(cliente), "value": str(cliente)}
                             for cliente in data_compartimento["Cliente"].dropna().unique()],
                    value="Todos",
                    placeholder="Selecione o Cliente"
                )
            ], style={"width": "32%", "display": "inline-block", "marginRight": "2%"}),

            # Filial
            html.Div([
                html.Label("Filial/Regional:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="filial-dropdown",
                    options=[{"label": "Todos", "value": "Todos"}],
                    value="Todos",
                    placeholder="Selecione a Filial/Regional"
                )
            ], style={"width": "32%", "display": "inline-block", "marginRight": "2%"}),

            # Tipo de Equipamento
            html.Div([
                html.Label("Tipo de Equipamento:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="tipo-equipamento-dropdown",
                    options=[{"label": "Todos", "value": "Todos"}] + 
                            [{"label": str(tipo), "value": str(tipo)}
                             for tipo in data_compartimento["Tipo De Equipamento"].dropna().unique()],
                    value="Todos",
                    placeholder="Selecione o Tipo"
                )
            ], style={"width": "32%", "display": "inline-block"}),
        ], style={"marginBottom": "20px"}),

        # Filtros da segunda linha
        html.Div([
            # Equipamento específico
            html.Div([
                html.Label("Equipamento:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="equipamento-dropdown",
                    options=[{"label": "Todos", "value": "Todos"}],
                    value="Todos",
                    placeholder="Selecione o Equipamento"
                )
            ], style={"width": "32%", "display": "inline-block", "marginRight": "2%"}),

            # Tipo de Compartimento
            html.Div([
                html.Label("Tipo de Compartimento:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="tipo-compartimento-dropdown",
                    options=[{"label": "Todos", "value": "Todos"}],
                    value="Todos",
                    placeholder="Selecione o Tipo de Compartimento"
                )
            ], style={"width": "32%", "display": "inline-block"}),
        ], style={"marginBottom": "20px"}),

        # Intervalo de tempo
        html.Div([
            html.Label("Intervalo de Previsão:", style={"fontWeight": "bold"}), html.Br(),
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=data_compartimento["Data de coleta"].min(),
                max_date_allowed=data_compartimento["Data de coleta"].max() + timedelta(days=365),
                start_date=data_compartimento["Data de coleta"].max(),
                end_date=data_compartimento["Data de coleta"].max() + timedelta(days=30),
                display_format='DD-MMM-YYYY',
                first_day_of_week=1
            )
        ])
    ], style={
        "padding": "20px",
        "backgroundColor": "#f8f9fa",
        "borderRadius": "10px",
        "boxShadow": "0px 0px 10px rgba(0,0,0,0.1)",
        "marginBottom": "20px"
    }),

    html.P(id="compartimento_atualizado"),

    # Adicionar nova amostra
    html.Div([
        dbc.Button("Adicionar nova amostra", id="open-popup", color="primary", className="mt-3"),
        dbc.Modal([
            dbc.ModalHeader("Título do Popup"),
            dbc.ModalBody([
                    dcc.Upload(
                        id="upload-csv",
                        children=html.Button("Carregar CSV"),
                        multiple=False,
                        style={"marginBottom": "15px"}
                    ),
                    html.Div(id="csv-upload-status", style={"marginBottom": "15px", "color": "green"}),
                    dbc.Row([
                        dbc.Col(popup_inputs[:len(popup_inputs)//2], width=6),
                        dbc.Col(popup_inputs[len(popup_inputs)//2:], width=6)
                    ])
                ], style={"padding": "100px"}),
            dbc.ModalFooter([
                dbc.Button("Salvar", id="save-inputs", color="success"),
                dbc.Button("Fechar", id="close-popup", color="secondary")
            ])
        ], id="popup-modal", is_open=False),

        # Área para mostrar os valores capturados
        html.Div(id="output-area"),

        # Armazena os valores dos inputs
        dcc.Store(id="stored-data")
    ], style={"padding": "20px", "marginBottom": "20px"}),

    # Métricas do modelo
    dcc.Store(id="stored-metrics"),
    html.Div([
        html.H4("Métricas do Modelo"),

        html.Div(id="loading-message", children="As métricas estão sendo calculadas...", style={"display": "block"}),

        html.Div(id="output-metrics")
    ], style={
            "backgroundColor": "#f8f9fa",
            "padding": "15px",
            "borderRadius": "10px",
            "boxShadow": "0px 2px 5px rgba(0,0,0,0.1)"
        }),


    # Alerta do Modo de Falha
    html.Div(id="alerta-modo-falha", style={"marginBottom": "20px", "padding": "20px"}),

    # Conteúdo Principal
    html.Div([
        # Histórico
        html.Div([
            html.H2("Histórico do Compartimento",
                    style={"color": "#2c3e50", "borderBottom": "2px solid #3498db"}),
            html.Div(id="historico-tabela")
        ], style={"marginBottom": "30px"}),

        # Previsões
        html.Div([
            html.H2("Modos de Falha Previstos no Intervalo",
                    style={"color": "#2c3e50", "borderBottom": "2px solid #3498db"}),
            html.Div(id="modos-previstos-intervalo")
        ], style={"marginBottom": "30px"}),

        # Importância das Características
        html.Div([
            html.H2("Importância das Características",
                    style={"color": "#2c3e50", "borderBottom": "2px solid #3498db"}),
            html.Div(id="importance-plot")
        ])
    ], style={
        "padding": "20px",
        "backgroundColor": "#ffffff",
        "borderRadius": "10px",
        "boxShadow": "0px 0px 10px rgba(0,0,0,0.1)"
    })
])

# Filtra o dataframe com o tipo de compartimentro selecionado
@app.callback(
    Output("compartimento_atualizado", "children"),
    Input("tipo-compartimento-dropdown", "value")  # Quando o tipo de compartimento for alterado
)
def atualizar_dados_por_compartimento(tipo_compartimento):
    # global data_compartimento
    # data_compartimento = filtrar_dados_por_compartimento(tipo_compartimento)

    return html.P(f"Dados atualizados para o tipo de compartimento: {tipo_compartimento}")

@app.callback(
    Output("stored-metrics", "data"),
    Input("tipo-compartimento-dropdown", "value")  # Quando o tipo de compartimento for alterado
)
def atualizar_metricas(tipo_compartimento):
    # Filtrar os dados de acordo com o tipo de compartimento
    filtered_data = data if tipo_compartimento == "Todos" else data[data["Tipo de Compartimento"] == tipo_compartimento]

    # Separar features e rótulos
    X = filtered_data[selected_features]
    
    # Verificar se as colunas dos modos de falha já existem no dataframe
    if not all([f"Mode_{mode}" in filtered_data.columns for mode in unique_modes_split]):
        print("Criando colunas binárias para Modos de Falha...")

        # Criar colunas binárias
        filtered_data["Modos de falha"] = filtered_data["Modos de falha"].astype(str).apply(
            lambda x: [mode.strip() for mode in x.split(".") if mode.strip()]
        )

        for mode in unique_modes_split:
            filtered_data[f"Mode_{mode}"] = filtered_data["Modos de falha"].apply(
                lambda x: 1 if mode in x else 0
            )

    y = filtered_data[[f"Mode_{mode}" for mode in unique_modes_split]]

    # Dividir os dados para validação (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo novamente
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    clf.fit(X_train, y_train)

    # Calcular métricas
    metricas = calcular_metricas(clf, X_test, y_test)

    return metricas

@app.callback(
    [Output("output-metrics", "children"), Output("loading-message", "style")],
    Input("stored-metrics", "data")
)
def exibir_metricas(metricas):
    if metricas is None:
        return html.P("Não foi possível calcular as métricas"), {"display": "block"}

    return html.Ul([
        html.Li(f"Acurácia: {metricas['Acurácia']}"),
        html.Li(f"Precisão: {metricas['Precisão']}"),
        html.Li(f"Recall: {metricas['Recall']}"),
        html.Li(f"F1-Score: {metricas['F1-Score']}")
    ]), {"display": "none"}




# Callback para capturar valores dos inputs e armazená-los
@app.callback(
    [Output("stored-data", "data"), Output("output-area", "children")],
    [Input("save-inputs", "n_clicks")],
    [State(f"input-{feature.lower().strip().replace(' ', '_').replace('[', '').replace(']', '').replace('/', '_').replace('.', '').replace(',', '').replace('%', '').replace('(', '').replace(')', '')}", "value") for feature in features]
)
def save_inputs(n_clicks, *input_values):
    if n_clicks and n_clicks > 0:
        # Garantir que os dados estejam no formato correto
        input_values = [str(value) if value is not None else "" for value in input_values]
        valores_csv = ",".join(input_values)
        
        with open("dados_popup.csv", "a") as file:
            file.write(valores_csv + "\n")
        
        save_to_excel("dados_popup.csv", "dados_popup.xlsx") # Substituir o segundo parâmetro pelo caminho do arquivo Excel contendo a base de dados
        
        return valores_csv, f"Valores armazenados e salvos na base de dados com sucesso!"
    
    return dash.no_update, dash.no_update

# Função para ler CSV e salvar em um arquivo Excel
def save_to_excel(csv_path, excel_path):
    csv_to_save = pd.read_csv(csv_path, header=None)  # Lê o CSV sem rótulos
    
    if os.path.exists(excel_path):
        df_excel = pd.read_excel(excel_path, header=None)  # Lê o arquivo Excel existente
        df_final = pd.concat([df_excel, csv_to_save], ignore_index=True)  # Adiciona os novos dados
    else:
        df_final = csv_to_save  # Se o arquivo não existir, cria um novo
    
    with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
        df_final.to_excel(writer, index=False, header=False)

@app.callback(
    Output("csv-upload-status", "children"),
    [Input("upload-csv", "contents")],
    [State("upload-csv", "filename")]
)
def read_csv_and_save(contents, filename):
    if contents is None:
        return dash.no_update

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        # Lendo o CSV diretamente
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=None, engine='python', header=None)
    except Exception as e:
        return f"Erro ao ler CSV: {str(e)}"

    if df.empty:
        return "Arquivo CSV inválido."

    # Salvar diretamente no Excel
    csv_temp_path = "dados_popup_temp.csv"
    df.to_csv(csv_temp_path, index=False, header=False)

    save_to_excel(csv_temp_path, "dados_popup.xlsx")

    # Excluir o arquivo temporário após salvar
    try:
        os.remove(csv_temp_path)
    except Exception as e:
        return f"Erro ao excluir arquivo temporário: {str(e)}"

    return f"Arquivo {filename} salvo com sucesso no banco de dados!"

# Callbacks para abrir e fechar o popup de adicionar nova amostra
@app.callback(
    Output("popup-modal", "is_open"),
    [Input("open-popup", "n_clicks"), Input("close-popup", "n_clicks")],
    [State("popup-modal", "is_open")]
)
def toggle_popup(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

# Callback para atualizar tipo de compartimento
@app.callback(
    Output("tipo-compartimento-dropdown", "options"),
    [
        Input("cliente-dropdown", "value"),
        Input("filial-dropdown", "value"),
        Input("tipo-equipamento-dropdown", "value"),
        Input("equipamento-dropdown", "value")
    ]
)
def update_tipo_compartimento(cliente, filial, tipo_equipamento, equipamento):
    # Começar com o dataset completo
    df_filtrado = data_compartimento.copy()

    # Aplicar filtros
    if cliente != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Cliente"] == cliente]
    if filial != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Filial / Regional / Obra solicitante"] == filial]
    if tipo_equipamento != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Tipo De Equipamento"] == tipo_equipamento]
    if equipamento != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Equipamento"] == equipamento]

    # Obter tipos de compartimentos únicos disponíveis após filtragem
    tipos_compartimento = df_filtrado["Tipo de Compartimento"].dropna().unique()
    tipos_compartimento = [str(tipo) for tipo in tipos_compartimento if not pd.isna(tipo)]

    # Criar lista de opções
    options = [{"label": "Todos", "value": "Todos"}]
    options.extend([{"label": tipo, "value": tipo} for tipo in sorted(tipos_compartimento)])

    return options


# Callback para atualizar filiais
@app.callback(
    Output("filial-dropdown", "options"),
    Input("cliente-dropdown", "value")
)
def update_filiais(cliente_selecionado):
    options = [{"label": "Todos", "value": "Todos"}]
    if cliente_selecionado and cliente_selecionado != "Todos":
        filiais = data_compartimento[data_compartimento["Cliente"] == cliente_selecionado]["Filial / Regional / Obra solicitante"].dropna().unique()
    else:
        filiais = data_compartimento["Filial / Regional / Obra solicitante"].dropna().unique()
    
    # Garantir que todos os valores sejam strings
    filiais = [str(filial) for filial in filiais if not pd.isna(filial)]
    
    options.extend([{"label": filial, "value": filial} for filial in sorted(filiais)])
    return options

# Callback para atualizar equipamentos
@app.callback(
    Output("equipamento-dropdown", "options"),
    [Input("cliente-dropdown", "value"),
     Input("filial-dropdown", "value"),
     Input("tipo-equipamento-dropdown", "value"),
     Input("tipo-compartimento-dropdown", "value")]
)
def update_equipamentos(cliente_selecionado, filial_selecionada, tipo_equipamento, tipo_compartimento):
    # Começar com o dataset completo
    df_filtrado = data_compartimento.copy()
    
    # Aplicar filtros
    if tipo_compartimento and tipo_compartimento != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Tipo de Compartimento"] == tipo_compartimento]

    if cliente_selecionado and cliente_selecionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Cliente"] == cliente_selecionado]
        
    if filial_selecionada and filial_selecionada != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Filial / Regional / Obra solicitante"] == filial_selecionada]
        
    if tipo_equipamento and tipo_equipamento != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Tipo De Equipamento"] == tipo_equipamento]
    
    # Obter lista de equipamentos únicos
    equipamentos = df_filtrado["Equipamento"].dropna().unique()
    
    # Garantir que todos os valores sejam strings
    equipamentos = [str(equip) for equip in equipamentos if not pd.isna(equip)]
    
    # Criar lista de opções
    options = [{"label": "Todos", "value": "Todos"}]
    options.extend([{"label": equip, "value": equip} for equip in sorted(equipamentos)])
    
    return options

# Callback para resetar o valor do equipamento quando os filtros mudarem
@app.callback(
    Output("equipamento-dropdown", "value"),
    [Input("cliente-dropdown", "value"),
     Input("filial-dropdown", "value"),
     Input("tipo-equipamento-dropdown", "value"),
     Input("tipo-compartimento-dropdown", "value")]
)
def reset_equipamento_value(cliente, filial, tipo, tipo_compartimento):
    return "Todos"

# Callback para atualizar alerta de modo de falha
@app.callback(
    Output("alerta-modo-falha", "children"),
    [Input("equipamento-dropdown", "value"),
     Input("date-picker-range", "start_date"),
     Input("date-picker-range", "end_date")]
)
def update_alerta_modo_falha(equipamento, start_date, end_date):
    
    if not all([equipamento, start_date, end_date]):
        return html.Div()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if equipamento == "Todos":
        equipamento_dados = data_compartimento[data_compartimento["Tipo de Compartimento"] == "motor"]
    else:
        equipamento_dados = data_compartimento[(data_compartimento["Equipamento"] == equipamento) & 
                                     (data_compartimento["Tipo de Compartimento"] == "motor")]

    if equipamento_dados.empty:
        return html.Div("Não há dados suficientes para fazer uma previsão.")

    # Encontrar a última coleta dentro do intervalo selecionado
    ultima_coleta = equipamento_dados[
        (equipamento_dados["Data de coleta"] <= end_date)
    ]["Data de coleta"].max()
    
    if pd.isnull(ultima_coleta):
        return html.Div("Não há dados de coleta dentro do intervalo selecionado.")

    # Usar os dados da última coleta para a previsão
    dados_ultima_coleta = equipamento_dados[equipamento_dados["Data de coleta"] == ultima_coleta].iloc[-1:]

    if dados_ultima_coleta.empty:
        return html.Div("Nenhum dado disponível para a última coleta.")
    
    # Verificar se todas as colunas de `selected_features` existem
    colunas_faltantes = [col for col in selected_features if col not in dados_ultima_coleta.columns]
    if colunas_faltantes:
        return html.Div(f"As colunas necessárias estão faltando: {', '.join(colunas_faltantes)}")
    
    X_prev = dados_ultima_coleta[selected_features].replace('Não informado', np.nan)
    X_prev = X_prev.fillna(X_prev.mean())
    X_prev = X_prev.apply(pd.to_numeric, errors='coerce').fillna(0)

    probas = np.array([estimator.predict_proba(X_prev)[:, 1] for estimator in multi_model.estimators_]).T[0]
    modo_mais_provavel = max(zip(unique_modes_split, probas), key=lambda x: x[1])

    # Calcular horas até a próxima falha
    horas_ate_falha = None
    if modo_mais_provavel[0] != "Sem perda de função parcial ou total evidente":
        dias_ate_falha = min((end_date - ultima_coleta).days, 365)  # Limitando a 1 ano
        horas_ate_falha = int(dias_ate_falha * 24 * modo_mais_provavel[1])

    if modo_mais_provavel[0] == "Sem perda de função parcial ou total evidente":
        cor_fundo = "#00C851"
        texto_adicional = ""
    else:
        cor_fundo = "#ff4444" if modo_mais_provavel[1] > 0.7 else "#ffbb33"
        texto_adicional = f" Horas estimadas até a próxima falha: {horas_ate_falha}" if horas_ate_falha else ""

    return html.Div([
        html.H3("Modo de Falha Mais Provável:", style={"marginBottom": "10px", "color": "#2c3e50"}),
        html.Div([
            html.Strong(modo_mais_provavel[0]),
            html.Span(f" (Probabilidade: {modo_mais_provavel[1]*100:.1f}%){texto_adicional}")
        ], style={
            "padding": "15px",
            "backgroundColor": cor_fundo,
            "color": "white",
            "borderRadius": "5px",
            "textAlign": "center",
            "fontSize": "18px",
            "boxShadow": "0px 0px 10px rgba(0,0,0,0.1)"
        }),
        html.Div(f"Baseado na última coleta em: {ultima_coleta.strftime('%d-%b-%Y')}", 
                 style={"marginTop": "10px", "fontSize": "14px", "color": "#7f8c8d"})
    ])

# Callback para atualizar histórico
@app.callback(
    Output("historico-tabela", "children"),
    Input("equipamento-dropdown", "value"),
    Input("tipo-compartimento-dropdown", "value"),
    Input("date-picker-range", "start_date"),
    Input("date-picker-range", "end_date")
)
def update_historico(equipamento, tipo_compartimento, start_date, end_date):
    if not equipamento:
        return html.Div("Por favor, selecione um equipamento.")

    # Converter as datas de entrada
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if equipamento == "Todos":
        historico = data_compartimento
    else:
        historico = data_compartimento[data_compartimento["Equipamento"] == equipamento]

    if tipo_compartimento == "Todos":
        historico = historico
    else:
        historico = historico[historico["Tipo de Compartimento"] == tipo_compartimento].copy()

    # Aplicar filtro de intervalo de datas
    historico = historico[
        (historico["Data de coleta"] >= start_date) &
        (historico["Data de coleta"] <= end_date)
    ]

    if historico.empty:
        return html.Div("Nenhum histórico encontrado para este equipamento no intervalo de tempo selecionado.")

    # Verificar se as colunas existem e são numéricas
    try:
        historico[selected_features] = historico[selected_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    except KeyError:
        return html.Div("Algumas colunas necessárias não foram encontradas no histórico.")

    # Verificar se a coluna "Data de coleta" está presente
    if "Data de coleta" not in historico.columns:
        return html.Div("A coluna 'Data de coleta' não foi encontrada no conjunto de dados.")

    return dcc.Graph(
        figure=px.line(
            historico,
            x="Data de coleta",
            y=selected_features,
            title=f"Histórico do Compartimento: {'Todos os Equipamentos' if equipamento == 'Todos' else equipamento}",
            template="simple_white"
        )
    )

# Callback para atualizar previsões de intervalo
@app.callback(
    Output("modos-previstos-intervalo", "children"),
    [Input("equipamento-dropdown", "value"),
     Input("date-picker-range", "start_date"),
     Input("date-picker-range", "end_date"),
     Input("tipo-compartimento-dropdown", "value")]
)
def update_previsoes_intervalo(equipamento, start_date, end_date, tipo_compartimento):
    if not all([equipamento, start_date, end_date, tipo_compartimento]):
        return html.Div("Por favor, selecione um equipamento, intervalo de tempo e compartimento.")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if equipamento == "Todos":
        equipamento_dados = data_compartimento
    else:
        equipamento_dados = data_compartimento[data_compartimento["Equipamento"] == equipamento].copy()

    if equipamento_dados.empty:
        return html.Div("Nenhum dado disponível para este equipamento.")
    
    if tipo_compartimento == "Todos":
        equipamento_dados = equipamento_dados
    else:
        equipamento_dados = equipamento_dados[equipamento_dados["Tipo de Compartimento"] == tipo_compartimento].copy()

    if equipamento_dados.empty:
        return html.Div("Nenhum dado disponível para este compartimento.")

    datas_previsao = pd.date_range(start=start_date, end=end_date, freq='7D')

    previsoes = []
    for data in datas_previsao:
        dados_anteriores = equipamento_dados[equipamento_dados["Data de coleta"] <= data].iloc[-1:]

        if not dados_anteriores.empty:
            X_prev = dados_anteriores[selected_features].replace('Não informado', np.nan)
            X_prev = X_prev.fillna(X_prev.mean())
            X_prev = X_prev.apply(pd.to_numeric, errors='coerce').fillna(0)

            probas = np.array([estimator.predict_proba(X_prev)[:, 1] for estimator in multi_model.estimators_]).T[0]

            previsoes.append({
                'Data': data,
                'Modos de Falha': dict(zip(unique_modes_split, probas))
            })

    if not previsoes:
        return html.Div("Não foi possível gerar previsões para o intervalo selecionado.")

    df_previsoes = pd.DataFrame([
        {'Data': prev['Data'], 'Probabilidade': prob, 'Modo': modo}
        for prev in previsoes
        for modo, prob in prev['Modos de Falha'].items()
    ])

    fig = px.line(
        df_previsoes,
        x='Data',
        y='Probabilidade',
        color='Modo',
        title=f'Evolução das Probabilidades de Falha - {"Todos os Equipamentos" if equipamento == "Todos" else equipamento}'
    )

    return dcc.Graph(figure=fig)

# Callbacks para atualizar o gráfico de importância das características
@app.callback(
    Output("importance-plot", "children"),
    [Input("cliente-dropdown", "value"),
     Input("filial-dropdown", "value"),
     Input("tipo-equipamento-dropdown", "value"),
     Input("equipamento-dropdown", "value"),
     Input("date-picker-range", "start_date"),
     Input("date-picker-range", "end_date")]
)
def update_importance_plot(cliente, filial, tipo_equip, equipamento, start_date, end_date):
    if not all([start_date, end_date]):
        return html.Div("Selecione um intervalo de datas.")

    # Converter datas
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Começar com todos os dados
    df_filtrado = data_compartimento.copy()

    # Aplicar filtros
    if cliente != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Cliente"] == cliente]
    if filial != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Filial / Regional / Obra solicitante"] == filial]
    if tipo_equip != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Tipo De Equipamento"] == tipo_equip]
    if equipamento != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Equipamento"] == equipamento]

    # Filtrar por data
    df_filtrado = df_filtrado[
        (df_filtrado["Data de coleta"] >= start_date) &
        (df_filtrado["Data de coleta"] <= end_date)
    ]

    if df_filtrado.empty:
        return html.Div("Não há dados suficientes para o período e filtros selecionados.")

    # Preparar dados para treinamento
    X = df_filtrado[selected_features].replace('Não informado', np.nan).fillna(df_filtrado[selected_features].mean())
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Preparar rótulos
    y = df_filtrado[[f"Mode_{mode}" for mode in unique_modes_split]].fillna(0)

    if len(X) < 2:  # Verificar se há dados suficientes para treinar
        return html.Div("Dados insuficientes para calcular importância das características.")

    try:
        # Treinar modelo com dados filtrados
        base_model = RandomForestClassifier(random_state=42, n_estimators=100)
        multi_model_filtered = MultiOutputClassifier(base_model)
        multi_model_filtered.fit(X, y)

        # Calcular importâncias
        importances = np.mean([estimator.feature_importances_ for estimator in multi_model_filtered.estimators_], axis=0)
        importances_percent = (importances / importances.sum()) * 100
        
        importances_df = pd.DataFrame({
            "Feature": selected_features,
            "Importance": importances,
            "Importance (%)": importances_percent
        }).sort_values(by="Importance", ascending=False)

        # Criar gráfico
        fig = px.bar(
            importances_df,
            x="Importance (%)",
            y="Feature",
            orientation="h",
            title="Importância das Características (%)"
        )

        # Atualizar layout do gráfico
        fig.update_layout(
            xaxis_title="Importância (%)",
            yaxis_title="Característica",
            font=dict(size=12),
            height=400
        )

        return dcc.Graph(figure=fig)

    except Exception as e:
        return html.Div(f"Erro ao calcular importância das características: {str(e)}")

# Inicialização do servidor
if __name__ == "__main__":
    port = 8050
    url = f"http://127.0.0.1:{port}/"
    print(f"Dashboard disponível em {url}")
    app.run_server(debug=True, port=port)