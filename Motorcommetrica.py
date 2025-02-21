import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import webbrowser
from datetime import datetime, timedelta
import locale

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

# Carregar os dados
file_path = r"Construtoras_classificado1.xlsx"
data = pd.read_excel(file_path)

# Converter a coluna de data usando a função personalizada
data["Data de coleta"] = data["Data de coleta"].apply(convert_date)

# Remover registros com datas inválidas
data = data[data["Data de coleta"].notna()]

# Filtrar dados do compartimento "motor"
data_motor = data[data["Tipo de Compartimento"] == "motor"]

# Garantir que as colunas essenciais não tenham valores nulos
data_motor = data_motor.dropna(subset=["Equipamento", "Data de coleta", "Modos de falha"])

# Ordenar os dados por equipamento e data
data_motor = data_motor.sort_values(by=["Equipamento", "Data de coleta"])

# Separar os modos de falha em múltiplos rótulos
data_motor["Modos de falha"] = data_motor["Modos de falha"].apply(
    lambda x: [mode.strip() for mode in str(x).split('.') if mode.strip()]
)

# Criar colunas binárias para cada modo de falha
unique_modes_split = sorted(set([mode for sublist in data_motor["Modos de falha"] for mode in sublist]))
for mode in unique_modes_split:
    data_motor[f"Mode_{mode}"] = data_motor["Modos de falha"].apply(
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

# Substituir valores inválidos por NaN e preencher com a média
X = data_motor[selected_features].replace('Não informado', np.nan).fillna(data_motor[selected_features].mean())

# Garantir que todas as colunas sejam numéricas
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Selecionar rótulos
y = data_motor[[f"Mode_{mode}" for mode in unique_modes_split]].fillna(0)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
base_model = RandomForestClassifier(random_state=42, n_estimators=100)
multi_model = MultiOutputClassifier(base_model)
multi_model.fit(X_train, y_train)

# Calcular métricas do modelo
y_pred = multi_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Inicializar o Dash
app = dash.Dash(__name__)

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

    # Métricas do Modelo
    html.Div([
        html.Div([
            html.H3("Métricas do Modelo", style={"color": "#2c3e50", "borderBottom": "2px solid #3498db"}),
            html.Div([
                html.Div(f"Acurácia: {accuracy:.2f}", style={"fontSize": "18px", "marginBottom": "10px"}),
                html.Div(f"Precisão: {precision:.2f}", style={"fontSize": "18px", "marginBottom": "10px"}),
                html.Div(f"Recall: {recall:.2f}", style={"fontSize": "18px", "marginBottom": "10px"}),
                html.Div(f"F1-Score: {f1:.2f}", style={"fontSize": "18px", "marginBottom": "10px"}),
            ], style={"padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "10px"})
        ], style={"marginBottom": "30px"})
    ]),

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
                             for cliente in data_motor["Cliente"].dropna().unique()],
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
                             for tipo in data_motor["Tipo De Equipamento"].dropna().unique()],
                    value="Todos",
                    placeholder="Selecione o Tipo"
                )
            ], style={"width": "32%", "display": "inline-block"}),
        ], style={"marginBottom": "20px"}),

        # Equipamento específico
        html.Div([
            html.Label("Equipamento:", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="equipamento-dropdown",
                options=[{"label": "Todos", "value": "Todos"}],
                value="Todos",
                placeholder="Selecione o Equipamento"
            )
        ], style={"marginBottom": "20px"}),

        # Intervalo de tempo
        html.Div([
            html.Label("Intervalo de Previsão:", style={"fontWeight": "bold"}),
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=data_motor["Data de coleta"].min(),
                max_date_allowed=data_motor["Data de coleta"].max() + timedelta(days=365),
                start_date=data_motor["Data de coleta"].max(),
                end_date=data_motor["Data de coleta"].max() + timedelta(days=30),
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

    # Alerta do Modo de Falha
    html.Div(id="alerta-modo-falha", style={"marginBottom": "20px"}),

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

# Callback para atualizar filiais
@app.callback(
    Output("filial-dropdown", "options"),
    Input("cliente-dropdown", "value")
)
def update_filiais(cliente_selecionado):
    options = [{"label": "Todos", "value": "Todos"}]
    if cliente_selecionado and cliente_selecionado != "Todos":
        filiais = data_motor[data_motor["Cliente"] == cliente_selecionado]["Filial / Regional / Obra solicitante"].dropna().unique()
    else:
        filiais = data_motor["Filial / Regional / Obra solicitante"].dropna().unique()
    
    # Garantir que todos os valores sejam strings
    filiais = [str(filial) for filial in filiais if not pd.isna(filial)]
    
    options.extend([{"label": filial, "value": filial} for filial in sorted(filiais)])
    return options
# Adicione este callback após os outros callbacks no seu código

@app.callback(
    Output("equipamento-dropdown", "options"),
    [Input("cliente-dropdown", "value"),
     Input("filial-dropdown", "value"),
     Input("tipo-equipamento-dropdown", "value")]
)
def update_equipamentos(cliente_selecionado, filial_selecionada, tipo_equipamento):
    # Começar com o dataset completo
    df_filtrado = data_motor.copy()
    
    # Aplicar filtros
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

# Também adicione um callback para resetar o valor do equipamento quando os filtros mudarem
@app.callback(
    Output("equipamento-dropdown", "value"),
    [Input("cliente-dropdown", "value"),
     Input("filial-dropdown", "value"),
     Input("tipo-equipamento-dropdown", "value")]
)
def reset_equipamento_value(cliente, filial, tipo):
    return "Todos"
# Callback para atualizar equipamentos
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
        equipamento_dados = data_motor[data_motor["Tipo de Compartimento"] == "motor"]
    else:
        equipamento_dados = data_motor[(data_motor["Equipamento"] == equipamento) & 
                                     (data_motor["Tipo de Compartimento"] == "motor")]

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
    Input("equipamento-dropdown", "value")
)
def update_historico(equipamento):
    if not equipamento:
        return html.Div("Por favor, selecione um equipamento.")

    if equipamento == "Todos":
        historico = data_motor
    else:
        historico = data_motor[data_motor["Equipamento"] == equipamento]

    if historico.empty:
        return html.Div("Nenhum histórico encontrado para este equipamento.")

    return dcc.Graph(
        figure=px.line(
            historico,
            x="Data de coleta",
            y=selected_features,
            title=f"Histórico do Compartimento: {'Todos os Equipamentos' if equipamento == 'Todos' else equipamento}"
        )
    )

# Callback para atualizar previsões
@app.callback(
    Output("modos-previstos-intervalo", "children"),
    [Input("equipamento-dropdown", "value"),
     Input("date-picker-range", "start_date"),
     Input("date-picker-range", "end_date")]
)
def update_previsoes_intervalo(equipamento, start_date, end_date):
    if not all([equipamento, start_date, end_date]):
        return html.Div("Por favor, selecione um equipamento e intervalo de tempo.")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if equipamento == "Todos":
        equipamento_dados = data_motor
    else:
        equipamento_dados = data_motor[data_motor["Equipamento"] == equipamento].copy()

    if equipamento_dados.empty:
        return html.Div("Nenhum dado disponível para este equipamento.")

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
    df_filtrado = data_motor.copy()

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
    webbrowser.open(url)
    app.run_server(debug=True, port=port)
