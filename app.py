# Pakete importieren
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math
import dash
from dash import Dash, dcc, html, Input, Output  # , State
import plotly.express as px
import dash_bootstrap_components as dbc

N = 100000  # Länge des Datensatzes

feat_1 = [0, 1]  # Wiederkehrendes Angebot
feat_2 = ["transparent", "grau", "weiß", "farbig"]  # Farbe
feat_3 = ["Level 1", "Level 2", "Level 3", "Level 4"]  # DIN SPEC 91446 Level
feat_4 = ["PET", "PP", "HDPE", "LDPE"]  # Sorte
feat_5 = [0, 1]  # Rezyklat-Quelle
feat_6 = ["Einmalig", "Wöchentlich", "Zweiwöchentlich", "Monatlich", "Quartalsweise", "Jährlich"]  # Angebotsfrequenz
feat_7 = ["DE", "FR", "PL", "NL"]
feat_8 = ["DE", "FR", "PL", "NL"]

id = list(range(N))
# Wählt die Werte der Features N mal zufällig aus
x = np.random.choice(feat_1, p=[0.7, 0.3], size=N)  # Wiederkehrend -> 70% Nein
y = np.random.choice(feat_2, size=N)  # Farbe gleichverteilt
u = np.random.choice(feat_3, size=N)  # DIN SPEC 91446 Level gleichverteilt
v = np.random.choice(feat_4, size=N)  # Kunststoffsorte gleichverteilt
w = np.random.choice(feat_5, p=[0.2, 0.8], size=N)  # Rezyklat Quelle: 80% Post-Industrial (=1)

versand = np.random.choice(feat_7, p=[0.5, 1 / 6, 1 / 6, 1 / 6], size=N)
ursprung = np.random.choice(feat_8, p=[0.5, 1 / 6, 1 / 6, 1 / 6], size=N)

z = []  # Angebotsfrequenz wird in Abhängigkeit von dem Feature Wiederkehrendes Angebot gefüllt.
for i in x:
    if i == 1:
        z.append(np.random.choice(feat_6, p=[0, 0.25, 0.2, 0.3, 0.15, 0.1]))
    else:
        z.append("Einmalig")

t = np.random.gamma(3, 10, size=N)  # Menge in Tonnen, 1-150 hauptsächlich zwischen 15 und 40 Tonnen


def preis(f_1, f_3, f_4, f_5, kont_variable):  # Preis in Abhängigkeit des Materials der Qualität und fehlender Angaben
    if f_4 == "PET":  # Preise nach Kunststoffsorte
        mu = 1700
        sigma = 170
    if f_4 == "PP":
        mu = 1300
        sigma = 130
    if f_4 == "HDPE":
        mu = 900
        sigma = 90
    if f_4 == "LDPE":
        mu = 1200
        sigma = 120
    # Passe Werte der Qualität an
    if f_3 == "Level 1":
        mu = 0.5 * mu
        sigma = np.sqrt(0.5) * sigma
    if f_3 == "Level 2":
        mu = 0.75 * mu
        sigma = np.sqrt(0.75) * sigma
    if f_3 == "Level 3":
        mu = mu
        sigma = sigma
    if f_3 == "Level 4":
        mu = 1.5 * mu
        sigma = np.sqrt(1.5) * sigma
    # Verringere Werte je nachdem, ob Angabe gemacht wurde
    if f_1 == 0:
        mu = 0.9 * mu
        sigma = np.sqrt(0.9) * sigma
    if f_1 == 1:
        mu = mu
        sigma = sigma
    # Verringere Werte je nachdem, ob Angabe gemacht wurde
    if f_5 == 0:  # Post-Consumer
        mu = 0.75 * mu
        sigma = np.sqrt(0.75) * sigma
    if f_5 == 1:  # Post-Industrial
        mu = mu
        sigma = sigma
    # Skaliere nach der kontinuierlichen Variable "Menge in Tonnen"
    mu = mu * (1 - 0.05 * ((20 - kont_variable) / 20))
    sigma = sigma * np.sqrt((1 - 0.05 * ((20 - kont_variable) / 20)))
    return np.random.normal(mu, sigma)


df = pd.DataFrame({"Wiederkehrendes Angebot": x, "Farbe": y, "DIN SPEC 91446 Level": u, "Sorte": v, "Rezyklat Quelle": w,
                   "Angebotsfrequenz": z, "Menge in Tonnen": t, "Versandstandort": versand,
                   "Materialursprung": ursprung})
df["Preis"] = [round(
    preis(df["Wiederkehrendes Angebot"][i], df["DIN SPEC 91446 Level"][i], df["Sorte"][i], df["Rezyklat Quelle"][i],
          df["Menge in Tonnen"][i]), 2) for i in range(N)]

# df.head(10)
#
# print(len(df["Preis"].loc[df["Wiederkehrendes Angebot"] == 0]),
#       len(df["Preis"].loc[df["Wiederkehrendes Angebot"] == 1]))

# Input
filter = {'Wiederkehrendes Angebot': 1, 'Farbe': 'grau', 'DIN SPEC 91446 Level': 'Level 2', 'Sorte': 'PET',
          'Rezyklat Quelle': 0, 'Angebotsfrequenz': 'Wöchentlich',
          'Menge in Tonnen': 7.35}  # Filter zum filtern der Daten
merkmale = ['Wiederkehrendes Angebot', 'Farbe', 'DIN SPEC 91446 Level', 'Sorte', 'Rezyklat Quelle', 'Angebotsfrequenz',
            'Menge in Tonnen']
feste_merkmale = ['Qualität', 'Sorte']


def generiere_statistik(filter, merkmale):
    mu = np.mean(filter_df(filter, merkmale)["Preis"])
    sigma = np.std(filter_df(filter, merkmale)["Preis"])
    return mu, sigma


def filter_df(filter, merkmale):
    fdf = df.copy()
    for m in merkmale:
        if fdf[m].dtype == 'float64':
            fdf = fdf.loc[fdf[m] <= int(filter[m]) + 5].loc[
                fdf[m] >= int(filter[m]) - 5]  # Kontinuierliche Variable -> Filter den Bereich +- 1
        else:
            fdf = fdf.loc[fdf[m] == filter[m]]
    return fdf


# Schreibe die Funktionen von oben etwas um
mindestdatenzahl = 10


def dash_visualisiere_binäre_ergebnisse(filter, merkmal, merkmale):
    """Visualisiert die Preisverteilungen eines binären Merkmals in Abhängigkeit seiner beiden Ausprägungen.
      Hierbei wird unterschieden, ob das Merkmal bereits den Wert 1 besitzt also vom User nicht mehr verbessert werden kann, oder nicht. Entsprechend wird nur die Preisverteilung oder der Preisunterschied ausgegeben."""
    
    b=merkmale.copy()

    if merkmal=="Wiederkehrendes Angebot" and "Angebotsfrequenz" in b:
      b.remove("Angebotsfrequenz")

    filter[merkmal] = 0
    merkmalsausprägungen = np.array([0, 1])
    x_input = [int(i) for i in filter_df(filter, b)["Preis"]]  # Auf ganze Zahlen gerundet für Visualisierung
    y_input = list(filter_df(filter, b)[merkmal])

    if x_input:
        m1 = np.mean(x_input)
        s1 = np.std(x_input)
    else:
        m1=0
        s1=0
    filter[merkmal] = 1
    x_input = x_input + [int(i) for i in filter_df(filter, b)["Preis"]]
    if len([int(i) for i in filter_df(filter, b)["Preis"]])>0:
        m2 = np.mean([int(i) for i in filter_df(filter, b)["Preis"]])
        s2 = np.std([int(i) for i in filter_df(filter, b)["Preis"]])
    else:
        m2=0
        s2=0
    y_input = y_input + list(filter_df(filter, b)[merkmal])
    filter[merkmal] = 0  # Wieder zurücksetzen

    if x_input and len(x_input) >= 2 * mindestdatenzahl:
        count = Counter(zip(x_input, y_input))
        xs = np.array([x for (x, y), c in count.items()])
        ys = np.array([y for (x, y), c in count.items()])
        cs = np.array([c for (x, y), c in count.items()])
        cmin = cs.min()
        cmax = cs.max()
        cmid = (cmin + cmax) / 2
        plotdaten = pd.DataFrame((xs, ys, cs)).transpose()
        plotdaten.columns = ["Preis in Euro", "y", "Anzahl"]
        fig = px.scatter(plotdaten, x="Preis in Euro", y="y", color="Anzahl", size="Anzahl",
                         hover_data={"Preis in Euro": True, "y": False, "Anzahl": True},
                         color_continuous_scale="viridis", template="ggplot2")
        fig.update_layout(yaxis=dict(title="", tickmode='array', tickvals=[0, 1], ticktext=["Nein", "Ja"]),
                          font=dict(size=18))
        fig.add_annotation(
            x=m1,
            y=0.2,
            xref="x",
            yref="y",
            text="Mittelwert = %.2f €\t,\t Standardabweichung= %.2f €" % (m1, s1),
            showarrow=False,
            font=dict(
                size=14,
                color="#ffffff"
            ),
            align="center",
            ax=20,
            ay=-30,
            bordercolor="#006D72",
            borderwidth=2,
            borderpad=4,
            bgcolor="#83B0B6",
            opacity=0.8
            )
        fig.add_annotation(
            x=m2,
            y=0.8,
            xref="x",
            yref="y",
            text="Mittelwert = %.2f €\t,\t Standardabweichung= %.2f €" % (m2, s2),
            showarrow=False,
            font=dict(
                size=14,
                color="#ffffff"
            ),
            align="center",
            ax=20,
            ay=-30,
            bordercolor="#006D72",
            borderwidth=2,
            borderpad=4,
            bgcolor="#83B0B6",
            opacity=0.8
            ) 
        return fig
    else:  # "Die Anzahl an Dateneinträgen ist zu gering, um eine statistische Aussage treffen zu können."
        return empty_plot()


def dash_visualisiere_kategoriale_ergebnisse(filter, merkmal, merkmale):
    """Funktion, die für ein kategoriales Merkmal die Abhängigkeit des Preises in den mit dem Filter gefilterten Daten
     von den Merkmalsausprägungen visualisiert."""
    X = []
    Y = []
    labels = []
    ticks = []
    i = 0
    a_values=set(df[merkmal])
    if merkmal == "DIN SPEC 91446 Level":
      a_values=['Level 1', 'Level 2', 'Level 3', 'Level 4']
    if merkmal == "Angebotsfrequenz":
      a_values=["Einmalig", "Wöchentlich", "Zweiwöchentlich", "Monatlich", "Quartalsweise", "Jährlich"]

    for a in a_values:
        b = merkmale.copy()
        b.remove(merkmal)
        data = filter_df(filter, b)  # Filter für alle Merkmale außer demjenigen, das hier untersucht wird
        X.append(list(data["Preis"].loc[data[merkmal] == a]))
        Y.append([i] * len(list(data["Preis"].loc[data[merkmal] == a])))
        labels.append(a)
        ticks.append(i)
        i = i + 1

    x_input = []
    y_input = []
    for k in range(len(X)):
        x_input = x_input + [int(j) for j in X[k]]
        y_input = y_input + Y[k]
    if len(x_input) >= len(X) * mindestdatenzahl:
        count = Counter(zip(x_input, y_input))
        xs = np.array([x for (x, y), c in count.items()])
        ys = np.array([y for (x, y), c in count.items()])
        cs = np.array([c for (x, y), c in count.items()])
        cmin = cs.min()
        cmax = cs.max()
        cmid = (cmin + cmax) / 2
        plotdaten = pd.DataFrame((xs, ys, cs)).transpose()
        plotdaten.columns = ["Preis in Euro", "y", "Anzahl"]
        fig = px.scatter(plotdaten, x="Preis in Euro", y="y", color="Anzahl", size="Anzahl",
                         hover_data={"Preis in Euro": True, "y": False, "Anzahl": True},
                         color_continuous_scale="viridis", template="ggplot2")
        fig.update_layout(yaxis=dict(title="", tickmode='array', tickvals=ticks, ticktext=labels),
                          font=dict(size=18))
        return fig
    else:  # "Die Anzahl an Dateneinträgen ist zu gering, um eine statistische Aussage treffen zu können."
        return empty_plot()


def empty_plot():
    fig = px.bar(template="ggplot2")
    fig.update_layout(yaxis_range=[0, 1])
    fig.update_layout(xaxis_range=[0, 1])
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="x",
        yref="y",
        text="Für diese Angaben liegen nicht genügend Daten vor,\n um eine statische Aussage treffen zu können.",
        showarrow=False,
        font=dict(
            size=14,
            color="#ffffff"
        ),
        align="center",
        ax=20,
        ay=-30,
        bordercolor="#006D72",
        borderwidth=2,
        borderpad=4,
        bgcolor="#83B0B6",
        opacity=0.8
    )
    return fig


def dash_visualisiere_kontinuierliche_ergebnisse(filter, merkmal, merkmale, binwidth):
    """Funktion, die für ein kontinuierliches Merkmal die Preisabhängigkeit prüft.
    Dazu werden Bins mit Länge binwidth erstellt und jeweils der mittlere Preis sowie die Standardabweichung bestimmt.
    Die Darstellung kann als Histogramm oder als Line Plot erfolgen."""
    b = merkmale.copy()
    b.remove(merkmal)
    data = filter_df(filter, b)  # Filter für alle Merkmale außer demjenigen, das hier untersucht wird

    mmin = math.floor(min(data[merkmal]))
    mmax = math.ceil(max(data[merkmal]))
    bw = binwidth  # Bin width → Breite der Intervalle für die Zielgröße
    X = np.arange(mmin, mmax + 0.01, bw)
    Y = []
    Y_err = []
    for i in range(len(X) - 1):
        Y.append(np.mean(data["Preis"].loc[data[merkmal] >= X[i]].loc[data[merkmal] <= X[i + 1]]))
        Y_err.append(np.std(data["Preis"].loc[data[merkmal] >= X[i]].loc[data[merkmal] <= X[i + 1]]))

    plotdaten = pd.DataFrame((X[:-1], Y, Y_err)).transpose()
    plotdaten.columns = [merkmal, "Preis in Euro", "Standardabweichung"]
    fig = px.bar(plotdaten, x=merkmal, y="Preis in Euro", error_y="Standardabweichung",
                 template="ggplot2", color_discrete_sequence=["#578D94"]*(len(X)-1))
    fig.update_layout(font=dict(size=18))

    return fig


def dash_visualisiere_merkmal(filter, m, merkmale):
    if m == "NA":
        return px.bar(template="ggplot2")
    elif len(set(df[m])) == 2:  # Wenn das Merkmal binär ist
        return dash_visualisiere_binäre_ergebnisse(filter, m, merkmale)  # Zeige den Preisunterschied
    elif 21 > len(set(df[m])) > 2:  # Wenn das Merkmal kategorial ist (nicht mehr als 20 Ausprägungen)
        return dash_visualisiere_kategoriale_ergebnisse(filter, m, merkmale)
    else:  # Für kontinuierliche Merkmale kann ein Histogramm oder ein Line Plot ausgegeben werden
        n_bins = 20
        return dash_visualisiere_kontinuierliche_ergebnisse(filter, m, merkmale,
                                                            (math.ceil(max(df[m])) - math.floor(min(df[m]))) / n_bins)


# Bilde das Dashboard
dash_app = Dash(external_stylesheets=[dbc.themes.FLATLY])
app = dash_app.server

dash_app.layout = html.Div([
    html.Br(),
    html.H1("Cyclops - ökonomische Bewertung", style={'text-align': 'center', 'color': '#006D72'}),
    html.Br(),
    html.Div([dcc.Markdown('''
#### Wie kann ich für meine Kunststoffabfälle oder Kunststoffrezyklate einen höheren Preis erzielen?    
Dieses Dashboard beantwortet diese Frage auf Grundlage der Daten von (hier simulierten) schon abgeschlossenen Transaktionen.   
Mit den Schaltflächen können Sie Daten zu Ihrem Angebot angeben, welche dann verwendet werden, um die vorhandenen Daten zu filtern. 
Das Dashboard zeigt Ihnen dann eine Übersicht über die Verteilung der Preise von ähnlichen Angeboten an. Es ist nicht notwendig, alle Felder auszufüllen.   
Je mehr Angaben getätigt werden, desto ähnlicher sind die Angebote, deren Preisverteilung angezeigt wird.    
Im nächsten Schritt können Sie sich anzeigen lassen, wie sich die Preisverteilung ändert, wenn Sie mehr Informationen zu Ihrem Angebot bereitstellen oder, wenn Sie eine Ihrer Angaben ändern.    
Damit können Sie dann sehen, welche Informationen Sie noch zu Ihrem Angebot hinzufügen können, um einen höheren Preis erzielen zu können.
Das Dashboard entstand im Rahmen des CYCLOPS Projektes, gefördert durch das Bundesministerium für Bildung und Forschung.'''),]),

    html.Br(),
    html.H4("Bitte die Produktangaben eintragen"),
    dcc.Dropdown(options=[{'label': 'Wiederkehrendes Angebot', 'value': 'NA'},
                          {'label': 'Nein', 'value': 0},
                          {'label': 'Ja', 'value': 1},
                          ], value='NA', id="Dropdown_v1"),
    dcc.Dropdown(options=[{'label': 'Angebotsfrequenz', 'value': 'NA'},
                          {'label': 'Einmalig', 'value': 'Einmalig'},
                          {'label': 'Wöchentlich', 'value': 'Wöchentlich'},
                          {'label': 'Zweiwöchentlich', 'value': 'Zweiwöchentlich'},
                          {'label': 'Quartalsweise', 'value': 'Quartalsweise'},
                          {'label': 'Jährlich', 'value': 'Jährlich'},
                          ], value='NA', id="Dropdown_v6"),
    dcc.Dropdown(options=[{'label': 'Farbe auswählen', 'value': 'NA'},
                          {'label': 'Grau', 'value': 'grau'},
                          {'label': 'Farbig', 'value': 'farbig'},
                          {'label': 'Transparent', 'value': 'transparent'},
                          {'label': 'Weiß', 'value': 'weiß'},
                          ], value='NA', id="Dropdown_v2"),
    dcc.Dropdown(options=[{'label': 'DIN SPEC 91446 Level auswählen *', 'value': 'NA'},
                          #{'label': html.Span(
                          #      [
                          #          html.Span('DIN SPEC 91446 Level auswählen *'),
                          #          html.Br(),
                          #          html.Span('''Die DIN SPEC 91446 ist eine Klassifizierung von Kunststoff-Rezyklaten durch Datenqualitätslevels
                          #          für die Verwendung und den (internetbasierten) Handel. Sie wurde von cirplus und DIN im August 2020 mit dem Ziel initiiert,
                          #          einen lange ersehnten Standard für die Industrie zu schaffen.''', style={'font-size': 12, 'padding-left': 0, 'color':'gray'}),
                          #          html.Br(),
                          #      ], style={'align-items': 'center', 'justify-content': 'center'}
                          #  ), 'value': 'NA'},
                          {'label': 'Level 1', 'value': 'Level 1'},
                          {'label': 'Level 2', 'value': 'Level 2'},
                          {'label': 'Level 3', 'value': 'Level 3'},
                          {'label': 'Level 4', 'value': 'Level 4'},
                          ], value='NA', id="Dropdown_v3", optionHeight=60),
    dcc.Dropdown(options=[{'label': 'Kunststoffsorte auswählen', 'value': 'NA'},
                          {'label': 'PET', 'value': 'PET'},
                          {'label': 'PP', 'value': 'PP'},
                          {'label': 'LDPE', 'value': 'LDPE'},
                          {'label': 'HDPE', 'value': 'HDPE'},
                          ], value='NA', id="Dropdown_v4"),
    dcc.Dropdown(options=[{'label': 'Rezyklat Quelle', 'value': 'NA'},
                          {'label': 'Post-Consumer', 'value': 0},
                          {'label': 'Post-Industrial', 'value': 1},
                          ], value='NA', id="Dropdown_v5"),
    dcc.Input(id="kontinuierliche_Variable", type="number", min=0, max=2000, placeholder="Menge in Tonnen eingegeben"),
    html.Br(),
    html.Div(dcc.Markdown('''
*  Die DIN SPEC 91446 ist eine Klassifizierung von Kunststoff-Rezyklaten durch Datenqualitätslevels für die Verwendung und den (internetbasierten) Handel. Sie wurde von cirplus und DIN im August 2020 mit dem Ziel initiiert, einen lange ersehnten Standard für die Industrie zu schaffen.'''),]),
    html.Br(),
    html.H4("Preisverteilung für Kunststoffe mit den gleichen Angaben", style={'text-align': 'center'}),
    dcc.Graph(id="Preisstatistik", figure={}),
    html.Br(),
    html.H4("Auswahl des Parameters, für den der Einfluss auf den Preis untersucht werden soll"),
    dcc.Dropdown(
        options=[{'label': 'Parameter auswählen', 'value': 'NA'},
                 {'label': 'DIN SPEC 91446 Level', 'value': 'DIN SPEC 91446 Level'},
                 {'label': 'Farbe', 'value': 'Farbe'},
                 {'label': 'Wiederkehrendes Angebot', 'value': 'Wiederkehrendes Angebot'},
                 {'label': 'Angebotsfrequenz', 'value': 'Angebotsfrequenz'},
                 {'label': 'Menge in Tonnen', 'value': 'Menge in Tonnen'}],
        value='NA', id="Parameter"),
    html.Br(),
    dcc.Graph(id="Preisunterschied", figure={}),
    html.Br(),
    html.Br(),
    html.Div([dcc.Markdown('''
#### Impressum:        
Kontakt: Phillip Bendix, Wuppertal Institut - [phillip.bendix@wupperinst.org](mailto:phillip.bendix@wupperinst.org)   
Umsetzung: Jonathan Kirchhoff, Maike Jansen, Phillip Bendix'''),])

    ])


@dash_app.callback(
    Output(component_id="Preisstatistik", component_property="figure"),
    Output(component_id="Preisunterschied", component_property="figure"),
    [Input("Dropdown_v1", "value"), Input("Dropdown_v2", "value"), Input("Dropdown_v3", "value"),
     Input("Dropdown_v4", "value"), Input("Dropdown_v5", "value"), Input("Dropdown_v6", "value"),
     Input("kontinuierliche_Variable", "value"), Input("Parameter", "value")],
)
def cb_render(v1, v2, v3, v4, v5, v6, v7, slct):
    merkmale = []
    filter = {}
    if v6 == "Einmalig":
        v1 = 0
    if v6 != "Einmalig" and v6 != "NA":
        v1 = 1
    if v1 != "NA":
        merkmale.append('Wiederkehrendes Angebot')
        filter['Wiederkehrendes Angebot'] = v1
    if v2 != "NA":
        merkmale.append('Farbe')
        filter['Farbe'] = v2
    if v3 != "NA":
        merkmale.append('DIN SPEC 91446 Level')
        filter['DIN SPEC 91446 Level'] = v3
    if v4 != "NA":
        merkmale.append('Sorte')
        filter['Sorte'] = v4
    if v5 != "NA":
        merkmale.append('Rezyklat Quelle')
        filter['Rezyklat Quelle'] = v5
    if v6 != "NA":
        merkmale.append('Angebotsfrequenz')
        filter['Angebotsfrequenz'] = v6
    if v7:
        merkmale.append('Menge in Tonnen')
        filter['Menge in Tonnen'] = v7

    # Erzeuge ersten Plot

    x_input = [int(i) for i in filter_df(filter, merkmale)["Preis"]]  # Auf ganze Zahlen gerundet für Visualisierung
    y_input = [1] * len(x_input)

    if x_input and len(x_input) >= mindestdatenzahl:
        count = Counter(zip(x_input, y_input))
        xs = np.array([x for (x, y), c in count.items()])
        ys = np.array([y for (x, y), c in count.items()])
        cs = np.array([c for (x, y), c in count.items()])
        cmin = cs.min()
        cmax = cs.max()
        cmid = (cmin + cmax) / 2
        plotdaten = pd.DataFrame((xs, ys, cs)).transpose()
        plotdaten.columns = ["Preis in Euro", "y", "Anzahl"]
        fig = px.scatter(plotdaten, x="Preis in Euro", y="y", color="Anzahl", size="Anzahl",
                         hover_data={"Preis in Euro": True, "y": False, "Anzahl": True},
                         color_continuous_scale="viridis", template="ggplot2")
        fig.update_layout(yaxis=dict(title="", tickmode='array', tickvals=[], ticktext=[]), font=dict(size=18))
        fig.update_layout(yaxis_range=[0.5, 1.5])
        fig.add_annotation(
            x=np.mean(x_input),
            y=0.8,
            xref="x",
            yref="y",
            text="Mittelwert = %.2f €\t,\t Standardabweichung= %.2f €" % (np.mean(x_input), np.std(x_input)),
            showarrow=False,
            font=dict(
                size=14,
                color="#ffffff"
            ),
            align="center",
            ax=20,
            ay=-30,
            bordercolor="#006D72",
            borderwidth=2,
            borderpad=4,
            bgcolor="#83B0B6",
            opacity=0.8
        )
    else:
        fig = empty_plot()

    # Erzeuge zweiten Plot
    ausgabe_parameter = "Ausgewählter Parameter: " + slct
    if slct not in merkmale:
        merkmale.append(slct)
    sec_fig = dash_visualisiere_merkmal(filter, slct, merkmale)

    return fig, sec_fig


if __name__ == "__main__":
    dash_app.run_server(debug=False)
