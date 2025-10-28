import requests
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib as mpl
import matplotlib.patheffects as pe

from matplotlib.ticker import FuncFormatter, MaxNLocator
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
session = requests.Session()
retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

def aplicar_tema_profissional():
    try:
        mpl.rcParams.update({
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#E5E7EB",
            "axes.linewidth": 1.0,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 12.5,
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#E5E7EB",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.grid": True,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "lines.linewidth": 2.2,
            "patch.edgecolor": "#ffffff",
            "text.color": "#111827",
        })
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=[
            "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
            "#EDC948", "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AB"
        ])
    except Exception:
        pass

aplicar_tema_profissional()

# ======================
# FUNÇÕES DE COLETA
# ======================

def coletar_dados_ibge():
    """
    Coleta dados do IBGE sobre população jovem (15 a 29 anos) e renda média.
    Fonte: API SIDRA/IBGE.
    """
    print("🔍 Coletando dados do IBGE...")
    try:
        tabela = 5437
        variavel = 5933
        classe_idade = 58
        cat_18_24 = 100052
        periodos = "201801-202502"
        url = f"https://servicodados.ibge.gov.br/api/v3/agregados/{tabela}/periodos/{periodos}/variaveis/{variavel}"
        params = {
            "localidades": "N1[all]",
            "classificacao": f"{classe_idade}[{cat_18_24}]"
        }
        r = session.get(url, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        registros = []
        for item in js:
            for resultado in item.get("resultados", []):
                for serie in resultado.get("series", []):
                    s = serie.get("serie", {})
                    for per, val in s.items():
                        if val is None:
                            continue
                        val_str = str(val)
                        val_num = pd.to_numeric(val_str.replace(".", "").replace(",", "."), errors="coerce")
                        if pd.isna(val_num):
                            continue
                        ano = int(str(per)[:4])
                        registros.append({"Ano": ano, "Renda Média Jovens (R$)": float(val_num)})
        if not registros:
            raise ValueError("Sem registros válidos do IBGE")
        df_q = pd.DataFrame(registros)
        df_ibge = df_q.groupby("Ano")["Renda Média Jovens (R$)"].mean().reset_index()
        return df_ibge
    except Exception:
        print("Aviso: IBGE indisponível no momento. Usando dados de fallback de renda.")
        return pd.DataFrame({
            "Ano": [2018, 2019, 2020, 2021, 2022],
            "Renda Média Jovens (R$)": [1600, 1650, 1500, 1580, 1700]
        })

def coletar_dados_bcb():
    """
    Coleta dados do Banco Central sobre endividamento das famílias.
    Fonte: SGS - Sistema Gerenciador de Séries Temporais.
    """
    print("💰 Coletando dados do Banco Central...")
    try:
        codigo_serie = 29037  # Endividamento das famílias (% da renda)
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados"
        response = session.get(url, params={"formato": "json"}, timeout=10)
        response.raise_for_status()
        data = pd.DataFrame(response.json())
        data["valor"] = data["valor"].astype(float)
        data["data"] = pd.to_datetime(data["data"], dayfirst=True)
        data.rename(columns={"valor": "Endividamento (%)"}, inplace=True)
        data["Ano"] = data["data"].dt.year
        df_bcb = data.groupby("Ano")["Endividamento (%)"].mean().reset_index()
        return df_bcb
    except Exception as e:
        print("Erro ao coletar dados do Banco Central:", e)
        return pd.DataFrame()

def coletar_dados_ipea(serie_codigo: str = None):
    """
    Coleta dados do IPEA sobre desemprego juvenil.
    Fonte: API IPEADATA.
    """
    print("📊 Coletando dados do IPEA...")
    try:
        if serie_codigo:
            url = f"http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{serie_codigo}')"
            r = session.get(url, params={"$select": "VALDATA,VALVALOR", "$orderby": "VALDATA", "$top": 300}, timeout=10)
            r.raise_for_status()
            js = r.json()
            if "value" in js and js["value"]:
                df = pd.DataFrame(js["value"])
                df["VALDATA"] = pd.to_datetime(df["VALDATA"], errors="coerce")
                df["VALVALOR"] = pd.to_numeric(df["VALVALOR"], errors="coerce")
                df["Ano"] = df["VALDATA"].dt.year
                df = df.groupby("Ano")["VALVALOR"].mean().reset_index()
                df.rename(columns={"VALVALOR": "Taxa Desemprego Jovem (%)"}, inplace=True)
                return df
        df_ipea = pd.DataFrame({
            "Ano": [2018, 2019, 2020, 2021, 2022],
            "Taxa Desemprego Jovem (%)": [21.4, 20.8, 27.3, 25.0, 22.1]
        })
        return df_ipea
    except Exception as e:
        print("Erro ao coletar dados do IPEA:", e)
        return pd.DataFrame()

# ======================
# PROCESSAMENTO
# ======================

def combinar_dados():
    df_ibge = coletar_dados_ibge()
    df_bcb = coletar_dados_bcb()
    df_ipea = coletar_dados_ipea()

    frames = [df_ibge, df_bcb, df_ipea]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    df_final = frames[0]
    for f in frames[1:]:
        df_final = pd.merge(df_final, f, on="Ano", how="outer")
    df_final = df_final.sort_values("Ano").reset_index(drop=True)

    anos_ref = set()
    if not df_ibge.empty and "Ano" in df_ibge:
        anos_ref |= set(df_ibge["Ano"].dropna().astype(int))
    if not df_ipea.empty and "Ano" in df_ipea:
        anos_ref |= set(df_ipea["Ano"].dropna().astype(int))
    if anos_ref:
        df_final = df_final[df_final["Ano"].astype(int).isin(sorted(anos_ref))]
    df_final = df_final.drop_duplicates(subset=["Ano"]).reset_index(drop=True)
    return df_final

# ======================
# ANÁLISE / INSIGHTS
# ======================

def gerar_insights(df):
    if df.empty:
        print("❌ Dados insuficientes para gerar insights")
        return
    print("\n📈 Gerando insights automáticos...\n")
    insights = []

    if "Ano" in df and not df["Ano"].isna().all():
        insights.append(f"Período analisado: {int(df['Ano'].min())}–{int(df['Ano'].max())}")
    if "Renda Média Jovens (R$)" in df:
        media_renda = df["Renda Média Jovens (R$)"].mean()
        insights.append(f"💵 Renda média dos jovens: R$ {media_renda:.2f}")
    if "Endividamento (%)" in df:
        media_endividamento = df["Endividamento (%)"].mean()
        insights.append(f"📉 Média do endividamento das famílias: {media_endividamento:.2f}%")
    if "Taxa Desemprego Jovem (%)" in df:
        media_desemprego = df["Taxa Desemprego Jovem (%)"].mean()
        insights.append(f"🧑‍💼 Média do desemprego jovem: {media_desemprego:.2f}%")

    # Correlação entre renda e endividamento
    if "Renda Média Jovens (R$)" in df and "Endividamento (%)" in df:
        correlacao = df["Renda Média Jovens (R$)"].corr(df["Endividamento (%)"])
        if pd.notna(correlacao):
            insights.append(f"🔗 Correlação renda x endividamento: {correlacao:.2f}")
    if "Taxa Desemprego Jovem (%)" in df and "Endividamento (%)" in df:
        c2 = df["Taxa Desemprego Jovem (%)"].corr(df["Endividamento (%)"])
        if pd.notna(c2):
            insights.append(f"🔗 Correlação desemprego x endividamento: {c2:.2f}")

    # Identificar ano mais crítico
    if "Endividamento (%)" in df:
        try:
            pior_ano = df.loc[df["Endividamento (%)"].idxmax(), "Ano"]
            insights.append(f"⚠️ Ano mais crítico em endividamento: {int(pior_ano)}")
            top = df.sort_values("Endividamento (%)", ascending=False)[["Ano", "Endividamento (%)"]].head(3)
            rank = ", ".join([f"{int(a)} ({v:.1f}%)" for a, v in zip(top["Ano"], top["Endividamento (%)"])])
            insights.append(f"Top 3 anos de endividamento: {rank}")
            if df.shape[0] >= 2:
                anos = df["Ano"].astype(float)
                serie = df["Endividamento (%)"].astype(float)
                try:
                    m, b = np.polyfit(anos, serie, 1)
                    insights.append(f"Tendência do endividamento (coeficiente anual): {m:.2f} p.p./ano")
                except Exception:
                    pass
        except Exception:
            pass

    if "Renda Média Jovens (R$)" in df and df.shape[0] >= 2:
        try:
            v0 = df.iloc[0]["Renda Média Jovens (R$)"]
            v1 = df.iloc[-1]["Renda Média Jovens (R$)"]
            anos_span = int(df.iloc[-1]["Ano"]) - int(df.iloc[0]["Ano"]) or 1
            var = ((v1 / v0) - 1) * 100 if v0 else float("nan")
            cagr = ((v1 / v0) ** (1 / anos_span) - 1) * 100 if v0 else float("nan")
            if pd.notna(var):
                insights.append(f"Variação da renda no período: {var:+.1f}%")
            if pd.notna(cagr):
                insights.append(f"CAGR da renda: {cagr:+.1f}% a.a.")
        except Exception:
            pass

    for i in insights:
        print(i)

# ======================
# VISUALIZAÇÃO
# ======================

def exibir_graficos(df):
    print("\n📊 Exibindo gráficos...\n")

    if df.empty or "Ano" not in df:
        return

    fig, ax1 = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)

    anos = df["Ano"].astype(int)

    cor_endiv = "#F28E2B"
    cor_desemp = "#E15759"
    cor_renda = "#4E79A7"

    # Eixo 1 (percentuais)
    if "Endividamento (%)" in df:
        ax1.plot(anos, df["Endividamento (%)"], marker="o", color=cor_endiv, label="Endividamento (%)")
    if "Taxa Desemprego Jovem (%)" in df:
        ax1.plot(anos, df["Taxa Desemprego Jovem (%)"], marker="^", color=cor_desemp, label="Desemprego Jovem (%)")
    ax1.set_xlabel("Ano")
    ax1.set_ylabel("%")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(axis="both", which="major", length=5, width=1)
    ax1.tick_params(axis="both", which="minor", length=3, width=0.8)
    ax1.margins(x=0.02)

    # Eixo 2 (renda em R$)
    ax2 = ax1.twinx()
    if "Renda Média Jovens (R$)" in df:
        ax2.plot(anos, df["Renda Média Jovens (R$)"], marker="s", color=cor_renda, label="Renda Média Jovens (R$)")
        ax2.set_ylabel("Renda (R$)")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: ("R$ " + format(y, ",.0f").replace(",", "X").replace(".", ",").replace("X", "."))))

    # Legenda combinada
    linhas, labels = [], []
    for ax in (ax1, ax2):
        h, l = ax.get_legend_handles_labels()
        linhas += h
        labels += l
    if labels:
        ax1.legend(linhas, labels, loc="upper left", ncols=1)

    ax1.set_title("Evolução dos indicadores dos jovens", pad=10)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    plt.show()

def exibir_insights_graficos(df):
    print("\n📊 Exibindo gráficos de insights...\n")
    if df.empty:
        return

    # Layout em mosaico: 3 KPIs em cima, 3 gráficos embaixo
    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=True)
    mosaic = [["kpi_renda", "kpi_endiv", "kpi_desemp"],
              ["scatter_re", "line_endiv", "scatter_ue"]]
    ax_map = fig.subplot_mosaic(mosaic)

    cor_renda = "#4E79A7"
    cor_endiv = "#F28E2B"
    cor_desemp = "#E15759"

    def fmt_moeda(v):
        return "R$ " + format(float(v), ",.0f").replace(",", "X").replace(".", ",").replace("X", ".")

    # KPI cards
    def kpi(ax, titulo, valor_str, cor, subtitulo=None):
        ax.set_axis_off()
        ax.set_facecolor("#FCFCFD")
        bbox = dict(boxstyle="round,pad=0.8", facecolor=cor, alpha=0.08, edgecolor=cor)
        ax.text(0.02, 0.70, titulo, fontsize=12, color=cor, weight="bold")
        ax.text(0.02, 0.28, valor_str, fontsize=22, color="#111", weight="bold", bbox=bbox)
        if subtitulo:
            ax.text(0.02, 0.08, subtitulo, fontsize=9.5, color="#555")

    if "Renda Média Jovens (R$)" in df:
        renda_media = df["Renda Média Jovens (R$)"].mean()
        subt = None
        try:
            dfo = df.sort_values("Ano")
            v0 = dfo.iloc[0]["Renda Média Jovens (R$)"]
            v1 = dfo.iloc[-1]["Renda Média Jovens (R$)"]
            anos_span = int(dfo.iloc[-1]["Ano"]) - int(dfo.iloc[0]["Ano"]) or 1
            var = ((v1 / v0) - 1) * 100 if v0 else float("nan")
            cagr = ((v1 / v0) ** (1 / anos_span) - 1) * 100 if v0 else float("nan")
            if pd.notna(var) and pd.notna(cagr):
                subt = f"Δ período {var:+.1f}% | CAGR {cagr:+.1f}% a.a."
        except Exception:
            pass
        kpi(ax_map["kpi_renda"], "Renda média", fmt_moeda(renda_media), cor_renda, subt)
    else:
        kpi(ax_map["kpi_renda"], "Renda média", "–", cor_renda)

    if "Endividamento (%)" in df:
        endiv_media = df['Endividamento (%)'].mean()
        subt = None
        try:
            dfo = df.sort_values("Ano")
            anos = dfo["Ano"].astype(float)
            serie = dfo['Endividamento (%)'].astype(float)
            m, b = np.polyfit(anos, serie, 1)
            subt = f"Tendência {m:+.2f} p.p./ano"
            idx = serie.idxmax()
            ano_pior = int(df.loc[idx, "Ano"]) if not pd.isna(idx) else None
            if ano_pior:
                subt += f" | Pior ano {ano_pior}"
        except Exception:
            pass
        kpi(ax_map["kpi_endiv"], "Endividamento médio", f"{endiv_media:.1f}%", cor_endiv, subt)
    else:
        kpi(ax_map["kpi_endiv"], "Endividamento médio", "–", cor_endiv)

    if "Taxa Desemprego Jovem (%)" in df:
        desemp_media = df['Taxa Desemprego Jovem (%)'].mean()
        subt = None
        try:
            dfo = df.sort_values("Ano")
            anos = dfo["Ano"].astype(float)
            serie = dfo['Taxa Desemprego Jovem (%)'].astype(float)
            m, b = np.polyfit(anos, serie, 1)
            subt = f"Tendência {m:+.2f} p.p./ano"
        except Exception:
            pass
        kpi(ax_map["kpi_desemp"], "Desemprego médio", f"{desemp_media:.1f}%", cor_desemp, subt)
    else:
        kpi(ax_map["kpi_desemp"], "Desemprego médio", "–", cor_desemp)

    # Scatter Renda x Endividamento
    ax = ax_map["scatter_re"]
    if "Renda Média Jovens (R$)" in df and "Endividamento (%)" in df:
        dados = df[["Renda Média Jovens (R$)", "Endividamento (%)"]].dropna()
        if len(dados) >= 2:
            x = dados["Renda Média Jovens (R$)"]
            y = dados["Endividamento (%)"]
            ax.scatter(x, y, color=cor_renda, s=80, alpha=0.95, edgecolors="#ffffff", linewidth=0.7)
            r = float(x.corr(y)) if not pd.isna(x.corr(y)) else float("nan")
            try:
                z = np.polyfit(x, y, 1)
                xp = np.linspace(x.min(), x.max(), 100)
                ax.plot(xp, np.poly1d(z)(xp), color="#374151", linestyle="--", linewidth=1)
            except Exception:
                pass
            ax.set_title(f"Renda x Endividamento (r={r:.2f})", pad=6)
            ax.set_xlabel("Renda (R$)")
            ax.set_ylabel("Endividamento (%)")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_moeda(v)))
        else:
            ax.text(0.5, 0.5, "Dados insuficientes", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "Sem dados", ha="center", va="center")
        ax.set_axis_off()

    # Linha Endividamento com destaque do pior ano
    ax = ax_map["line_endiv"]
    if "Ano" in df and "Endividamento (%)" in df and not df["Endividamento (%)"].isna().all():
        anos = df["Ano"].astype(int)
        serie = df["Endividamento (%)"]
        ax.plot(anos, serie, marker="o", color=cor_endiv)
        try:
            idx = serie.idxmax()
            ano_pior = int(df.loc[idx, "Ano"])
            val_pior = float(df.loc[idx, "Endividamento (%)"])
            ax.scatter([ano_pior], [val_pior], color="#d62728", s=90, zorder=3)
            ax.annotate(
                f"Pior ano: {ano_pior}\n{val_pior:.1f}%",
                xy=(ano_pior, val_pior), xytext=(10, 10), textcoords="offset points",
                ha="left", color="#d62728",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff", alpha=0.8, edgecolor="#d62728")
            )
        except Exception:
            pass
        ax.set_title("Endividamento por ano", pad=6)
        ax.set_xlabel("Ano")
        ax.set_ylabel("%")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax.text(0.5, 0.5, "Sem dados", ha="center", va="center")
        ax.set_axis_off()

    # Scatter Desemprego x Endividamento
    ax = ax_map["scatter_ue"]
    if "Taxa Desemprego Jovem (%)" in df and "Endividamento (%)" in df:
        dados = df[["Taxa Desemprego Jovem (%)", "Endividamento (%)"]].dropna()
        if len(dados) >= 2:
            x = dados["Taxa Desemprego Jovem (%)"]
            y = dados["Endividamento (%)"]
            ax.scatter(x, y, color=cor_desemp, s=80, alpha=0.95, edgecolors="#ffffff", linewidth=0.7)
            r = float(x.corr(y)) if not pd.isna(x.corr(y)) else float("nan")
            try:
                z = np.polyfit(x, y, 1)
                xp = np.linspace(x.min(), x.max(), 100)
                ax.plot(xp, np.poly1d(z)(xp), color="#374151", linestyle="--", linewidth=1)
            except Exception:
                pass
            ax.set_title(f"Desemprego x Endividamento (r={r:.2f})", pad=6)
            ax.set_xlabel("Desemprego (%)")
            ax.set_ylabel("Endividamento (%)")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        else:
            ax.text(0.5, 0.5, "Dados insuficientes", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "Sem dados", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle("Insights visuais", fontsize=14, fontweight="bold")
    plt.show()

def coletar_dados_churn():
    rng = np.random.default_rng(42)
    n = 100
    tipo_plano = rng.choice(["Mensal", "Anual"], size=n)
    tempo_contrato = rng.integers(1, 73, size=n)
    reclamacoes = rng.integers(0, 6, size=n)
    preco = np.round(rng.uniform(50.0, 150.0, size=n), 2)
    churn = np.zeros(n, dtype=int)
    mask_m = tipo_plano == "Mensal"
    mask_a = tipo_plano == "Anual"
    cnt_m = int(mask_m.sum())
    cnt_a = int(mask_a.sum())
    q_m = (cnt_m * 40) // 100
    q_a = (cnt_a * 10) // 100
    if q_m > 0:
        churn[rng.choice(np.where(mask_m)[0], size=q_m, replace=False)] = 1
    if q_a > 0:
        churn[rng.choice(np.where(mask_a)[0], size=q_a, replace=False)] = 1
    df = pd.DataFrame({
        "Tempo de Contrato (meses)": tempo_contrato,
        "Tipo de Plano": tipo_plano,
        "Reclamações Registradas": reclamacoes,
        "Preço do Plano (R$)": preco,
        "Churn": churn
    })
    return df

def analisar_churn(df_churn):
    if df_churn is None or df_churn.empty:
        print("❌ Dados de churn indisponíveis")
        return
    taxa_media = float(df_churn["Churn"].mean() * 100.0)
    print(f"\nChurn médio geral: {taxa_media:.1f}%")
    taxas = df_churn.groupby("Tipo de Plano")["Churn"].mean().mul(100.0)
    tx_mensal = float(taxas.get("Mensal", float("nan")))
    tx_anual = float(taxas.get("Anual", float("nan")))
    print(f"Comparativo por tipo de plano - Mensal: {tx_mensal:.1f}% | Anual: {tx_anual:.1f}%")
    corr = df_churn["Reclamações Registradas"].corr(df_churn["Churn"])
    if pd.isna(corr):
        corr = float("nan")
    print(f"Correlação Reclamações x Churn: {corr:.2f}")

def exibir_grafico_churn(df_churn):
    if df_churn is None or df_churn.empty:
        return
    taxas = df_churn.groupby("Tipo de Plano")["Churn"].mean().mul(100.0)
    taxas = taxas.reindex(["Mensal", "Anual"])

    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    cores = ["#E15759", "#4E79A7"]
    bars = ax.bar(taxas.index, taxas.values, color=cores[:len(taxas)], edgecolor="#ffffff", linewidth=1.2, width=0.55)
    ax.set_ylabel("Churn (%)")
    ax.set_title("Churn por Tipo de Plano", pad=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))

    ymax = np.nanmax(taxas.values) if len(taxas.values) else 0
    if np.isfinite(ymax) and ymax > 0:
        ax.set_ylim(0, ymax * 1.15)

    try:
        labels = [f"{v:.1f}%" if not np.isnan(v) else "" for v in taxas.values]
        ax.bar_label(bars, labels=labels, padding=6, fontsize=10)
    except Exception:
        for i, v in enumerate(taxas.values):
            if not np.isnan(v):
                ax.text(i, v + max(1, v*0.03), f"{v:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.show()

def segmentar_perfis_jovens(df_final):
    cols = ["Renda Média Jovens (R$)", "Endividamento (%)", "Taxa Desemprego Jovem (%)"]
    if df_final is None or df_final.empty or not all(c in df_final.columns for c in cols):
        return df_final.copy() if df_final is not None else pd.DataFrame()
    dados = df_final[cols].dropna()
    df_out = df_final.copy()
    if dados.empty:
        df_out["Cluster"] = pd.NA
        return df_out
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(dados)
    df_out["Cluster"] = pd.NA
    df_out.loc[dados.index, "Cluster"] = labels
    return df_out

def exibir_clusterizacao(df_clusterizado):
    cols = ["Endividamento (%)", "Taxa Desemprego Jovem (%)", "Cluster"]
    if df_clusterizado is None or df_clusterizado.empty or not all(c in df_clusterizado.columns for c in cols):
        return
    dfp = df_clusterizado.dropna(subset=["Cluster", "Endividamento (%)", "Taxa Desemprego Jovem (%)"])
    if dfp.empty:
        return

    x = dfp["Taxa Desemprego Jovem (%)"].astype(float)
    y = dfp["Endividamento (%)"].astype(float)
    cl = dfp["Cluster"].astype(int)

    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    cmap = mpl.cm.get_cmap("Set2", int(dfp["Cluster"].nunique()))
    scatter = ax.scatter(x, y, c=cl, cmap=cmap, s=90, alpha=0.95, edgecolors="#ffffff", linewidth=0.7)

    # Centroides (medias por cluster) nos eixos escolhidos
    centers = dfp.groupby("Cluster").agg({
        "Taxa Desemprego Jovem (%)": "mean",
        "Endividamento (%)": "mean"
    }).reset_index()
    ax.scatter(
        centers["Taxa Desemprego Jovem (%)"], centers["Endividamento (%)"],
        marker="X", s=160, c=centers["Cluster"].astype(int), cmap=cmap,
        edgecolors="#333333", linewidth=1.0, zorder=3
    )
    for _, row in centers.iterrows():
        ax.annotate(
            f"C{int(row['Cluster'])}",
            xy=(row["Taxa Desemprego Jovem (%)"], row["Endividamento (%)"]),
            xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, weight="bold",
            color="#111", path_effects=[pe.withStroke(linewidth=3, foreground="#ffffff")]
        )

    ax.set_xlabel("Taxa Desemprego Jovem (%)")
    ax.set_ylabel("Endividamento (%)")
    ax.set_title("Clusterização de Tendências (3 Perfis)", pad=6)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # Legenda manual (cores dos clusters + centroides)
    handles, labels = [], []
    for i in sorted(dfp["Cluster"].unique()):
        color = cmap(int(i))
        handles.append(plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markeredgecolor="#ffffff", markersize=8))
        labels.append(f"Cluster {int(i)}")
    handles.append(plt.Line2D([0], [0], marker='X', color='none', markerfacecolor="#666666", markeredgecolor="#333333", markersize=9))
    labels.append("Centróides")
    ax.legend(handles, labels, title="Perfis", frameon=False)

    plt.show()


def treinar_avaliar_modelo_churn(df_churn):
    cols = [
        "Tempo de Contrato (meses)",
        "Tipo de Plano",
        "Reclamações Registradas",
        "Preço do Plano (R$)"
    ]
    if df_churn is None or df_churn.empty or not all(c in df_churn.columns for c in cols + ["Churn"]):
        print("❌ Dados de churn indisponíveis para treino")
        return None
    X = df_churn[cols].copy()
    y = df_churn["Churn"].astype(int)
    num_cols = ["Tempo de Contrato (meses)", "Reclamações Registradas", "Preço do Plano (R$)"]
    cat_cols = ["Tipo de Plano"]
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first"), cat_cols)
    ])
    model = Pipeline(steps=[
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    strat = y if y.nunique() > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=strat)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print("\n📌 Avaliação do modelo de Churn")
    acc = accuracy_score(y_te, y_pred)
    print(f"Acurácia: {acc:.3f}")
    y_proba = None
    try:
        y_proba = model.predict_proba(X_te)[:, 1]
    except Exception:
        pass
    if y_proba is not None and len(np.unique(y_te)) > 1:
        try:
            auc = roc_auc_score(y_te, y_proba)
            print(f"ROC AUC: {auc:.3f}")
        except Exception:
            pass
    try:
        print("Matriz de confusão:\n", confusion_matrix(y_te, y_pred))
        print("\nRelatório de classificação:\n", classification_report(y_te, y_pred, digits=3))
    except Exception:
        pass
    try:
        feats = model.named_steps["prep"].get_feature_names_out()
        coefs = model.named_steps["clf"].coef_[0]
        idx = np.argsort(np.abs(coefs))[::-1][:8]
        print("\nCoeficientes (impacto no risco de churn):")
        for i in idx:
            print(f"- {feats[i]}: {coefs[i]:+.3f}")
    except Exception:
        pass
    return model


def prescrever_acoes_churn(model, df_churn, top_k: int = 15):
    cols = [
        "Tempo de Contrato (meses)",
        "Tipo de Plano",
        "Reclamações Registradas",
        "Preço do Plano (R$)"
    ]
    if model is None or df_churn is None or df_churn.empty:
        print("❌ Modelo ou dados de churn indisponíveis")
        return
    X_all = df_churn[cols].copy()
    try:
        base_probs = model.predict_proba(X_all)[:, 1]
    except Exception:
        print("❌ Modelo não suporta predição de probabilidade")
        return
    base_exp = float(base_probs.sum())
    ordem = np.argsort(-base_probs)
    mask_mensal = df_churn["Tipo de Plano"].astype(str).eq("Mensal").to_numpy()
    idx_mensal = [int(i) for i in ordem if mask_mensal[int(i)]]
    k1 = int(min(top_k, len(idx_mensal)))
    if k1 > 0:
        df_c1 = X_all.copy()
        df_c1.loc[idx_mensal[:k1], "Tipo de Plano"] = "Anual"
        new1 = float(model.predict_proba(df_c1)[:, 1].sum())
        red1 = base_exp - new1
    else:
        red1 = float("nan")
    k2 = int(min(top_k, len(ordem)))
    df_c2 = X_all.copy()
    if k2 > 0:
        idx2 = ordem[:k2]
        vals = df_c2.loc[idx2, "Reclamações Registradas"].astype(int) - 1
        df_c2.loc[idx2, "Reclamações Registradas"] = np.clip(vals, 0, None)
        new2 = float(model.predict_proba(df_c2)[:, 1].sum())
        red2 = base_exp - new2
    else:
        red2 = float("nan")
    df_c3 = X_all.copy()
    if k2 > 0:
        idx3 = ordem[:k2]
        df_c3.loc[idx3, "Preço do Plano (R$)"] = df_c3.loc[idx3, "Preço do Plano (R$)"].astype(float) * 0.9
        new3 = float(model.predict_proba(df_c3)[:, 1].sum())
        red3 = base_exp - new3
    else:
        red3 = float("nan")
    print("\n📌 Prescrição de ações de retenção (what-if)")
    print(f"Churn esperado (antes): {base_exp:.2f}")
    if not np.isnan(red1):
        print(f"1) Converter top {k1} mensais para anual → redução esperada: {red1:.2f}")
    if not np.isnan(red2):
        print(f"2) Tratar 1 reclamação dos top {k2} de risco → redução esperada: {red2:.2f}")
    if not np.isnan(red3):
        print(f"3) Desconto de 10% no preço para top {k2} de risco → redução esperada: {red3:.2f}")
    reducoes = [(red1, "Converter para anual"), (red2, "Reduzir reclamações"), (red3, "Desconto 10%")]
    reducoes = [(r, nome) for r, nome in reducoes if not (isinstance(r, float) and np.isnan(r))]
    if reducoes:
        rmax, nome = max(reducoes, key=lambda t: t[0])
        print(f"\nRecomendação: priorizar '{nome}' (maior redução estimada de {rmax:.2f}).")


def prever_indicadores(df_final, anos_a_prever: int = 3):
    if df_final is None or df_final.empty or "Ano" not in df_final:
        print("❌ df_final indisponível")
        return pd.DataFrame()
    dfo = df_final.sort_values("Ano")
    cols = ["Endividamento (%)", "Taxa Desemprego Jovem (%)", "Renda Média Jovens (R$)"]
    cols = [c for c in cols if c in dfo.columns]
    if not cols:
        print("❌ Sem séries para prever")
        return pd.DataFrame()
    ultimo = int(dfo["Ano"].max())
    anos_fut = list(range(ultimo + 1, ultimo + 1 + anos_a_prever))
    prev = {"Ano": anos_fut}
    for c in cols:
        serie = dfo[["Ano", c]].dropna()
        if serie.shape[0] >= 2:
            x = serie["Ano"].astype(float).to_numpy()
            y = serie[c].astype(float).to_numpy()
            try:
                m, b = np.polyfit(x, y, 1)
                prev[c] = [float(m * a + b) for a in anos_fut]
            except Exception:
                prev[c] = [float("nan")] * len(anos_fut)
        else:
            prev[c] = [float("nan")] * len(anos_fut)
    df_prev = pd.DataFrame(prev)
    print("\n📌 Previsões (tendência linear)")
    print(df_prev)
    return df_prev


def prescrever_politicas_macro(df_final):
    if df_final is None or df_final.empty:
        print("❌ df_final indisponível")
        return
    dfo = df_final.sort_values("Ano")
    def slope(col):
        s = dfo[["Ano", col]].dropna()
        if s.shape[0] < 2:
            return float("nan")
        x = s["Ano"].astype(float)
        y = s[col].astype(float)
        try:
            m, b = np.polyfit(x, y, 1)
            return float(m)
        except Exception:
            return float("nan")
    s_end = slope("Endividamento (%)") if "Endividamento (%)" in dfo else float("nan")
    s_des = slope("Taxa Desemprego Jovem (%)") if "Taxa Desemprego Jovem (%)" in dfo else float("nan")
    s_ren = slope("Renda Média Jovens (R$)") if "Renda Média Jovens (R$)" in dfo else float("nan")
    print("\n📌 Recomendações prescritivas (baseadas em tendências)")
    if pd.notna(s_end) and s_end > 0.2:
        print("- Endividamento em alta: priorizar educação financeira e renegociação de dívidas.")
    if pd.notna(s_des) and s_des > 0.2:
        print("- Desemprego jovem em alta: ampliar qualificação e intermediação de emprego.")
    if pd.notna(s_ren) and s_ren < -50:
        print("- Queda de renda: considerar subsídios/auxílio focalizado para jovens.")
    if "Taxa Desemprego Jovem (%)" in dfo and "Endividamento (%)" in dfo:
        s = dfo[["Taxa Desemprego Jovem (%)", "Endividamento (%)"]].dropna()
        if s.shape[0] >= 2:
            c = float(s.corr().iloc[0, 1])
            if pd.notna(c) and c > 0.3:
                print("- Correlação desemprego x endividamento positiva: coordenar políticas de crédito com emprego.")


def recomendar_acoes_clusters(df_clusterizado):
    if df_clusterizado is None or df_clusterizado.empty or "Cluster" not in df_clusterizado:
        print("❌ Clusterização indisponível")
        return
    d = df_clusterizado.dropna(subset=["Cluster"]).copy()
    if d.empty:
        print("❌ Sem rótulos de cluster para recomendar")
        return
    centers = d.groupby("Cluster").agg({
        "Taxa Desemprego Jovem (%)": "mean",
        "Endividamento (%)": "mean"
    })
    med_des = float(d["Taxa Desemprego Jovem (%)"].median()) if "Taxa Desemprego Jovem (%)" in d else float("nan")
    med_end = float(d["Endividamento (%)"].median()) if "Endividamento (%)" in d else float("nan")
    print("\n📌 Recomendações por perfil (cluster)")
    for cl, row in centers.iterrows():
        des = float(row.get("Taxa Desemprego Jovem (%)", float("nan")))
        endv = float(row.get("Endividamento (%)", float("nan")))
        if pd.notna(des) and pd.notna(endv):
            if des >= med_des and endv >= med_end:
                rec = "foco em preço/acessibilidade e educação financeira"
            elif des >= med_des and endv < med_end:
                rec = "foco em empregabilidade e capacitação"
            elif des < med_des and endv >= med_end:
                rec = "foco em renegociação e bundles de valor"
            else:
                rec = "foco em upsell de planos premium com benefícios"
            print(f"- Cluster {int(cl)}: {rec} (centro: desemp={des:.1f}%, endiv={endv:.1f}%)")

def menu_interativo(df_final):
    modelo_churn = None
    df_churn_cache = None
    df_cluster_cache = None
    while True:
        print("\n===== Menu =====")
        print("1) Projeto 1: Análise de Churn")
        print("2) Projeto 2: Clusterização de Perfis Financeiros")
        print("3) Insights automáticos")
        print("4) Gráficos macro (evolução)")
        print("5) Mostrar dados combinados (df_final)")
        print("6) Modo Guia (passo a passo)")
        print("7) Churn: Treinar modelo preditivo")
        print("8) Churn: Prescrição de ações (what-if)")
        print("9) Macro: Prever próximos anos")
        print("10) Macro: Recomendações prescritivas")
        print("11) Cluster: Recomendações por perfil")
        print("0) Sair")
        try:
            op = input("Escolha uma opção: ").strip()
        except EOFError:
            op = "0"
        if op == "1":
            print("\n▶ Projeto 1: Análise Preditiva de Churn")
            df_churn_cache = coletar_dados_churn()
            analisar_churn(df_churn_cache)
            try:
                v = input("Deseja exibir o gráfico de churn? (s/n): ").strip().lower()
            except EOFError:
                v = "n"
            if v == "s":
                exibir_grafico_churn(df_churn_cache)
            try:
                input("Pressione Enter para voltar ao menu...")
            except EOFError:
                pass
        elif op == "2":
            print("\n▶ Projeto 2: Clusterização de Perfis Financeiros")
            df_cluster_cache = segmentar_perfis_jovens(df_final)
            try:
                print(df_cluster_cache[[
                    "Ano",
                    "Endividamento (%)",
                    "Taxa Desemprego Jovem (%)",
                    "Renda Média Jovens (R$)",
                    "Cluster"
                ]])
            except Exception:
                print(df_cluster_cache.head())
            try:
                v = input("Deseja exibir a visualização da clusterização? (s/n): ").strip().lower()
            except EOFError:
                v = "n"
            if v == "s":
                exibir_clusterizacao(df_cluster_cache)
            try:
                input("Pressione Enter para voltar ao menu...")
            except EOFError:
                pass
        elif op == "3":
            print("\n📈 Insights automáticos")
            gerar_insights(df_final)
            try:
                v = input("Deseja exibir os gráficos de insights? (s/n): ").strip().lower()
            except EOFError:
                v = "n"
            if v == "s":
                exibir_insights_graficos(df_final)
            try:
                input("Pressione Enter para voltar ao menu...")
            except EOFError:
                pass
        elif op == "4":
            exibir_graficos(df_final)
            try:
                input("Pressione Enter para voltar ao menu...")
            except EOFError:
                pass
        elif op == "5":
            print(df_final)
            try:
                input("Pressione Enter para voltar ao menu...")
            except EOFError:
                pass
        elif op == "6":
            modo_guia(df_final)
        elif op == "7":
            if df_churn_cache is None:
                df_churn_cache = coletar_dados_churn()
            modelo_churn = treinar_avaliar_modelo_churn(df_churn_cache)
            try:
                input("Pressione Enter para voltar ao menu...")
            except EOFError:
                pass
        elif op == "8":
            if df_churn_cache is None:
                df_churn_cache = coletar_dados_churn()
            if modelo_churn is None:
                modelo_churn = treinar_avaliar_modelo_churn(df_churn_cache)
            prescrever_acoes_churn(modelo_churn, df_churn_cache)
            try:
                input("Pressione Enter para voltar ao menu...")
            except EOFError:
                pass
        elif op == "9":
            prever_indicadores(df_final, anos_a_prever=3)
            try:
                input("Pressione Enter para voltar ao menu...")
            except EOFError:
                pass
        elif op == "10":
            prescrever_politicas_macro(df_final)
            try:
                input("Pressione Enter para voltar ao menu...")
            except EOFError:
                pass
        elif op == "11":
            if df_cluster_cache is None:
                df_cluster_cache = segmentar_perfis_jovens(df_final)
            recomendar_acoes_clusters(df_cluster_cache)
            try:
                input("Pressione Enter para voltar ao menu...")
            except EOFError:
                pass
        elif op == "0":
            break
        else:
            print("Opção inválida.")

def modo_guia(df_final):
    idx = 0
    df_churn = None
    df_clusterizado = None
    steps = [
        ("Churn: análise (texto)", lambda: (coletar_dados_churn(), "churn")),
        ("Churn: gráfico", "plot_churn"),
        ("Clusterização: segmentação (tabela)", "cluster_tab"),
        ("Clusterização: gráfico", "plot_cluster"),
        ("Insights automáticos (texto)", "insights_txt"),
        ("Gráficos de insights", "plot_insights"),
        ("Gráficos macro (evolução)", "plot_macro"),
    ]
    while True:
        titulo, acao = steps[idx]
        print(f"\n--- Passo {idx+1}/{len(steps)}: {titulo} ---")
        if acao == "plot_churn":
            if df_churn is None:
                df_churn = coletar_dados_churn()
                analisar_churn(df_churn)
            exibir_grafico_churn(df_churn)
        elif acao == "cluster_tab":
            if df_clusterizado is None:
                df_clusterizado = segmentar_perfis_jovens(df_final)
            try:
                print(df_clusterizado[[
                    "Ano",
                    "Endividamento (%)",
                    "Taxa Desemprego Jovem (%)",
                    "Renda Média Jovens (R$)",
                    "Cluster"
                ]])
            except Exception:
                print(df_clusterizado.head())
        elif acao == "plot_cluster":
            if df_clusterizado is None:
                df_clusterizado = segmentar_perfis_jovens(df_final)
            exibir_clusterizacao(df_clusterizado)
        elif acao == "insights_txt":
            gerar_insights(df_final)
        elif acao == "plot_insights":
            exibir_insights_graficos(df_final)
        elif acao == "plot_macro":
            exibir_graficos(df_final)
        else:
            df_churn, _ = steps[0][1]() if callable(steps[0][1]) else (df_churn, None)
            analisar_churn(df_churn)
        try:
            cmd = input("[n] próximo | [p] voltar | [r] repetir | [s] sair: ").strip().lower()
        except EOFError:
            cmd = "s"
        if cmd == "n":
            idx = (idx + 1) % len(steps)
        elif cmd == "p":
            idx = (idx - 1) % len(steps)
        elif cmd == "r":
            continue
        elif cmd == "s":
            break
        else:
            print("Opção inválida.")

# ======================
# EXECUÇÃO PRINCIPAL
# ======================

if __name__ == "__main__":
    df_final = combinar_dados()
    if not df_final.empty:
        print("\n✅ Dados coletados com sucesso!")
        menu_interativo(df_final)
    else:
        print("❌ Não foi possível gerar os dados.")
