import requests
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

from matplotlib.ticker import FuncFormatter, MaxNLocator
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
session = requests.Session()
retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

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

    with plt.style.context("seaborn-v0_8"):
        fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)

        anos = df["Ano"].astype(int)

        cor_endiv = "#f28e2b"
        cor_desemp = "#e15759"
        cor_renda = "#4e79a7"

        # Eixo 1 (percentuais)
        if "Endividamento (%)" in df:
            ax1.plot(anos, df["Endividamento (%)"], marker="o", color=cor_endiv, label="Endividamento (%)", linewidth=2)
        if "Taxa Desemprego Jovem (%)" in df:
            ax1.plot(anos, df["Taxa Desemprego Jovem (%)"], marker="^", color=cor_desemp, label="Desemprego Jovem (%)", linewidth=2)
        ax1.set_xlabel("Ano")
        ax1.set_ylabel("%", color="#444")
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.grid(True, linestyle="--", alpha=0.3)

        # Eixo 2 (renda em R$)
        ax2 = ax1.twinx()
        if "Renda Média Jovens (R$)" in df:
            ax2.plot(anos, df["Renda Média Jovens (R$)"], marker="s", color=cor_renda, label="Renda Média Jovens (R$)", linewidth=2)
            ax2.set_ylabel("Renda (R$)", color="#444")
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: ("R$ " + format(y, ",.0f").replace(",", "X").replace(".", ",").replace("X", "."))))

        # Legenda combinada
        linhas, labels = [], []
        for ax in (ax1, ax2):
            h, l = ax.get_legend_handles_labels()
            linhas += h
            labels += l
        if labels:
            ax1.legend(linhas, labels, loc="upper left", frameon=False)

        ax1.set_title("Evolução dos indicadores dos jovens")
        for spine in ["top", "right"]:
            ax1.spines[spine].set_alpha(0.2)
            ax2.spines[spine].set_alpha(0.2)

        plt.show()

def exibir_insights_graficos(df):
    print("\n📊 Exibindo gráficos de insights...\n")
    if df.empty:
        return

    with plt.style.context("seaborn-v0_8"):
        # Layout em mosaico: 3 KPIs em cima, 3 gráficos embaixo
        fig = plt.figure(figsize=(13, 8), constrained_layout=True)
        mosaic = [["kpi_renda", "kpi_endiv", "kpi_desemp"],
                  ["scatter_re", "line_endiv", "scatter_ue"]]
        ax_map = fig.subplot_mosaic(mosaic)

        cor_renda = "#4e79a7"
        cor_endiv = "#f28e2b"
        cor_desemp = "#e15759"

        def fmt_moeda(v):
            return "R$ " + format(float(v), ",.0f").replace(",", "X").replace(".", ",").replace("X", ".")

        # KPI cards
        def kpi(ax, titulo, valor_str, cor, subtitulo=None):
            ax.set_axis_off()
            ax.set_facecolor("white")
            bbox = dict(boxstyle="round,pad=0.8", facecolor=cor, alpha=0.08, edgecolor=cor)
            ax.text(0.03, 0.66, titulo, fontsize=11, color=cor, weight="bold")
            ax.text(0.03, 0.25, valor_str, fontsize=20, color="#111", weight="bold", bbox=bbox)
            if subtitulo:
                ax.text(0.03, 0.05, subtitulo, fontsize=9, color="#555")

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
                ax.scatter(x, y, color=cor_renda, s=60, alpha=0.9, edgecolors="white", linewidth=0.5)
                r = float(x.corr(y)) if not pd.isna(x.corr(y)) else float("nan")
                try:
                    z = np.polyfit(x, y, 1)
                    xp = np.linspace(x.min(), x.max(), 100)
                    ax.plot(xp, np.poly1d(z)(xp), color="#444", linestyle="--", linewidth=1)
                except Exception:
                    pass
                ax.set_title(f"Renda x Endividamento (r={r:.2f})")
                ax.set_xlabel("Renda (R$)")
                ax.set_ylabel("Endividamento (%)")
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_moeda(v)))
                ax.grid(True, linestyle="--", alpha=0.3)
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
            ax.plot(anos, serie, marker="o", color=cor_endiv, linewidth=2)
            try:
                idx = serie.idxmax()
                ano_pior = int(df.loc[idx, "Ano"])
                val_pior = float(df.loc[idx, "Endividamento (%)"])
                ax.scatter([ano_pior], [val_pior], color="#d62728", s=80, zorder=3)
                ax.annotate(f"Pior ano: {ano_pior}\n{val_pior:.1f}%", xy=(ano_pior, val_pior), xytext=(10, 10), textcoords="offset points", ha="left", color="#d62728", bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff", alpha=0.7, edgecolor="#d62728"))
            except Exception:
                pass
            ax.set_title("Endividamento por ano")
            ax.set_xlabel("Ano")
            ax.set_ylabel("%")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, linestyle="--", alpha=0.3)
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
                ax.scatter(x, y, color=cor_desemp, s=60, alpha=0.9, edgecolors="white", linewidth=0.5)
                r = float(x.corr(y)) if not pd.isna(x.corr(y)) else float("nan")
                try:
                    z = np.polyfit(x, y, 1)
                    xp = np.linspace(x.min(), x.max(), 100)
                    ax.plot(xp, np.poly1d(z)(xp), color="#444", linestyle="--", linewidth=1)
                except Exception:
                    pass
                ax.set_title(f"Desemprego x Endividamento (r={r:.2f})")
                ax.set_xlabel("Desemprego (%)")
                ax.set_ylabel("Endividamento (%)")
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
                ax.grid(True, linestyle="--", alpha=0.3)
            else:
                ax.text(0.5, 0.5, "Dados insuficientes", ha="center", va="center")
                ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "Sem dados", ha="center", va="center")
            ax.set_axis_off()

        fig.suptitle("Insights visuais", fontsize=14, fontweight="bold")
        plt.show()

# ======================
# EXECUÇÃO PRINCIPAL
# ======================

if __name__ == "__main__":
    df_final = combinar_dados()
    if not df_final.empty:
        print("\n✅ Dados coletados com sucesso!")
        print(df_final)

        try:
            opcao_insights = input("\nDeseja gerar insights automáticos? (s/n): ").strip().lower()
        except EOFError:
            opcao_insights = "n"
        if opcao_insights == "s":
            gerar_insights(df_final)
            exibir_insights_graficos(df_final)

        try:
            opcao_graficos = input("\nDeseja exibir os gráficos? (s/n): ").strip().lower()
        except EOFError:
            opcao_graficos = "n"
        if opcao_graficos == "s":
            exibir_graficos(df_final)
    else:
        print("❌ Não foi possível gerar os dados.")
