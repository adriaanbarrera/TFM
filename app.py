"""
app.py - TFM Prediccion Electrica Madrid
Ejecutar con: python -m streamlit run app.py
"""

import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests, time

st.set_page_config(
    page_title='Prediccion Electrica Madrid',
    page_icon='zap', layout='wide',
    initial_sidebar_state='collapsed',
)

AZUL   = '#2563EB'
AMBAR  = '#F59E0B'
VERDE  = '#10B981'
ROJO   = '#EF4444'
GRIS_F = '#F8FAFC'
GRIS_B = '#E2E8F0'
TEXTO  = '#0F172A'
TEXTO_S= '#64748B'

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {TEXTO}; }}
  .stApp {{ background: {GRIS_F}; }}
  .hero {{
    background: linear-gradient(135deg, {AZUL} 0%, #1D4ED8 100%);
    border-radius: 16px; padding: 28px 36px; margin-bottom: 20px; color: white;
  }}
  .hero h1 {{ font-size: 1.8rem; font-weight: 700; margin: 0 0 4px 0; letter-spacing: -0.5px; }}
  .hero p  {{ font-size: 0.95rem; margin: 0; opacity: 0.85; }}
  .tarjeta {{
    background: white; border-radius: 12px; padding: 18px 22px;
    border: 1px solid {GRIS_B}; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .etiqueta {{
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: {TEXTO_S}; margin-bottom: 6px;
  }}
  .valor {{
    font-size: 1.65rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; line-height: 1; margin-bottom: 4px;
  }}
  .subtexto {{ font-size: 0.78rem; color: {TEXTO_S}; }}
  .verde {{ color: {VERDE}; }}
  .rojo  {{ color: {ROJO};  }}
  .azul  {{ color: {AZUL};  }}
  .ambar {{ color: {AMBAR}; }}
  .seccion {{ font-size: 0.95rem; font-weight: 600; margin: 24px 0 10px 0; }}
  .consejo-box {{
    background: #F0FDF4; border-radius: 10px; padding: 16px 18px;
    border: 1px solid #BBF7D0; font-size: 0.85rem; line-height: 1.8;
  }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATOS
# ==============================================================================

def obtener_precio_actual():
    """Obtiene el precio de la hora actual desde OMIE."""
    now   = pd.Timestamp.now(tz='Europe/Madrid').floor('h')
    fecha = now.date()
    hora  = now.hour  # 0-23

    fname  = f'marginalpdbc_{fecha.strftime("%Y%m%d")}.1'
    params = {'parents[]': 'marginalpdbc', 'filename': fname}
    try:
        r = requests.get('https://www.omie.es/es/file-download',
                         params=params, timeout=15)
        if r.status_code != 200:
            return None
        lineas = r.text.strip().split('\n')
        for linea in lineas[1:]:
            if linea.startswith('*'):
                continue
            partes = [p.strip() for p in linea.split(';')]
            if len(partes) < 5:
                continue
            try:
                h = int(partes[3]) - 1  # OMIE usa horas 1-24, convertimos a 0-23
                if h == hora:
                    return float(partes[4].replace(',', '.'))
            except Exception:
                continue
    except Exception:
        pass
    return None


@st.cache_data(ttl=1800)
def obtener_demanda_historica(horas=48):
    """Obtiene demanda historica desde REE con fallback a e.sios."""
    ESIOS_KEY = os.getenv('ESIOS_API_KEY', '')
    now   = pd.Timestamp.now(tz='Europe/Madrid').floor('h')
    start = now - pd.Timedelta(hours=horas)

    df_demanda = pd.DataFrame()

    # Intento 1: apidatos
    try:
        headers = {'Accept': 'application/json'}
        if ESIOS_KEY:
            headers['x-api-key'] = ESIOS_KEY
        r = requests.get(
            'https://apidatos.ree.es/es/datos/demanda/demanda-tiempo-real',
            params={
                'start_date': start.strftime('%Y-%m-%dT%H:%M'),
                'end_date':   now.strftime('%Y-%m-%dT%H:%M'),
                'time_trunc': 'hour', 'geo_limit': 'ccaa', 'geo_ids': 8741,
            },
            headers=headers, timeout=20
        )
        if r.status_code == 200:
            for bloque in r.json().get('included', []):
                if bloque.get('type') == 'Demanda real':
                    tmp = pd.DataFrame(bloque['attributes']['values'])
                    tmp['datetime'] = pd.to_datetime(
                        tmp['datetime'], utc=True).dt.tz_convert('Europe/Madrid')
                    df_demanda = (tmp[['datetime', 'value']]
                                  .rename(columns={'value': 'demanda_mw'})
                                  .sort_values('datetime')
                                  .reset_index(drop=True))
                    break
    except Exception:
        pass

    # Intento 2: e.sios con token
    if df_demanda.empty and ESIOS_KEY:
        try:
            all_dfs = []
            cursor  = start
            while cursor <= now:
                window_end = min(cursor + pd.Timedelta(days=29), now)
                r = requests.get(
                    'https://api.esios.ree.es/indicators/1293',
                    headers={
                        'Accept':    'application/json; application/vnd.esios-api-v1+json',
                        'x-api-key': ESIOS_KEY,
                    },
                    params={
                        'start_date': cursor.strftime('%Y-%m-%dT%H:%M:%S'),
                        'end_date':   window_end.strftime('%Y-%m-%dT%H:%M:%S'),
                        'time_trunc': 'hour',
                        'geo_ids[]':  8741,
                    },
                    timeout=30
                )
                if r.status_code == 200:
                    valores = r.json()['indicator']['values']
                    if valores:
                        tmp = pd.DataFrame(valores)
                        tmp['datetime'] = pd.to_datetime(
                            tmp['datetime'], utc=True).dt.tz_convert('Europe/Madrid')
                        all_dfs.append(tmp[['datetime', 'value']].rename(
                            columns={'value': 'demanda_mw'}))
                cursor += pd.Timedelta(days=30)
                time.sleep(0.3)
            if all_dfs:
                df_demanda = (pd.concat(all_dfs)
                              .drop_duplicates('datetime')
                              .sort_values('datetime')
                              .reset_index(drop=True))
                df_demanda = df_demanda[df_demanda['datetime'] >= start]
        except Exception:
            pass

    return df_demanda


@st.cache_data(ttl=3600)
def obtener_predicciones():
    try:
        from predict import Predictor
        decision = {'modelos_produccion': {'precio': 'xgboost', 'demanda': 'xgboost'}}
        path = Path('data/models/decision_final.json')
        if path.exists():
            with open(path) as f:
                decision = json.load(f)
        mp = decision.get('modelos_produccion', {}).get('precio',  'xgboost')
        md = decision.get('modelos_produccion', {}).get('demanda', 'xgboost')
        return Predictor(modelo_precio=mp, modelo_demanda=md).predecir()
    except Exception as e:
        st.error(f'Error: {e}')
        return {}


# ==============================================================================
# GRAFICAS
# ==============================================================================

def layout_grafica():
    return dict(
        height=240, margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor='white', paper_bgcolor='white',
        hovermode='x unified',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=GRIS_B, zeroline=False),
        font=dict(family='Inter', size=11, color=TEXTO),
        showlegend=False,
    )

def grafica_prediccion_precio(pred):
    ts   = pd.to_datetime(pred['timestamps'])
    vals = np.array(pred['precio'])
    fig  = go.Figure()
    margen = vals * 0.10
    fig.add_trace(go.Scatter(
        x=list(ts) + list(ts[::-1]),
        y=list(vals+margen) + list((vals-margen)[::-1]),
        fill='toself', fillcolor='rgba(245,158,11,0.10)',
        line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=vals, mode='lines',
        line=dict(color=AMBAR, width=2.5),
        hovertemplate='%{x|%d/%m %H:%M}<br><b>%{y:.1f} EUR/MWh</b><extra></extra>'
    ))
    fig.update_layout(**layout_grafica(), yaxis_title='EUR/MWh')
    return fig

def grafica_historico_demanda(df):
    fig = go.Figure()
    if df.empty:
        fig.update_layout(**layout_grafica(), yaxis_title='MW')
        fig.add_annotation(text='Sin datos historicos disponibles',
                           xref='paper', yref='paper', x=0.5, y=0.5,
                           showarrow=False, font_color=TEXTO_S)
        return fig
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['demanda_mw'],
        mode='lines', line=dict(color=VERDE, width=2.5),
        fill='tozeroy', fillcolor='rgba(16,185,129,0.06)',
        hovertemplate='%{x|%d/%m %H:%M}<br><b>%{y:,.0f} MW</b><extra></extra>'
    ))
    fig.update_layout(**layout_grafica(), yaxis_title='MW')
    return fig

def grafica_prediccion_demanda(pred):
    ts   = pd.to_datetime(pred['timestamps'])
    vals = np.array(pred['demanda'])
    fig  = go.Figure()
    margen = vals * 0.05
    fig.add_trace(go.Scatter(
        x=list(ts) + list(ts[::-1]),
        y=list(vals+margen) + list((vals-margen)[::-1]),
        fill='toself', fillcolor='rgba(239,68,68,0.08)',
        line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=vals, mode='lines',
        line=dict(color=ROJO, width=2.5),
        hovertemplate='%{x|%d/%m %H:%M}<br><b>%{y:,.0f} MW</b><extra></extra>'
    ))
    fig.update_layout(**layout_grafica(), yaxis_title='MW')
    return fig

def grafica_perfil_precio(pred):
    ts   = pd.to_datetime(pred['timestamps'])[:24]
    vals = np.array(pred['precio'])[:24]
    horas = [t.strftime('%H:%M') for t in ts]
    p33 = np.percentile(vals, 33)
    p66 = np.percentile(vals, 66)
    colores = [
        'rgba(16,185,129,0.85)' if v <= p33 else
        'rgba(245,158,11,0.85)' if v <= p66 else
        'rgba(239,68,68,0.85)'
        for v in vals
    ]
    fig = go.Figure(go.Bar(
        x=horas, y=vals, marker_color=colores,
        hovertemplate='%{x}<br><b>%{y:.1f} EUR/MWh</b><extra></extra>',
    ))
    fig.add_hline(y=np.mean(vals), line_dash='dash', line_color=TEXTO_S,
                  line_width=1.2,
                  annotation_text=f'Media: {np.mean(vals):.1f} EUR/MWh',
                  annotation_font_size=10, annotation_font_color=TEXTO_S)
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis_title='EUR/MWh',
        xaxis=dict(showgrid=False, tickfont_size=9),
        yaxis=dict(showgrid=True, gridcolor=GRIS_B),
        font=dict(family='Inter', size=11, color=TEXTO),
        showlegend=False,
    )
    return fig


# ==============================================================================
# LAYOUT
# ==============================================================================

ahora = pd.Timestamp.now(tz='Europe/Madrid')
dias_es  = ['Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo']
meses_es = ['enero','febrero','marzo','abril','mayo','junio',
            'julio','agosto','septiembre','octubre','noviembre','diciembre']
fecha_str = (f'{dias_es[ahora.dayofweek]}, {ahora.day} de '
             f'{meses_es[ahora.month-1]} de {ahora.year} - {ahora.strftime("%H:%M")}')

st.markdown(f'<div class="hero"><h1>Prediccion Electrica - Madrid</h1><p>{fecha_str}</p></div>',
            unsafe_allow_html=True)

col_btn, _ = st.columns([1, 5])
with col_btn:
    if st.button('Actualizar', use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.spinner('Cargando datos...'):
    precio_actual  = obtener_precio_actual()
    df_demanda_hist = obtener_demanda_historica(horas=48)
    pred = obtener_predicciones()

if not pred:
    st.error('No se pudieron generar predicciones.')
    st.stop()

ts_pred   = pd.to_datetime(pred['timestamps'])
vals_p    = np.array(pred['precio'])
vals_d    = np.array(pred['demanda'])
idx_min_p = int(np.argmin(vals_p))
idx_max_p = int(np.argmax(vals_p))
idx_min_d = int(np.argmin(vals_d))
idx_max_d = int(np.argmax(vals_d))


# ── PRECIO ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="seccion">Precio electrico (EUR/MWh)</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="tarjeta">
      <div class="etiqueta">Precio minimo predicho</div>
      <div class="valor verde">{vals_p[idx_min_p]:.1f} <small>EUR/MWh</small></div>
      <div class="subtexto">{ts_pred[idx_min_p].strftime('%d/%m a las %H:%M')}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="tarjeta">
      <div class="etiqueta">Precio maximo predicho</div>
      <div class="valor rojo">{vals_p[idx_max_p]:.1f} <small>EUR/MWh</small></div>
      <div class="subtexto">{ts_pred[idx_max_p].strftime('%d/%m a las %H:%M')}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="tarjeta">
      <div class="etiqueta">Precio medio 48h</div>
      <div class="valor azul">{np.mean(vals_p):.1f} <small>EUR/MWh</small></div>
      <div class="subtexto">= {np.mean(vals_p)/1000:.4f} EUR/kWh</div>
    </div>""", unsafe_allow_html=True)
with c4:
    if precio_actual is not None:
        st.markdown(f"""<div class="tarjeta">
          <div class="etiqueta">Precio ahora - OMIE</div>
          <div class="valor ambar">{precio_actual:.1f} <small>EUR/MWh</small></div>
          <div class="subtexto">= {precio_actual/1000:.4f} EUR/kWh</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="tarjeta">
          <div class="etiqueta">Precio ahora - OMIE</div>
          <div class="valor" style="color:{TEXTO_S}">--</div>
          <div class="subtexto">No disponible</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

st.caption('Prediccion de precio - proximas 48 horas (XGBoost - banda = +/-10%)')
st.plotly_chart(grafica_prediccion_precio(pred),
                use_container_width=True, config={'displayModeBar': False})


# ── DEMANDA ────────────────────────────────────────────────────────────────────
st.markdown('<div class="seccion">Demanda electrica (MW)</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="tarjeta">
      <div class="etiqueta">Demanda minima predicha</div>
      <div class="valor verde">{vals_d[idx_min_d]:,.0f} <small>MW</small></div>
      <div class="subtexto">{ts_pred[idx_min_d].strftime('%d/%m a las %H:%M')}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="tarjeta">
      <div class="etiqueta">Demanda maxima predicha</div>
      <div class="valor rojo">{vals_d[idx_max_d]:,.0f} <small>MW</small></div>
      <div class="subtexto">{ts_pred[idx_max_d].strftime('%d/%m a las %H:%M')}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="tarjeta">
      <div class="etiqueta">Demanda media 48h</div>
      <div class="valor azul">{np.mean(vals_d):,.0f} <small>MW</small></div>
      <div class="subtexto">Prediccion XGBoost</div>
    </div>""", unsafe_allow_html=True)
with c4:
    if not df_demanda_hist.empty:
        d_act = df_demanda_hist['demanda_mw'].iloc[-1]
        st.markdown(f"""<div class="tarjeta">
          <div class="etiqueta">Demanda ahora - REE</div>
          <div class="valor ambar">{d_act:,.0f} <small>MW</small></div>
          <div class="subtexto">Tiempo real</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="tarjeta">
          <div class="etiqueta">Demanda ahora - REE</div>
          <div class="valor" style="color:{TEXTO_S}">--</div>
          <div class="subtexto">No disponible</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

col_h, col_p = st.columns(2)
with col_h:
    st.caption('Demanda real - ultimas 48 horas (fuente: REE / e.sios)')
    st.plotly_chart(grafica_historico_demanda(df_demanda_hist),
                    use_container_width=True, config={'displayModeBar': False})
with col_p:
    st.caption('Prediccion de demanda - proximas 48 horas (XGBoost - banda = +/-5%)')
    st.plotly_chart(grafica_prediccion_demanda(pred),
                    use_container_width=True, config={'displayModeBar': False})


# ── PERFIL HORARIO ─────────────────────────────────────────────────────────────
st.markdown('<div class="seccion">Cuando es mas barato usar electricidad?</div>',
            unsafe_allow_html=True)
st.caption('Prediccion de precio por hora para las proximas 24 horas - '
           'Verde: precio bajo / Amarillo: precio medio / Rojo: precio alto')

col_graf, col_consejo = st.columns([3, 1])
with col_graf:
    st.plotly_chart(grafica_perfil_precio(pred),
                    use_container_width=True, config={'displayModeBar': False})
with col_consejo:
    ahorro_pct = (
        (vals_p[idx_max_p] - vals_p[idx_min_p]) / vals_p[idx_max_p] * 100
        if vals_p[idx_max_p] > 0 else 0
    )
    st.markdown(f"""<div class="consejo-box">
      <b>Consejo de ahorro</b><br><br>
      Pon la lavadora o el<br>
      lavavajillas a las<br>
      <b style="color:{VERDE}">{ts_pred[idx_min_p].strftime('%H:%M del %d/%m')}</b>.<br><br>
      Evita usarlos a las<br>
      <b style="color:{ROJO}">{ts_pred[idx_max_p].strftime('%H:%M del %d/%m')}</b>.<br><br>
      Ahorro potencial:<br>
      hasta <b style="color:{VERDE}">{ahorro_pct:.0f}%</b> respecto<br>
      a la hora mas cara.
    </div>""", unsafe_allow_html=True)


# ── FOOTER ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:36px; padding-top:14px; border-top:1px solid {GRIS_B};
     font-size:0.73rem; color:{TEXTO_S}; text-align:center; line-height:1.8">
  Fuentes: REE / e.sios - OMIE - Open-Meteo |
  Modelo: XGBoost - Horizonte 48h - Datos 2023-2025 Madrid |
  TFM Ciencia de Datos e IA
</div>
""", unsafe_allow_html=True)
