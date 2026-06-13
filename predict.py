"""
predict.py - Motor de inferencia TFM
Uso: python predict.py --modelo_precio xgboost --modelo_demanda xgboost
"""

import os, json, pickle, warnings, argparse, requests, time
from pathlib import Path

import pandas as pd
import numpy as np
import holidays

warnings.filterwarnings('ignore')

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'data' / 'models'

HORIZONTE  = 48
LOOKBACK   = 168
LAT_MADRID = 40.4168
LON_MADRID = -3.7038
GEO_ID_MAD = 8741

FEATURE_COLS = [
    'hora', 'dia_semana', 'mes', 'semana_anio', 'anio',
    'es_festivo', 'dias_hasta_festivo', 'demanda_mw', 'precio_eur_mwh',
    'temperatura_c', 'humedad_pct', 'viento_kmh', 'irradiacion_wm2',
    'nubosidad_pct', 'precipitacion_mm', 'precio_eur_kwh',
    'hora_sin', 'hora_cos', 'dia_sin', 'dia_cos', 'mes_sin', 'mes_cos',
    'es_vispera_festivo', 'es_post_festivo',
    'precio_lag_1h', 'demanda_lag_1h', 'precio_lag_2h', 'demanda_lag_2h',
    'precio_lag_3h', 'demanda_lag_3h', 'precio_lag_24h', 'demanda_lag_24h',
    'precio_lag_48h', 'demanda_lag_48h', 'precio_lag_168h', 'demanda_lag_168h',
    'precio_roll_mean_6h', 'demanda_roll_mean_6h', 'precio_roll_std_6h',
    'precio_roll_max_6h', 'precio_roll_min_6h',
    'precio_roll_mean_24h', 'demanda_roll_mean_24h', 'precio_roll_std_24h',
    'precio_roll_max_24h', 'precio_roll_min_24h',
    'precio_roll_mean_168h', 'demanda_roll_mean_168h', 'precio_roll_std_168h',
    'precio_roll_max_168h', 'precio_roll_min_168h',
    'precio_ratio_24h', 'precio_diff_1h', 'precio_diff_24h', 'demanda_diff_1h',
    'precio_pct_1h', 'precio_pct_24h',
    'temperatura_c2', 'temp_x_hora', 'irrad_x_hora', 'es_hora_solar',
]

H_CORTO = list(range(0, 12))
H_LARGO = list(range(12, 48))


# ==============================================================================
# DESCARGA DE DATOS
# ==============================================================================

class DataFetcher:

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(override=False)
        self.esios_key = os.getenv('ESIOS_API_KEY', '')

    def fetch_demanda_ree(self, horas=LOOKBACK + HORIZONTE):
        now   = pd.Timestamp.now(tz='Europe/Madrid').floor('h')
        start = now - pd.Timedelta(hours=horas)

        resultado = pd.DataFrame(columns=['datetime', 'demanda_mw'])

        # Intento 1: apidatos
        try:
            headers = {'Accept': 'application/json'}
            if self.esios_key:
                headers['x-api-key'] = self.esios_key
            r = requests.get(
                'https://apidatos.ree.es/es/datos/demanda/demanda-tiempo-real',
                params={
                    'start_date': start.strftime('%Y-%m-%dT%H:%M'),
                    'end_date':   now.strftime('%Y-%m-%dT%H:%M'),
                    'time_trunc': 'hour', 'geo_limit': 'ccaa', 'geo_ids': GEO_ID_MAD,
                },
                headers=headers, timeout=20
            )
            if r.status_code == 200:
                for bloque in r.json().get('included', []):
                    if bloque.get('type') == 'Demanda real':
                        tmp = pd.DataFrame(bloque['attributes']['values'])
                        tmp['datetime'] = pd.to_datetime(
                            tmp['datetime'], utc=True).dt.tz_convert('Europe/Madrid')
                        resultado = (tmp[['datetime', 'value']]
                                     .rename(columns={'value': 'demanda_mw'})
                                     .sort_values('datetime')
                                     .reset_index(drop=True))
                        break
        except Exception as e:
            print(f'  REE apidatos: {e}')

        # Intento 2: e.sios con token
        if resultado.empty and self.esios_key:
            try:
                all_dfs = []
                cursor  = start
                while cursor <= now:
                    window_end = min(cursor + pd.Timedelta(days=29), now)
                    r = requests.get(
                        'https://api.esios.ree.es/indicators/1293',
                        headers={
                            'Accept':    'application/json; application/vnd.esios-api-v1+json',
                            'x-api-key': self.esios_key,
                        },
                        params={
                            'start_date': cursor.strftime('%Y-%m-%dT%H:%M:%S'),
                            'end_date':   window_end.strftime('%Y-%m-%dT%H:%M:%S'),
                            'time_trunc': 'hour',
                            'geo_ids[]':  GEO_ID_MAD,
                        },
                        timeout=30
                    )
                    if r.status_code == 200:
                        valores = r.json()['indicator']['values']
                        if valores:
                            tmp = pd.DataFrame(valores)
                            tmp['datetime'] = pd.to_datetime(
                                tmp['datetime'], utc=True).dt.tz_convert('Europe/Madrid')
                            all_dfs.append(
                                tmp[['datetime', 'value']].rename(
                                    columns={'value': 'demanda_mw'}))
                    cursor += pd.Timedelta(days=30)
                    time.sleep(0.3)
                if all_dfs:
                    resultado = (pd.concat(all_dfs)
                                 .drop_duplicates('datetime')
                                 .sort_values('datetime')
                                 .reset_index(drop=True))
            except Exception as e:
                print(f'  REE e.sios: {e}')

        return resultado

    def fetch_precio_omie(self, dias=10):
        resultado = pd.DataFrame(columns=['datetime', 'precio_eur_mwh'])
        all_dfs   = []
        fechas    = pd.date_range(
            end=pd.Timestamp.now().floor('D'), periods=dias, freq='D'
        )
        for fecha in fechas:
            fname  = f'marginalpdbc_{fecha.strftime("%Y%m%d")}.1'
            params = {'parents[]': 'marginalpdbc', 'filename': fname}
            try:
                r = requests.get(
                    'https://www.omie.es/es/file-download',
                    params=params, timeout=20)
                r.raise_for_status()
                lineas = r.text.strip().split('\n')
                datos  = [l.split(';') for l in lineas[1:]
                          if not l.startswith('*') and len(l) > 5]
                df = pd.DataFrame(
                    datos,
                    columns=['anio','mes','dia','hora','precio_es','precio_pt','_'])
                df = df[['anio','mes','dia','hora','precio_es']].apply(
                    lambda c: c.str.strip())
                df[['anio','mes','dia','hora']] = df[['anio','mes','dia','hora']].astype(int)
                df['precio_es'] = df['precio_es'].str.replace(',', '.').astype(float)
                df['hora'] = df['hora'] - 1
                df['datetime'] = (
                    pd.to_datetime(df[['anio','mes','dia']].rename(
                        columns={'anio':'year','mes':'month','dia':'day'}
                    )) + pd.to_timedelta(df['hora'], unit='h')
                )
                df['datetime'] = df['datetime'].dt.tz_localize(
                    'Europe/Madrid', ambiguous='NaT', nonexistent='shift_forward')
                all_dfs.append(df[['datetime','precio_es']].dropna())
                time.sleep(0.3)
            except Exception:
                pass

        if all_dfs:
            resultado = (pd.concat(all_dfs)
                         .drop_duplicates('datetime')
                         .sort_values('datetime')
                         .rename(columns={'precio_es': 'precio_eur_mwh'})
                         .reset_index(drop=True))
        return resultado

    def fetch_meteorologia(self, horas_pasado=LOOKBACK):
        now  = pd.Timestamp.now(tz='Europe/Madrid').floor('h')
        VARS = ('temperature_2m,relativehumidity_2m,windspeed_10m,'
                'shortwave_radiation,cloudcover,precipitation')

        df_hist = pd.DataFrame()
        try:
            r = requests.get(
                'https://archive-api.open-meteo.com/v1/archive',
                params={
                    'latitude':   LAT_MADRID, 'longitude': LON_MADRID,
                    'start_date': (now - pd.Timedelta(hours=horas_pasado)).date().isoformat(),
                    'end_date':   now.date().isoformat(),
                    'hourly':     VARS, 'timezone': 'Europe/Madrid',
                }, timeout=30)
            r.raise_for_status()
            df_hist = pd.DataFrame(r.json()['hourly'])
            df_hist['datetime'] = pd.to_datetime(df_hist['time']).dt.tz_localize(
                'Europe/Madrid', ambiguous='NaT', nonexistent='shift_forward')
            df_hist = df_hist.dropna(subset=['datetime'])
        except Exception as e:
            print(f'  Open-Meteo historico: {e}')

        df_fc = pd.DataFrame()
        try:
            r = requests.get(
                'https://api.open-meteo.com/v1/forecast',
                params={
                    'latitude':      LAT_MADRID, 'longitude': LON_MADRID,
                    'hourly':        VARS, 'timezone': 'Europe/Madrid',
                    'forecast_days': 3,
                }, timeout=30)
            r.raise_for_status()
            df_fc = pd.DataFrame(r.json()['hourly'])
            df_fc['datetime'] = pd.to_datetime(df_fc['time']).dt.tz_localize(
                'Europe/Madrid', ambiguous='NaT', nonexistent='shift_forward')
            df_fc = df_fc.dropna(subset=['datetime'])
        except Exception as e:
            print(f'  Open-Meteo forecast: {e}')

        frames = [f for f in [df_hist, df_fc] if not f.empty]
        if not frames:
            return pd.DataFrame()

        df = (pd.concat(frames)
              .drop(columns=['time'], errors='ignore')
              .drop_duplicates('datetime')
              .sort_values('datetime')
              .rename(columns={
                  'temperature_2m':      'temperatura_c',
                  'relativehumidity_2m': 'humedad_pct',
                  'windspeed_10m':       'viento_kmh',
                  'shortwave_radiation': 'irradiacion_wm2',
                  'cloudcover':          'nubosidad_pct',
                  'precipitation':       'precipitacion_mm',
              })
              .reset_index(drop=True))
        return df

    def fetch_calendario(self, fecha_ini, fecha_fin):
        festivos_es = {}
        for ano in range(fecha_ini.year, fecha_fin.year + 1):
            festivos_es.update(holidays.Spain(years=ano, subdiv='MD'))
        fechas_festivo = set(festivos_es.keys())

        idx = pd.date_range(
            fecha_ini.tz_localize(None) if fecha_ini.tzinfo else fecha_ini,
            fecha_fin.tz_localize(None) if fecha_fin.tzinfo else fecha_fin,
            freq='h', tz='Europe/Madrid'
        )
        df = pd.DataFrame({'datetime': idx})
        df['fecha']       = df['datetime'].dt.date
        df['hora']        = df['datetime'].dt.hour
        df['dia_semana']  = df['datetime'].dt.dayofweek
        df['mes']         = df['datetime'].dt.month
        df['anio']        = df['datetime'].dt.year
        df['semana_anio'] = df['datetime'].dt.isocalendar().week.astype(int)
        df['es_festivo']  = df['fecha'].isin(fechas_festivo).astype(int)

        dias_s = sorted(fechas_festivo)
        def dias_hasta(f):
            fut = [d for d in dias_s if d >= f]
            return (fut[0] - f).days if fut else 365
        mapa = {f: dias_hasta(f) for f in df['fecha'].unique()}
        df['dias_hasta_festivo'] = df['fecha'].map(mapa)
        df['es_vispera_festivo'] = (df['dias_hasta_festivo'] == 1).astype(int)
        df['es_post_festivo']    = (
            (df['es_festivo'].shift(24, fill_value=0) == 1) &
            (df['es_festivo'] == 0)
        ).astype(int)
        df.drop(columns=['fecha'], inplace=True)
        return df


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

class FeatureEngineer:

    def construir_features(self, df):
        df = df.copy().sort_values('datetime').reset_index(drop=True)

        # Garantizar columnas target aunque no haya datos
        for col in ['demanda_mw', 'precio_eur_mwh']:
            if col not in df.columns:
                df[col] = np.nan

        # Meteorologia por defecto si falta
        for col in ['temperatura_c','humedad_pct','viento_kmh',
                    'irradiacion_wm2','nubosidad_pct','precipitacion_mm']:
            if col not in df.columns:
                df[col] = 0.0

        # Encoding ciclico
        df['hora_sin']   = np.sin(2 * np.pi * df['hora']        / 24)
        df['hora_cos']   = np.cos(2 * np.pi * df['hora']        / 24)
        df['dia_sin']    = np.sin(2 * np.pi * df['dia_semana']  / 7)
        df['dia_cos']    = np.cos(2 * np.pi * df['dia_semana']  / 7)
        df['mes_sin']    = np.sin(2 * np.pi * df['mes']         / 12)
        df['mes_cos']    = np.cos(2 * np.pi * df['mes']         / 12)

        df['precio_eur_kwh'] = df['precio_eur_mwh'] / 1000

        # Lags
        for lag in [1, 2, 3, 24, 48, 168]:
            df[f'precio_lag_{lag}h']  = df['precio_eur_mwh'].shift(lag)
            df[f'demanda_lag_{lag}h'] = df['demanda_mw'].shift(lag)

        # Rolling
        for w in [6, 24, 168]:
            df[f'precio_roll_mean_{w}h']  = df['precio_eur_mwh'].shift(1).rolling(w, min_periods=1).mean()
            df[f'precio_roll_std_{w}h']   = df['precio_eur_mwh'].shift(1).rolling(w, min_periods=1).std()
            df[f'precio_roll_max_{w}h']   = df['precio_eur_mwh'].shift(1).rolling(w, min_periods=1).max()
            df[f'precio_roll_min_{w}h']   = df['precio_eur_mwh'].shift(1).rolling(w, min_periods=1).min()
            df[f'demanda_roll_mean_{w}h'] = df['demanda_mw'].shift(1).rolling(w, min_periods=1).mean()

        df['precio_ratio_24h'] = df['precio_lag_1h'] / (df['precio_roll_mean_24h'] + 1e-6)

        # Diferencias
        df['precio_diff_1h']  = df['precio_eur_mwh'].diff(1)
        df['precio_diff_24h'] = df['precio_eur_mwh'].diff(24)
        df['demanda_diff_1h'] = df['demanda_mw'].diff(1)
        df['precio_pct_1h']   = df['precio_eur_mwh'].pct_change(1).replace([np.inf,-np.inf], 0).fillna(0)
        df['precio_pct_24h']  = df['precio_eur_mwh'].pct_change(24).replace([np.inf,-np.inf], 0).fillna(0)

        # Interacciones
        df['temperatura_c2']  = df['temperatura_c'] ** 2
        df['temp_x_hora']     = df['temperatura_c'] * df['hora']
        df['irrad_x_hora']    = df['irradiacion_wm2'] * df['hora']
        df['es_hora_solar']   = df['hora'].between(10, 16).astype(int)

        return df


# ==============================================================================
# PREDICTOR
# ==============================================================================

class Predictor:

    MODELOS_VALIDOS = ('xgboost', 'lstm')

    def __init__(self, modelo_precio='xgboost', modelo_demanda='xgboost'):
        assert modelo_precio  in self.MODELOS_VALIDOS
        assert modelo_demanda in self.MODELOS_VALIDOS
        self.modelo_precio  = modelo_precio
        self.modelo_demanda = modelo_demanda
        print(f'Cargando modelos: precio={modelo_precio} | demanda={modelo_demanda}')
        self._cargar_scalers()
        self._cargar_modelos()
        self.fetcher  = DataFetcher()
        self.engineer = FeatureEngineer()
        print('Predictor listo')

    def _cargar_scalers(self):
        with open(MODELS_DIR / 'scaler_precio.pkl',  'rb') as f:
            self.scaler_precio  = pickle.load(f)
        with open(MODELS_DIR / 'scaler_demanda.pkl', 'rb') as f:
            self.scaler_demanda = pickle.load(f)

    def _cargar_modelos(self):
        self.model_p = self._cargar_un_modelo(self.modelo_precio,  'precio')
        self.model_d = self._cargar_un_modelo(self.modelo_demanda, 'demanda')

    def _cargar_un_modelo(self, tipo, variable):
        if tipo == 'xgboost':
            with open(MODELS_DIR / f'xgb_{variable}_corto.pkl', 'rb') as f:
                corto = pickle.load(f)
            with open(MODELS_DIR / f'xgb_{variable}_largo.pkl', 'rb') as f:
                largo = pickle.load(f)
            return {'tipo': 'xgboost', 'corto': corto, 'largo': largo}
        elif tipo == 'lstm':
            import tensorflow as tf
            model = tf.keras.models.load_model(
                MODELS_DIR / f'v2_lstm_{variable}.keras')
            return {'tipo': 'lstm', 'model': model}
        raise ValueError(f'Tipo desconocido: {tipo}')

    def _predecir_xgboost(self, modelo_dict, X_row, scaler):
        pred_corto = np.column_stack([m.predict(X_row) for m in modelo_dict['corto']])
        pred_largo = np.column_stack([m.predict(X_row) for m in modelo_dict['largo']])
        pred_sc    = np.concatenate([pred_corto, pred_largo], axis=1)
        return scaler.inverse_transform(pred_sc)[0]

    def _predecir_lstm(self, modelo_dict, df_hist, scaler):
        import tensorflow as tf
        SCALER_COLS = [
            'demanda_mw', 'precio_eur_mwh', 'temperatura_c', 'humedad_pct',
            'viento_kmh', 'irradiacion_wm2', 'nubosidad_pct', 'precipitacion_mm',
            'precio_eur_kwh', 'precio_lag_1h', 'demanda_lag_1h', 'precio_lag_2h',
            'demanda_lag_2h', 'precio_lag_3h', 'demanda_lag_3h', 'precio_lag_24h',
            'demanda_lag_24h', 'precio_lag_48h', 'demanda_lag_48h', 'precio_lag_168h',
            'demanda_lag_168h', 'precio_roll_mean_6h', 'demanda_roll_mean_6h',
            'precio_roll_std_6h', 'precio_roll_max_6h', 'precio_roll_min_6h',
            'precio_roll_mean_24h', 'demanda_roll_mean_24h', 'precio_roll_std_24h',
            'precio_roll_max_24h', 'precio_roll_min_24h', 'precio_roll_mean_168h',
            'demanda_roll_mean_168h', 'precio_roll_std_168h', 'precio_roll_max_168h',
            'precio_roll_min_168h', 'precio_ratio_24h', 'precio_diff_1h',
            'precio_diff_24h', 'demanda_diff_1h', 'precio_pct_1h', 'precio_pct_24h',
            'temperatura_c2', 'temp_x_hora', 'irrad_x_hora', 'irrad_solar_activa'
        ]
        with open(MODELS_DIR / 'scaler_features.pkl', 'rb') as f:
            sc_feat = pickle.load(f)
        for col in SCALER_COLS:
            if col not in df_hist.columns:
                df_hist[col] = 0.0
        X_scaled = sc_feat.transform(df_hist[SCALER_COLS].fillna(0).values)
        X_seq    = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, -1).astype(np.float32)
        pred_sc  = modelo_dict['model'].predict(X_seq, verbose=0)
        return scaler.inverse_transform(pred_sc)[0]
    
    def predecir(self):
        print('\nIniciando pipeline de prediccion...')
        now       = pd.Timestamp.now(tz='Europe/Madrid').floor('h')
        fecha_ini = now - pd.Timedelta(hours=LOOKBACK)
        fecha_fin = now + pd.Timedelta(hours=HORIZONTE)

        # 1. Descarga
        print('\nDescargando datos frescos...')
        print('  Demanda REE...')
        df_ree   = self.fetcher.fetch_demanda_ree(horas=LOOKBACK + 24)
        print('  Precio OMIE...')
        df_omie  = self.fetcher.fetch_precio_omie(dias=10)
        print('  Meteorologia Open-Meteo...')
        df_meteo = self.fetcher.fetch_meteorologia(horas_pasado=LOOKBACK)
        print('  Calendario...')
        df_cal   = self.fetcher.fetch_calendario(fecha_ini, fecha_fin)

        # 2. Merge
        df = df_cal.copy()

        if not df_ree.empty:
            tmp = df_ree.copy()
            tmp['datetime'] = tmp['datetime'].dt.floor('h')
            df = df.merge(tmp, on='datetime', how='left')
        else:
            df['demanda_mw'] = np.nan

        if not df_omie.empty:
            tmp = df_omie.copy()
            tmp['datetime'] = tmp['datetime'].dt.floor('h')
            df = df.merge(tmp, on='datetime', how='left')
        else:
            df['precio_eur_mwh'] = np.nan

        if not df_meteo.empty:
            tmp = df_meteo.copy()
            tmp['datetime'] = tmp['datetime'].dt.floor('h')
            df = df.merge(tmp, on='datetime', how='left')

        # Imputar meteorologia
        for col in ['temperatura_c','humedad_pct','viento_kmh',
                    'irradiacion_wm2','nubosidad_pct','precipitacion_mm']:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit=6).fillna(0)
            else:
                df[col] = 0.0

        # 3. Feature engineering
        print('\nCalculando features...')
        df_feat = self.engineer.construir_features(df)
        df_hist = df_feat[df_feat['datetime'] <= now].copy()

        if len(df_hist) < 2:
            raise ValueError('Datos historicos insuficientes.')

        # 4. Preparar X
        for col in FEATURE_COLS:
            if col not in df_hist.columns:
                df_hist[col] = 0.0

        X_row = df_hist[FEATURE_COLS].fillna(0).values[-1:]

        # 5. Prediccion
        print(f'\nPrediciendo precio con {self.modelo_precio}...')
        if self.model_p['tipo'] == 'xgboost':
            pred_precio = self._predecir_xgboost(self.model_p, X_row, self.scaler_precio)
        else:
            pred_precio = self._predecir_lstm(self.model_p, df_hist, self.scaler_precio)

        print(f'Prediciendo demanda con {self.modelo_demanda}...')
        if self.model_d['tipo'] == 'xgboost':
            pred_demanda = self._predecir_xgboost(self.model_d, X_row, self.scaler_demanda)
        else:
            pred_demanda = self._predecir_lstm(self.model_d, df_hist, self.scaler_demanda)

        # 6. Resultado
        timestamps = [
            (now + pd.Timedelta(hours=h+1)).isoformat()
            for h in range(HORIZONTE)
        ]
        pred_precio_list  = [round(float(v), 2) for v in pred_precio]
        pred_demanda_list = [round(float(v), 2) for v in pred_demanda]

        ts        = pd.to_datetime(timestamps)
        idx_min_p = int(np.argmin(pred_precio))
        idx_max_p = int(np.argmax(pred_precio))
        idx_min_d = int(np.argmin(pred_demanda))
        idx_max_d = int(np.argmax(pred_demanda))

        resultado = {
            'timestamps':       timestamps,
            'precio':           pred_precio_list,
            'demanda':          pred_demanda_list,
            'hora_barata':      ts[idx_min_p].strftime('%H:%M'),
            'hora_cara':        ts[idx_max_p].strftime('%H:%M'),
            'precio_min':       pred_precio_list[idx_min_p],
            'precio_max':       pred_precio_list[idx_max_p],
            'precio_medio':     round(float(np.mean(pred_precio)), 2),
            'demanda_min':      pred_demanda_list[idx_min_d],
            'demanda_max':      pred_demanda_list[idx_max_d],
            'hora_min_demanda': ts[idx_min_d].strftime('%H:%M'),
            'hora_max_demanda': ts[idx_max_d].strftime('%H:%M'),
            'demanda_media':    round(float(np.mean(pred_demanda)), 2),
            'modelo_precio':    self.modelo_precio,
            'modelo_demanda':   self.modelo_demanda,
            'generado_en':      now.isoformat(),
        }

        print(f'\nPrediccion completada')
        print(f'  Hora mas barata:  {resultado["hora_barata"]} ({resultado["precio_min"]:.1f} EUR/MWh)')
        print(f'  Hora mas cara:    {resultado["hora_cara"]} ({resultado["precio_max"]:.1f} EUR/MWh)')
        print(f'  Precio medio 48h: {resultado["precio_medio"]:.1f} EUR/MWh')
        return resultado

    def predecir_a_dataframe(self):
        res = self.predecir()
        return pd.DataFrame({
            'datetime':                pd.to_datetime(res['timestamps']),
            'precio_predicho_eur_mwh': res['precio'],
            'demanda_predicha_mw':     res['demanda'],
        })


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelo_precio',  default='xgboost',
                        choices=Predictor.MODELOS_VALIDOS)
    parser.add_argument('--modelo_demanda', default='xgboost',
                        choices=Predictor.MODELOS_VALIDOS)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    predictor = Predictor(args.modelo_precio, args.modelo_demanda)
    resultado = predictor.predecir()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        print(f'\nResultado guardado en {args.output}')
    else:
        print('\nPrimeras 6 predicciones:')
        for i in range(6):
            print(f'  {resultado["timestamps"][i]}  '
                  f'Precio: {resultado["precio"][i]:6.1f} EUR/MWh  '
                  f'Demanda: {resultado["demanda"][i]:,.0f} MW')
