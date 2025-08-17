import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
from io import StringIO
import datetime
import calendar
from datetime import datetime, timedelta
import os
import h3
import zipfile
import tempfile
import json
import geopandas as gpd
import shutil
import xml.etree.ElementTree as ET

from streamlit import subheader

# Configuração da página
st.set_page_config(
    page_title="Atlas das Aves Aquáticas do Rio de Janeiro",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar API key do Mapbox
os.environ[
    "MAPBOX_API_KEY"] = "pk.eyJ1IjoiY2FseXB0dXJhIiwiYSI6ImNpdjV2MjhyNDAxaWMyb3MydHVvdTNhYXEifQ.zYAN0zIEFHZImB5xE_U3qg"


# Função para carregar dados de observações de aves
@st.cache_data(ttl=3600)
def carregar_dados_aves():
    """Carrega dados de observações de aves do Google Sheets"""
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR1fE-OFPIe79oj0dX8jN_5Zi0zzg7nV64gLE1UiOo16jZnJ8H2ml0OLzgryU2819HKPu3BZxbEZx_7/pub?output=csv"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            csv_content = StringIO(response.content.decode('utf-8'))
            df = pd.read_csv(csv_content)

            # Processar as datas
            df['eventDate'] = pd.to_datetime(df['eventDate'], errors='coerce')
            df['ano'] = df['eventDate'].dt.year
            df['mes'] = df['eventDate'].dt.month
            df['mes_nome'] = df['eventDate'].dt.month_name()
            df['ano_mes'] = df['eventDate'].dt.to_period('M')

            # Garantir que latitude e longitude são numéricas
            df['decimalLatitude'] = pd.to_numeric(df['decimalLatitude'], errors='coerce')
            df['decimalLongitude'] = pd.to_numeric(df['decimalLongitude'], errors='coerce')

            # Remover registros com coordenadas inválidas
            df = df.dropna(subset=['decimalLatitude', 'decimalLongitude', 'eventDate'])

            # Limpar nomes de espécies e municípios
            if 'vernacularName' in df.columns:
                df['vernacularName'] = df['vernacularName'].str.strip()
            if 'species' in df.columns:
                df['species'] = df['species'].str.strip()
            if 'level2Name' in df.columns:
                df['level2Name'] = df['level2Name'].str.strip()

            return df
        else:
            st.error(f"Erro ao carregar dados: Status {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


# NOVA FUNÇÃO: Verificar se espécie é ameaçada
def verificar_especie_ameacada(row):
    """Verifica se uma espécie é considerada ameaçada em qualquer das listas"""
    colunas_ameaca = ['IUCN 2021', 'MMA 2022', 'Ameaçadas RJ 2001']

    for coluna in colunas_ameaca:
        if coluna in row and pd.notna(row[coluna]) and str(row[coluna]).strip() not in ['', 'LC', 'NA', '0']:
            return True
    return False


# NOVA FUNÇÃO: Verificar se espécie é migratória
def verificar_especie_migratoria(row):
    """Verifica se uma espécie é considerada migratória em qualquer das listas"""
    # Verificar CBRO 2021 (0 ou 1)
    if 'Migratórias CBRO 2021' in row and pd.notna(row['Migratórias CBRO 2021']):
        if str(row['Migratórias CBRO 2021']).strip() == '1':
            return True

    # Verificar Somenzari et al. 2017 (qualquer status que não seja vazio/NA)
    if 'Migratórias Somenzari et al. 2017' in row and pd.notna(row['Migratórias Somenzari et al. 2017']):
        status = str(row['Migratórias Somenzari et al. 2017']).strip()
        if status not in ['', 'NA', '0', 'nan']:
            return True

    return False


# Função para carregar limites municipais do RJ
@st.cache_data(ttl=3600 * 24)
def carregar_limites_rj():
    """Carrega limites municipais do Rio de Janeiro do Google Drive"""
    file_id = "17NkmuXiouhD38Ty-SMKxg9qokEXgvPu8"
    url_shapefile = f"https://drive.google.com/uc?id={file_id}&export=download"

    try:
        response = requests.get(url_shapefile, timeout=60)
        if response.status_code != 200:
            st.warning("Não foi possível carregar os limites municipais do RJ.")
            return None

        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "rj_municipios.zip")
        with open(zip_path, 'wb') as f:
            f.write(response.content)

        extract_dir = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        shp_files = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.endswith('.shp'):
                    shp_files.append(os.path.join(root, file))

        if not shp_files:
            st.warning("Arquivo shapefile (.shp) não encontrado no ZIP.")
            return None

        shp_path = shp_files[0]
        gdf = gpd.read_file(shp_path)

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        geojson = json.loads(gdf.to_json())
        shutil.rmtree(temp_dir)
        return geojson

    except Exception as e:
        st.warning(f"Erro ao carregar limites municipais: {e}")
        return None


# NOVA FUNÇÃO MELHORADA: Processar geometrias completas
def processar_geometria_completa(geometry):
    """Processa todas as partes de uma geometria (Polygon ou MultiPolygon)"""
    coords_list = []

    try:
        if geometry.geom_type == 'Polygon':
            # Adicionar exterior ring
            coords = [[x, y] for x, y in geometry.exterior.coords]
            if len(coords) >= 4:
                coords_list.append(coords)

            # CRÍTICO: Adicionar interior rings (holes) para municípios costeiros
            for interior in geometry.interiors:
                hole_coords = [[x, y] for x, y in interior.coords]
                if len(hole_coords) >= 4:
                    coords_list.append(hole_coords)

        elif geometry.geom_type == 'MultiPolygon':
            # CRÍTICO: Processar TODOS os polígonos (ilhas, recortes costeiros)
            for polygon in geometry.geoms:
                # Exterior ring de cada polígono
                coords = [[x, y] for x, y in polygon.exterior.coords]
                if len(coords) >= 4:
                    coords_list.append(coords)

                # Interior rings (holes) de cada polígono
                for interior in polygon.interiors:
                    hole_coords = [[x, y] for x, y in interior.coords]
                    if len(hole_coords) >= 4:
                        coords_list.append(hole_coords)

    except Exception as e:
        print(f"Erro ao processar geometria: {e}")

    return coords_list


# Função para filtrar dados por período
def filtrar_por_periodo(df, periodo_selecionado, data_inicio=None, data_fim=None):
    """Filtra o DataFrame por período selecionado"""
    hoje = datetime.now()

    if periodo_selecionado == "Últimos 15 dias":
        data_limite = hoje - timedelta(days=15)
        return df[df['eventDate'] >= data_limite]
    elif periodo_selecionado == "Último mês":
        data_limite = hoje - timedelta(days=30)
        return df[df['eventDate'] >= data_limite]
    elif periodo_selecionado == "Série completa":
        return df
    elif periodo_selecionado == "Período personalizado":
        if data_inicio and data_fim:
            return df[(df['eventDate'] >= pd.to_datetime(data_inicio)) &
                      (df['eventDate'] <= pd.to_datetime(data_fim))]
        else:
            return df
    return df


# Função para filtrar dados por município
def filtrar_por_municipio(df, municipio_selecionado):
    """Filtra o DataFrame por município selecionado"""
    if municipio_selecionado == "Todos os municípios":
        return df
    else:
        return df[df['level2Name'] == municipio_selecionado]


# Função para filtrar dados por espécie
def filtrar_por_especie(df, especie_selecionada):
    """Filtra o DataFrame por espécie selecionada"""
    if especie_selecionada == "Todas as espécies":
        return df
    else:
        return df[df['species'] == especie_selecionada]


# FUNÇÃO MELHORADA: Calcular riqueza de espécies usando H3 com análises especializadas
def calcular_riqueza_h3(df, tamanho_hex="1km", metodo_prioritario="Total de espécies"):
    """Calcula riqueza de espécies usando sistema H3 com análises especializadas"""
    if df.empty:
        return pd.DataFrame()

    resolucoes_h3 = {
        "1km": 7, "2km": 6, "5km": 5,
    }

    resolution = resolucoes_h3.get(tamanho_hex, 7)

    df_h3 = df.copy()
    df_h3['h3_index'] = df_h3.apply(
        lambda row: h3.latlng_to_cell(row['decimalLatitude'], row['decimalLongitude'], resolution),
        axis=1
    )

    # Marcar espécies ameaçadas e migratórias
    df_h3['is_ameacada'] = df_h3.apply(verificar_especie_ameacada, axis=1)
    df_h3['is_migratoria'] = df_h3.apply(verificar_especie_migratoria, axis=1)

    # Agrupamento com diferentes métricas
    if metodo_prioritario == "Espécies ameaçadas":
        # Filtrar apenas espécies ameaçadas para contagem principal
        df_ameacadas = df_h3[df_h3['is_ameacada'] == True]

        # Agrupamento principal com todas as espécies
        agrupamento = df_h3.groupby('h3_index').agg({
            'vernacularName': [lambda x: x.nunique(), lambda x: sorted(x.unique())],
            'species': lambda x: sorted(x.unique()),
            'eventDate': ['count', 'min', 'max'],
            'level2Name': lambda x: ', '.join(sorted(x.unique())[:3]),
            'is_ameacada': 'sum'  # Número de registros de espécies ameaçadas
        }).reset_index()

        # Corrigir nomes das colunas do MultiIndex
        agrupamento.columns = ['h3_index', 'riqueza_especies_total', 'especies_completas',
                               'especies_cientificas_completas',
                               'total_registros', 'data_min', 'data_max', 'municipios', 'registros_ameacadas']

        # Calcular riqueza de espécies ameaçadas por hexágono separadamente
        if not df_ameacadas.empty:
            riqueza_ameacadas = df_ameacadas.groupby('h3_index')['vernacularName'].nunique().reset_index()
            riqueza_ameacadas.columns = ['h3_index', 'riqueza_especies']

            # Merge com agrupamento principal
            agrupamento = agrupamento.merge(riqueza_ameacadas, on='h3_index', how='left')
        else:
            # Se não há espécies ameaçadas, criar coluna com zeros
            agrupamento['riqueza_especies'] = 0

        # Preencher valores nulos com zero
        agrupamento['riqueza_especies'] = agrupamento['riqueza_especies'].fillna(0)

    elif metodo_prioritario == "Espécies migratórias":
        # Filtrar apenas espécies migratórias para contagem principal
        df_migratorias = df_h3[df_h3['is_migratoria'] == True]

        # Agrupamento principal com todas as espécies
        agrupamento = df_h3.groupby('h3_index').agg({
            'vernacularName': [lambda x: x.nunique(), lambda x: sorted(x.unique())],
            'species': lambda x: sorted(x.unique()),
            'eventDate': ['count', 'min', 'max'],
            'level2Name': lambda x: ', '.join(sorted(x.unique())[:3]),
            'is_migratoria': 'sum'  # Número de registros de espécies migratórias
        }).reset_index()

        # Corrigir nomes das colunas do MultiIndex
        agrupamento.columns = ['h3_index', 'riqueza_especies_total', 'especies_completas',
                               'especies_cientificas_completas',
                               'total_registros', 'data_min', 'data_max', 'municipios', 'registros_migratorias']

        # Calcular riqueza de espécies migratórias por hexágono separadamente
        if not df_migratorias.empty:
            riqueza_migratorias = df_migratorias.groupby('h3_index')['vernacularName'].nunique().reset_index()
            riqueza_migratorias.columns = ['h3_index', 'riqueza_especies']

            # Merge com agrupamento principal
            agrupamento = agrupamento.merge(riqueza_migratorias, on='h3_index', how='left')
        else:
            # Se não há espécies migratórias, criar coluna com zeros
            agrupamento['riqueza_especies'] = 0

        # Preencher valores nulos com zero
        agrupamento['riqueza_especies'] = agrupamento['riqueza_especies'].fillna(0)

    else:  # "Total de espécies" (método original)
        agrupamento = df_h3.groupby('h3_index').agg({
            'vernacularName': [lambda x: x.nunique(), lambda x: sorted(x.unique())],
            'species': lambda x: sorted(x.unique()),
            'eventDate': ['count', 'min', 'max'],
            'level2Name': lambda x: ', '.join(sorted(x.unique())[:3])
        }).reset_index()

        # Corrigir nomes das colunas do MultiIndex
        agrupamento.columns = ['h3_index', 'riqueza_especies', 'especies_completas', 'especies_cientificas_completas',
                               'total_registros', 'data_min', 'data_max', 'municipios']

    # Criar listas resumidas para tooltip (primeiras 3 espécies)
    agrupamento['especies_cientificas'] = agrupamento['especies_cientificas_completas'].apply(
        lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '')
    )

    agrupamento['periodo'] = agrupamento['data_min'].dt.strftime('%m/%Y') + ' a ' + agrupamento['data_max'].dt.strftime(
        '%m/%Y')

    # NOVO: Gerar IDs sequenciais ordenados por riqueza (decrescente)
    agrupamento_ordenado = agrupamento.sort_values('riqueza_especies', ascending=False).reset_index(drop=True)

    # Prefixo baseado no tamanho e método
    if metodo_prioritario == "Espécies ameaçadas":
        if tamanho_hex == "2km":
            prefixo = "A2K_"
        elif tamanho_hex == "5km":
            prefixo = "A5K_"
        else:
            prefixo = "A"
    elif metodo_prioritario == "Espécies migratórias":
        if tamanho_hex == "2km":
            prefixo = "M2K_"
        elif tamanho_hex == "5km":
            prefixo = "M5K_"
        else:
            prefixo = "M"
    else:  # Total de espécies
        if tamanho_hex == "2km":
            prefixo = "H2K_"
        elif tamanho_hex == "5km":
            prefixo = "H5K_"
        else:
            prefixo = "H"

    agrupamento_ordenado['hexagono_id'] = agrupamento_ordenado.index.map(lambda x: f"{prefixo}{x + 1:04d}")

    # Calcular coordenadas e polígonos
    coordenadas = []
    polygons = []
    areas_info = []

    for h3_index in agrupamento_ordenado['h3_index']:
        lat, lon = h3.cell_to_latlng(h3_index)
        coordenadas.append((lat, lon))

        boundary = h3.cell_to_boundary(h3_index)
        polygon = [[float(coord[1]), float(coord[0])] for coord in boundary]
        polygons.append(polygon)

        area_km2 = h3.cell_area(h3_index, unit='km^2')
        areas_info.append(f"{tamanho_hex} (~{area_km2:.1f}km²)")

    agrupamento_ordenado['latitude'] = [coord[0] for coord in coordenadas]
    agrupamento_ordenado['longitude'] = [coord[1] for coord in coordenadas]
    agrupamento_ordenado['polygon'] = polygons
    agrupamento_ordenado['area_info'] = areas_info

    agrupamento_ordenado['riqueza_especies'] = agrupamento_ordenado['riqueza_especies'].astype(int)
    agrupamento_ordenado['total_registros'] = agrupamento_ordenado['total_registros'].astype(int)
    agrupamento_ordenado['h3_index'] = agrupamento_ordenado['h3_index'].astype(str)

    return agrupamento_ordenado


# Função para calcular riqueza por município
def calcular_riqueza_municipio(df):
    """Calcula riqueza de espécies por município"""
    if df.empty:
        return pd.DataFrame()

    riqueza_municipio = df.groupby('level2Name').agg({
        'vernacularName': lambda x: x.nunique(),
        'eventDate': ['count', 'min', 'max'],
        'species': lambda x: ', '.join(sorted(x.unique())[:5])
    }).reset_index()

    riqueza_municipio.columns = ['municipio', 'riqueza_especies', 'total_registros', 'data_min', 'data_max',
                                 'especies_cientificas']

    riqueza_municipio['periodo'] = riqueza_municipio['data_min'].dt.strftime('%m/%Y') + ' a ' + riqueza_municipio[
        'data_max'].dt.strftime('%m/%Y')

    return riqueza_municipio.sort_values('riqueza_especies', ascending=False)


# NOVA FUNÇÃO MELHORADA: Calcular riqueza por município com geometrias completas
def calcular_riqueza_municipio_mapa_espacial(df, limites_geojson):
    """Calcula riqueza de espécies por município com processamento completo de geometrias costeiras"""
    if df.empty or limites_geojson is None:
        return pd.DataFrame()

    try:
        gdf_pontos = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['decimalLongitude'], df['decimalLatitude']),
            crs='EPSG:4326'
        )

        gdf_municipios = gpd.GeoDataFrame.from_features(
            limites_geojson['features'],
            crs='EPSG:4326'
        )

        campo_nome = None
        for campo in ['NOME', 'NM_MUNICIP', 'NM_MUN', 'NAME', 'MUNICIPIO', 'nome']:
            if campo in gdf_municipios.columns:
                campo_nome = campo
                break

        if not campo_nome:
            st.warning("Não foi possível identificar o campo do nome no shapefile.")
            return pd.DataFrame()

        gdf_municipios = gdf_municipios.rename(columns={campo_nome: 'nome_municipio'})

        with st.spinner(f"Processando {len(gdf_pontos)} pontos com análise espacial otimizada..."):
            gdf_join = gpd.sjoin(gdf_pontos, gdf_municipios, how='left', predicate='within')

        gdf_com_municipio = gdf_join[gdf_join['nome_municipio'].notna()].copy()

        if gdf_com_municipio.empty:
            st.warning("Nenhum ponto foi encontrado dentro dos limites municipais.")
            return pd.DataFrame()

        riqueza_municipio = gdf_com_municipio.groupby('nome_municipio').agg({
            'vernacularName': lambda x: x.nunique(),
            'eventDate': ['count', 'min', 'max'],
            'species': lambda x: ', '.join(sorted(x.unique())[:5]),
            'index_right': 'first'
        }).reset_index()

        riqueza_municipio.columns = ['municipio', 'riqueza_especies', 'total_registros', 'data_min', 'data_max',
                                     'especies_cientificas', 'municipio_idx']

        riqueza_municipio['periodo'] = riqueza_municipio['data_min'].dt.strftime('%m/%Y') + ' a ' + riqueza_municipio[
            'data_max'].dt.strftime('%m/%Y')

        # NOVA ABORDAGEM: Processar TODAS as geometrias
        municipios_com_geometria = []

        for _, row in riqueza_municipio.iterrows():
            municipio_idx = int(row['municipio_idx'])
            municipio_geom = gdf_municipios.iloc[municipio_idx]

            # Usar a nova função para processar geometrias completas
            coords_list = processar_geometria_completa(municipio_geom.geometry)

            if coords_list:
                # Criar entrada para cada parte do município
                for i, coords in enumerate(coords_list):
                    municipios_com_geometria.append({
                        'municipio': row['municipio'],
                        'riqueza_especies': int(row['riqueza_especies']),
                        'total_registros': int(row['total_registros']),
                        'periodo': row['periodo'],
                        'especies_cientificas': row['especies_cientificas'],
                        'polygon': coords
                    })

        return pd.DataFrame(municipios_com_geometria)

    except Exception as e:
        st.error(f"Erro na análise espacial: {e}")
        return pd.DataFrame()


# NOVA FUNÇÃO: Adicionar limites municipais completos
def adicionar_limites_municipais_completos(layers, limites_geojson):
    """Adiciona limites municipais processando TODAS as geometrias"""
    if not limites_geojson:
        return layers

    try:
        path_data = []

        for feature in limites_geojson['features']:
            geometry = feature['geometry']

            if geometry['type'] == 'Polygon':
                for ring in geometry['coordinates']:
                    if len(ring) >= 4:
                        path_data.append({
                            'path': ring,
                            'color': [128, 128, 128, 150],
                            'width': 1
                        })

            elif geometry['type'] == 'MultiPolygon':
                for polygon in geometry['coordinates']:
                    for ring in polygon:
                        if len(ring) >= 4:
                            path_data.append({
                                'path': ring,
                                'color': [128, 128, 128, 150],
                                'width': 1
                            })

        if path_data:
            layers.append(pdk.Layer(
                'PathLayer',
                data=path_data,
                get_path='path',
                get_color='color',
                get_width='width',
                width_scale=1,
                width_min_pixels=1,
                pickable=False,
                auto_highlight=False
            ))

    except Exception as e:
        print(f"Erro ao adicionar limites municipais: {e}")

    return layers


# Função MELHORADA para gerar mapa de aves
def gerar_mapa_aves(df, tipo_mapa, estilo_mapa, transparencia=0.8, tamanho_hex="1km", limites_geojson=None,
                    usar_areas_prioritarias=False, metodo_prioritario=None, hexagono_destacado=None):
    """Gera mapa de observações de aves com processamento melhorado de geometrias costeiras e destaque de hexágono"""
    if df.empty or 'decimalLatitude' not in df.columns or 'decimalLongitude' not in df.columns:
        return None, None

    df_mapa = df.dropna(subset=['decimalLatitude', 'decimalLongitude']).copy()

    if len(df_mapa) == 0:
        return None, None

    # Calcular centro e zoom
    if not df_mapa.empty:
        min_lat = df_mapa['decimalLatitude'].min()
        max_lat = df_mapa['decimalLatitude'].max()
        min_lon = df_mapa['decimalLongitude'].min()
        max_lon = df_mapa['decimalLongitude'].max()

        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        max_range = max(lat_range, lon_range)

        if max_range > 10:
            zoom = 5
        elif max_range > 5:
            zoom = 6
        elif max_range > 2:
            zoom = 7
        elif max_range > 1:
            zoom = 8
        elif max_range > 0.5:
            zoom = 9
        else:
            zoom = 10

        zoom = max(5, zoom - 0.5)
    else:
        center_lat = -22.9068
        center_lon = -43.1729
        zoom = 7

    estilos_mapa = {
        "Satélite": "mapbox://styles/mapbox/satellite-v9",
        "Claro": "mapbox://styles/mapbox/light-v11",
        "Escuro": "mapbox://styles/mapbox/dark-v11"
    }

    estilo_url = estilos_mapa.get(estilo_mapa, "mapbox://styles/mapbox/satellite-v9")

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
        bearing=0
    )

    layers = []
    dados_hexagonos = None  # Para retornar dados dos hexágonos

    if tipo_mapa == "Mapa de pontos":
        df_mapa['data_formatada'] = df_mapa['eventDate'].dt.strftime('%d/%m/%Y')

        layers.append(pdk.Layer(
            'ScatterplotLayer',
            data=df_mapa,
            get_position=['decimalLongitude', 'decimalLatitude'],
            get_color=[0, 150, 255, 200],
            get_radius=50,
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_scale=6,
            radius_min_pixels=3,
            radius_max_pixels=15,
            line_width_min_pixels=1,
        ))

        tooltip = {
            "html": """
            <div style="padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px; max-width: 300px;">
              <b>Espécie:</b> {vernacularName}<br/>
              <b>Nome científico:</b> {species}<br/>
              <b>Data:</b> {data_formatada}<br/>
              <b>Município:</b> {level2Name}<br/>
              <b>Localidade:</b> {locality}<br/>
              <b>Catálogo:</b> {catalogNumber}
            </div>
            """,
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

    elif tipo_mapa == "Mapa de calor":
        layers.append(pdk.Layer(
            'HeatmapLayer',
            data=df_mapa,
            get_position=['decimalLongitude', 'decimalLatitude'],
            opacity=0.8,
            get_weight=1,
            radiusPixels=60,
            color_range=[
                [255, 255, 178], [254, 204, 92], [253, 141, 60], [240, 59, 32], [189, 0, 38]
            ]
        ))

        tooltip = {
            "html": """
            <div style="padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px;">
              <b>Visualização de densidade de observações</b><br/>
              <b>Dica:</b> Cores mais quentes = maior densidade
            </div>
            """,
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

    elif tipo_mapa == "Mapa por municípios":
        # Usar a função MELHORADA
        df_municipios_riqueza = calcular_riqueza_municipio_mapa_espacial(df_mapa, limites_geojson)

        if df_municipios_riqueza.empty:
            st.warning("Não foi possível calcular riqueza por municípios para os dados filtrados.")
            return None, None

        max_riqueza = df_municipios_riqueza['riqueza_especies'].max() if len(df_municipios_riqueza) > 0 else 1

        def get_color_by_richness(riqueza, max_riqueza):
            normalized = min(riqueza / max_riqueza, 1.0)
            if normalized <= 0.2:
                return [255, 255, 178, int(255 * transparencia)]
            elif normalized <= 0.4:
                return [254, 204, 92, int(255 * transparencia)]
            elif normalized <= 0.6:
                return [253, 141, 60, int(255 * transparencia)]
            elif normalized <= 0.8:
                return [240, 59, 32, int(255 * transparencia)]
            else:
                return [189, 0, 38, int(255 * transparencia)]

        df_municipios_riqueza['fill_color'] = df_municipios_riqueza['riqueza_especies'].apply(
            lambda x: get_color_by_richness(x, max_riqueza)
        )

        layers.append(pdk.Layer(
            'PolygonLayer',
            data=df_municipios_riqueza,
            get_polygon='polygon',
            get_fill_color='fill_color',
            get_line_color=[255, 255, 255, 200],
            line_width=2,
            pickable=True,
            auto_highlight=True,
            get_elevation=0,
        ))

        tooltip = {
            "html": """
            <div style="padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px; max-width: 350px;">
              <b>Município:</b> {municipio}<br/>
              <b>Riqueza de espécies:</b> {riqueza_especies}<br/>
              <b>Total de registros:</b> {total_registros}<br/>
              <b>Período:</b> {periodo}<br/>
              <b>Espécies (científico):</b> {especies_cientificas}
            </div>
            """,
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

    elif tipo_mapa == "Mapa de hexágono":
        df_h3_riqueza = calcular_riqueza_h3(df_mapa, tamanho_hex, metodo_prioritario)
        dados_hexagonos = df_h3_riqueza  # Armazenar para consulta posterior

        if df_h3_riqueza.empty:
            st.warning("Não foi possível calcular hexágonos H3 para os dados filtrados.")
            return None, None

        if usar_areas_prioritarias:
            max_riqueza_total = df_h3_riqueza['riqueza_especies'].max()
            limite_50_porcento = max_riqueza_total * 0.5
            df_h3_riqueza = df_h3_riqueza[df_h3_riqueza['riqueza_especies'] >= limite_50_porcento].copy()

            if df_h3_riqueza.empty:
                st.warning("Nenhum hexágono atende aos critérios de áreas prioritárias.")
                return None, None

            # Cores específicas para cada método de priorização
            if metodo_prioritario == "Espécies ameaçadas":
                def get_color_areas_prioritarias(riqueza, max_riqueza):
                    porcentagem = (riqueza / max_riqueza) * 100
                    if porcentagem >= 83.2:
                        return [139, 0, 0, int(255 * transparencia)]  # Vermelho escuro para ameaçadas
                    elif porcentagem >= 66.6:
                        return [220, 20, 60, int(255 * transparencia)]  # Crimson
                    else:
                        return [255, 69, 0, int(255 * transparencia)]  # Laranja avermelhado
            elif metodo_prioritario == "Espécies migratórias":
                def get_color_areas_prioritarias(riqueza, max_riqueza):
                    porcentagem = (riqueza / max_riqueza) * 100
                    if porcentagem >= 83.2:
                        return [0, 0, 139, int(255 * transparencia)]  # Azul escuro para migratórias
                    elif porcentagem >= 66.6:
                        return [30, 144, 255, int(255 * transparencia)]  # Azul dodger
                    else:
                        return [135, 206, 250, int(255 * transparencia)]  # Azul céu claro
            else:  # Total de espécies (cores originais)
                def get_color_areas_prioritarias(riqueza, max_riqueza):
                    porcentagem = (riqueza / max_riqueza) * 100
                    if porcentagem >= 83.2:
                        return [220, 20, 60, int(255 * transparencia)]
                    elif porcentagem >= 66.6:
                        return [255, 165, 0, int(255 * transparencia)]
                    else:
                        return [255, 255, 0, int(255 * transparencia)]

            df_h3_riqueza['fill_color'] = df_h3_riqueza['riqueza_especies'].apply(
                lambda x: get_color_areas_prioritarias(x, max_riqueza_total)
            )

            def get_categoria_prioridade(riqueza, max_riqueza):
                porcentagem = (riqueza / max_riqueza) * 100
                if porcentagem >= 83.2:
                    return f"Alta prioridade (≥83.2%): {porcentagem:.1f}%"
                elif porcentagem >= 66.6:
                    return f"Média prioridade (66.6-83.2%): {porcentagem:.1f}%"
                else:
                    return f"Baixa prioridade (50-66.6%): {porcentagem:.1f}%"

            df_h3_riqueza['categoria_prioridade'] = df_h3_riqueza['riqueza_especies'].apply(
                lambda x: get_categoria_prioridade(x, max_riqueza_total)
            )

            total_hexagonos = len(df_h3_riqueza)
            alta_prioridade = len(df_h3_riqueza[df_h3_riqueza['riqueza_especies'] >= max_riqueza_total * 0.832])
            media_prioridade = len(df_h3_riqueza[(df_h3_riqueza['riqueza_especies'] >= max_riqueza_total * 0.666) &
                                                 (df_h3_riqueza['riqueza_especies'] < max_riqueza_total * 0.832)])
            baixa_prioridade = len(df_h3_riqueza[(df_h3_riqueza['riqueza_especies'] >= max_riqueza_total * 0.5) &
                                                 (df_h3_riqueza['riqueza_especies'] < max_riqueza_total * 0.666)])

            # Texto específico baseado no método
            if metodo_prioritario == "Espécies ameaçadas":
                metrica_nome = "espécies ameaçadas"
                contexto = "Hexágonos com maior concentração de espécies ameaçadas"
            elif metodo_prioritario == "Espécies migratórias":
                metrica_nome = "espécies migratórias"
                contexto = "Hexágonos com maior concentração de espécies migratórias"
            else:
                metrica_nome = "espécies"
                contexto = "Hexágonos com maior riqueza total de espécies"

            st.info(f"""
            **Áreas Prioritárias Identificadas - {metrica_nome.title()}:**
            - {contexto}
            - Alta prioridade: {alta_prioridade} hexágonos
            - Média prioridade: {media_prioridade} hexágonos  
            - Baixa prioridade: {baixa_prioridade} hexágonos
            - **Total:** {total_hexagonos} hexágonos prioritários
            - **Valor máximo:** {max_riqueza_total} {metrica_nome}
            """)
        else:
            max_riqueza = df_h3_riqueza['riqueza_especies'].max() if len(df_h3_riqueza) > 0 else 1

            def get_color_by_richness(riqueza, max_riqueza):
                normalized = min(riqueza / max_riqueza, 1.0)
                if normalized <= 0.2:
                    return [255, 255, 178, int(255 * transparencia)]
                elif normalized <= 0.4:
                    return [254, 204, 92, int(255 * transparencia)]
                elif normalized <= 0.6:
                    return [253, 141, 60, int(255 * transparencia)]
                elif normalized <= 0.8:
                    return [240, 59, 32, int(255 * transparencia)]
                else:
                    return [189, 0, 38, int(255 * transparencia)]

            df_h3_riqueza['fill_color'] = df_h3_riqueza['riqueza_especies'].apply(
                lambda x: get_color_by_richness(x, max_riqueza)
            )

        # NOVO: Aplicar destaque visual - sempre bordas brancas normais
        df_h3_riqueza['line_color'] = [[0, 0, 0, 0]] * len(df_h3_riqueza)  # Transparente padrão
        df_h3_riqueza['line_width'] = [0] * len(df_h3_riqueza)  # Largura padrão

        # Camada principal com todos os hexágonos
        layers.append(pdk.Layer(
            'PolygonLayer',
            data=df_h3_riqueza,
            get_polygon='polygon',
            get_fill_color='fill_color',
            get_line_color='line_color',
            get_line_width='line_width',
            line_width_scale=1,
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=False,
            get_elevation=0,
        ))

        # NOVA CAMADA: Hexágono destacado por cima
        if hexagono_destacado:
            hexagono_destaque = df_h3_riqueza[df_h3_riqueza['hexagono_id'] == hexagono_destacado].copy()
            if not hexagono_destaque.empty:
                # Configurar destaque visual
                hexagono_destaque['line_color_destaque'] = [[255, 255, 0, 255]] * len(hexagono_destaque)  # Amarelo neon
                hexagono_destaque['line_width_destaque'] = [50] * len(hexagono_destaque)  # Borda grossa
                hexagono_destaque['fill_color_destaque'] = hexagono_destaque['fill_color']  # Mesma cor de preenchimento

                layers.append(pdk.Layer(
                    'PolygonLayer',
                    data=hexagono_destaque,
                    get_polygon='polygon',
                    get_fill_color='fill_color_destaque',
                    get_line_color='line_color_destaque',
                    get_line_width='line_width_destaque',
                    line_width_scale=1,
                    line_width_min_pixels=1,
                    pickable=True,
                    auto_highlight=False,
                    get_elevation=1,  # Ligeiramente elevado para ficar por cima
                ))

        # Tooltips específicos para cada método
        if usar_areas_prioritarias:
            if metodo_prioritario == "Espécies ameaçadas":
                tooltip = {
                    "html": """
                    <div style="padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px; max-width: 350px;">
                      <b>{hexagono_id}</b> - Espécies Ameaçadas<br/>
                      <b>{categoria_prioridade}</b><br/>
                      <b>Espécies ameaçadas:</b> {riqueza_especies}<br/>
                      <b>Total de registros:</b> {total_registros}<br/>
                      <b>Período:</b> {periodo}<br/>
                      <b>Municípios:</b> {municipios}<br/>
                      <b>Área:</b> {area_info}<br/>
                      <b>ID H3:</b> {h3_index}
                    </div>
                    """,
                    "style": {"backgroundColor": "darkred", "color": "white"}
                }
            elif metodo_prioritario == "Espécies migratórias":
                tooltip = {
                    "html": """
                    <div style="padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px; max-width: 350px;">
                      <b>{hexagono_id}</b> - Espécies Migratórias<br/>
                      <b>{categoria_prioridade}</b><br/>
                      <b>Espécies migratórias:</b> {riqueza_especies}<br/>
                      <b>Total de registros:</b> {total_registros}<br/>
                      <b>Período:</b> {periodo}<br/>
                      <b>Municípios:</b> {municipios}<br/>
                      <b>Área:</b> {area_info}<br/>
                      <b>ID H3:</b> {h3_index}
                    </div>
                    """,
                    "style": {"backgroundColor": "darkblue", "color": "white"}
                }
            else:  # Total de espécies
                tooltip = {
                    "html": """
                    <div style="padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px; max-width: 350px;">
                      <b>{hexagono_id}</b><br/>
                      <b>{categoria_prioridade}</b><br/>
                      <b>Riqueza de espécies:</b> {riqueza_especies}<br/>
                      <b>Total de registros:</b> {total_registros}<br/>
                      <b>Período:</b> {periodo}<br/>
                      <b>Municípios:</b> {municipios}<br/>
                      <b>Área:</b> {area_info}<br/>
                      <b>ID H3:</b> {h3_index}
                    </div>
                    """,
                    "style": {"backgroundColor": "steelblue", "color": "white"}
                }
        else:
            tooltip = {
                "html": """
                <div style="padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px; max-width: 350px;">
                  <b>{hexagono_id}</b><br/>
                  <b>Riqueza de espécies:</b> {riqueza_especies}<br/>
                  <b>Total de registros:</b> {total_registros}<br/>
                  <b>Período:</b> {periodo}<br/>
                  <b>Municípios:</b> {municipios}<br/>
                  <b>Espécies (científico):</b> {especies_cientificas}<br/>
                  <b>Área:</b> {area_info}<br/>
                  <b>ID H3:</b> {h3_index}
                </div>
                """,
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }

    # Adicionar limites municipais MELHORADOS
    if limites_geojson and tipo_mapa != "Mapa por municípios":
        layers = adicionar_limites_municipais_completos(layers, limites_geojson)

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=estilo_url,
        map_provider="mapbox",
        tooltip=tooltip
    )

    return r, dados_hexagonos


# NOVA FUNÇÃO: Consultar hexágono por ID
def consultar_hexagono_por_id(dados_hexagonos, hexagono_id):
    """Consulta informações detalhadas de um hexágono específico pelo ID"""
    if dados_hexagonos is None or dados_hexagonos.empty:
        return None

    hexagono_id = hexagono_id.upper().strip()
    hexagono = dados_hexagonos[dados_hexagonos['hexagono_id'] == hexagono_id]

    if hexagono.empty:
        return None

    return hexagono.iloc[0]


# Função para gerar gráfico temporal
def gerar_grafico_temporal(df, tipo_grafico="Número de registros"):
    """Gera gráfico temporal de aves"""
    if df.empty or 'eventDate' not in df.columns:
        return None

    hoje = datetime.now()
    data_limite = hoje - timedelta(days=5 * 365)
    df_temporal = df[df['eventDate'] >= data_limite].copy()

    if df_temporal.empty:
        return None

    if tipo_grafico == "Número de registros":
        contagem_anual = df_temporal.groupby(df_temporal['eventDate'].dt.year).size().reset_index()
        contagem_anual.columns = ['Ano', 'Quantidade']
        y_label = 'Número de Registros'
        title = 'Número de Registros por Ano - Últimos 5 Anos'
    else:
        contagem_anual = df_temporal.groupby(df_temporal['eventDate'].dt.year)['vernacularName'].nunique().reset_index()
        contagem_anual.columns = ['Ano', 'Quantidade']
        y_label = 'Número de Espécies'
        title = 'Riqueza de Espécies por Ano - Últimos 5 Anos'

    fig = px.bar(
        contagem_anual,
        x='Ano',
        y='Quantidade',
        title=title,
        color='Quantidade',
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        xaxis_title='Ano',
        yaxis_title=y_label,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig


# ==================== FUNÇÕES DE EXPORTAÇÃO KMZ ATUALIZADAS ====================

def adicionar_estilos_kml(document, transparencia):
    """Adiciona estilos para os diferentes elementos incluindo novos métodos de priorização"""

    # Estilo para hexágonos - diferentes intensidades de riqueza
    cores_riqueza = [
        ("baixa", "50ffff32", int(255 * transparencia)),  # Amarelo claro
        ("media_baixa", "50ffdc5c", int(255 * transparencia)),  # Amarelo
        ("media", "508d5cfd", int(255 * transparencia)),  # Laranja
        ("media_alta", "50323cef", int(255 * transparencia)),  # Vermelho claro
        ("alta", "501e14bd", int(255 * transparencia))  # Vermelho escuro
    ]

    for nome, cor_fill, alpha in cores_riqueza:
        style = ET.SubElement(document, "Style", id=f"hex_{nome}")
        poly_style = ET.SubElement(style, "PolyStyle")
        color = ET.SubElement(poly_style, "color")
        color.text = cor_fill
        fill = ET.SubElement(poly_style, "fill")
        fill.text = "1"
        outline = ET.SubElement(poly_style, "outline")
        outline.text = "1"

        line_style = ET.SubElement(style, "LineStyle")
        line_color = ET.SubElement(line_style, "color")
        line_color.text = "ff000000"  # Preto
        width = ET.SubElement(line_style, "width")
        width.text = "1"

    # Estilos para municípios
    cores_municipios = [
        ("mun_baixa", "50ffff32"),  # Amarelo claro
        ("mun_media_baixa", "50ffdc5c"),  # Amarelo
        ("mun_media", "508d5cfd"),  # Laranja
        ("mun_media_alta", "50323cef"),  # Vermelho claro
        ("mun_alta", "501e14bd")  # Vermelho escuro
    ]

    for nome, cor_fill in cores_municipios:
        style_mun = ET.SubElement(document, "Style", id=nome)
        poly_style_mun = ET.SubElement(style_mun, "PolyStyle")
        color_mun = ET.SubElement(poly_style_mun, "color")
        color_mun.text = cor_fill
        fill_mun = ET.SubElement(poly_style_mun, "fill")
        fill_mun.text = "1"
        outline_mun = ET.SubElement(poly_style_mun, "outline")
        outline_mun.text = "1"

        line_style_mun = ET.SubElement(style_mun, "LineStyle")
        line_color_mun = ET.SubElement(line_style_mun, "color")
        line_color_mun.text = "ff000000"  # Preto
        width_mun = ET.SubElement(line_style_mun, "width")
        width_mun.text = "2"

    # Estilos para áreas prioritárias - RIQUEZA TOTAL
    estilos_prioridade_total = [
        ("prioridade_total_alta", "99143cdc"),  # Vermelho intenso
        ("prioridade_total_media", "9900a5ff"),  # Laranja
        ("prioridade_total_baixa", "9900ffff")  # Amarelo
    ]

    for nome, cor_fill in estilos_prioridade_total:
        style_prior = ET.SubElement(document, "Style", id=nome)
        poly_style_prior = ET.SubElement(style_prior, "PolyStyle")
        color_prior = ET.SubElement(poly_style_prior, "color")
        color_prior.text = cor_fill
        fill_prior = ET.SubElement(poly_style_prior, "fill")
        fill_prior.text = "1"
        outline_prior = ET.SubElement(poly_style_prior, "outline")
        outline_prior.text = "1"

        line_style_prior = ET.SubElement(style_prior, "LineStyle")
        line_color_prior = ET.SubElement(line_style_prior, "color")
        line_color_prior.text = "ccffffff"  # Branco
        width_prior = ET.SubElement(line_style_prior, "width")
        width_prior.text = "1"

    # NOVOS ESTILOS: Áreas prioritárias - ESPÉCIES AMEAÇADAS (tons de vermelho)
    estilos_prioridade_ameacadas = [
        ("prioridade_ameacadas_alta", "998b0000"),  # Vermelho escuro intenso
        ("prioridade_ameacadas_media", "99dc143c"),  # Crimson
        ("prioridade_ameacadas_baixa", "99ff4500")  # Laranja avermelhado
    ]

    for nome, cor_fill in estilos_prioridade_ameacadas:
        style_prior = ET.SubElement(document, "Style", id=nome)
        poly_style_prior = ET.SubElement(style_prior, "PolyStyle")
        color_prior = ET.SubElement(poly_style_prior, "color")
        color_prior.text = cor_fill
        fill_prior = ET.SubElement(poly_style_prior, "fill")
        fill_prior.text = "1"
        outline_prior = ET.SubElement(poly_style_prior, "outline")
        outline_prior.text = "1"

        line_style_prior = ET.SubElement(style_prior, "LineStyle")
        line_color_prior = ET.SubElement(line_style_prior, "color")
        line_color_prior.text = "ccffffff"  # Branco
        width_prior = ET.SubElement(line_style_prior, "width")
        width_prior.text = "1"

    # NOVOS ESTILOS: Áreas prioritárias - ESPÉCIES MIGRATÓRIAS (tons de azul)
    estilos_prioridade_migratorias = [
        ("prioridade_migratorias_alta", "9900008b"),  # Azul escuro intenso
        ("prioridade_migratorias_media", "991e90ff"),  # Azul dodger
        ("prioridade_migratorias_baixa", "9987cefa")  # Azul céu claro
    ]

    for nome, cor_fill in estilos_prioridade_migratorias:
        style_prior = ET.SubElement(document, "Style", id=nome)
        poly_style_prior = ET.SubElement(style_prior, "PolyStyle")
        color_prior = ET.SubElement(poly_style_prior, "color")
        color_prior.text = cor_fill
        fill_prior = ET.SubElement(poly_style_prior, "fill")
        fill_prior.text = "1"
        outline_prior = ET.SubElement(poly_style_prior, "outline")
        outline_prior.text = "1"

        line_style_prior = ET.SubElement(style_prior, "LineStyle")
        line_color_prior = ET.SubElement(line_style_prior, "color")
        line_color_prior.text = "ccffffff"  # Branco
        width_prior = ET.SubElement(line_style_prior, "width")
        width_prior.text = "1"

    # Estilo para limites administrativos
    style_limite = ET.SubElement(document, "Style", id="limite_admin")
    line_style_limite = ET.SubElement(style_limite, "LineStyle")
    line_color_limite = ET.SubElement(line_style_limite, "color")
    line_color_limite.text = "ff808080"  # Cinza
    width_limite = ET.SubElement(line_style_limite, "width")
    width_limite.text = "2"


def adicionar_hexagonos_kml(folder_principal, dados_hex, tamanho, usar_areas_prioritarias,
                            metodo_prioritario="Total de espécies"):
    """Adiciona pasta de hexágonos de um tamanho específico com suporte aos novos métodos"""

    # Nome da pasta baseado no método
    if metodo_prioritario == "Espécies ameaçadas":
        nome_pasta = f"Hexágonos {tamanho} - Ameaçadas ({len(dados_hex)} hexágonos)"
    elif metodo_prioritario == "Espécies migratórias":
        nome_pasta = f"Hexágonos {tamanho} - Migratórias ({len(dados_hex)} hexágonos)"
    else:
        nome_pasta = f"Hexágonos {tamanho} ({len(dados_hex)} hexágonos)"

    folder_hex = ET.SubElement(folder_principal, "Folder")
    folder_name = ET.SubElement(folder_hex, "name")
    folder_name.text = nome_pasta

    # Calcular riqueza máxima para normalização das cores
    max_riqueza = dados_hex['riqueza_especies'].max() if len(dados_hex) > 0 else 1

    for _, row in dados_hex.iterrows():
        placemark = ET.SubElement(folder_hex, "Placemark")

        # Nome do hexágono baseado no método
        name = ET.SubElement(placemark, "name")
        if metodo_prioritario == "Espécies ameaçadas":
            if usar_areas_prioritarias and 'categoria_prioridade' in row:
                name.text = f"{row['hexagono_id']} - {row['riqueza_especies']} esp. ameaçadas ({row['categoria_prioridade'].split(':')[0]})"
            else:
                name.text = f"{row['hexagono_id']} - {row['riqueza_especies']} esp. ameaçadas"
        elif metodo_prioritario == "Espécies migratórias":
            if usar_areas_prioritarias and 'categoria_prioridade' in row:
                name.text = f"{row['hexagono_id']} - {row['riqueza_especies']} esp. migratórias ({row['categoria_prioridade'].split(':')[0]})"
            else:
                name.text = f"{row['hexagono_id']} - {row['riqueza_especies']} esp. migratórias"
        else:  # Total de espécies
            if usar_areas_prioritarias and 'categoria_prioridade' in row:
                name.text = f"{row['hexagono_id']} - {row['riqueza_especies']} espécies ({row['categoria_prioridade'].split(':')[0]})"
            else:
                name.text = f"{row['hexagono_id']} - {row['riqueza_especies']} espécies"

        # Descrição detalhada baseada no método
        description = ET.SubElement(placemark, "description")
        especies_lista = ""
        if isinstance(row['especies_completas'], list) and len(row['especies_completas']) > 0:
            especies_lista = "<br/>".join([f"• {esp}" for esp in row['especies_completas'][:10]])
            if len(row['especies_completas']) > 10:
                especies_lista += f"<br/>... e mais {len(row['especies_completas']) - 10} espécies"

        if metodo_prioritario == "Espécies ameaçadas":
            description.text = f"""
            <![CDATA[
            <h3>{row['hexagono_id']} - Espécies Ameaçadas</h3>
            <p><b>Espécies ameaçadas:</b> {row['riqueza_especies']}</p>
            <p><b>Total de registros:</b> {row['total_registros']}</p>
            <p><b>Período:</b> {row['periodo']}</p>
            <p><b>Municípios:</b> {row['municipios']}</p>
            <p><b>Área:</b> {row['area_info']}</p>
            <p><b>ID H3:</b> {row['h3_index']}</p>
            <hr>
            <h4>Todas as Espécies Observadas:</h4>
            {especies_lista}
            ]]>
            """
        elif metodo_prioritario == "Espécies migratórias":
            description.text = f"""
            <![CDATA[
            <h3>{row['hexagono_id']} - Espécies Migratórias</h3>
            <p><b>Espécies migratórias:</b> {row['riqueza_especies']}</p>
            <p><b>Total de registros:</b> {row['total_registros']}</p>
            <p><b>Período:</b> {row['periodo']}</p>
            <p><b>Municípios:</b> {row['municipios']}</p>
            <p><b>Área:</b> {row['area_info']}</p>
            <p><b>ID H3:</b> {row['h3_index']}</p>
            <hr>
            <h4>Todas as Espécies Observadas:</h4>
            {especies_lista}
            ]]>
            """
        else:  # Total de espécies
            description.text = f"""
            <![CDATA[
            <h3>{row['hexagono_id']}</h3>
            <p><b>Riqueza de espécies:</b> {row['riqueza_especies']}</p>
            <p><b>Total de registros:</b> {row['total_registros']}</p>
            <p><b>Período:</b> {row['periodo']}</p>
            <p><b>Municípios:</b> {row['municipios']}</p>
            <p><b>Área:</b> {row['area_info']}</p>
            <p><b>ID H3:</b> {row['h3_index']}</p>
            <hr>
            <h4>Espécies Observadas:</h4>
            {especies_lista}
            ]]>
            """

        # Estilo baseado na riqueza
        style_url = ET.SubElement(placemark, "styleUrl")
        normalized_riqueza = row['riqueza_especies'] / max_riqueza
        if normalized_riqueza <= 0.2:
            style_url.text = "#hex_baixa"
        elif normalized_riqueza <= 0.4:
            style_url.text = "#hex_media_baixa"
        elif normalized_riqueza <= 0.6:
            style_url.text = "#hex_media"
        elif normalized_riqueza <= 0.8:
            style_url.text = "#hex_media_alta"
        else:
            style_url.text = "#hex_alta"

        # Geometria do polígono
        polygon = ET.SubElement(placemark, "Polygon")
        exterior = ET.SubElement(polygon, "outerBoundaryIs")
        linear_ring = ET.SubElement(exterior, "LinearRing")
        coordinates = ET.SubElement(linear_ring, "coordinates")

        # Converter coordenadas do polígono
        coords_str = ""
        for coord in row['polygon']:
            coords_str += f"{coord[0]},{coord[1]},0 "
        # Fechar o polígono
        if row['polygon']:
            coords_str += f"{row['polygon'][0][0]},{row['polygon'][0][1]},0"

        coordinates.text = coords_str.strip()


def adicionar_municipios_kml(document, dados_municipios):
    """Adiciona pasta de municípios com cores baseadas na riqueza de espécies"""

    folder_mun = ET.SubElement(document, "Folder")
    folder_name = ET.SubElement(folder_mun, "name")
    folder_name.text = f"Riqueza por Municípios ({dados_municipios['municipio'].nunique()} municípios)"

    # Agrupar por município para evitar duplicatas
    municipios_agrupados = dados_municipios.groupby('municipio').agg({
        'riqueza_especies': 'first',
        'total_registros': 'first',
        'periodo': 'first',
        'especies_cientificas': 'first',
        'polygon': list
    }).reset_index()

    # Calcular riqueza máxima para normalização das cores
    max_riqueza_municipios = municipios_agrupados['riqueza_especies'].max() if len(municipios_agrupados) > 0 else 1

    def get_style_municipio_by_richness(riqueza, max_riqueza):
        """Retorna o estilo baseado na riqueza do município"""
        normalized = min(riqueza / max_riqueza, 1.0)
        if normalized <= 0.2:
            return "#mun_baixa"
        elif normalized <= 0.4:
            return "#mun_media_baixa"
        elif normalized <= 0.6:
            return "#mun_media"
        elif normalized <= 0.8:
            return "#mun_media_alta"
        else:
            return "#mun_alta"

    for _, row in municipios_agrupados.iterrows():
        placemark = ET.SubElement(folder_mun, "Placemark")

        # Nome do município
        name = ET.SubElement(placemark, "name")
        name.text = f"{row['municipio']} - {row['riqueza_especies']} espécies"

        # Descrição
        description = ET.SubElement(placemark, "description")
        description.text = f"""
        <![CDATA[
        <h3>{row['municipio']}</h3>
        <p><b>Riqueza de espécies:</b> {row['riqueza_especies']}</p>
        <p><b>Total de registros:</b> {row['total_registros']}</p>
        <p><b>Período:</b> {row['periodo']}</p>
        <p><b>Principais espécies:</b> {row['especies_cientificas']}</p>
        <p><b>Percentual da riqueza máxima:</b> {(row['riqueza_especies'] / max_riqueza_municipios) * 100:.1f}%</p>
        ]]>
        """

        # Estilo baseado na riqueza
        style_url = ET.SubElement(placemark, "styleUrl")
        style_url.text = get_style_municipio_by_richness(row['riqueza_especies'], max_riqueza_municipios)

        # Geometria - pode ter múltiplos polígonos por município
        if len(row['polygon']) == 1:
            # Município simples
            polygon = ET.SubElement(placemark, "Polygon")
            exterior = ET.SubElement(polygon, "outerBoundaryIs")
            linear_ring = ET.SubElement(exterior, "LinearRing")
            coordinates = ET.SubElement(linear_ring, "coordinates")

            coords_str = ""
            for coord in row['polygon'][0]:
                coords_str += f"{coord[0]},{coord[1]},0 "
            if row['polygon'][0]:
                coords_str += f"{row['polygon'][0][0][0]},{row['polygon'][0][0][1]},0"

            coordinates.text = coords_str.strip()
        else:
            # Município com múltiplas partes (MultiPolygon)
            multigeometry = ET.SubElement(placemark, "MultiGeometry")
            for poly_coords in row['polygon']:
                polygon = ET.SubElement(multigeometry, "Polygon")
                exterior = ET.SubElement(polygon, "outerBoundaryIs")
                linear_ring = ET.SubElement(exterior, "LinearRing")
                coordinates = ET.SubElement(linear_ring, "coordinates")

                coords_str = ""
                for coord in poly_coords:
                    coords_str += f"{coord[0]},{coord[1]},0 "
                if poly_coords:
                    coords_str += f"{poly_coords[0][0]},{poly_coords[0][1]},0"

                coordinates.text = coords_str.strip()


def adicionar_areas_prioritarias_kml(document, dados_hex, tamanho_hex, metodo_prioritario="Total de espécies"):
    """Adiciona pasta de áreas prioritárias para um tamanho específico de hexágono com suporte aos novos métodos"""

    # Sempre processar áreas prioritárias na exportação KMZ
    # Filtrar hexágonos prioritários (>= 50% da riqueza máxima)
    max_riqueza = dados_hex['riqueza_especies'].max()
    limite_50_porcento = max_riqueza * 0.5
    hex_prioritarios = dados_hex[dados_hex['riqueza_especies'] >= limite_50_porcento].copy()

    if hex_prioritarios.empty:
        return

    # Classificar em categorias
    def get_categoria_prioridade(riqueza, max_riqueza):
        porcentagem = (riqueza / max_riqueza) * 100
        if porcentagem >= 83.2:
            return "Alta Prioridade"
        elif porcentagem >= 66.6:
            return "Média Prioridade"
        else:
            return "Baixa Prioridade"

    def get_style_prioridade(categoria, metodo):
        if metodo == "Espécies ameaçadas":
            if categoria == "Alta Prioridade":
                return "#prioridade_ameacadas_alta"
            elif categoria == "Média Prioridade":
                return "#prioridade_ameacadas_media"
            else:
                return "#prioridade_ameacadas_baixa"
        elif metodo == "Espécies migratórias":
            if categoria == "Alta Prioridade":
                return "#prioridade_migratorias_alta"
            elif categoria == "Média Prioridade":
                return "#prioridade_migratorias_media"
            else:
                return "#prioridade_migratorias_baixa"
        else:  # Total de espécies
            if categoria == "Alta Prioridade":
                return "#prioridade_total_alta"
            elif categoria == "Média Prioridade":
                return "#prioridade_total_media"
            else:
                return "#prioridade_total_baixa"

    hex_prioritarios['categoria'] = hex_prioritarios['riqueza_especies'].apply(
        lambda x: get_categoria_prioridade(x, max_riqueza)
    )

    # Encontrar ou criar pasta principal de áreas prioritárias
    pasta_principal = None
    for child in document:
        if child.tag == "Folder":
            for name_elem in child:
                if name_elem.tag == "name" and "Áreas Prioritárias" in name_elem.text:
                    pasta_principal = child
                    break
            if pasta_principal is not None:
                break

    if pasta_principal is None:
        pasta_principal = ET.SubElement(document, "Folder")
        folder_name_principal = ET.SubElement(pasta_principal, "name")
        folder_name_principal.text = "Áreas Prioritárias"

    # Criar subpasta para este tamanho de hexágono e método
    folder_tamanho = ET.SubElement(pasta_principal, "Folder")
    folder_name_tamanho = ET.SubElement(folder_tamanho, "name")

    if metodo_prioritario == "Espécies ameaçadas":
        folder_name_tamanho.text = f"Ameaçadas {tamanho_hex} ({len(hex_prioritarios)} hexágonos)"
    elif metodo_prioritario == "Espécies migratórias":
        folder_name_tamanho.text = f"Migratórias {tamanho_hex} ({len(hex_prioritarios)} hexágonos)"
    else:
        folder_name_tamanho.text = f"Riqueza Total {tamanho_hex} ({len(hex_prioritarios)} hexágonos)"

    # Criar subpastas por categoria
    for categoria in ["Alta Prioridade", "Média Prioridade", "Baixa Prioridade"]:
        hex_categoria = hex_prioritarios[hex_prioritarios['categoria'] == categoria]
        if hex_categoria.empty:
            continue

        subfolder = ET.SubElement(folder_tamanho, "Folder")
        subfolder_name = ET.SubElement(subfolder, "name")
        if categoria == "Alta Prioridade":
            subfolder_name.text = f"Alta Prioridade (≥83.2%) - {len(hex_categoria)} hexágonos"
        elif categoria == "Média Prioridade":
            subfolder_name.text = f"Média Prioridade (66.6-83.2%) - {len(hex_categoria)} hexágonos"
        else:
            subfolder_name.text = f"Baixa Prioridade (50-66.6%) - {len(hex_categoria)} hexágonos"

        for _, row in hex_categoria.iterrows():
            placemark = ET.SubElement(subfolder, "Placemark")

            name = ET.SubElement(placemark, "name")
            if metodo_prioritario == "Espécies ameaçadas":
                name.text = f"{row['hexagono_id']} - {row['riqueza_especies']} esp. ameaçadas"
            elif metodo_prioritario == "Espécies migratórias":
                name.text = f"{row['hexagono_id']} - {row['riqueza_especies']} esp. migratórias"
            else:
                name.text = f"{row['hexagono_id']} - {row['riqueza_especies']} espécies"

            description = ET.SubElement(placemark, "description")

            if metodo_prioritario == "Espécies ameaçadas":
                description.text = f"""
                <![CDATA[
                <h3>Área Prioritária {tamanho_hex}: {categoria}</h3>
                <p><b>Foco:</b> Espécies Ameaçadas</p>
                <p><b>Hexágono:</b> {row['hexagono_id']}</p>
                <p><b>Espécies ameaçadas:</b> {row['riqueza_especies']}</p>
                <p><b>Percentual:</b> {(row['riqueza_especies'] / max_riqueza) * 100:.1f}% da riqueza máxima</p>
                <p><b>Registros:</b> {row['total_registros']}</p>
                <p><b>Municípios:</b> {row['municipios']}</p>
                <p><b>Tamanho:</b> {row['area_info']}</p>
                ]]>
                """
            elif metodo_prioritario == "Espécies migratórias":
                description.text = f"""
                <![CDATA[
                <h3>Área Prioritária {tamanho_hex}: {categoria}</h3>
                <p><b>Foco:</b> Espécies Migratórias</p>
                <p><b>Hexágono:</b> {row['hexagono_id']}</p>
                <p><b>Espécies migratórias:</b> {row['riqueza_especies']}</p>
                <p><b>Percentual:</b> {(row['riqueza_especies'] / max_riqueza) * 100:.1f}% da riqueza máxima</p>
                <p><b>Registros:</b> {row['total_registros']}</p>
                <p><b>Municípios:</b> {row['municipios']}</p>
                <p><b>Tamanho:</b> {row['area_info']}</p>
                ]]>
                """
            else:
                description.text = f"""
                <![CDATA[
                <h3>Área Prioritária {tamanho_hex}: {categoria}</h3>
                <p><b>Foco:</b> Riqueza Total de Espécies</p>
                <p><b>Hexágono:</b> {row['hexagono_id']}</p>
                <p><b>Riqueza:</b> {row['riqueza_especies']} espécies</p>
                <p><b>Percentual:</b> {(row['riqueza_especies'] / max_riqueza) * 100:.1f}% da riqueza máxima</p>
                <p><b>Registros:</b> {row['total_registros']}</p>
                <p><b>Municípios:</b> {row['municipios']}</p>
                <p><b>Tamanho:</b> {row['area_info']}</p>
                ]]>
                """

            # Estilo baseado na categoria de prioridade e método
            style_url = ET.SubElement(placemark, "styleUrl")
            style_url.text = get_style_prioridade(categoria, metodo_prioritario)

            # Geometria
            polygon = ET.SubElement(placemark, "Polygon")
            exterior = ET.SubElement(polygon, "outerBoundaryIs")
            linear_ring = ET.SubElement(exterior, "LinearRing")
            coordinates = ET.SubElement(linear_ring, "coordinates")

            coords_str = ""
            for coord in row['polygon']:
                coords_str += f"{coord[0]},{coord[1]},0 "
            if row['polygon']:
                coords_str += f"{row['polygon'][0][0]},{row['polygon'][0][1]},0"

            coordinates.text = coords_str.strip()


def adicionar_limites_administrativos_kml(document, limites_geojson):
    """Adiciona limites administrativos"""

    folder_limites = ET.SubElement(document, "Folder")
    folder_name = ET.SubElement(folder_limites, "name")
    folder_name.text = "Limites Administrativos"

    subfolder = ET.SubElement(folder_limites, "Folder")
    subfolder_name = ET.SubElement(subfolder, "name")
    subfolder_name.text = "Contornos Municipais RJ"

    for feature in limites_geojson['features']:
        placemark = ET.SubElement(subfolder, "Placemark")

        # Nome do limite
        properties = feature.get('properties', {})
        nome_municipio = None
        for campo in ['NOME', 'NM_MUNICIP', 'NM_MUN', 'NAME', 'MUNICIPIO', 'nome']:
            if campo in properties:
                nome_municipio = properties[campo]
                break

        name = ET.SubElement(placemark, "name")
        name.text = nome_municipio or "Limite Municipal"

        # Estilo
        style_url = ET.SubElement(placemark, "styleUrl")
        style_url.text = "#limite_admin"

        # Geometria
        geometry = feature['geometry']
        if geometry['type'] == 'Polygon':
            for ring in geometry['coordinates']:
                polygon = ET.SubElement(placemark, "Polygon")
                exterior = ET.SubElement(polygon, "outerBoundaryIs")
                linear_ring = ET.SubElement(exterior, "LinearRing")
                coordinates = ET.SubElement(linear_ring, "coordinates")

                coords_str = " ".join([f"{coord[0]},{coord[1]},0" for coord in ring])
                coordinates.text = coords_str

        elif geometry['type'] == 'MultiPolygon':
            multigeometry = ET.SubElement(placemark, "MultiGeometry")
            for polygon_coords in geometry['coordinates']:
                for ring in polygon_coords:
                    polygon = ET.SubElement(multigeometry, "Polygon")
                    exterior = ET.SubElement(polygon, "outerBoundaryIs")
                    linear_ring = ET.SubElement(exterior, "LinearRing")
                    coordinates = ET.SubElement(linear_ring, "coordinates")

                    coords_str = " ".join([f"{coord[0]},{coord[1]},0" for coord in ring])
                    coordinates.text = coords_str


def adicionar_informacoes_dataset_kml(document, df_filtrado, periodo_selecionado,
                                      municipio_selecionado, especie_selecionada, usar_areas_prioritarias):
    """Adiciona pasta com informações do dataset"""

    folder_info = ET.SubElement(document, "Folder")
    folder_name = ET.SubElement(folder_info, "name")
    folder_name.text = "Informações do Dataset"

    # Estatísticas gerais
    placemark_stats = ET.SubElement(folder_info, "Placemark")
    name_stats = ET.SubElement(placemark_stats, "name")
    name_stats.text = "Estatísticas Gerais"

    description_stats = ET.SubElement(placemark_stats, "description")

    # Calcular estatísticas
    total_registros = len(df_filtrado)
    total_especies = df_filtrado['vernacularName'].nunique()
    total_municipios = df_filtrado['level2Name'].nunique()
    data_mais_antiga = df_filtrado['eventDate'].min().strftime('%d/%m/%Y')
    data_mais_recente = df_filtrado['eventDate'].max().strftime('%d/%m/%Y')

    # NOVAS ESTATÍSTICAS: Espécies ameaçadas e migratórias
    df_com_ameacadas = df_filtrado.copy()
    df_com_ameacadas['is_ameacada'] = df_com_ameacadas.apply(verificar_especie_ameacada, axis=1)
    df_com_ameacadas['is_migratoria'] = df_com_ameacadas.apply(verificar_especie_migratoria, axis=1)

    especies_ameacadas = df_com_ameacadas[df_com_ameacadas['is_ameacada']]['vernacularName'].nunique()
    especies_migratorias = df_com_ameacadas[df_com_ameacadas['is_migratoria']]['vernacularName'].nunique()

    description_stats.text = f"""
    <![CDATA[
    <h3>Estatísticas do Dataset</h3>
    <p><b>Total de registros:</b> {total_registros:,}</p>
    <p><b>Riqueza de espécies:</b> {total_especies}</p>
    <p><b>Espécies ameaçadas:</b> {especies_ameacadas}</p>
    <p><b>Espécies migratórias:</b> {especies_migratorias}</p>
    <p><b>Municípios com registros:</b> {total_municipios}</p>
    <p><b>Período dos dados:</b> {data_mais_antiga} a {data_mais_recente}</p>
    <hr>
    <h3>Filtros Aplicados</h3>
    <p><b>Período:</b> {periodo_selecionado}</p>
    <p><b>Município:</b> {municipio_selecionado}</p>
    <p><b>Espécie:</b> {especie_selecionada}</p>
    <p><b>Áreas prioritárias:</b> Todas incluídas (riqueza total, ameaçadas e migratórias)</p>
    <hr>
    <h3>Geração do Arquivo</h3>
    <p><b>Data/Hora:</b> {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}</p>
    <p><b>Fonte:</b> Atlas das Aves Aquáticas do Rio de Janeiro</p>
    <p><b>Versão:</b> Com análises de espécies ameaçadas e migratórias</p>
    ]]>
    """

    # Coordenadas centrais do RJ para o placemark
    point_stats = ET.SubElement(placemark_stats, "Point")
    coordinates_stats = ET.SubElement(point_stats, "coordinates")
    coordinates_stats.text = "-43.1729,-22.9068,0"  # Centro do RJ


def gerar_arquivo_kmz(kml_root, df_filtrado):
    """Gera o arquivo KMZ final"""

    # Criar diretório temporário
    with tempfile.TemporaryDirectory() as temp_dir:
        # Salvar KML
        kml_path = os.path.join(temp_dir, "doc.kml")
        tree = ET.ElementTree(kml_root)
        tree.write(kml_path, encoding='utf-8', xml_declaration=True)

        # Criar arquivo KMZ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        kmz_filename = f"atlas_aves_completo_{timestamp}.kmz"
        kmz_path = os.path.join(temp_dir, kmz_filename)

        with zipfile.ZipFile(kmz_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
            kmz.write(kml_path, "doc.kml")

        # Ler o arquivo para retornar como bytes
        with open(kmz_path, 'rb') as f:
            kmz_data = f.read()

        return kmz_data


def exportar_atlas_completo_kmz(df_filtrado, limites_geojson, usar_areas_prioritarias=False,
                                metodo_prioritario=None, transparencia=0.8,
                                periodo_selecionado="Série completa", municipio_selecionado="Todos os municípios",
                                especie_selecionada="Todas as espécies"):
    """
    Exporta atlas completo com hexágonos, municípios e áreas prioritárias em formato KMZ
    ATUALIZADO: Inclui análises de espécies ameaçadas e migratórias
    """

    if df_filtrado.empty:
        return None

    # 1. Preparar todos os datasets para os três métodos
    dados_hex_1km_total = calcular_riqueza_h3(df_filtrado, "1km", "Total de espécies")
    dados_hex_2km_total = calcular_riqueza_h3(df_filtrado, "2km", "Total de espécies")
    dados_hex_5km_total = calcular_riqueza_h3(df_filtrado, "5km", "Total de espécies")

    dados_hex_1km_ameacadas = calcular_riqueza_h3(df_filtrado, "1km", "Espécies ameaçadas")
    dados_hex_2km_ameacadas = calcular_riqueza_h3(df_filtrado, "2km", "Espécies ameaçadas")
    dados_hex_5km_ameacadas = calcular_riqueza_h3(df_filtrado, "5km", "Espécies ameaçadas")

    dados_hex_1km_migratorias = calcular_riqueza_h3(df_filtrado, "1km", "Espécies migratórias")
    dados_hex_2km_migratorias = calcular_riqueza_h3(df_filtrado, "2km", "Espécies migratórias")
    dados_hex_5km_migratorias = calcular_riqueza_h3(df_filtrado, "5km", "Espécies migratórias")

    dados_municipios = calcular_riqueza_municipio_mapa_espacial(df_filtrado, limites_geojson)

    # 2. Criar estrutura KML
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, "Document")

    # Nome do documento
    doc_name = ET.SubElement(document, "name")
    doc_name.text = "Atlas das Aves Aquáticas - Rio de Janeiro (Completo)"

    # Descrição do documento
    doc_description = ET.SubElement(document, "description")
    doc_description.text = f"""
    <![CDATA[
    <h2>Atlas das Aves Aquáticas do Rio de Janeiro</h2>
    <p><b>Versão Completa com Análises Especializadas</b></p>
    <p><b>Período:</b> {periodo_selecionado}</p>
    <p><b>Município:</b> {municipio_selecionado}</p>
    <p><b>Espécie:</b> {especie_selecionada}</p>
    <p><b>Total de registros:</b> {len(df_filtrado)}</p>
    <p><b>Riqueza de espécies:</b> {df_filtrado['vernacularName'].nunique()}</p>
    <p><b>Gerado em:</b> {datetime.now().strftime('%d/%m/%Y às %H:%M')}</p>
    <hr>
    <p><b>Conteúdo:</b> Hexágonos de biodiversidade, análises de espécies ameaçadas e migratórias, áreas prioritárias</p>
    ]]>
    """

    # 3. Adicionar estilos
    adicionar_estilos_kml(document, transparencia)

    # 4. Adicionar pasta principal de hexágonos por RIQUEZA TOTAL
    folder_hex_principal = ET.SubElement(document, "Folder")
    hex_name = ET.SubElement(folder_hex_principal, "name")
    hex_name.text = "Hexágonos - Riqueza Total de Espécies"

    # 4.1 Hexágonos riqueza total
    if not dados_hex_1km_total.empty:
        adicionar_hexagonos_kml(folder_hex_principal, dados_hex_1km_total, "1km", False, "Total de espécies")
    if not dados_hex_2km_total.empty:
        adicionar_hexagonos_kml(folder_hex_principal, dados_hex_2km_total, "2km", False, "Total de espécies")
    if not dados_hex_5km_total.empty:
        adicionar_hexagonos_kml(folder_hex_principal, dados_hex_5km_total, "5km", False, "Total de espécies")

    # 5. NOVA SEÇÃO: Adicionar pasta principal de hexágonos por ESPÉCIES AMEAÇADAS
    folder_hex_ameacadas = ET.SubElement(document, "Folder")
    hex_ameacadas_name = ET.SubElement(folder_hex_ameacadas, "name")
    hex_ameacadas_name.text = "Hexágonos - Espécies Ameaçadas"

    if not dados_hex_1km_ameacadas.empty:
        adicionar_hexagonos_kml(folder_hex_ameacadas, dados_hex_1km_ameacadas, "1km", False, "Espécies ameaçadas")
    if not dados_hex_2km_ameacadas.empty:
        adicionar_hexagonos_kml(folder_hex_ameacadas, dados_hex_2km_ameacadas, "2km", False, "Espécies ameaçadas")
    if not dados_hex_5km_ameacadas.empty:
        adicionar_hexagonos_kml(folder_hex_ameacadas, dados_hex_5km_ameacadas, "5km", False, "Espécies ameaçadas")

    # 6. NOVA SEÇÃO: Adicionar pasta principal de hexágonos por ESPÉCIES MIGRATÓRIAS
    folder_hex_migratorias = ET.SubElement(document, "Folder")
    hex_migratorias_name = ET.SubElement(folder_hex_migratorias, "name")
    hex_migratorias_name.text = "Hexágonos - Espécies Migratórias"

    if not dados_hex_1km_migratorias.empty:
        adicionar_hexagonos_kml(folder_hex_migratorias, dados_hex_1km_migratorias, "1km", False, "Espécies migratórias")
    if not dados_hex_2km_migratorias.empty:
        adicionar_hexagonos_kml(folder_hex_migratorias, dados_hex_2km_migratorias, "2km", False, "Espécies migratórias")
    if not dados_hex_5km_migratorias.empty:
        adicionar_hexagonos_kml(folder_hex_migratorias, dados_hex_5km_migratorias, "5km", False, "Espécies migratórias")

    # 7. Adicionar municípios
    if not dados_municipios.empty:
        adicionar_municipios_kml(document, dados_municipios)

    # 8. Adicionar áreas prioritárias para TODOS os métodos e tamanhos
    # 8.1 Áreas prioritárias por riqueza total
    if not dados_hex_1km_total.empty:
        adicionar_areas_prioritarias_kml(document, dados_hex_1km_total, "1km", "Total de espécies")
    if not dados_hex_2km_total.empty:
        adicionar_areas_prioritarias_kml(document, dados_hex_2km_total, "2km", "Total de espécies")
    if not dados_hex_5km_total.empty:
        adicionar_areas_prioritarias_kml(document, dados_hex_5km_total, "5km", "Total de espécies")

    # 8.2 Áreas prioritárias por espécies ameaçadas
    if not dados_hex_1km_ameacadas.empty:
        adicionar_areas_prioritarias_kml(document, dados_hex_1km_ameacadas, "1km", "Espécies ameaçadas")
    if not dados_hex_2km_ameacadas.empty:
        adicionar_areas_prioritarias_kml(document, dados_hex_2km_ameacadas, "2km", "Espécies ameaçadas")
    if not dados_hex_5km_ameacadas.empty:
        adicionar_areas_prioritarias_kml(document, dados_hex_5km_ameacadas, "5km", "Espécies ameaçadas")

    # 8.3 Áreas prioritárias por espécies migratórias
    if not dados_hex_1km_migratorias.empty:
        adicionar_areas_prioritarias_kml(document, dados_hex_1km_migratorias, "1km", "Espécies migratórias")
    if not dados_hex_2km_migratorias.empty:
        adicionar_areas_prioritarias_kml(document, dados_hex_2km_migratorias, "2km", "Espécies migratórias")
    if not dados_hex_5km_migratorias.empty:
        adicionar_areas_prioritarias_kml(document, dados_hex_5km_migratorias, "5km", "Espécies migratórias")

    # 9. Adicionar limites administrativos
    if limites_geojson:
        adicionar_limites_administrativos_kml(document, limites_geojson)

    # 10. Adicionar informações do dataset
    adicionar_informacoes_dataset_kml(document, df_filtrado, periodo_selecionado,
                                      municipio_selecionado, especie_selecionada, True)

    # 11. Gerar arquivo KMZ
    return gerar_arquivo_kmz(kml, df_filtrado)


# ==================== INTERFACE PRINCIPAL ATUALIZADA ====================

# Interface principal
def main():
    st.title("Atlas das Aves Aquáticas do Rio de Janeiro")
    st.markdown("---")

    with st.spinner("Carregando dados de observações de aves e limites municipais..."):
        df = carregar_dados_aves()
        limites_geojson = carregar_limites_rj()

    if df.empty:
        st.error("Não foi possível carregar os dados. Verifique sua conexão de internet.")
        return

    # Sidebar - Filtros
    with st.sidebar:
        st.title("Filtros de Análise")
        st.markdown("---")

        st.markdown("### Período de Análise")
        periodo_opcoes = ["Série completa", "Último mês", "Últimos 15 dias", "Período personalizado"]
        periodo_selecionado = st.selectbox("Selecione o período:", periodo_opcoes, index=0)

        data_inicio = None
        data_fim = None
        if periodo_selecionado == "Período personalizado":
            col1, col2 = st.columns(2)
            with col1:
                data_inicio = st.date_input("Data inicial:")
            with col2:
                data_fim = st.date_input("Data final:")

        st.markdown("### Município")
        municipios_disponíveis = ["Todos os municípios"] + sorted(df['level2Name'].unique().tolist())
        municipio_selecionado = st.selectbox("Selecione o município:", municipios_disponíveis, index=0)

        st.markdown("### Espécie")
        especies_disponíveis = ["Todas as espécies"] + sorted(df['species'].unique().tolist())
        especie_selecionada = st.selectbox("Selecione a espécie:", especies_disponíveis, index=0)

        st.markdown("---")

        st.markdown("### Análise de Áreas Prioritárias")
        usar_areas_prioritarias = st.checkbox("Ativar análise de áreas prioritárias")

        metodo_prioritario = None
        if usar_areas_prioritarias:
            metodo_prioritario = st.selectbox(
                "Método de priorização:",
                ["Total de espécies", "Espécies ameaçadas", "Espécies migratórias"],
                index=0,
                help="Escolha o critério para identificar áreas prioritárias"
            )

            if metodo_prioritario == "Espécies ameaçadas":
                st.info("🚨 Analisando hexágonos com maior concentração de espécies ameaçadas (IUCN, MMA, RJ)")
            elif metodo_prioritario == "Espécies migratórias":
                st.info("🦅 Analisando hexágonos com maior concentração de espécies migratórias (CBRO, Somenzari)")
            else:
                st.info("📊 Analisando hexágonos com maior riqueza total de espécies")

        st.markdown("---")

        # Aplicar filtros
        df_filtrado = df.copy()
        df_filtrado = filtrar_por_periodo(df_filtrado, periodo_selecionado, data_inicio, data_fim)
        df_filtrado = filtrar_por_municipio(df_filtrado, municipio_selecionado)
        df_filtrado = filtrar_por_especie(df_filtrado, especie_selecionada)

        # Exibir estatísticas ATUALIZADAS
        total_registros = len(df_filtrado)
        total_especies = df_filtrado['vernacularName'].nunique() if not df_filtrado.empty else 0

        if not df_filtrado.empty:
            data_mais_antiga = df_filtrado['eventDate'].min().strftime('%d/%m/%Y')
            data_mais_recente = df_filtrado['eventDate'].max().strftime('%d/%m/%Y')
            municipios_com_registros = df_filtrado['level2Name'].nunique()

            # NOVAS MÉTRICAS: Espécies ameaçadas e migratórias
            df_temp = df_filtrado.copy()
            df_temp['is_ameacada'] = df_temp.apply(verificar_especie_ameacada, axis=1)
            df_temp['is_migratoria'] = df_temp.apply(verificar_especie_migratoria, axis=1)

            especies_ameacadas = df_temp[df_temp['is_ameacada']]['vernacularName'].nunique()
            especies_migratorias = df_temp[df_temp['is_migratoria']]['vernacularName'].nunique()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total de Registros", total_registros)
                st.metric("Riqueza de Espécies", total_especies)
                st.metric("Municípios", municipios_com_registros)

            with col2:
                st.metric("Esp. Ameaçadas", especies_ameacadas, help="Espécies em alguma categoria de ameaça")
                st.metric("Esp. Migratórias", especies_migratorias, help="Espécies com comportamento migratório")

            st.text(f"Período: {data_mais_antiga} a {data_mais_recente}")

            st.markdown("---")
            st.info(
                "🔬 **Versão Atualizada:** Inclui análises de espécies ameaçadas (IUCN, MMA, RJ) e migratórias (CBRO, Somenzari)")
        else:
            st.warning("Nenhum registro encontrado com os filtros aplicados.")

    # Layout principal - Controles do mapa
    map_col1, map_col2, map_col3 = st.columns(3)

    with map_col1:
        if usar_areas_prioritarias:
            tipo_mapa = "Mapa de hexágono"
            st.selectbox(
                "Tipo de Mapa:",
                ["Mapa de hexágono"],
                index=0,
                disabled=True,
                help="Tipo de mapa fixado para análise de áreas prioritárias"
            )
        else:
            tipo_mapa = st.selectbox(
                "Tipo de Mapa:",
                ["Mapa de hexágono", "Mapa por municípios", "Mapa de calor", "Mapa de pontos"],
                index=0
            )

    with map_col2:
        estilo_mapa = st.selectbox(
            "Estilo do Mapa:",
            ["Satélite", "Claro", "Escuro"]
        )

    with map_col3:
        if tipo_mapa in ["Mapa de hexágono", "Mapa por municípios"]:
            transparencia = st.slider(
                "Transparência:",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="Ajuste a transparência dos polígonos"
            )

            if tipo_mapa == "Mapa de hexágono":
                tamanho_hex = st.selectbox(
                    "Tamanho dos hexágonos:",
                    options=["1km", "2km", "5km"],
                    index=0,
                    help="Selecione o tamanho dos hexágonos"
                )
            else:
                tamanho_hex = "1km"
        else:
            transparencia = 0.8
            tamanho_hex = "1km"

    st.markdown("### Visualização do Mapa")

    # Gerar e exibir mapa
    mapa_pydeck = None
    dados_hexagonos = None
    hexagono_destacado = st.session_state.get('hexagono_consultado_id', None)

    if not df_filtrado.empty:
        mapa_pydeck, dados_hexagonos = gerar_mapa_aves(df_filtrado, tipo_mapa, estilo_mapa, transparencia, tamanho_hex,
                                                       limites_geojson,
                                                       usar_areas_prioritarias, metodo_prioritario, hexagono_destacado)
        if mapa_pydeck:
            st.pydeck_chart(mapa_pydeck)
        else:
            st.warning("Não foi possível gerar o mapa com os dados filtrados.")
    else:
        st.warning("Nenhum dado disponível para exibir no mapa.")

    # Seção de exportação ATUALIZADA
    st.markdown("---")
    st.markdown("### Exportação de Dados")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Exportar Mapa Atual (HTML)", help="Salva apenas o mapa visualizado"):
            if not df_filtrado.empty and mapa_pydeck:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"mapa_aves_rj_{timestamp}.html"
                    html_content = mapa_pydeck.to_html(as_string=True)

                    st.download_button(
                        label="Baixar Arquivo HTML",
                        data=html_content,
                        file_name=filename,
                        mime="text/html",
                        help="Clique para baixar o mapa como arquivo HTML interativo"
                    )

                    st.success("Mapa HTML gerado! Clique no botão acima para baixar.")
                    st.info("""
                    **Como usar o arquivo HTML:**
                    1. Baixe o arquivo HTML
                    2. Abra com qualquer navegador web
                    3. O mapa será totalmente interativo
                    4. Funciona offline!
                    """)

                except Exception as e:
                    st.error(f"Erro ao gerar HTML: {e}")
            else:
                st.warning("Nenhum mapa disponível para exportação.")

    with col2:
        if st.button("Exportar Atlas Completo (KMZ)",
                     help="Todas as análises: riqueza total, espécies ameaçadas e migratórias"):
            if not df_filtrado.empty:
                with st.spinner("Gerando atlas completo com todas as análises... Isso pode levar alguns minutos."):
                    try:
                        kmz_data = exportar_atlas_completo_kmz(
                            df_filtrado,
                            limites_geojson,
                            usar_areas_prioritarias,
                            metodo_prioritario,
                            transparencia,
                            periodo_selecionado,
                            municipio_selecionado,
                            especie_selecionada
                        )

                        if kmz_data:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"atlas_aves_completo_{timestamp}.kmz"

                            st.download_button(
                                label="Baixar Atlas KMZ Completo",
                                data=kmz_data,
                                file_name=filename,
                                mime="application/vnd.google-earth.kmz",
                                help="Arquivo completo com todas as análises especializadas"
                            )

                            st.success("Atlas KMZ Completo gerado com sucesso!")

                            # Informações sobre o conteúdo ATUALIZADO
                            hex_1km_total = calcular_riqueza_h3(df_filtrado, "1km", "Total de espécies")
                            hex_1km_ameacadas = calcular_riqueza_h3(df_filtrado, "1km", "Espécies ameaçadas")
                            hex_1km_migratorias = calcular_riqueza_h3(df_filtrado, "1km", "Espécies migratórias")

                            info_content = f"""
                            **Conteúdo do Atlas KMZ Completo:**

                            **Hexágonos por Riqueza Total:**
                            - Hexágonos 1km: {len(hex_1km_total)} hexágonos
                            - Hexágonos 2km: {len(calcular_riqueza_h3(df_filtrado, "2km", "Total de espécies"))} hexágonos  
                            - Hexágonos 5km: {len(calcular_riqueza_h3(df_filtrado, "5km", "Total de espécies"))} hexágonos

                            **Hexágonos por Espécies Ameaçadas:**
                            - Hexágonos 1km: {len(hex_1km_ameacadas)} hexágonos
                            - Hexágonos 2km: {len(calcular_riqueza_h3(df_filtrado, "2km", "Espécies ameaçadas"))} hexágonos
                            - Hexágonos 5km: {len(calcular_riqueza_h3(df_filtrado, "5km", "Espécies ameaçadas"))} hexágonos

                            **Hexágonos por Espécies Migratórias:**
                            - Hexágonos 1km: {len(hex_1km_migratorias)} hexágonos
                            - Hexágonos 2km: {len(calcular_riqueza_h3(df_filtrado, "2km", "Espécies migratórias"))} hexágonos
                            - Hexágonos 5km: {len(calcular_riqueza_h3(df_filtrado, "5km", "Espécies migratórias"))} hexágonos

                            **Outros Componentes:**
                            - Municípios: {df_filtrado['level2Name'].nunique()} municípios
                            - Limites administrativos incluídos
                            - Todas as áreas prioritárias incluídas para cada método
                            """

                            st.info(info_content)

                            st.info("""
                            **🆕 Novidades da Versão Completa:**
                            - ✅ Análise de espécies ameaçadas (IUCN 2021, MMA 2022, RJ 2001)
                            - ✅ Análise de espécies migratórias (CBRO 2021, Somenzari 2017)
                            - ✅ Áreas prioritárias específicas para cada tipo de análise
                            - ✅ Cores distintivas para cada categoria (vermelho=ameaçadas, azul=migratórias)
                            - ✅ Funciona completamente offline no Google Earth, QGIS ou GPS
                            """)

                        else:
                            st.error("Erro ao gerar arquivo KMZ.")

                    except Exception as e:
                        st.error(f"Erro ao gerar KMZ: {e}")
                        st.error("Verifique se todos os dados estão carregados corretamente.")
            else:
                st.warning("Nenhum dado disponível para exportação.")

    # SEÇÃO: Consulta de Hexágonos ATUALIZADA
    if dados_hexagonos is not None and not dados_hexagonos.empty and tipo_mapa == "Mapa de hexágono":
        st.markdown("---")
        st.markdown("### Consulta de Hexágonos")

        col1, col2 = st.columns([1, 3])

        with col1:
            hexagono_input = st.text_input(
                "Digite o ID do hexágono:",
                placeholder="Ex: H0001, A0001, M0001",
                help="Digite o ID do hexágono que aparece no tooltip do mapa"
            )

            if st.button("Consultar"):
                if hexagono_input:
                    hexagono_info = consultar_hexagono_por_id(dados_hexagonos, hexagono_input)
                    if hexagono_info is not None:
                        st.session_state.hexagono_consultado = hexagono_info
                        st.session_state.hexagono_consultado_id = hexagono_input.upper().strip()
                        st.rerun()  # Recarregar para atualizar o mapa
                    else:
                        st.error(f"Hexágono {hexagono_input} não encontrado.")
                        if 'hexagono_consultado' in st.session_state:
                            del st.session_state.hexagono_consultado
                        if 'hexagono_consultado_id' in st.session_state:
                            del st.session_state.hexagono_consultado_id
                else:
                    st.warning("Digite um ID de hexágono válido.")

        with col2:
            if 'hexagono_consultado' in st.session_state:
                hex_info = st.session_state.hexagono_consultado

                # Botão para limpar destaque
                if st.button("Limpar Destaque"):
                    if 'hexagono_consultado' in st.session_state:
                        del st.session_state.hexagono_consultado
                    if 'hexagono_consultado_id' in st.session_state:
                        del st.session_state.hexagono_consultado_id
                    st.rerun()

                # Informações básicas em colunas
                info_col1, info_col2, info_col3 = st.columns(3)

                with info_col1:
                    st.metric("ID", hex_info['hexagono_id'])

                    # Mostrar métrica específica baseada no método
                    if metodo_prioritario == "Espécies ameaçadas":
                        st.metric("Esp. Ameaçadas", f"{hex_info['riqueza_especies']}")
                    elif metodo_prioritario == "Espécies migratórias":
                        st.metric("Esp. Migratórias", f"{hex_info['riqueza_especies']}")
                    else:
                        st.metric("Riqueza", f"{hex_info['riqueza_especies']} espécies")

                with info_col2:
                    st.metric("Registros", hex_info['total_registros'])
                    st.metric("Área", hex_info['area_info'])

                with info_col3:
                    st.metric("Período", hex_info['periodo'])
                    st.metric("Municípios", hex_info['municipios'])

                # Lista completa de espécies
                st.markdown("#### Lista Completa de Espécies")

                especies_tab1, especies_tab2 = st.tabs(["Nomes Populares", "Nomes Científicos"])

                with especies_tab1:
                    especies_populares = hex_info['especies_completas']
                    if isinstance(especies_populares, list):
                        for i, especie in enumerate(especies_populares, 1):
                            st.write(f"{i:2d}. {especie}")
                    else:
                        st.write("Nenhuma espécie encontrada")

                with especies_tab2:
                    especies_cientificas = hex_info['especies_cientificas_completas']
                    if isinstance(especies_cientificas, list):
                        for i, especie in enumerate(especies_cientificas, 1):
                            st.write(f"{i:2d}. *{especie}*")
                    else:
                        st.write("Nenhuma espécie encontrada")

                # Informações técnicas
                with st.expander("Informações Técnicas"):
                    metodo_descricao = ""
                    if metodo_prioritario == "Espécies ameaçadas":
                        metodo_descricao = "\nMétodo: Espécies ameaçadas (IUCN, MMA, RJ)"
                    elif metodo_prioritario == "Espécies migratórias":
                        metodo_descricao = "\nMétodo: Espécies migratórias (CBRO, Somenzari)"
                    else:
                        metodo_descricao = "\nMétodo: Riqueza total de espécies"

                    st.code(f"""
ID H3: {hex_info['h3_index']}
Coordenadas: {hex_info['latitude']:.6f}, {hex_info['longitude']:.6f}
Área aproximada: {hex_info['area_info']}
Total de espécies: {len(hex_info['especies_completas']) if isinstance(hex_info['especies_completas'], list) else 0}
Total de registros: {hex_info['total_registros']}{metodo_descricao}
                    """)
            else:
                metodo_atual = metodo_prioritario if usar_areas_prioritarias else "Riqueza total"
                st.info(
                    f"Digite um ID de hexágono acima para ver informações detalhadas.\n\n**Método atual:** {metodo_atual}")

        # Lista de todos os hexágonos disponíveis
        with st.expander("Lista de Todos os Hexágonos"):
            df_lista = dados_hexagonos[['hexagono_id', 'riqueza_especies', 'total_registros', 'municipios']].copy()

            # Renomear coluna baseado no método
            if metodo_prioritario == "Espécies ameaçadas":
                df_lista.columns = ['ID', 'Esp. Ameaçadas', 'Registros', 'Municípios']
            elif metodo_prioritario == "Espécies migratórias":
                df_lista.columns = ['ID', 'Esp. Migratórias', 'Registros', 'Municípios']
            else:
                df_lista.columns = ['ID', 'Riqueza', 'Registros', 'Municípios']

            st.dataframe(df_lista, use_container_width=True)

    # Análises complementares
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Análise Temporal")

        tipo_grafico = st.selectbox(
            "Tipo de análise temporal:",
            ["Número de registros", "Número de espécies"],
            index=0
        )

        fig_temporal = gerar_grafico_temporal(df_filtrado, tipo_grafico)
        if fig_temporal:
            st.plotly_chart(fig_temporal, use_container_width=True)
        else:
            st.warning("Não foi possível gerar o gráfico temporal.")

    with col2:
        st.subheader("Top 10 Municípios")

        if not df_filtrado.empty:
            riqueza_municipios = calcular_riqueza_municipio(df_filtrado)
            if not riqueza_municipios.empty:
                top_10 = riqueza_municipios.head(10)

                st.markdown("**Por Riqueza de Espécies:**")
                for _, row in top_10.iterrows():
                    st.text(f"{row['municipio']}: {row['riqueza_especies']} espécies")
            else:
                st.text("Nenhum dado disponível")
        else:
            st.text("Nenhum dado disponível")

    # Seção de espécies mais observadas ATUALIZADA
    st.markdown("---")
    st.subheader("Espécies Mais Observadas")

    if not df_filtrado.empty:
        # Tabs para diferentes visualizações
        tab1, tab2, tab3 = st.tabs(["📊 Mais Observadas", "🚨 Ameaçadas", "🦅 Migratórias"])

        with tab1:
            especies_observadas = df_filtrado['vernacularName'].value_counts().head(10).reset_index()
            especies_observadas.columns = ['Espécie', 'Registros']

            fig_especies = px.bar(
                especies_observadas,
                x='Registros',
                y='Espécie',
                orientation='h',
                title='Top 10 Espécies Mais Observadas',
                color='Registros',
                color_continuous_scale='Blues'
            )

            fig_especies.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )

            st.plotly_chart(fig_especies, use_container_width=True)

        with tab2:
            # Análise de espécies ameaçadas
            df_temp = df_filtrado.copy()
            df_temp['is_ameacada'] = df_temp.apply(verificar_especie_ameacada, axis=1)

            especies_ameacadas_obs = df_temp[df_temp['is_ameacada']]['vernacularName'].value_counts().head(10)

            if not especies_ameacadas_obs.empty:
                especies_ameacadas_df = especies_ameacadas_obs.reset_index()
                especies_ameacadas_df.columns = ['Espécie', 'Registros']

                fig_ameacadas = px.bar(
                    especies_ameacadas_df,
                    x='Registros',
                    y='Espécie',
                    orientation='h',
                    title='Top 10 Espécies Ameaçadas Mais Observadas',
                    color='Registros',
                    color_continuous_scale='Reds'
                )

                fig_ameacadas.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=400
                )

                st.plotly_chart(fig_ameacadas, use_container_width=True)
                st.info(
                    f"📊 **Total de espécies ameaçadas observadas:** {df_temp[df_temp['is_ameacada']]['vernacularName'].nunique()}")
            else:
                st.warning("Nenhuma espécie ameaçada encontrada nos dados filtrados.")

        with tab3:
            # Análise de espécies migratórias
            df_temp = df_filtrado.copy()
            df_temp['is_migratoria'] = df_temp.apply(verificar_especie_migratoria, axis=1)

            especies_migratorias_obs = df_temp[df_temp['is_migratoria']]['vernacularName'].value_counts().head(10)

            if not especies_migratorias_obs.empty:
                especies_migratorias_df = especies_migratorias_obs.reset_index()
                especies_migratorias_df.columns = ['Espécie', 'Registros']

                fig_migratorias = px.bar(
                    especies_migratorias_df,
                    x='Registros',
                    y='Espécie',
                    orientation='h',
                    title='Top 10 Espécies Migratórias Mais Observadas',
                    color='Registros',
                    color_continuous_scale='Blues'
                )

                fig_migratorias.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=400
                )

                st.plotly_chart(fig_migratorias, use_container_width=True)
                st.info(
                    f"📊 **Total de espécies migratórias observadas:** {df_temp[df_temp['is_migratoria']]['vernacularName'].nunique()}")
            else:
                st.warning("Nenhuma espécie migratória encontrada nos dados filtrados.")
    else:
        st.warning("Nenhum dado disponível para análise de espécies.")

    # Tabela de dados detalhados ATUALIZADA
    if st.expander("Dados Detalhados"):
        if not df_filtrado.empty:
            df_exibir = df_filtrado[
                ['eventDate', 'vernacularName', 'species', 'level2Name', 'locality', 'decimalLatitude',
                 'decimalLongitude']].copy()

            # Adicionar colunas de status
            df_exibir_temp = df_filtrado.copy()
            df_exibir_temp['is_ameacada'] = df_exibir_temp.apply(verificar_especie_ameacada, axis=1)
            df_exibir_temp['is_migratoria'] = df_exibir_temp.apply(verificar_especie_migratoria, axis=1)

            df_exibir['Status_Ameaca'] = df_exibir_temp['is_ameacada'].apply(lambda x: "🚨 Ameaçada" if x else "")
            df_exibir['Status_Migracao'] = df_exibir_temp['is_migratoria'].apply(lambda x: "🦅 Migratória" if x else "")

            df_exibir['eventDate'] = df_exibir['eventDate'].dt.strftime('%d/%m/%Y')
            df_exibir.columns = ['Data', 'Nome Popular', 'Nome Científico', 'Município', 'Localidade', 'Latitude',
                                 'Longitude', 'Status Ameaça', 'Status Migração']

            st.dataframe(df_exibir, use_container_width=True, height=300)

            csv = df_exibir.to_csv(index=False)
            st.download_button(
                label="Baixar dados filtrados (CSV)",
                data=csv,
                file_name=f"aves_rj_completo_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Inclui status de ameaça e migração"
            )
        else:
            st.warning("Nenhum dado disponível para exibição.")

    # NOVA SEÇÃO: Resumo das Análises Especializadas
    if not df_filtrado.empty:
        st.markdown("---")
        st.markdown("### 📊 Resumo das Análises Especializadas")

        col1, col2, col3 = st.columns(3)

        # Preparar dados para análise
        df_temp = df_filtrado.copy()
        df_temp['is_ameacada'] = df_temp.apply(verificar_especie_ameacada, axis=1)
        df_temp['is_migratoria'] = df_temp.apply(verificar_especie_migratoria, axis=1)

        with col1:
            st.markdown("#### 🚨 Espécies Ameaçadas")
            especies_ameacadas_total = df_temp[df_temp['is_ameacada']]['vernacularName'].nunique()
            registros_ameacadas = len(df_temp[df_temp['is_ameacada']])
            percentual_ameacadas = (especies_ameacadas_total / df_temp['vernacularName'].nunique() * 100) if df_temp[
                                                                                                                 'vernacularName'].nunique() > 0 else 0

            st.metric("Espécies Ameaçadas", especies_ameacadas_total)
            st.metric("Registros", registros_ameacadas)
            st.metric("% do Total", f"{percentual_ameacadas:.1f}%")
            st.caption("Baseado em IUCN 2021, MMA 2022, RJ 2001")

        with col2:
            st.markdown("#### 🦅 Espécies Migratórias")
            especies_migratorias_total = df_temp[df_temp['is_migratoria']]['vernacularName'].nunique()
            registros_migratorias = len(df_temp[df_temp['is_migratoria']])
            percentual_migratorias = (especies_migratorias_total / df_temp['vernacularName'].nunique() * 100) if \
            df_temp['vernacularName'].nunique() > 0 else 0

            st.metric("Espécies Migratórias", especies_migratorias_total)
            st.metric("Registros", registros_migratorias)
            st.metric("% do Total", f"{percentual_migratorias:.1f}%")
            st.caption("Baseado em CBRO 2021, Somenzari 2017")

        with col3:
            st.markdown("#### 📈 Sobreposição")
            # Espécies que são tanto ameaçadas quanto migratórias
            especies_ambas = df_temp[(df_temp['is_ameacada']) & (df_temp['is_migratoria'])]['vernacularName'].nunique()
            registros_ambas = len(df_temp[(df_temp['is_ameacada']) & (df_temp['is_migratoria'])])

            st.metric("Ameaçadas + Migratórias", especies_ambas)
            st.metric("Registros", registros_ambas)
            if especies_ameacadas_total > 0:
                percentual_sobreposicao = (especies_ambas / especies_ameacadas_total * 100)
                st.metric("% das Ameaçadas", f"{percentual_sobreposicao:.1f}%")
            else:
                st.metric("% das Ameaçadas", "0%")
            st.caption("Espécies com dupla vulnerabilidade")

    # Rodapé ATUALIZADO
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; font-size: 0.8rem; color: gray;">
            Atlas das Aves Aquáticas do Rio de Janeiro | Versão Completa com Análises Especializadas<br/>
            🔬 Inclui análises de espécies ameaçadas (IUCN, MMA, RJ) e migratórias (CBRO, Somenzari)<br/>
            📊 Use os filtros na barra lateral para explorar os dados por período, município e espécie<br/>
            🗺️ Exportação KMZ completa com todas as camadas e análises prioritárias<br/>
            Desenvolvido para a conservação das aves aquáticas
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
