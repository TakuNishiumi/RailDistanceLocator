import geopandas
import matplotlib.pyplot as plt
import japanize_matplotlib
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points, snap, split, linemerge
import networkx as nx
import itertools # for combining generators

# --- 緯度経度から適切な平面直角座標系 (JGD2011) のEPSGコードを決定 ---
def get_japan_plane_rectangular_crs_epsg(lon, lat):
    """
    与えられた緯度経度に基づいて、対応する日本測地系2011 (JGD2011) の
    平面直角座標系のEPSGコードを返します。

    Args:
        lon (float): 経度
        lat (float): 緯度

    Returns:
        str: EPSGコード文字列 (例: "EPSG:6674")。見つからない場合はNone。
    """
    # 各系の中心経度と適用範囲（おおよそ）
    # 参照: https://www.gsi.go.jp/sokuchikijun/jgd2011-coordinate-system.html
    
    if 128.0 <= lon < 130.5: return "EPSG:6669" # 系1: 129.5°E
    elif 130.5 <= lon < 132.0: return "EPSG:6670" # 系2: 131°E
    elif 132.0 <= lon < 133.5: return "EPSG:6671" # 系3: 132.5°E
    elif 133.5 <= lon < 135.0: return "EPSG:6672" # 系4: 134°E
    elif 135.0 <= lon < 136.5: return "EPSG:6673" # 系5: 135.5°E
    elif 136.5 <= lon < 138.0: return "EPSG:6674" # 系6: 137°E (京都はここに含まれることが多い)
    elif 138.0 <= lon < 139.5: return "EPSG:6675" # 系7: 138.5°E
    elif 139.5 <= lon < 141.0: return "EPSG:6676" # 系8: 140°E (東京はここに含まれることが多い)
    elif 140.25 <= lon < 141.75: return "EPSG:6677" # 系9: 140.75°E (東京都の一部と伊豆諸島)
    elif 141.0 <= lon < 143.0: return "EPSG:6678" # 系10: 142°E
    elif 143.0 <= lon < 144.0: return "EPSG:6679" # 系11: 143°E
    elif 144.0 <= lon < 145.0: return "EPSG:6680" # 系12: 144°E
    elif 126.0 <= lon < 128.0: return "EPSG:6681" # 系13: 127°E (沖縄)
    elif 123.5 <= lon < 125.0: return "EPSG:6682" # 系14: 124°E (与那国島など)
    elif 153.5 <= lon < 154.5: return "EPSG:6683" # 系15: 154°E (南鳥島)
    elif 135.5 <= lon < 136.5 and lat < 30: return "EPSG:6684" # 系16: 136°E (硫黄島)
    elif 147.5 <= lon < 148.5: return "EPSG:6685" # 系17: 148°E (沖ノ鳥島)
    elif 138.5 <= lon < 139.5 and lat < 30: return "EPSG:6686" # 系18: 139°E (小笠原諸島)
    elif 141.5 <= lon < 142.5: return "EPSG:6687" # 系19: 142°E (東北の一部)

    print(f"警告: 緯度経度 ({lon}, {lat}) に対応する平面直角座標系が見つかりませんでした。デフォルトで 'EPSG:6674' を使用します。")
    return "EPSG:6674" # デフォルトとして京都周辺の系を使用

# --- データ読み込み関数 ---
def load_and_filter_data(railroad_section_path, station_path, line_name, company_name=None):
    try:
        railroad_sections_all = geopandas.read_file(railroad_section_path)
        stations_all = geopandas.read_file(station_path)

        target_sections = railroad_sections_all[railroad_sections_all['N02_003'] == line_name]
        if company_name:
            target_sections = target_sections[target_sections['N02_004'] == company_name]

        target_stations = stations_all[stations_all['N02_003'] == line_name]
        if company_name:
            target_stations = target_stations[target_stations['N02_004'] == company_name]

        if target_sections.empty:
            print(f"データが空です: 路線名 '{line_name}', 会社名 '{company_name}'")
            return None, None
        
        return target_sections, target_stations

    except FileNotFoundError:
        print(f"エラー: '{railroad_section_path}' と '{station_path}' が見つかりません。")
        return None, None
    except Exception as e:
        print(f"データ読み込み中にエラーが発生しました: {e}")
        return None, None

# --- NEW: 鉄道区間GeoDataFrameからNetworkXグラフを構築する関数 ---
def build_railroad_graph(sections_gdf, tolerance=0.1):
    """
    鉄道区間GeoDataFrameからNetworkXグラフを構築します。
    ノードは線分の端点および交点、エッジは線分を表します。
    より正確な接続のためにsnapとsplitを使用します。

    Args:
        sections_gdf (geopandas.GeoDataFrame): 鉄道区間GeoDataFrame (メートルCRS)。
        tolerance (float): 線路の端点を接続するための許容誤差（メートル）。

    Returns:
        networkx.Graph: 構築されたグラフ。
        dict: 座標からノードIDへのマッピング。
        dict: ノードIDから座標へのマッピング。
    """
    G = nx.Graph()
    node_id_counter = 0
    coords_to_node_id = {}
    node_id_to_coords = {}

    def get_node_id(coord):
        # nonlocalキーワードを追加
        nonlocal node_id_counter, coords_to_node_id, node_id_to_coords

        # 既存のノードがあればそのIDを、なければ新しいIDを割り当てる
        for existing_coord, node_id in coords_to_node_id.items():
            if Point(coord).distance(Point(existing_coord)) < tolerance: # 既存ノードとの距離が許容範囲内なら同じノードとみなす
                return node_id
        
        # 新しいノード
        coords_to_node_id[coord] = node_id_counter
        node_id_to_coords[node_id_counter] = coord
        G.add_node(node_id_counter, x=coord[0], y=coord[1], pos=(coord[0], coord[1]))
        node_id_counter += 1
        return coords_to_node_id[coord]

    all_lines = []
    for geom in sections_gdf.geometry:
        if geom.geom_type == 'LineString':
            all_lines.append(geom)
        elif geom.geom_type == 'MultiLineString':
            all_lines.extend(list(geom.geoms))

    # 各LineStringを分解し、端点をノードとして追加
    for original_line in all_lines:
        current_line = original_line
        
        # 内部ノードと端点を追加
        for i in range(len(current_line.coords) - 1):
            start_coord = current_line.coords[i]
            end_coord = current_line.coords[i+1]
            
            u = get_node_id(start_coord)
            v = get_node_id(end_coord)
            
            segment_geom = LineString([start_coord, end_coord])
            weight = segment_geom.length
            
            # 同じエッジがなければ追加
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=weight, geometry=segment_geom)
    
    # 全てのラインのユニークな端点と交点を取得し、ノードとして追加し直す
    # より複雑なネットワークの場合、topojson-likeな処理が必要になるが、今回は簡易的に
    # `linemerge`でジオメトリを整理し、そのLineString/MultiLineStringの端点を使う
    merged_lines = linemerge(sections_gdf.geometry.unary_union)
    
    if isinstance(merged_lines, LineString):
        merged_lines = [merged_lines]
    elif isinstance(merged_lines, MultiLineString):
        merged_lines = list(merged_lines.geoms)

    for line in merged_lines:
        for i in range(len(line.coords) - 1):
            start_coord = line.coords[i]
            end_coord = line.coords[i+1]
            
            u = get_node_id(start_coord)
            v = get_node_id(end_coord)
            
            segment_geom = LineString([start_coord, end_coord])
            weight = segment_geom.length
            
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=weight, geometry=segment_geom)

    return G, coords_to_node_id, node_id_to_coords

# --- 点をグラフの最も近いエッジにスナップする関数 ---
def snap_point_to_graph_edge(G, point_proj, tolerance=0.1):
    """
    指定された点をグラフの最も近いエッジ上にスナップし、
    その点に一時的なノードを追加してグラフに接続します。

    Args:
        G (networkx.Graph): 鉄道ネットワークグラフ。
        point_proj (shapely.geometry.Point): スナップする点（投影CRS）。
        tolerance (float): エッジをスナップ/分割する際の許容誤差（メートル）。

    Returns:
        tuple: (スナップされたノードID, 追加されたノードの座標)。
               スナップできない場合は (None, None)。
    """
    closest_edge = None
    min_dist = float('inf')
    
    for u, v, data in G.edges(data=True):
        if 'geometry' in data:
            dist = point_proj.distance(data['geometry'])
            if dist < min_dist:
                min_dist = dist
                closest_edge = (u, v, data['geometry'])
    
    if closest_edge is None:
        return None, None # 最も近いエッジが見つからない

    u_node, v_node, edge_geom = closest_edge
    
    # 点が既存のノードに十分近い場合はそのノードを返す
    # 既存のノードもPointオブジェクトとして比較
    if point_proj.distance(Point(G.nodes[u_node]['x'], G.nodes[u_node]['y'])) < tolerance:
        return u_node, (G.nodes[u_node]['x'], G.nodes[u_node]['y'])
    if point_proj.distance(Point(G.nodes[v_node]['x'], G.nodes[v_node]['y'])) < tolerance:
        return v_node, (G.nodes[v_node]['x'], G.nodes[v_node]['y'])

    # エッジ上にスナップする点
    snapped_point_on_edge = nearest_points(edge_geom, point_proj)[0]

    # 新しいノードIDを割り当てる
    # 既存のノードIDと重複しないように、文字列と既存ノードIDの最大値+1を使用
    # NetworkXのノードはHashableであれば何でも良いので、文字列もOK
    max_int_node_id = -1
    for node_id in G.nodes:
        if isinstance(node_id, int):
            max_int_node_id = max(max_int_node_id, node_id)
    new_node_id_base = max_int_node_id + 1
    new_node_id = f"snap_node_{new_node_id_base}_{snapped_point_on_edge.x:.2f}_{snapped_point_on_edge.y:.2f}" 

    # 既存のエッジを削除し、新しいノードを挟んで2つの新しいエッジを作成
    # エッジが存在することを確認してから削除
    if G.has_edge(u_node, v_node):
        G.remove_edge(u_node, v_node)

    # 新しいノードを追加
    G.add_node(new_node_id, x=snapped_point_on_edge.x, y=snapped_point_on_edge.y, pos=(snapped_point_on_edge.x, snapped_point_on_edge.y))

    # 新しい2つのエッジを追加
    geom_uv = LineString([(G.nodes[u_node]['x'], G.nodes[u_node]['y']), (snapped_point_on_edge.x, snapped_point_on_edge.y)])
    geom_vw = LineString([(snapped_point_on_edge.x, snapped_point_on_edge.y), (G.nodes[v_node]['x'], G.nodes[v_node]['y'])])
    
    G.add_edge(u_node, new_node_id, weight=geom_uv.length, geometry=geom_uv)
    G.add_edge(new_node_id, v_node, weight=geom_vw.length, geometry=geom_vw)
    
    return new_node_id, (snapped_point_on_edge.x, snapped_point_on_edge.y)


# --- 2つの駅間の線路に沿った距離を計算する関数 (NetworkX版) ---
# G_networkx, sections_proj を引数に追加
def calculate_distance_along_railroad(G_networkx, sections_proj, stations_gdf, station1_name, station2_name, ax=None):
    """
    2つの駅間の線路に沿った最短距離をNetworkXを用いて計算します。

    Args:
        G_networkx (networkx.Graph): 構築済みの鉄道ネットワークグラフ（投影CRS）。
        sections_proj (geopandas.GeoDataFrame): 鉄道区間データを含むGeoDataFrame（投影CRS）。
        stations_gdf (geopandas.GeoDataFrame): 駅データを含むGeoDataFrame（元のCRS）。
        station1_name (str): 1つ目の駅の名称。
        station2_name (str): 2つ目の駅の名称。
        ax (matplotlib.axes.Axes, optional): プロットに追加する場合のAxesオブジェクト。デフォルトはNone。

    Returns:
        float: 2つの駅間の線路に沿った距離（メートル）。計算できない場合はNone。
        shapely.geometry.LineString: 計算された経路のLineString (プロット用)。
    """
    if stations_gdf is None or stations_gdf.empty:
        print("駅データが提供されていないか、空です。")
        return None, None
    if sections_proj is None or sections_proj.empty:
        print("鉄道区間データが提供されていないか、空です。")
        return None, None

    station1_data = stations_gdf[stations_gdf['N02_005'] == station1_name]
    station2_data = stations_gdf[stations_gdf['N02_005'] == station2_name]

    if station1_data.empty:
        print(f"駅 '{station1_name}' が見つかりませんでした。")
        return None, None
    if station2_data.empty:
        print(f"駅 '{station2_name}' が見つかりませんでした。")
        return None

    station1_lonlat_centroid = station1_data.geometry.centroid.iloc[0]
    station2_lonlat_centroid = station2_data.geometry.centroid.iloc[0]

    # データの中央に近い緯度経度を計算し、適切なCRSを自動選択
    # （グラフ構築時にCRS変換しているので、ここではstations_gdfのCRSを起点に変換する）
    center_lon, center_lat = station1_lonlat_centroid.x, station1_lonlat_centroid.y
    # sections_projのCRSが既に設定されているはずなのでそれを使用
    selected_target_crs_meters = sections_proj.crs 

    print(f"距離計算のために使用される投影座標系: {selected_target_crs_meters}")

    # stations_gdfをグラフと同じ投影CRSに変換
    stations_proj = stations_gdf.to_crs(selected_target_crs_meters)

    station1_proj_centroid = stations_proj[stations_proj['N02_005'] == station1_name].geometry.centroid.iloc[0]
    station2_proj_centroid = stations_proj[stations_proj['N02_005'] == station2_name].geometry.centroid.iloc[0]

    # グラフのコピーを作成し、駅をスナップ
    G = G_networkx.copy() # スナップによってグラフを変更するのでコピーしておく

    # 駅をグラフ上のエッジにスナップし、一時的なノードを追加
    start_snap_node, start_snap_coords = snap_point_to_graph_edge(G, station1_proj_centroid)
    end_snap_node, end_snap_coords = snap_point_to_graph_edge(G, station2_proj_centroid)

    if start_snap_node is None or end_snap_node is None:
        print("駅を線路上にスナップできませんでした。")
        return None, None
    
    # 最短経路を探索
    try:
        path_nodes = nx.shortest_path(G, source=start_snap_node, target=end_snap_node, weight='weight')
        print(f"探索された経路ノード数: {len(path_nodes)}")
    except nx.NetworkXNoPath:
        print(f"駅 '{station1_name}' と駅 '{station2_name}' の間に経路が見つかりませんでした。")
        return None, None
    except Exception as e:
        print(f"最短経路の探索中にエラーが発生しました: {e}")
        return None, None

    # 経路の総距離とLineStringジオメトリを再構築
    total_distance = 0.0
    
    # スナップされた開始点の座標から経路を始める
    full_path_coords = []

    # 最初のノードの座標を追加 (スナップ点)
    if start_snap_node in G.nodes:
        full_path_coords.append((G.nodes[start_snap_node]['x'], G.nodes[start_snap_node]['y']))

    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        
        edge_data = G.get_edge_data(u, v)
        if edge_data and 'geometry' in edge_data:
            segment_geom = edge_data['geometry']
            total_distance += edge_data['weight']
            
            # セグメントの座標を適切な順序で追加
            # 現在のノードuの座標とセグメントの開始/終了点のどちらが近いかを確認
            u_coords_in_graph = (G.nodes[u]['x'], G.nodes[u]['y'])
            v_coords_in_graph = (G.nodes[v]['x'], G.nodes[v]['y']) # 次のノードの座標も利用

            segment_start_coord = segment_geom.coords[0]
            segment_end_coord = segment_geom.coords[-1]

            # セグメントの開始点がuに、終了点がvに近いか
            if Point(u_coords_in_graph).distance(Point(segment_start_coord)) < 1e-6 and \
               Point(v_coords_in_graph).distance(Point(segment_end_coord)) < 1e-6:
                coords_to_add = list(segment_geom.coords)
            # セグメントの開始点がvに、終了点がuに近いか（逆順）
            elif Point(u_coords_in_graph).distance(Point(segment_end_coord)) < 1e-6 and \
                 Point(v_coords_in_graph).distance(Point(segment_start_coord)) < 1e-6:
                coords_to_add = list(segment_geom.coords)[::-1]
            else:
                # 予期せぬケース (スナップノードとエッジジオメトリの関連付けが複雑な場合など)
                # シンプルに、現在のノードと次のノードの座標を追加
                coords_to_add = [u_coords_in_graph, v_coords_in_graph]
            
            # 重複する座標を避けて追加
            for coord in coords_to_add:
                # Pointオブジェクトに変換して距離比較
                if not full_path_coords or Point(coord).distance(Point(full_path_coords[-1])) > 1e-6:
                    full_path_coords.append(coord)
    
    # 最後のノードの座標がまだ追加されていなければ追加 (スナップ点)
    if end_snap_node in G.nodes:
        final_coord = (G.nodes[end_snap_node]['x'], G.nodes[end_snap_node]['y'])
        # Pointオブジェクトに変換して距離比較
        if not full_path_coords or Point(final_coord).distance(Point(full_path_coords[-1])) > 1e-6:
            full_path_coords.append(final_coord)

    # 経路のLineStringを再構築
    if len(full_path_coords) < 2:
        print("経路の座標が不足しています。")
        return None, None
        
    path_line_proj = LineString(full_path_coords)

    print(f"駅 '{station1_name}' と駅 '{station2_name}' 間の線路に沿った距離: {total_distance:.2f} メートル")
    
    # プロットに追加する場合
    if ax:
        # 経路をプロット（元のCRSに戻して）
        geopandas.GeoSeries([path_line_proj], crs=selected_target_crs_meters).to_crs(stations_gdf.crs).plot( # sections_gdf.crs -> stations_gdf.crs に変更
            ax=ax, color='green', linewidth=4, linestyle='-', alpha=0.7, label=f'経路: {station1_name} -> {station2_name}'
        )
        # スナップされた駅の位置をプロット
        geopandas.GeoSeries([Point(start_snap_coords), Point(end_snap_coords)], crs=selected_target_crs_meters).to_crs(stations_gdf.crs).plot( # sections_gdf.crs -> stations_gdf.crs に変更
            ax=ax, marker='P', color='orange', markersize=150, edgecolor='black', linewidth=1, label='スナップされた駅位置'
        )
        ax.text(station1_lonlat_centroid.x, station1_lonlat_centroid.y, station1_name, fontsize=10, ha='center', va='bottom', color='purple')
        ax.text(station2_lonlat_centroid.x, station2_lonlat_centroid.y, station2_name, fontsize=10, ha='center', va='bottom', color='purple')

    # LineStringをGeoSeriesに変換し、to_crsを適用して元のCRSに戻す
    path_line_lonlat = geopandas.GeoSeries([path_line_proj], crs=selected_target_crs_meters).to_crs(stations_gdf.crs).iloc[0] # sections_gdf.crs -> stations_gdf.crs に変更
    return total_distance, path_line_lonlat # プロット用に元のCRSに戻したLineStringも返す

# --- 線路上で指定距離を移動した地点の計算 (NetworkX版) ---
# G_networkx, sections_proj を引数に追加
def get_point_along_railroad_line(G_networkx, sections_proj, stations_gdf, start_station_name, end_station_name, distance_meters, ax=None):
    """
    指定した開始駅Aから終了駅Bへ向かう線路上で、指定した距離だけ進んだ地点の座標を計算します。
    NetworkXを用いて経路を特定し、その経路のジオメトリに沿って距離を計算します。

    Args:
        G_networkx (networkx.Graph): 構築済みの鉄道ネットワークグラフ（投影CRS）。
        sections_proj (geopandas.GeoDataFrame): 鉄道区間データを含むGeoDataFrame（投影CRS）。
        stations_gdf (geopandas.GeoDataFrame): 駅データを含むGeoDataFrame（元のCRS）。
        start_station_name (str): 開始駅Aの名称。
        end_station_name (str): 終了駅Bの名称。
        distance_meters (float): 駅Aから駅B方向へ線路上を進む距離（メートル）。
        ax (matplotlib.axes.Axes, optional): プロットに追加する場合のAxesオブジェクト。デフォルトはNone。

    Returns:
        tuple: (経度, 緯度) のタプル。計算できない場合は (None, None)。
    """
    if stations_gdf is None or stations_gdf.empty:
        print("駅データが提供されていないか、空です。")
        return None, None
    if sections_proj is None or sections_proj.empty:
        print("鉄道区間データが提供されていないか、空です。")
        return None, None

    start_station_data = stations_gdf[stations_gdf['N02_005'] == start_station_name]
    end_station_data = stations_gdf[stations_gdf['N02_005'] == end_station_name]

    if start_station_data.empty:
        print(f"開始駅 '{start_station_name}' が見つかりませんでした。")
        return None, None
    if end_station_data.empty:
        print(f"終了駅 '{end_station_name}' が見つかりませんでした。")
        return None, None

    start_point_lonlat = start_station_data.geometry.centroid.iloc[0]
    end_point_lonlat = end_station_data.geometry.centroid.iloc[0]

    # データの中央に近い緯度経度を計算し、適切なCRSを自動選択
    # （グラフ構築時にCRS変換しているので、ここではsections_projのCRSを起点に変換する）
    center_lon, center_lat = start_point_lonlat.x, start_point_lonlat.y
    selected_target_crs_meters = sections_proj.crs # sections_projのCRSが既に設定されているはずなのでそれを使用

    print(f"線路上地点計算のために使用される投影座標系: {selected_target_crs_meters}")

    # stations_gdfをグラフと同じ投影CRSに変換
    stations_proj = stations_gdf.to_crs(selected_target_crs_meters)

    start_point_proj_centroid = stations_proj[stations_proj['N02_005'] == start_station_name].geometry.centroid.iloc[0]
    end_point_proj_centroid = stations_proj[stations_proj['N02_005'] == end_station_name].geometry.centroid.iloc[0]

    # グラフのコピーを作成し、駅をスナップ
    G = G_networkx.copy() # スナップによってグラフを変更するのでコピーしておく

    # 駅をグラフ上のエッジにスナップし、一時的なノードを追加
    start_snap_node, start_snap_coords = snap_point_to_graph_edge(G, start_point_proj_centroid)
    end_snap_node, end_snap_coords = snap_point_to_graph_edge(G, end_point_proj_centroid)

    if start_snap_node is None or end_snap_node is None:
        print("駅を線路上にスナップできませんでした。")
        return None, None
    
    # 最短経路を探索
    try:
        path_nodes = nx.shortest_path(G, source=start_snap_node, target=end_snap_node, weight='weight')
        print(f"探索された経路ノード数: {len(path_nodes)}")
    except nx.NetworkXNoPath:
        print(f"駅 '{start_station_name}' と駅 '{end_station_name}' の間に経路が見つかりませんでした。")
        return None, None
    except Exception as e:
        print(f"最短経路の探索中にエラーが発生しました: {e}")
        return None, None

    # 経路のLineStringジオメトリを再構築
    full_path_coords = []
    if start_snap_node in G.nodes:
        full_path_coords.append((G.nodes[start_snap_node]['x'], G.nodes[start_snap_node]['y']))

    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        
        edge_data = G.get_edge_data(u, v)
        if edge_data and 'geometry' in edge_data:
            segment_geom = edge_data['geometry']
            
            u_coords_in_graph = (G.nodes[u]['x'], G.nodes[u]['y'])
            v_coords_in_graph = (G.nodes[v]['x'], G.nodes[v]['y'])

            segment_start_coord = segment_geom.coords[0]
            segment_end_coord = segment_geom.coords[-1]

            if Point(u_coords_in_graph).distance(Point(segment_start_coord)) < 1e-6 and \
               Point(v_coords_in_graph).distance(Point(segment_end_coord)) < 1e-6:
                coords_to_add = list(segment_geom.coords)
            elif Point(u_coords_in_graph).distance(Point(segment_end_coord)) < 1e-6 and \
                 Point(v_coords_in_graph).distance(Point(segment_start_coord)) < 1e-6:
                coords_to_add = list(segment_geom.coords)[::-1]
            else:
                coords_to_add = [u_coords_in_graph, v_coords_in_graph] # Fallback

            for coord in coords_to_add:
                # Pointオブジェクトに変換して距離比較
                if not full_path_coords or Point(coord).distance(Point(full_path_coords[-1])) > 1e-6:
                    full_path_coords.append(coord)
    
    if end_snap_node in G.nodes:
        final_coord = (G.nodes[end_snap_node]['x'], G.nodes[end_snap_node]['y'])
        # Pointオブジェクトに変換して距離比較
        if not full_path_coords or Point(final_coord).distance(Point(full_path_coords[-1])) > 1e-6:
            full_path_coords.append(final_coord)

    if len(full_path_coords) < 2:
        print("経路の座標が不足しています。")
        return None, None
        
    path_line_proj = LineString(full_path_coords)

    # 補間する最終的な距離を計算
    target_distance_on_line = distance_meters

    # LineStringの長さを超えないように調整
    line_total_length = path_line_proj.length
    
    if target_distance_on_line < 0:
        print(f"指定距離が負の値になり、経路の始点より手前です。経路の始点の座標を返します。")
        calculated_point_proj = path_line_proj.interpolate(0)
    elif target_distance_on_line > line_total_length:
        print(f"指定距離 {distance_meters}m が経路の総延長 {line_total_length:.2f}m を超えています。")
        print("経路の終端の座標を返します。")
        calculated_point_proj = path_line_proj.interpolate(line_total_length)
    else:
        calculated_point_proj = path_line_proj.interpolate(target_distance_on_line)
    
    # 計算された点を元のCRSに戻す
    calculated_point_lonlat_gs = geopandas.GeoSeries([calculated_point_proj], crs=selected_target_crs_meters)
    calculated_point_lonlat = calculated_point_lonlat_gs.to_crs(stations_gdf.crs).iloc[0] # sections_gdf.crs -> stations_gdf.crs に変更

    print(f"線路上で駅 '{start_station_name}' から '{end_station_name}' 方向へ {distance_meters}m 進んだ地点の座標: ")
    print(f"  経度: {calculated_point_lonlat.x}, 緯度: {calculated_point_lonlat.y}")

    # プロットに追加する場合
    if ax:
        # 計算された地点をプロットに追加
        geopandas.GeoSeries([calculated_point_lonlat], crs=stations_gdf.crs).plot( # sections_gdf.crs -> stations_gdf.crs に変更
            ax=ax, marker='X', color='lime', markersize=250, edgecolor='black', linewidth=1, label=f'線路上{distance_meters}m地点'
        )
        # 経路全体をプロット（デバッグ用）
        # path_line_projを元のCRSに戻してプロット
        path_line_lonlat_for_plot = geopandas.GeoSeries([path_line_proj], crs=selected_target_crs_meters).to_crs(stations_gdf.crs).iloc[0] # sections_gdf.crs -> stations_gdf.crs に変更
        geopandas.GeoSeries([path_line_lonlat_for_plot], crs=stations_gdf.crs).plot(
            ax=ax, color='green', linewidth=3, linestyle='--', label=f'経路: {start_station_name} -> {end_station_name}'
        )
        # 駅の重心とスナップ点をプロット
        geopandas.GeoSeries([start_point_lonlat, end_point_lonlat], crs=stations_gdf.crs).plot(
            ax=ax, marker='x', color='purple', markersize=100, linewidth=2, label='始点・終点駅重心'
        )
        ax.text(start_point_lonlat.x, start_point_lonlat.y, 'A', fontsize=12, ha='center', va='bottom', color='purple')
        ax.text(end_point_lonlat.x, end_point_lonlat.y, 'B', fontsize=12, ha='center', va='bottom', color='purple')

        geopandas.GeoSeries([Point(start_snap_coords), Point(end_snap_coords)], crs=selected_target_crs_meters).to_crs(stations_gdf.crs).plot( # sections_gdf.crs -> stations_gdf.crs に変更
            ax=ax, marker='P', color='orange', markersize=150, edgecolor='black', linewidth=1, label='スナップされた駅位置'
        )
        ax.legend()

    return calculated_point_lonlat.x, calculated_point_lonlat.y


# --- 使用例 ---
if __name__ == "__main__":
    railroad_section_file = './data/N02-22_RailroadSection.shp'
    station_file = './data/N02-22_Station.shp'

    # 表示したい路線と会社名を指定
    line_name = '東海道線'
    company_name = '西日本旅客鉄道' 

    # データをロード
    sections, stations = load_and_filter_data(railroad_section_file, station_file, line_name, company_name)

    if sections is not None and stations is not None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12)) 

        # 路線をプロット
        plot_label_section = f'{line_name} 鉄道区間'
        if company_name:
            plot_label_section += f' ({company_name})'
        sections.plot(ax=ax, color='blue', linewidth=2, label=plot_label_section)

        # 駅をプロット（LineStringの場合は重心を使用）
        if not stations.empty:
            station_centroids = stations.geometry.centroid
            
            plot_label_station = f'{line_name} 駅 (重心)'
            if company_name:
                plot_label_station += f' ({company_name})'
            geopandas.GeoSeries(station_centroids).plot(ax=ax, marker='o', color='red', markersize=50, label=plot_label_station)
            
            # 駅名をラベルとして追加
            for x, y, label in zip(station_centroids.x, station_centroids.y, stations['N02_005']):
                ax.text(x, y, label, fontsize=8, ha='right')
        
        # --- グラフの初期構築 (一度だけ実行) ---
        # まず、中心の緯度経度から適切なCRSを決定し、セクションデータをそのCRSに変換
        # 駅データはまだ元のCRSのままでよい
        if not stations.empty:
            sample_station_lonlat = stations.geometry.centroid.iloc[0]
            initial_target_crs_meters = get_japan_plane_rectangular_crs_epsg(sample_station_lonlat.x, sample_station_lonlat.y)
        else:
            print("警告: 駅データが空のため、デフォルトのCRS 'EPSG:6674' を使用します。")
            initial_target_crs_meters = "EPSG:6674" # デフォルト

        print(f"\n初期グラフ構築のために選択された投影座標系: {initial_target_crs_meters}")
        sections_proj_for_graph = sections.to_crs(initial_target_crs_meters)
        
        # グラフを構築
        print("鉄道グラフを構築中...")
        G_railroad, coords_to_node_id, node_id_to_coords = build_railroad_graph(sections_proj_for_graph)
        print(f"グラフ構築完了: ノード数={G_railroad.number_of_nodes()}, エッジ数={G_railroad.number_of_edges()}")


        # --- 2つの駅間の線路に沿った距離の計算 ---
        print("\n--- 2つの駅間の線路に沿った距離の計算 (NetworkX使用) ---")
        station1_for_distance = '京都'
        station2_for_distance = '大阪' # 例として大阪駅
        
        # 構築済みのグラフと投影済みセクションデータを渡す
        distance_between_stations, path_geom_for_distance = calculate_distance_along_railroad(
                                                                        G_railroad, 
                                                                        sections_proj_for_graph, 
                                                                        stations, # stationsは元のCRSのまま渡す
                                                                        station1_for_distance, 
                                                                        station2_for_distance, 
                                                                        ax=ax)
        if distance_between_stations is not None:
            print(f"計算結果: {station1_for_distance} - {station2_for_distance} 間の線路距離 = {distance_between_stations:.2f} m")
            # プロットに経路を描画する部分はcalculate_distance_along_railroad内で実行される

        # --- 線路上で指定距離を移動した地点の計算 (NetworkX使用) ---
        print("\n--- 線路上で指定距離を移動した地点の計算 (NetworkX使用) ---")
        start_station_name_example = '京都' 
        end_station_name_example = '大阪'   
        distance_to_calculate_point = 5000 # 5000メートル (5km)

        # 構築済みのグラフと投影済みセクションデータを渡す
        calculated_coords_on_line = get_point_along_railroad_line(
                                                                    G_railroad, 
                                                                    sections_proj_for_graph, 
                                                                    stations, # stationsは元のCRSのまま渡す
                                                                    start_station_name_example, 
                                                                    end_station_name_example, 
                                                                    distance_to_calculate_point, 
                                                                    ax=ax)
        
        # タイトルと凡例を設定
        plot_title = f'{line_name} 鉄道路線の表示とNetworkXによる経路'
        if company_name:
            plot_title += f' ({company_name})'
        ax.set_title(plot_title)
        ax.set_xlabel('経度')
        ax.set_ylabel('緯度')
        ax.legend()
        ax.grid(True)
        plt.show()
    else:
        print("データをロードできませんでした。表示をスキップします。")