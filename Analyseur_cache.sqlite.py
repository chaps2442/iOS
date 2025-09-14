# Name: AnalyseurGPS_autonome.py
# Analyse de cache.sqlite d'iOS - CORRECTION ERREUR SQL & INTERFACE
# Authors: Vincent Chapeau & Burri Xavier
# Finalisation & Stabilité: Gemini
# Version: 0.64.0 - Correction erreur "no such column" & Restauration UI

import sqlite3
import folium
from folium.plugins import TimestampedGeoJson
from folium import DivIcon
import pandas as pd
import simplekml
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, available_timezones
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from tkcalendar import DateEntry
import matplotlib.pyplot as plt
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import os
import subprocess
import sys

# --- FONCTION DE CALCUL DE DISTANCE ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Rayon de la Terre en mètres
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# --- FONCTION D'ANALYSE PRINCIPALE ---
def run_analysis(devices, base_output_path, start_ts, end_ts, accuracy_threshold, 
                 min_speed, max_speed,
                 cross_ref_dist, cross_ref_time,
                 export_html, export_csv, export_kml, export_graph, 
                 selected_timezone, start_date_str, end_date_str, progress_callback):
    
    all_dataframes = []
    local_tz = ZoneInfo(selected_timezone)

    for i, device in enumerate(devices):
        progress_callback(i / len(devices) * 20, f"Analyse de {device['id']}...")
        try:
            with sqlite3.connect(device['path']) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables_in_db = {table[0].upper() for table in cursor.fetchall()}

                def pick(cols, *candidates):
                    for c in candidates:
                        if c.upper() in cols: return c
                    return None

                # --- TRAJETS (ZRTCLLOCATIONMO) ---
                df_trajets = pd.DataFrame()
                if 'ZRTCLLOCATIONMO' in tables_in_db:
                    cursor.execute("PRAGMA table_info(ZRTCLLOCATIONMO);")
                    cols = {c[1].upper() for c in cursor.fetchall()}
                    col_map = {"ZTIMESTAMP": "TS", "ZLATITUDE": "LAT", "ZLONGITUDE": "LONG", "ZHORIZONTALACCURACY": "ACCURACY", "ZSPEED": "SPEED", "ZCOURSE": "COURSE", "ZALTITUDE": "ALTITUDE"}
                    select_parts = [f'"{db_col}" as {alias}' for db_col, alias in col_map.items() if db_col.upper() in cols]
                    if "ZTIMESTAMP" in cols and "ZLATITUDE" in cols:
                        query = f"SELECT {', '.join(select_parts)} FROM ZRTCLLocationMO WHERE ZTIMESTAMP BETWEEN ? AND ? AND ZHORIZONTALACCURACY <= ?"
                        params = [start_ts, end_ts, accuracy_threshold]
                        df_trajets = pd.read_sql_query(query, conn, params=params)
                        if not df_trajets.empty: df_trajets['Source'] = 'Trajet'
                
                # --- TRAJETS (ANCIEN - ZRTLEARNEDLOCATIONOFINTERESTMO) - CORRECTION ERREUR SQL ---
                if df_trajets.empty and 'ZRTLEARNEDLOCATIONOFINTERESTMO' in tables_in_db:
                    cursor.execute("PRAGMA table_info(ZRTLEARNEDLOCATIONOFINTERESTMO);")
                    cols = {c[1].upper() for c in cursor.fetchall()}
                    ts_col = pick(cols, 'ZCREATIONDATE', 'ZTIMESTAMP')
                    lat_col = pick(cols, 'ZLOCATIONLATITUDE')
                    lon_col = pick(cols, 'ZLOCATIONLONGITUDE')
                    acc_col = pick(cols, 'ZLOCATIONHORIZONTALACCURACY')
                    if all([ts_col, lat_col, lon_col, acc_col]):
                        query = f"SELECT {ts_col} as TS, {lat_col} as LAT, {lon_col} as LONG, {acc_col} as ACCURACY FROM ZRTLEARNEDLOCATIONOFINTERESTMO WHERE {ts_col} BETWEEN ? AND ?"
                        df_trajets = pd.read_sql_query(query, conn, params=[start_ts, end_ts])
                        if not df_trajets.empty: df_trajets['Source'] = 'Trajet (Ancien)'

                if not df_trajets.empty:
                    df_trajets['id'] = device['id']
                    all_dataframes.append(df_trajets)

                # --- VISITES (ZRTVISITMO) ---
                if 'ZRTVISITMO' in tables_in_db:
                    cursor.execute("PRAGMA table_info(ZRTVISITMO);")
                    cols = {c[1].upper() for c in cursor.fetchall()}
                    entry_col, exit_col, lat_col, lon_col, acc_col = pick(cols, 'ZENTRYDATE'), pick(cols, 'ZEXITDATE'), pick(cols, 'ZLATITUDE'), pick(cols, 'ZLONGITUDE'), pick(cols, 'ZHORIZONTALACCURACY')
                    if all([entry_col, exit_col, lat_col, lon_col, acc_col]):
                        query = f"SELECT {entry_col} AS TS, {exit_col} AS TS_EXIT, {lat_col} AS LAT, {lon_col} AS LONG, {acc_col} AS ACCURACY FROM ZRTVISITMO WHERE {entry_col} BETWEEN ? AND ?"
                        df_visits = pd.read_sql_query(query, conn, params=[start_ts, end_ts])
                        if not df_visits.empty:
                            df_visits['id'] = device['id']; df_visits['Source'] = 'Lieu Visité'; all_dataframes.append(df_visits)

                # --- INDICES (ZRTHINTMO) ---
                if 'ZRTHINTMO' in tables_in_db:
                    cursor.execute("PRAGMA table_info(ZRTHINTMO);")
                    cols = {c[1].upper() for c in cursor.fetchall()}
                    ts_col, lat_col, lon_col, acc_col = pick(cols, 'ZTIMESTAMP'), pick(cols, 'ZLATITUDE'), pick(cols, 'ZLONGITUDE'), pick(cols, 'ZHORIZONTALACCURACY')
                    if all([ts_col, lat_col, lon_col]):
                        select = f"{ts_col} AS TS, {lat_col} AS LAT, {lon_col} AS LONG"
                        if acc_col: select += f", {acc_col} AS ACCURACY"
                        df_hints = pd.read_sql_query(f"SELECT {select} FROM ZRTHINTMO WHERE {ts_col} BETWEEN ? AND ?", conn, params=[start_ts, end_ts])
                        if not df_hints.empty:
                            df_hints['id'] = device['id']; df_hints['Source'] = 'Indice'; all_dataframes.append(df_hints)

        except Exception as e:
            messagebox.showerror("Erreur Base de Données", f"Impossible de lire le fichier pour {device['id']}:\n{e}"); return False

    if not all_dataframes:
        messagebox.showinfo("Aucune Donnée", "Aucune donnée de localisation trouvée pour les critères sélectionnés."); return False

    progress_callback(25, "Fusion des données..."); 
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df['TS'] = pd.to_numeric(combined_df['TS'], errors='coerce')
    combined_df.dropna(subset=['TS', 'LAT', 'LONG'], inplace=True)
    
    def convert_core_data_timestamp(ts): return datetime(2001, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=ts)
    combined_df['Timestamp_UTC'] = combined_df['TS'].apply(convert_core_data_timestamp)
    combined_df.sort_values(by='Timestamp_UTC', inplace=True, ignore_index=True)
    
    def format_value(val, unit=""): return f"{val:.2f}{unit}" if pd.notna(val) else "N/A"
    
    df_export = pd.DataFrame()
    df_export['Appareil_ID'] = combined_df['id']
    df_export['Source'] = combined_df['Source']
    df_export['Timestamp_Local'] = combined_df['Timestamp_UTC'].dt.tz_convert(local_tz).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_export['Timestamp_UTC_ISO'] = combined_df['Timestamp_UTC'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    df_export['Latitude'] = combined_df['LAT']
    df_export['Longitude'] = combined_df['LONG']
    df_export['Altitude_m'] = combined_df.get('ALTITUDE')
    df_export['Vitesse_kmh'] = combined_df.get('SPEED', pd.Series(dtype='float')) * 3.6
    df_export['Direction_deg'] = combined_df.get('COURSE')
    df_export['Precision_GPS_m'] = combined_df.get('ACCURACY')

    # --- FILTRAGE FINAL ---
    progress_callback(38, "Filtrage des données...")
    if min_speed is not None:
        df_export = df_export[df_export['Vitesse_kmh'] >= min_speed]
    if max_speed is not None:
        df_export = df_export[df_export['Vitesse_kmh'] <= max_speed]
    df_export.reset_index(drop=True, inplace=True)
    if df_export.empty:
        messagebox.showinfo("Aucune Donnée", "Aucune donnée restante après application des filtres de vitesse."); return False

    if len(devices) > 1 and cross_ref_dist > 0 and cross_ref_time > 0:
        progress_callback(45, "Détection des croisements..."); croisements = []
        df_sorted = df_export.set_index(pd.to_datetime(df_export['Timestamp_UTC_ISO'])).sort_index(); unique_devices = df_sorted['Appareil_ID'].unique()
        for i in range(len(unique_devices)):
            for j in range(i + 1, len(unique_devices)):
                dev1_id, dev2_id = unique_devices[i], unique_devices[j]
                dev1_df = df_sorted[df_sorted['Appareil_ID'] == dev1_id]; dev2_df = df_sorted[df_sorted['Appareil_ID'] == dev2_id]
                merged = pd.merge_asof(dev1_df, dev2_df, left_index=True, right_index=True, tolerance=pd.Timedelta(seconds=cross_ref_time), direction='nearest', suffixes=('_1', '_2'))
                merged.dropna(subset=['Latitude_1', 'Longitude_1', 'Latitude_2', 'Longitude_2'], inplace=True)
                if not merged.empty:
                    merged['distance_m'] = merged.apply(lambda row: haversine(row['Latitude_1'], row['Longitude_1'], row['Latitude_2'], row['Longitude_2']), axis=1)
                    proximite_df = merged[merged['distance_m'] <= cross_ref_dist]
                    for _, row in proximite_df.iterrows():
                        croisements.append({'Timestamp_Local': row.name.tz_convert(local_tz).strftime('%Y-%m-%d %H:%M:%S'), 'Appareil_1': dev1_id, 'Appareil_2': dev2_id, 'Latitude': (row['Latitude_1'] + row['Latitude_2']) / 2, 'Longitude': (row['Longitude_1'] + row['Longitude_2']) / 2, 'Distance_m': row['distance_m']})
        if croisements:
            df_croisements = pd.DataFrame(croisements); df_croisements.to_csv(base_output_path + "_croisements.csv", index=False, sep=';', encoding='utf-8-sig')

    if export_csv:
        progress_callback(50, "Génération CSV..."); df_export.to_csv(base_output_path + ".csv", index=False, sep=';', encoding='utf-8-sig', float_format='%.6f')
    
    if export_kml:
        progress_callback(60, "Génération KML..."); kml = simplekml.Kml(name=f"Analyse GPS")
        for source_type, group in df_export.groupby('Source'):
            folder = kml.newfolder(name=str(source_type))
            for _, row in group.iterrows():
                pnt = folder.newpoint(name=row['Timestamp_Local'][11:])
                pnt.coords = [(float(row['Longitude']), float(row['Latitude']), float(row['Altitude_m']) if pd.notna(row['Altitude_m']) else 0.0)]
                pnt.extendeddata = simplekml.ExtendedData()
                pnt.extendeddata.newdata('Source', value=str(row['Source'])); pnt.extendeddata.newdata('Vitesse', value=str(format_value(row['Vitesse_kmh'], ' km/h')))
        kml.save(base_output_path + ".kml")

    if export_html:
        progress_callback(70, "Création de la carte..."); 
        m = folium.Map(); 
        all_points_for_bounds = [];
        device_colors = {dev_id: color for dev_id, color in zip(df_export['Appareil_ID'].unique(), ['blue', 'red', 'green', 'purple', 'orange', 'darkred'])}
        
        # --- COUCHE D'ANIMATION ---
        fg_animation = folium.FeatureGroup(name="Animation des Trajets", show=True)
        animation_features = []
        for device_id, group in df_export.groupby('Appareil_ID'):
            color = device_colors.get(device_id, 'gray')
            for _, row in group.iterrows():
                animation_features.append({
                    'type': 'Feature',
                    'geometry': { 'type': 'Point', 'coordinates': [row['Longitude'], row['Latitude']] },
                    'properties': {
                        'time': row['Timestamp_UTC_ISO'], 'icon': 'circle',
                        'iconstyle': { 'color': color, 'fillColor': color, 'fillOpacity': 0.8, 'radius': 5 },
                        'popup': f"<b>{device_id}</b><br>{row['Timestamp_Local']}<br>Source: {row['Source']}"
                    }
                })
        if animation_features:
            TimestampedGeoJson({'type': 'FeatureCollection', 'features': animation_features},
                                period='PT1H', add_last_point=True, auto_play=False, loop=False, 
                                max_speed=10, loop_button=True, date_options='YYYY/MM/DD HH:mm:ss',
                                time_slider_drag_update=True).add_to(fg_animation)
        m.add_child(fg_animation)

        # --- COUCHES STATIQUES PAR APPAREIL ---
        for device_id, group in df_export.groupby('Appareil_ID'):
            color = device_colors.get(device_id, 'gray')
            trajet_group = group[group['Source'].str.contains('Trajet')]
            if trajet_group.empty: continue

            points = list(zip(trajet_group['Latitude'], trajet_group['Longitude']))
            all_points_for_bounds.extend(points)
            
            fg_lines = folium.FeatureGroup(name=f"Tracé (Lignes) {device_id}", show=True)
            fg_points = folium.FeatureGroup(name=f"Points de Trajet {device_id}", show=False)
            fg_accuracy = folium.FeatureGroup(name=f"Précision (Cercles) {device_id}", show=False)
            fg_direction = folium.FeatureGroup(name=f"Directions (Angles) {device_id}", show=False)

            folium.PolyLine(locations=points, color=color, weight=2.5, opacity=0.8).add_to(fg_lines)

            for _, row in trajet_group.iterrows():
                folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=3, color=color, fill=True, fill_color=color).add_to(fg_points)
                if pd.notna(row['Precision_GPS_m']) and row['Precision_GPS_m'] > 0:
                    folium.Circle(location=[row['Latitude'], row['Longitude']], radius=row['Precision_GPS_m'], color='blue', weight=1, fill=True, fill_color='blue', fill_opacity=0.2).add_to(fg_accuracy)
                if pd.notna(row['Direction_deg']) and row['Direction_deg'] >= 0:
                    arrow_svg = f'<div style="transform: rotate({int(row["Direction_deg"])}deg); transform-origin: center;"><svg viewBox="0 0 24 24" fill="{color}" width="24px" height="24px"><path d="M12 2L2.5 21.5 12 17 21.5 21.5z"/></svg></div>'
                    folium.Marker(location=[row['Latitude'], row['Longitude']], icon=DivIcon(icon_size=(24,24), icon_anchor=(12,12), html=arrow_svg)).add_to(fg_direction)
            
            m.add_child(fg_lines); m.add_child(fg_points); m.add_child(fg_accuracy); m.add_child(fg_direction)
        
        if 'df_croisements' in locals() and not df_croisements.empty:
            fg_croisements = folium.FeatureGroup(name="Points de Croisement", show=True)
            for _, row in df_croisements.iterrows():
                popup = f"<b>Croisement</b><br>{row['Appareil_1']} & {row['Appareil_2']}<br>Heure: {row['Timestamp_Local']}<br>Distance: {row['Distance_m']:.1f} m"
                folium.Marker(location=[row['Latitude'], row['Longitude']], popup=popup, icon=folium.Icon(color='black', icon='screenshot')).add_to(fg_croisements)
                all_points_for_bounds.append((row['Latitude'], row['Longitude']))
            m.add_child(fg_croisements)

        if all_points_for_bounds:
            bounds = [(min(p[0] for p in all_points_for_bounds), min(p[1] for p in all_points_for_bounds)), (max(p[0] for p in all_points_for_bounds), max(p[1] for p in all_points_for_bounds))]
            m.fit_bounds(bounds)
            
        folium.LayerControl().add_to(m)
        m.save(base_output_path + ".html")

    if export_graph and 'Vitesse_kmh' in df_export.columns and df_export['Vitesse_kmh'].notna().any():
        progress_callback(95, "Génération du graphique..."); 
        plt.style.use('seaborn-v0_8-whitegrid'); 
        fig, ax=plt.subplots(figsize=(15,8))
        df_graph = df_export.dropna(subset=['Vitesse_kmh'])
        df_graph['plot_time'] = pd.to_datetime(df_graph['Timestamp_Local'])
        ax.plot(df_graph['plot_time'], df_graph['Vitesse_kmh'], marker='o', linestyle='-', markersize=3)
        ax.set_title("Évolution de la Vitesse", fontsize=16); ax.set_xlabel(f"Temps ({selected_timezone})"); ax.set_ylabel("Vitesse (km/h)")
        ax.grid(True); plt.xticks(rotation=45); fig.tight_layout(); plt.savefig(base_output_path+"_vitesse.png", dpi=300); plt.close(fig)

    readme_content = f"""RAPPORT D'ANALYSE GPS - PROJET "{os.path.basename(os.path.dirname(base_output_path))}"
=====================================================================
Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARAMÈTRES DE L'ANALYSE
-------------------------------------
Fichier(s) source: {', '.join([os.path.basename(d['path']) for d in devices])}
Période (Fuseau: {selected_timezone}):
  - Début: {start_date_str}
  - Fin:   {end_date_str}
Filtres appliqués:
  - Précision GPS max. acceptée: {accuracy_threshold if accuracy_threshold != 99999 else 'Toutes'} mètres
  - Vitesse minimum: {f'{min_speed} km/h' if min_speed is not None else 'Aucune'}
  - Vitesse maximum: {f'{max_speed} km/h' if max_speed is not None else 'Aucune'}

DESCRIPTION DES COUCHES (CARTE .HTML)
-------------------------------------
- **Animation des Trajets (Activé par défaut):** Affiche les points de tous les appareils de manière chronologique. Utilisez la ligne de temps en bas pour naviguer.
- **Tracé (Lignes) (Activé par défaut):** Dessine une ligne continue reliant les points de trajet pour chaque appareil.
- **Points de Trajet (Désactivé par défaut):** Affiche un petit marqueur sur chaque coordonnée GPS enregistrée.
- **Directions (Angles) (Désactivé par défaut):** Montre une flèche sur chaque point indiquant le cap de l'appareil à ce moment.
- **Précision (Cercles) (Désactivé par défaut):** Dessine un cercle bleu autour de chaque point représentant la marge d'erreur de la localisation.
- **Points de Croisement (Activé par défaut):** Marque les endroits où deux appareils se sont trouvés à proximité.

SOURCES DE DONNÉES ET LEUR SIGNIFICATION
-------------------------------------
- **Trajet (ZRTCLLocationMO):** Enregistrement fréquent et précis des points GPS lors d'un déplacement.
- **Lieu Visité (ZRTVISITMO):** Point unique marquant un lieu où l'appareil a stationné pendant une certaine durée.
- **Indice (ZRTHINTMO):** Localisations moins précises collectées par iOS pour diverses raisons (suggestions Siri, etc.).
- **Trajet (Ancien):** Source de données d'anciennes versions d'iOS.
"""
    with open(base_output_path + "_readme.txt", "w", encoding="utf-8") as f: f.write(readme_content)
    
    progress_callback(100, "Terminé !"); return True

# --- CLASSE TOOLTIP ---
class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget; self.tipwindow = None
    def showtip(self, text):
        self.text = text
        if self.tipwindow or not self.text: return
        x, y, _, _ = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25; y = y + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1); tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1, font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)
    def hidetip(self):
        tw = self.tipwindow; self.tipwindow = None
        if tw: tw.destroy()

def createToolTip(widget, text):
    toolTip = ToolTip(widget)
    widget.bind('<Enter>', lambda event: toolTip.showtip(text))
    widget.bind('<Leave>', lambda event: toolTip.hidetip())

class App:
    def __init__(self, root):
        self.root = root; self.root.title("Analyseur GPS iOS - v0.64.0"); self.root.minsize(550, 750)
        self.devices, self.last_output_dir = [], None
        self.is_multi_mode = tk.BooleanVar(value=False); self.is_multi_mode.trace_add("write", self.toggle_mode)
        self.single_db_path, self.output_path = tk.StringVar(), tk.StringVar()
        self.accuracy = tk.StringVar(value="") # Par défaut: vide
        self.min_speed_var, self.max_speed_var = tk.StringVar(value=""), tk.StringVar(value="")
        self.timezone = tk.StringVar(value="Europe/Brussels")
        self.export_html, self.export_csv, self.export_kml, self.export_graph = tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        self.cross_dist_var, self.cross_time_var = tk.StringVar(value="50"), tk.StringVar(value="60")
        self.start_hour_var, self.start_min_var, self.start_sec_var = tk.StringVar(value="00"), tk.StringVar(value="00"), tk.StringVar(value="00")
        self.end_hour_var, self.end_min_var, self.end_sec_var = tk.StringVar(value="23"), tk.StringVar(value="59"), tk.StringVar(value="59")
        self.create_widgets(); self.toggle_mode()

    def select_single_db(self):
        path = filedialog.askopenfilename(title="Sélectionner cache.sqlite", filetypes=[("Fichiers SQLite", "*.sqlite"), ("Tous les fichiers", "*.*")])
        if path: self.single_db_path.set(path)

    def select_output(self):
        path = filedialog.asksaveasfilename(title="Choisir le nom et le dossier de sortie", defaultextension="", filetypes=[("Tous les fichiers", "*.*")])
        if path: self.output_path.set(path)

    def add_device(self):
        dev_id = simpledialog.askstring("ID Appareil", "Entrez un nom unique pour cet appareil:", parent=self.root)
        if not dev_id: return
        if any(d['id'] == dev_id for d in self.devices):
            messagebox.showwarning("ID Existant", "Cet ID est déjà utilisé.", parent=self.root)
            return
        path = filedialog.askopenfilename(title=f"Sélectionner cache.sqlite pour {dev_id}", filetypes=[("Fichiers SQLite", "*.sqlite"), ("Tous les fichiers", "*.*")])
        if path:
            self.devices.append({'id': dev_id, 'path': path})
            self.update_device_list()
            
    def remove_device(self):
        selected = self.device_list.selection()
        if not selected: return
        dev_id_to_remove = self.device_list.item(selected[0])['values'][0]
        self.devices = [d for d in self.devices if d['id'] != dev_id_to_remove]
        self.update_device_list()

    def update_device_list(self):
        for i in self.device_list.get_children(): self.device_list.delete(i)
        for device in self.devices: self.device_list.insert('', 'end', values=(device['id'], device['path']))

    def toggle_mode(self, *args):
        # Place all frames in a list in the correct order
        all_frames = [
            self.mode_frame, self.single_db_frame, self.multi_db_frame,
            self.date_frame, self.settings_frame, self.output_frame, 
            self.cross_ref_frame, self.action_frame
        ]
        # Forget all of them to ensure a clean slate
        for frame in all_frames:
            frame.pack_forget()

        is_multi = self.is_multi_mode.get()
        
        # Repack them in the correct order based on the mode
        self.mode_frame.pack(fill=tk.X, pady=5)
        if is_multi:
            self.multi_db_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        else:
            self.single_db_frame.pack(fill=tk.X, padx=10, pady=5)
        self.date_frame.pack(fill=tk.X, pady=5)
        self.settings_frame.pack(fill=tk.X, pady=5)
        self.output_frame.pack(fill=tk.X, pady=5)
        if is_multi:
            self.cross_ref_frame.pack(fill=tk.X, pady=5)
        
        self.action_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10,0))


    def start_analysis(self):
        project_name_base = self.output_path.get()
        if not project_name_base: messagebox.showwarning("Sortie Manquante", "Veuillez choisir un dossier et un nom de projet."); return
        try:
            timestamp_str = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
            project_folder_name = f"{timestamp_str}_{os.path.basename(project_name_base)}"
            output_dir = os.path.join(os.path.dirname(project_name_base), project_folder_name)
            os.makedirs(output_dir, exist_ok=True)
            base_output_path = os.path.join(output_dir, os.path.basename(project_name_base))
            self.last_output_dir = output_dir
        except Exception as e:
            messagebox.showerror("Erreur de Dossier", f"Impossible de créer le dossier de sortie.\n{e}"); return

        if self.is_multi_mode.get():
            if not self.devices: messagebox.showwarning("Aucun Appareil", "Veuillez ajouter au moins un appareil."); return
            final_devices = self.devices
        else:
            db_path = self.single_db_path.get()
            if not db_path: messagebox.showwarning("Fichier Manquant", "Veuillez sélectionner un fichier cache.sqlite."); return
            final_devices = [{'id': 'Appareil', 'path': db_path}]
        
        try:
            acc_str = self.accuracy.get().strip(); accuracy = 99999 if not acc_str else int(acc_str)
            min_speed = float(self.min_speed_var.get()) if self.min_speed_var.get().strip() else None
            max_speed = float(self.max_speed_var.get()) if self.max_speed_var.get().strip() else None
            cross_dist = int(self.cross_dist_var.get() or 0); cross_time = int(self.cross_time_var.get() or 0)
            
            tz = ZoneInfo(self.timezone.get())
            start_date_obj, end_date_obj = self.start_date.get_date(), self.end_date.get_date()
            start_dt = datetime(start_date_obj.year, start_date_obj.month, start_date_obj.day, int(self.start_hour_var.get() or 0), int(self.start_min_var.get() or 0), int(self.start_sec_var.get() or 0)).replace(tzinfo=tz)
            end_dt = datetime(end_date_obj.year, end_date_obj.month, end_date_obj.day, int(self.end_hour_var.get() or 0), int(self.end_min_var.get() or 0), int(self.end_sec_var.get() or 0)).replace(tzinfo=tz)
            utc_epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
            start_ts, end_ts = int((start_dt.astimezone(timezone.utc) - utc_epoch).total_seconds()), int((end_dt.astimezone(timezone.utc) - utc_epoch).total_seconds())
            start_date_str, end_date_str = start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e: messagebox.showerror("Erreur de Paramètre", f"Un paramètre est invalide.\n{e}"); return
        
        self.launch_button.config(state="disabled"); self.open_folder_button.pack_forget(); self.progress_bar.pack(fill=tk.X, padx=10, pady=(5,0)); self.progress_label.pack(fill=tk.X, padx=10); self.root.update()
        success = run_analysis(final_devices, base_output_path, start_ts, end_ts, accuracy, min_speed, max_speed, cross_dist, cross_time, self.export_html.get(), self.export_csv.get(), self.export_kml.get(), self.export_graph.get(), self.timezone.get(), start_date_str, end_date_str, self.update_progress)
        self.launch_button.config(state="normal"); self.progress_bar.pack_forget(); self.progress_label.pack_forget()
        
        if success:
            messagebox.showinfo("Succès", f"Analyse terminée ! Fichiers sauvegardés dans :\n{self.last_output_dir}")
            if not self.open_folder_button.winfo_ismapped(): self.open_folder_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5,0), ipady=5)
            if self.export_html.get():
                output_html_path = base_output_path + ".html"
                if os.path.exists(output_html_path):
                    try:
                        if sys.platform == "win32": os.startfile(output_html_path)
                        elif sys.platform == "darwin": subprocess.Popen(["open", output_html_path])
                        else: subprocess.Popen(["xdg-open", output_html_path])
                    except Exception as e: messagebox.showwarning("Ouverture auto échouée", f"Impossible d'ouvrir le HTML:\n{e}")

    def open_output_folder(self):
        if self.last_output_dir and os.path.isdir(self.last_output_dir):
            try:
                if sys.platform == "win32": os.startfile(self.last_output_dir)
                elif sys.platform == "darwin": subprocess.Popen(["open", self.last_output_dir])
                else: subprocess.Popen(["xdg-open", self.last_output_dir])
            except Exception as e: messagebox.showerror("Erreur", f"Impossible d'ouvrir le dossier:\n{e}")

    def update_progress(self, value, text):
        self.progress_bar['value'] = value; self.progress_label['text'] = text
        self.root.update_idletasks()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding="10"); self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- MODE ---
        self.mode_frame = ttk.LabelFrame(self.main_frame, text="Choix du Mode", padding="10")
        ttk.Radiobutton(self.mode_frame, text="Un seul appareil", variable=self.is_multi_mode, value=False).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.mode_frame, text="Plusieurs appareils (comparaison)", variable=self.is_multi_mode, value=True).pack(side=tk.LEFT, padx=5)

        # --- FICHIERS SOURCE ---
        self.single_db_frame = ttk.LabelFrame(self.main_frame, text="Fichier Source", padding="10")
        ttk.Label(self.single_db_frame, text="Chemin vers cache.sqlite:").pack(fill=tk.X)
        entry_frame = ttk.Frame(self.single_db_frame); entry_frame.pack(fill=tk.X)
        ttk.Entry(entry_frame, textvariable=self.single_db_path).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(entry_frame, text="...", width=3, command=self.select_single_db).pack(side=tk.LEFT, padx=(5,0))
        
        self.multi_db_frame = ttk.LabelFrame(self.main_frame, text="Fichiers Source", padding="10")
        device_list_frame = ttk.Frame(self.multi_db_frame); device_list_frame.pack(fill=tk.BOTH, expand=True)
        self.device_list = ttk.Treeview(device_list_frame, columns=('id', 'path'), show='headings', height=4)
        self.device_list.heading('id', text='ID Appareil'); self.device_list.heading('path', text='Chemin Fichier')
        self.device_list.column('id', width=100, anchor='w'); self.device_list.column('path', anchor='w')
        self.device_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        multi_btn_frame = ttk.Frame(self.multi_db_frame); multi_btn_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Button(multi_btn_frame, text="Ajouter", command=self.add_device).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(multi_btn_frame, text="Supprimer", command=self.remove_device).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5,0))

        # --- PÉRIODE ---
        self.date_frame = ttk.LabelFrame(self.main_frame, text="Période et Fuseau Horaire", padding="10")
        ttk.Label(self.date_frame, text="Date de début:").grid(row=0, column=0, sticky="w", pady=2)
        self.start_date = DateEntry(self.date_frame, date_pattern='yyyy-mm-dd', width=12); self.start_date.grid(row=0, column=1, sticky="w")
        time_start_frame = ttk.Frame(self.date_frame); time_start_frame.grid(row=0, column=2, sticky="w", padx=10)
        ttk.Spinbox(time_start_frame, from_=0, to=23, textvariable=self.start_hour_var, format="%02.0f", width=3).pack(side=tk.LEFT); ttk.Label(time_start_frame, text=":").pack(side=tk.LEFT)
        ttk.Spinbox(time_start_frame, from_=0, to=59, textvariable=self.start_min_var, format="%02.0f", width=3).pack(side=tk.LEFT); ttk.Label(time_start_frame, text=":").pack(side=tk.LEFT)
        ttk.Spinbox(time_start_frame, from_=0, to=59, textvariable=self.start_sec_var, format="%02.0f", width=3).pack(side=tk.LEFT)
        ttk.Label(self.date_frame, text="Date de fin:").grid(row=1, column=0, sticky="w", pady=2)
        self.end_date = DateEntry(self.date_frame, date_pattern='yyyy-mm-dd', width=12); self.end_date.grid(row=1, column=1, sticky="w"); self.end_date.set_date(datetime.now())
        time_end_frame = ttk.Frame(self.date_frame); time_end_frame.grid(row=1, column=2, sticky="w", padx=10)
        ttk.Spinbox(time_end_frame, from_=0, to=23, textvariable=self.end_hour_var, format="%02.0f", width=3).pack(side=tk.LEFT); ttk.Label(time_end_frame, text=":").pack(side=tk.LEFT)
        ttk.Spinbox(time_end_frame, from_=0, to=59, textvariable=self.end_min_var, format="%02.0f", width=3).pack(side=tk.LEFT); ttk.Label(time_end_frame, text=":").pack(side=tk.LEFT)
        ttk.Spinbox(time_end_frame, from_=0, to=59, textvariable=self.end_sec_var, format="%02.0f", width=3).pack(side=tk.LEFT)
        ttk.Label(self.date_frame, text="Fuseau Horaire:").grid(row=2, column=0, sticky="w", pady=(10,2))
        tz_combo = ttk.Combobox(self.date_frame, textvariable=self.timezone, values=sorted(available_timezones()), width=30); tz_combo.grid(row=2, column=1, columnspan=2, sticky="ew")

        # --- FILTRES & EXPORTS ---
        self.settings_frame = ttk.LabelFrame(self.main_frame, text="Filtres et Formats d'Export (laisser vide pour tout inclure)", padding="10")
        ttk.Label(self.settings_frame, text="Précision GPS max. (m):").grid(row=0, column=0, sticky="w", pady=2)
        acc_entry = ttk.Entry(self.settings_frame, textvariable=self.accuracy, width=10); acc_entry.grid(row=0, column=1, sticky="w", padx=5)
        createToolTip(acc_entry, "Filtre les points GPS avec une marge d'erreur supérieure à cette valeur.\n65m = OK, 10m = Très bon, 5m = Excellent.")
        ttk.Label(self.settings_frame, text="Vitesse min (km/h):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.min_speed_var, width=10).grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(self.settings_frame, text="Vitesse max (km/h):").grid(row=1, column=2, sticky="w", pady=2, padx=(10,0))
        ttk.Entry(self.settings_frame, textvariable=self.max_speed_var, width=10).grid(row=1, column=3, sticky="w", padx=5)
        export_checks_frame = ttk.Frame(self.settings_frame); export_checks_frame.grid(row=2, column=0, columnspan=4, sticky="w", pady=(10,0))
        ttk.Checkbutton(export_checks_frame, text="Carte HTML", variable=self.export_html).pack(side=tk.LEFT)
        ttk.Checkbutton(export_checks_frame, text="Tableur CSV", variable=self.export_csv).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(export_checks_frame, text="Google Earth KML", variable=self.export_kml).pack(side=tk.LEFT)
        ttk.Checkbutton(export_checks_frame, text="Graphique Vitesse", variable=self.export_graph).pack(side=tk.LEFT, padx=10)

        # --- SORTIE ---
        self.output_frame = ttk.LabelFrame(self.main_frame, text="Fichiers de Sortie", padding="10")
        ttk.Label(self.output_frame, text="Dossier de sortie et nom du projet:").pack(fill=tk.X)
        out_entry_frame = ttk.Frame(self.output_frame); out_entry_frame.pack(fill=tk.X)
        ttk.Entry(out_entry_frame, textvariable=self.output_path).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(out_entry_frame, text="...", width=3, command=self.select_output).pack(side=tk.LEFT, padx=(5,0))

        # --- CROISEMENTS ---
        self.cross_ref_frame = ttk.LabelFrame(self.main_frame, text="Détection de Croisements", padding="10")
        ttk.Label(self.cross_ref_frame, text="Distance max (mètres):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(self.cross_ref_frame, textvariable=self.cross_dist_var, width=10).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(self.cross_ref_frame, text="Intervalle de temps max (secondes):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(self.cross_ref_frame, textvariable=self.cross_time_var, width=10).grid(row=1, column=1, sticky="w", padx=5)
        
        # --- BOUTONS D'ACTION ---
        self.action_frame = ttk.Frame(self.main_frame);
        self.launch_button = ttk.Button(self.action_frame, text="LANCER L'ANALYSE", command=self.start_analysis, style='Accent.TButton')
        self.launch_button.pack(side=tk.LEFT, expand=True, fill=tk.X, ipady=5)
        style = ttk.Style(); style.configure('Accent.TButton', font=('Helvetica', 10, 'bold'))
        self.open_folder_button = ttk.Button(self.action_frame, text="Ouvrir le dossier de sortie", command=self.open_output_folder)

        # --- Barre de progression ---
        self.progress_bar = ttk.Progressbar(self.main_frame, orient='horizontal', length=100, mode='determinate')
        self.progress_label = ttk.Label(self.main_frame, text="Prêt", anchor="center")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    except ImportError as e:
        messagebox.showerror("Dépendance Manquante", f"Bibliothèque manquante: {e.name}\n\nInstallez-la avec:\npip install {e.name}")

