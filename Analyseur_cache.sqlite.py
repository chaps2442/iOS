# Name: AnalyseurGPS_autonome.py
# Analyse de cache.sqlite d'iOS 17
# Authors: Vincent Chapeau & Burri Xavier
# vincent.chapeau@teeltechcanada.com
# df.xavierburri@protonmail.ch
# Version: 22.1 - Correctif de la fonction d'analyse
#
# --- PRÉREQUIS D'INSTALLATION ---
# pip install folium pandas tkcalendar simplekml matplotlib
# ------------------------------------

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
import os
import subprocess
import sys

def run_analysis(devices, base_output_path, start_ts, end_ts, accuracy_threshold, min_speed_kmh, max_speed_kmh, export_html, export_csv, export_kml, export_graph, selected_timezone, start_date_str, end_date_str, progress_callback):
    all_dfs = []
    local_tz = ZoneInfo(selected_timezone)
    min_speed_mps = float(min_speed_kmh) / 3.6 if min_speed_kmh else None
    max_speed_mps = float(max_speed_kmh) / 3.6 if max_speed_kmh else None

    # 1. Traitement de chaque appareil
    for i, device in enumerate(devices):
        progress_callback(i / len(devices) * 20, f"Lecture de {device['id']}...")
        try:
            with sqlite3.connect(device['path']) as conn:
                base_query = f"""
                    SELECT ZLATITUDE as LAT, ZLONGITUDE as LONG, ZHORIZONTALACCURACY as ACCURACY,
                           ZSPEED as SPEED, ZTIMESTAMP as TS, ZCOURSE as COURSE, ZALTITUDE as ALTITUDE,
                           ZVERTICALACCURACY as V_ACCURACY, ZSPEEDACCURACY as SPEED_ACCURACY
                    FROM ZRTCLLOCATIONMO
                    WHERE ZTIMESTAMP BETWEEN {start_ts} AND {end_ts}
                      AND ZHORIZONTALACCURACY <= {accuracy_threshold}
                """
                if min_speed_mps is not None: base_query += f" AND ZSPEED >= {min_speed_mps}"
                if max_speed_mps is not None: base_query += f" AND ZSPEED <= {max_speed_mps}"
                base_query += " ORDER BY ZTIMESTAMP ASC"
                df = pd.read_sql_query(base_query, conn)
                if not df.empty:
                    df['id'] = device['id']
                    all_dfs.append(df)
        except Exception as e:
            messagebox.showerror("Erreur Base de Données", f"Impossible de lire le fichier pour {device['id']}:\n{e}"); return False
            
    if not all_dfs:
        messagebox.showinfo("Aucune Donnée", "Aucune donnée de localisation trouvée pour les critères sélectionnés."); return False

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['TS'] = pd.to_numeric(combined_df['TS'], errors='coerce')
    combined_df.dropna(subset=['TS'], inplace=True)
    combined_df.sort_values(by='TS', inplace=True)
    
    if combined_df.empty:
        messagebox.showinfo("Aucune Donnée Valide", "Aucune donnée avec un timestamp valide n'a été trouvée après nettoyage."); return False

    def convert_core_data_timestamp(ts): return datetime(2001, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=ts)
    def convert_mps_to_kmh(mps): return mps * 3.6 if pd.notna(mps) and mps >= 0 else np.nan
    def format_value(val, unit=""): return f"{val:.2f}{unit}" if pd.notna(val) and val >= 0 else "N/A"

    # 2. Exports de données
    df_export = pd.DataFrame()
    if export_csv or export_kml or export_graph:
        progress_callback(30, "Préparation des données pour les exports...")
        df_export['Appareil_ID'] = combined_df['id']
        df_export[f'Timestamp_{local_tz.key.replace("/", "_")}'] = combined_df['TS'].apply(lambda ts: convert_core_data_timestamp(ts).astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S'))
        df_export['Timestamp_UTC'] = combined_df['TS'].apply(lambda ts: convert_core_data_timestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
        df_export['Latitude'] = combined_df['LAT']; df_export['Longitude'] = combined_df['LONG']; df_export['Altitude_m'] = combined_df['ALTITUDE']
        df_export['Vitesse_kmh'] = combined_df['SPEED'].apply(convert_mps_to_kmh); df_export['Direction_deg'] = combined_df['COURSE']
        df_export['Precision_GPS_m'] = combined_df['ACCURACY']; df_export['Precision_Altitude_m'] = combined_df['V_ACCURACY']; df_export['Precision_Vitesse_kmh'] = combined_df['SPEED_ACCURACY'].apply(convert_mps_to_kmh)
        
        if export_csv:
            progress_callback(40, "Génération du fichier CSV...")
            df_export.to_csv(base_output_path + ".csv", index=False, sep=';', encoding='utf-8-sig', float_format='%.6f')

        if export_kml:
            progress_callback(50, "Génération du fichier KML...")
            kml = simplekml.Kml(name=f"Analyse GPS - {os.path.basename(base_output_path)}")
            for device_id, group in df_export.groupby('Appareil_ID'):
                folder = kml.newfolder(name=device_id)
                for _, row in group.iterrows():
                    pnt = folder.newpoint(name=row[f'Timestamp_{local_tz.key.replace("/", "_")}'][11:])
                    pnt.coords = [(row['Longitude'], row['Latitude'], row['Altitude_m'])]
                    pnt.extendeddata = simplekml.ExtendedData()
                    pnt.extendeddata.newdata('Appareil', value=str(row['Appareil_ID']))
                    pnt.extendeddata.newdata('Heure Locale', value=str(row[f'Timestamp_{local_tz.key.replace("/", "_")}']))
                    pnt.extendeddata.newdata('Heure UTC', value=str(row['Timestamp_UTC']))
                    pnt.extendeddata.newdata('Vitesse', value=format_value(row['Vitesse_kmh'], ' km/h'))
                    pnt.extendeddata.newdata('Direction', value=format_value(row['Direction_deg'], '°'))
                    pnt.extendeddata.newdata('Altitude', value=format_value(row['Altitude_m'], ' m'))
                    pnt.extendeddata.newdata('Précision GPS', value=f"±{format_value(row['Precision_GPS_m'], ' m')}")
            kml.save(base_output_path + ".kml")

    # 3. Génération de la carte HTML
    if export_html:
        progress_callback(60, "Création de la carte animée...")
        map_center = [combined_df['LAT'].mean(), combined_df['LONG'].mean()]
        m = folium.Map(location=map_center, zoom_start=15)
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred']
        
        all_features_animation = []
        for i, device_id in enumerate(combined_df['id'].unique()):
            device_df = combined_df[combined_df['id'] == device_id]
            if device_df.empty: continue
            color = colors[i % len(colors)]
            
            fg_line = folium.FeatureGroup(name=f"Trajet {device_id}", show=False)
            fg_direction = folium.FeatureGroup(name=f"Directions {device_id}", show=False)
            fg_accuracy = folium.FeatureGroup(name=f"Précision {device_id}", show=False)
            
            points = list(zip(device_df['LAT'], device_df['LONG']))
            fg_line.add_child(folium.PolyLine(locations=points, color=color, weight=2.5, opacity=0.8))
            
            for _, row in device_df.iterrows():
                if pd.notna(row['COURSE']) and row['COURSE'] >= 0:
                    arrow_svg = f"""
                        <div style="transform: rotate({int(row['COURSE'])}deg); width: 24px; height: 24px; transform-origin: center center;">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}" stroke="white" stroke-width="1.5">
                                <path d="M12 2L2 22L12 17L22 22L12 2Z" />
                            </svg>
                        </div>"""
                    folium.Marker(
                        location=[row['LAT'], row['LONG']],
                        icon=DivIcon(icon_size=(24,24), icon_anchor=(12,12), html=arrow_svg)
                    ).add_to(fg_direction)

                if pd.notna(row['ACCURACY']) and row['ACCURACY'] > 0:
                    folium.Circle(location=[row['LAT'], row['LONG']], radius=row['ACCURACY'], color=color, fill_color=color, fill_opacity=0.1, weight=1).add_to(fg_accuracy)
            
            m.add_child(fg_line); m.add_child(fg_direction); m.add_child(fg_accuracy)
            
            for _, row in device_df.iterrows():
                utc_time = convert_core_data_timestamp(row['TS']); local_time = utc_time.astimezone(local_tz)
                popup = (f"<b>{row['id']}</b><br><hr>"
                         f"Date & Heure ({local_tz.key.split('/')[-1]}):<br><b>{local_time.strftime('%Y-%m-%d %H:%M:%S')}</b><br>"
                         f"Vitesse: {format_value(convert_mps_to_kmh(row['SPEED']), ' km/h')}<br>"
                         f"<b>Direction: {format_value(row['COURSE'], '°')}</b><br>"
                         f"Altitude: {format_value(row['ALTITUDE'], ' m')}<br>"
                         f"Précision: ±{format_value(row['ACCURACY'], ' m')}")
                all_features_animation.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [row['LONG'], row['LAT']]},'properties': {'time': utc_time.isoformat(), 'icon': 'circle', 'iconstyle': {'fillColor': color, 'fillOpacity': 0.8, 'stroke': 'true', 'radius': 7},'popup': popup}})
        
        if all_features_animation: TimestampedGeoJson({'type': 'FeatureCollection', 'features': all_features_animation}, period='PT1S', add_last_point=True, auto_play=False, loop=False, max_speed=10, loop_button=True, date_options='HH:mm:ss', time_slider_drag_update=True).add_to(m)
        folium.LayerControl().add_to(m)
        m.save(base_output_path + ".html")

    # 4. Génération du graphique de vitesse
    if export_graph:
        progress_callback(95, "Génération du graphique de vitesse...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred']
        for i, device_id in enumerate(df_export['Appareil_ID'].unique()):
            device_df_graph = df_export[df_export['Appareil_ID'] == device_id].copy()
            device_df_graph['plot_time'] = pd.to_datetime(device_df_graph[f'Timestamp_{local_tz.key.replace("/", "_")}'])
            ax.plot(device_df_graph['plot_time'], device_df_graph['Vitesse_kmh'], label=device_id, color=colors[i % len(colors)], marker='o', linestyle='-', markersize=3)
        ax.set_title(f"Évolution de la Vitesse - {os.path.basename(base_output_path)}", fontsize=16)
        ax.set_xlabel(f"Temps ({selected_timezone})", fontsize=12); ax.set_ylabel("Vitesse (km/h)", fontsize=12)
        ax.legend(); ax.grid(True); ax.tick_params(axis='x', rotation=45); fig.tight_layout()
        plt.savefig(base_output_path + "_vitesse.png", dpi=300); plt.close(fig)

    # 5. Génération du fichier README
    accuracy_display = "Toute" if accuracy_threshold >= 99999 else f"<= {accuracy_threshold} m"
    min_speed_display = "Aucun" if min_speed_kmh is None else f">= {min_speed_kmh} km/h"
    max_speed_display = "Aucune" if max_speed_kmh is None else f"<= {max_speed_kmh} km/h"
    source_files_str = ', '.join([os.path.basename(d['path']) for d in devices])

    readme_content = f"""RAPPORT D'ANALYSE GPS - PROJET "{os.path.basename(base_output_path)}"
=======================================================
Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARAMÈTRES DE L'ANALYSE UTILISÉS
-------------------------------------
Fichier(s) source: {source_files_str}
Période (Fuseau: {selected_timezone}):
  - Début: {start_date_str}
  - Fin:   {end_date_str}
Filtres:
  - Précision GPS max: {accuracy_display}
  - Vitesse min: {min_speed_display}
  - Vitesse max: {max_speed_display}

DESCRIPTION DES COLONNES (Fichiers .CSV et .KML)
-------------------------------------------------
- Appareil_ID: L'identifiant unique donné à chaque appareil lors de l'analyse.
- Timestamp_{local_tz.key.replace("/", "_")}: Date et heure dans le fuseau horaire sélectionné ({selected_timezone}).
- Timestamp_UTC: Date et heure en Temps Universel Coordonné (UTC). C'est la référence absolue.
- Latitude: Coordonnée GPS Nord-Sud (en degrés décimaux).
- Longitude: Coordonnée GPS Est-Ouest (en degrés décimaux).
- Altitude_m: Altitude au-dessus du niveau de la mer, en mètres.
- Vitesse_kmh: Vitesse calculée, en kilomètres par heure.
- Direction_deg: Cap du mouvement en degrés par rapport au Nord (0°=Nord, 90°=Est, 180°=Sud, 270°=Ouest).
- Precision_GPS_m: Marge d'erreur horizontale de la position GPS, en mètres. Un chiffre bas indique une meilleure précision.
- Precision_Altitude_m: Marge d'erreur verticale de l'altitude, en mètres.
- Precision_Vitesse_kmh: Marge d'erreur de la mesure de vitesse, en km/h.

FICHIERS GÉNÉRÉS
-----------------
- .html: Une carte interactive et animée, visible dans un navigateur web.
- .csv: Un fichier tableur contenant toutes les données extraites (Excel, etc.).
- .kml: Un fichier pour les logiciels de type SIG (Google Earth Pro).
- _vitesse.png: Un graphique de l'évolution de la vitesse au cours du temps.
"""
    with open(base_output_path + "_readme.txt", "w", encoding="utf-8") as f: f.write(readme_content)
    
    progress_callback(100, "Terminé !"); messagebox.showinfo("Succès", "Analyse terminée !")
    return True

class App:
    def __init__(self, root):
        self.root = root; self.root.title("Analyseur GPS iOS - Édition Finale"); self.root.minsize(550, 680)
        self.devices = []; self.is_multi_mode = tk.BooleanVar(value=False); self.is_multi_mode.trace_add("write", self.toggle_mode)
        self.single_db_path = tk.StringVar(); self.output_path = tk.StringVar()
        self.accuracy = tk.StringVar(value=""); self.timezone = tk.StringVar(value="Europe/Brussels")
        self.export_html = tk.BooleanVar(value=True); self.export_csv = tk.BooleanVar(value=True); self.export_kml = tk.BooleanVar(value=True); self.export_graph = tk.BooleanVar(value=True)
        self.start_hour_var, self.start_min_var, self.start_sec_var = tk.StringVar(value="00"), tk.StringVar(value="00"), tk.StringVar(value="00")
        self.end_hour_var, self.end_min_var, self.end_sec_var = tk.StringVar(value="23"), tk.StringVar(value="59"), tk.StringVar(value="59")
        self.min_speed_var, self.max_speed_var = tk.StringVar(), tk.StringVar()
        self.last_output_dir = None
        self.create_widgets(); self.toggle_mode()

    def _validate_time_input(self, P, max_val_str):
        max_val = int(max_val_str)
        if P == "" or (P.isdigit() and len(P) <= 2 and int(P) <= max_val): return True
        return False

    def _auto_advance(self, event, next_widget):
        if len(event.widget.get()) >= 2:
            next_widget.focus_set(); next_widget.selection_range(0, tk.END)

    def toggle_mode(self, *args):
        if self.is_multi_mode.get(): self.single_device_frame.pack_forget(); self.multi_device_frame.pack(fill=tk.X, pady=5)
        else: self.multi_device_frame.pack_forget(); self.single_device_frame.pack(fill=tk.X, pady=5)
        
    def add_device(self):
        path = filedialog.askopenfilename(title="Sélectionner le fichier cache.sqlite", filetypes=[("Database Files", "*.sqlite *.db"), ("All files", "*.*")]);
        if not path: return
        identifier = simpledialog.askstring("Identifiant", "Entrez un nom pour cet appareil (ex: Victime):", parent=self.root)
        if not identifier: return
        self.devices.append({'id': identifier, 'path': path}); self.update_device_list()
        
    def remove_device(self):
        selected_indices = self.device_listbox.curselection()
        if not selected_indices: return
        for i in sorted(selected_indices, reverse=True): del self.devices[i]
        self.update_device_list()
        
    def update_device_list(self):
        self.device_listbox.delete(0, tk.END)
        for device in self.devices: self.device_listbox.insert(tk.END, f"{device['id']} ({os.path.basename(device['path'])})")
        
    def start_analysis(self):
        if self.is_multi_mode.get():
            if not self.devices: messagebox.showwarning("Aucun Appareil", "Veuillez ajouter au moins un appareil."); return
            final_devices = self.devices
        else:
            if not self.single_db_path.get(): messagebox.showwarning("Fichier Manquant", "Veuillez sélectionner une base de données."); return
            final_devices = [{'id': 'Appareil', 'path': self.single_db_path.get()}]
        if not self.output_path.get(): messagebox.showwarning("Sortie Manquante", "Veuillez définir un nom de projet pour la sortie."); return
        try:
            acc_str = self.accuracy.get().strip()
            accuracy = 99999 if not acc_str or acc_str.upper() in ['TOUT', 'ALL'] else int(acc_str)
            min_speed = self.min_speed_var.get().strip() or None; max_speed = self.max_speed_var.get().strip() or None
            if min_speed: float(min_speed)
            if max_speed: float(max_speed)
            tz = ZoneInfo(self.timezone.get())
            start_date_str = f"{self.start_date.get_date()} {self.start_hour_var.get()}:{self.start_min_var.get()}:{self.start_sec_var.get()}"
            end_date_str = f"{self.end_date.get_date()} {self.end_hour_var.get()}:{self.end_min_var.get()}:{self.end_sec_var.get()}"
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S').astimezone(tz)
            end_dt = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').astimezone(tz)
            start_ts = int((start_dt.astimezone(timezone.utc) - datetime(2001, 1, 1, tzinfo=timezone.utc)).total_seconds())
            end_ts = int((end_dt.astimezone(timezone.utc) - datetime(2001, 1, 1, tzinfo=timezone.utc)).total_seconds())
        except ValueError: messagebox.showerror("Erreur de Valeur", "La précision et les vitesses doivent être des nombres valides."); return
        except Exception as e: messagebox.showerror("Erreur de Date", f"Format de date invalide.\n{e}"); return
        
        self.launch_button.config(state="disabled")
        self.open_folder_button.pack_forget()
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(5,0)); self.progress_label.pack(fill=tk.X, padx=10); self.root.update()
        
        success = run_analysis(final_devices, self.output_path.get(), start_ts, end_ts, accuracy, min_speed, max_speed, self.export_html.get(), self.export_csv.get(), self.export_kml.get(), self.export_graph.get(), self.timezone.get(), start_date_str, end_date_str, self.update_progress)
        
        self.launch_button.config(state="normal"); self.progress_bar.pack_forget(); self.progress_label.pack_forget()

        if success:
            self.last_output_dir = os.path.dirname(self.output_path.get())
            self.open_folder_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5,5), ipady=5)
        
    def update_progress(self, value, text):
        self.progress_bar['value'] = value; self.progress_label['text'] = text; self.root.update_idletasks()
        
    def open_output_folder(self):
        if not self.last_output_dir or not os.path.isdir(self.last_output_dir):
            messagebox.showwarning("Dossier introuvable", "Le dossier de sortie n'a pas pu être trouvé.")
            return
        if sys.platform == "win32": os.startfile(self.last_output_dir)
        elif sys.platform == "darwin": subprocess.Popen(["open", self.last_output_dir])
        else: subprocess.Popen(["xdg-open", self.last_output_dir])

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        mode_frame = ttk.LabelFrame(main_frame, text="1. Mode d'analyse", padding="10"); mode_frame.pack(fill=tk.X, pady=3)
        ttk.Checkbutton(mode_frame, text="Synchroniser plusieurs appareils", variable=self.is_multi_mode).pack(anchor="w")
        source_frame = ttk.LabelFrame(main_frame, text="2. Source(s) de données", padding="10"); source_frame.pack(fill=tk.X, pady=3)
        self.single_device_frame = ttk.Frame(source_frame, padding=5); ttk.Label(self.single_device_frame, text="Base de données:").grid(row=0, column=0, sticky="w"); ttk.Entry(self.single_device_frame, textvariable=self.single_db_path).grid(row=0, column=1, sticky="ew"); ttk.Button(self.single_device_frame, text="...", width=3, command=lambda: self.single_db_path.set(filedialog.askopenfilename(title="Select DB", filetypes=[("DB Files", "*.sqlite *.db"), ("All", "*.*")]))).grid(row=0, column=2); self.single_device_frame.columnconfigure(1, weight=1)
        self.multi_device_frame = ttk.Frame(source_frame, padding=5); self.device_listbox = tk.Listbox(self.multi_device_frame, height=3); self.device_listbox.pack(fill=tk.X, expand=True, pady=5); btn_frame = ttk.Frame(self.multi_device_frame); btn_frame.pack(fill=tk.X); ttk.Button(btn_frame, text="Ajouter un appareil...", command=self.add_device).pack(side=tk.LEFT, expand=True, fill=tk.X); ttk.Button(btn_frame, text="Supprimer la sélection", command=self.remove_device).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        time_frame = ttk.LabelFrame(main_frame, text="3. Période", padding="10"); time_frame.pack(fill=tk.X, pady=3)
        vcmd_hour = (self.root.register(self._validate_time_input), '%P', '23'); vcmd_min_sec = (self.root.register(self._validate_time_input), '%P', '59')
        ttk.Label(time_frame, text="Début:").grid(row=0, column=0); self.start_date = DateEntry(time_frame, date_pattern='y-mm-dd'); self.start_date.grid(row=0, column=1)
        self.start_hour_spin = ttk.Spinbox(time_frame, from_=0, to=23, width=3, format="%02.0f", textvariable=self.start_hour_var, validate="key", validatecommand=vcmd_hour); self.start_hour_spin.grid(row=0, column=2)
        self.start_min_spin = ttk.Spinbox(time_frame, from_=0, to=59, width=3, format="%02.0f", textvariable=self.start_min_var, validate="key", validatecommand=vcmd_min_sec); self.start_min_spin.grid(row=0, column=3)
        self.start_sec_spin = ttk.Spinbox(time_frame, from_=0, to=59, width=3, format="%02.0f", textvariable=self.start_sec_var, validate="key", validatecommand=vcmd_min_sec); self.start_sec_spin.grid(row=0, column=4)
        ttk.Label(time_frame, text="Fin:").grid(row=1, column=0); self.end_date = DateEntry(time_frame, date_pattern='y-mm-dd'); self.end_date.grid(row=1, column=1)
        self.end_hour_spin = ttk.Spinbox(time_frame, from_=0, to=23, width=3, format="%02.0f", textvariable=self.end_hour_var, validate="key", validatecommand=vcmd_hour); self.end_hour_spin.grid(row=1, column=2)
        self.end_min_spin = ttk.Spinbox(time_frame, from_=0, to=59, width=3, format="%02.0f", textvariable=self.end_min_var, validate="key", validatecommand=vcmd_min_sec); self.end_min_spin.grid(row=1, column=3)
        self.end_sec_spin = ttk.Spinbox(time_frame, from_=0, to=59, width=3, format="%02.0f", textvariable=self.end_sec_var, validate="key", validatecommand=vcmd_min_sec); self.end_sec_spin.grid(row=1, column=4)
        self.start_hour_spin.bind('<KeyRelease>', lambda e: self._auto_advance(e, self.start_min_spin)); self.start_min_spin.bind('<KeyRelease>', lambda e: self._auto_advance(e, self.start_sec_spin))
        self.end_hour_spin.bind('<KeyRelease>', lambda e: self._auto_advance(e, self.end_min_spin)); self.end_min_spin.bind('<KeyRelease>', lambda e: self._auto_advance(e, self.end_sec_spin))
        ttk.Label(time_frame, text="Fuseau:").grid(row=2, column=0); ttk.Combobox(time_frame, textvariable=self.timezone, values=sorted(list(available_timezones()))).grid(row=2,column=1,columnspan=4,sticky="ew")
        
        filter_frame = ttk.LabelFrame(main_frame, text="4. Filtres (laisser vide pour ignorer)", padding="10"); filter_frame.pack(fill=tk.X, pady=3)
        ttk.Label(filter_frame, text="Précision GPS max (m):").grid(row=0, column=0, sticky="w"); ttk.Entry(filter_frame, textvariable=self.accuracy, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(filter_frame, text="Vitesse min (km/h):").grid(row=1, column=0, sticky="w", pady=(5,0)); ttk.Entry(filter_frame, textvariable=self.min_speed_var, width=10).grid(row=1, column=1, sticky="w", pady=(5,0))
        ttk.Label(filter_frame, text="Vitesse max (km/h):").grid(row=1, column=2, sticky="w", padx=(10,0), pady=(5,0)); ttk.Entry(filter_frame, textvariable=self.max_speed_var, width=10).grid(row=1, column=3, sticky="w", pady=(5,0))
        
        output_frame = ttk.LabelFrame(main_frame, text="5. Sortie", padding="10"); output_frame.pack(fill=tk.X, pady=3)
        ttk.Label(output_frame, text="Formats:").grid(row=0, column=0, sticky="w", padx=5)
        format_checks_frame = ttk.Frame(output_frame); ttk.Checkbutton(format_checks_frame, text="HTML", variable=self.export_html).pack(side=tk.LEFT); ttk.Checkbutton(format_checks_frame, text="CSV", variable=self.export_csv).pack(side=tk.LEFT, padx=5); ttk.Checkbutton(format_checks_frame, text="KML", variable=self.export_kml).pack(side=tk.LEFT); ttk.Checkbutton(format_checks_frame, text="Graphique", variable=self.export_graph).pack(side=tk.LEFT, padx=5)
        format_checks_frame.grid(row=0, column=1, sticky="w")
        ttk.Label(output_frame, text="Nom projet sortie:").grid(row=1, column=0, sticky="w", padx=5, pady=5); ttk.Entry(output_frame, textvariable=self.output_path).grid(row=1, column=1, sticky="ew", pady=5); ttk.Button(output_frame, text="Choisir...", width=8, command=lambda: self.output_path.set(filedialog.asksaveasfilename(title="Output name"))).grid(row=1, column=2, padx=5, pady=5); output_frame.columnconfigure(1, weight=1)
        
        self.action_button_frame = ttk.Frame(main_frame); self.action_button_frame.pack(pady=10, fill=tk.X)
        self.launch_button = ttk.Button(self.action_button_frame, text="Lancer l'Analyse", command=self.start_analysis, style="Accent.TButton"); self.launch_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5,5), ipady=5); ttk.Style().configure("Accent.TButton", font=("Helvetica", 12, "bold"))
        self.open_folder_button = ttk.Button(self.action_button_frame, text="Ouvrir le dossier de sortie", command=self.open_output_folder)
        
        self.progress_bar = ttk.Progressbar(main_frame, orient='horizontal', mode='determinate'); self.progress_label = ttk.Label(main_frame, text="", font=("Helvetica", 8, "italic"))
        author_frame = ttk.Frame(main_frame); author_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        ttk.Label(author_frame, text="Développé par V. Chapeau & X. Burri - Testé avec iOS 17", font=("Helvetica", 8)).pack()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    except ImportError as e:
        error_msg = f"Une bibliothèque est manquante: {e.name}\n\nVeuillez l'installer avec la commande:\npip install {e.name}"
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Dépendance Manquante", error_msg)
