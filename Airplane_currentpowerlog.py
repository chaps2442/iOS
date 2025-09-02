# -----------------------------------------------------------------------------
# Auteur : Vincent Chapeau
# Contact : vincent.chapeau@teeltechcanada.com
# Version : 1.0
# Date : 02/09/2025
#
# Description :
# Cet outil analyse la base de données CurrentPowerlog.PLSQL d'une extraction
# iOS pour identifier les activations et désactivations du Mode Avion.
# Il crée une timeline chronologique en agrégeant plusieurs sources de données :
#   1. Preuve Directe : L'état "airplaneMode" dans la table TelephonyActivity.
#   2. Preuves de Corroboration : L'extinction des puces Wi-Fi et Bluetooth.
#   3. Preuve contextuelle : L'état général du réseau cellulaire.
#
# L'interface graphique, construite avec Tkinter, affiche cette timeline et
# surligne les moments où les événements coïncident, fournissant une forte
# probabilité d'une activation manuelle du Mode Avion. Le script permet
# également d'exporter les résultats en format TXT ou CSV.
# -----------------------------------------------------------------------------

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sqlite3
import pandas as pd
from datetime import timedelta

# --- Fonctions d'analyse (corrigées et plus robustes) ---

def execute_query(conn, query, error_message):
    """Exécute une requête SQL et gère les erreurs de manière générique."""
    try:
        return pd.read_sql_query(query, conn)
    except sqlite3.Error as e:
        # Affiche une seule fois l'erreur pour ne pas spammer l'utilisateur
        if not hasattr(execute_query, "shown_errors"):
            execute_query.shown_errors = set()
        if error_message not in execute_query.shown_errors:
            messagebox.showwarning("Avertissement de Requête", f"{error_message}: La table n'a pas été trouvée ou une erreur est survenue. L'analyse continue sans ces données.\n\nDétail: {e}")
            execute_query.shown_errors.add(error_message)
        return pd.DataFrame()

def get_airplane_mode_events(conn):
    """
    PREUVE DIRECTE : Extrait les événements ON/OFF de la colonne airplaneMode.
    Source: PLBBAgent_EventPoint_TelephonyActivity
    """
    query = """
    SELECT
        datetime(timestamp, 'unixepoch') as 'Timestamp',
        airplaneMode AS 'State'
    FROM
        PLBBAgent_EventPoint_TelephonyActivity
    WHERE
        airplaneMode IS NOT NULL AND airplaneMode != '';
    """
    df = execute_query(conn, query, "Impossible de lire TelephonyActivity.airplaneMode")
    if not df.empty:
        df['EventType'] = df['State'].apply(lambda x: f"Mode Avion {x.upper()}")
        df['Source'] = 'Cellulaire (Direct)'
        return df[['Timestamp', 'EventType', 'Source']]
    return pd.DataFrame()

def get_cellular_status_events(conn):
    """
    PREUVE INDIRECTE : Extrait les changements d'état du réseau (data).
    Source: PLBBAgent_EventPoint_TelephonyActivity
    """
    query = """
    SELECT
        datetime(timestamp, 'unixepoch') as 'Timestamp',
        dataStatus AS 'Status'
    FROM
        PLBBAgent_EventPoint_TelephonyActivity
    WHERE
        Status IS NOT NULL;
    """
    df = execute_query(conn, query, "Impossible de lire TelephonyActivity.dataStatus")
    if not df.empty:
        df['EventType'] = df['Status'].apply(
            lambda x: 'Réseau Cellulaire INACTIF' if x in ('Detached', 'Suspended') else 'Réseau Cellulaire ACTIF'
        )
        df['Source'] = 'Cellulaire (Statut)'
        return df[['Timestamp', 'EventType', 'Source']]
    return pd.DataFrame()

def get_bluetooth_power_events(conn):
    """
    PREUVE DE CORROBORATION : Extrait les moments où le Bluetooth est désactivé.
    Source: PLBluetoothAgent_EventForward_DeviceState
    """
    query = """
    SELECT
        datetime(timestamp, 'unixepoch') as 'Timestamp'
    FROM
        PLBluetoothAgent_EventForward_DeviceState
    WHERE
        DevicePowered = 0;
    """
    df = execute_query(conn, query, "Impossible de lire l'état du Bluetooth")
    if not df.empty:
        df['EventType'] = 'Bluetooth OFF'
        df['Source'] = 'Bluetooth'
        return df
    return pd.DataFrame()
    
def get_wifi_power_events(conn):
    """
    PREUVE DE CORROBORATION : Extrait les moments où le Wi-Fi est désactivé.
    Source: PLWifiAgent_EventBackward_DiffProperties (Corrigé pour utiliser une table existante)
    """
    query = """
    SELECT
        datetime(timestamp, 'unixepoch') as 'Timestamp'
    FROM
        PLWifiAgent_EventBackward_DiffProperties
    WHERE
        WifiPowered = 0;
    """
    df = execute_query(conn, query, "Impossible de lire l'état du Wi-Fi")
    if not df.empty:
        df['EventType'] = 'Wi-Fi OFF'
        df['Source'] = 'Wi-Fi'
        return df
    return pd.DataFrame()

def analyze_powerlog(db_path, result_text, status_label):
    """
    Fonction principale qui orchestre l'analyse et l'affichage.
    """
    status_label.config(text="Analyse en cours... Connexion à la base de données.")
    result_text.config(state=tk.NORMAL)
    result_text.delete('1.0', tk.END)
    result_text.update()

    # Effacer les erreurs précédentes pour la nouvelle analyse
    if hasattr(execute_query, "shown_errors"):
        delattr(execute_query, "shown_errors")

    try:
        # Ouvre la base de données en mode lecture seule pour plus de sécurité
        with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
            status_label.config(text="Analyse en cours... Extraction des données.")
            result_text.update()

            # 1. Extraire toutes les sources de données
            airplane_df = get_airplane_mode_events(conn)
            cellular_df = get_cellular_status_events(conn)
            bluetooth_df = get_bluetooth_power_events(conn)
            wifi_df = get_wifi_power_events(conn)
    except sqlite3.Error as e:
        messagebox.showerror("Erreur Fichier", f"Impossible d'ouvrir ou lire le fichier : {e}")
        status_label.config(text="Erreur.")
        result_text.config(state=tk.DISABLED)
        return

    # 2. Combiner et trier
    all_events = pd.concat([airplane_df, cellular_df, bluetooth_df, wifi_df], ignore_index=True)
    
    if all_events.empty:
        status_label.config(text="Analyse terminée. Aucun événement pertinent trouvé.")
        result_text.insert(tk.END, "Aucun événement pertinent n'a été trouvé.")
        result_text.config(state=tk.DISABLED)
        return
        
    all_events.dropna(subset=['Timestamp'], inplace=True)
    all_events['Timestamp'] = pd.to_datetime(all_events['Timestamp'])
    all_events = all_events.sort_values(by='Timestamp').drop_duplicates().reset_index(drop=True)
    
    # 3. Identifier les clusters (Signature du Mode Avion)
    status_label.config(text="Analyse en cours... Identification des corrélations.")
    result_text.update()
    
    time_window = timedelta(seconds=2)
    all_events['is_cluster'] = False

    airplane_on_events = all_events[all_events['EventType'] == 'Mode Avion ON']
    
    for index, event in airplane_on_events.iterrows():
        start_time = event['Timestamp'] - time_window
        end_time = event['Timestamp'] + time_window
        nearby_events_indices = all_events[
            (all_events['Timestamp'] >= start_time) & (all_events['Timestamp'] <= end_time)
        ].index
        
        all_events.loc[nearby_events_indices, 'is_cluster'] = True

    # 4. Afficher
    result_text.tag_config('title', font=('Helvetica', 14, 'bold', 'underline'), justify='center')
    result_text.tag_config('airplane_on', foreground='#e63946', font=('Courier New', 10, 'bold')) # Rouge
    result_text.tag_config('airplane_off', foreground='#e63946') # Rouge
    result_text.tag_config('wifi', foreground='#1d3557')
    result_text.tag_config('bluetooth', foreground='#2a9d8f')
    result_text.tag_config('cellular', foreground='#fca311') # Orange
    result_text.tag_config('cluster_bg', background='#fff3b0')

    result_text.insert(tk.END, "Timeline des États des Radios\n", 'title')
    result_text.insert(tk.END, "(Les activations du Mode Avion corrélées sont surlignées en jaune)\n\n")

    for _, row in all_events.iterrows():
        timestamp_str = row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        line = f"{timestamp_str} -- {row['EventType']:<28} (Source: {row['Source']})\n"
        
        tags = []
        if 'Mode Avion ON' in row['EventType']:
            tags.append('airplane_on')
        elif 'Mode Avion' in row['EventType']:
            tags.append('airplane_off')
        elif 'Wi-Fi' in row['EventType']:
            tags.append('wifi')
        elif 'Bluetooth' in row['EventType']:
            tags.append('bluetooth')
        else:
            tags.append('cellular')
        
        if row['is_cluster']:
            tags.append('cluster_bg')
            
        result_text.insert(tk.END, line, tuple(tags))

    result_text.config(state=tk.DISABLED)
    status_label.config(text="Analyse terminée.")
    PowerlogAnalyzerApp.last_results_df = all_events


# --- Interface Graphique (Tkinter) ---

class PowerlogAnalyzerApp:
    last_results_df = None

    def __init__(self, root):
        self.root = root
        self.root.title("Analyseur PowerLog pour Mode Avion (v3.1)")
        self.root.geometry("950x700")
        
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.load_button = ttk.Button(control_frame, text="1. Charger Fichier", command=self.load_file)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.analyze_button = ttk.Button(control_frame, text="2. Lancer l'analyse", command=self.run_analysis, state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=5)

        self.export_button = ttk.Button(control_frame, text="3. Exporter", command=self.export_timeline, state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        self.file_path_label = ttk.Label(control_frame, text="Aucun fichier chargé", foreground="grey", wraplength=450)
        self.file_path_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.result_text = tk.Text(result_frame, wrap=tk.NONE, font=("Courier New", 10), state=tk.DISABLED)
        v_scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        h_scroll = ttk.Scrollbar(result_frame, orient=tk.HORIZONTAL, command=self.result_text.xview)
        self.result_text.config(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(main_frame, text="Prêt", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.db_path = ""

    def load_file(self):
        path = filedialog.askopenfilename(
            title="Sélectionnez le fichier CurrentPowerlog.PLSQL",
            filetypes=(("Database Files", "*.plsql *.sqlite *.db"), ("Tous les fichiers", "*.*"))
        )
        if path:
            self.db_path = path
            self.file_path_label.config(text=path)
            self.analyze_button.config(state=tk.NORMAL)
            self.export_button.config(state=tk.DISABLED)
            self.status_label.config(text="Fichier chargé. Prêt pour l'analyse.")
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete('1.0', tk.END)
            self.result_text.config(state=tk.DISABLED)

    def run_analysis(self):
        if self.db_path:
            analyze_powerlog(self.db_path, self.result_text, self.status_label)
            if PowerlogAnalyzerApp.last_results_df is not None and not PowerlogAnalyzerApp.last_results_df.empty:
                self.export_button.config(state=tk.NORMAL)
        else:
            messagebox.showwarning("Aucun fichier", "Veuillez d'abord charger un fichier.")
            
    def export_timeline(self):
        if PowerlogAnalyzerApp.last_results_df is None or PowerlogAnalyzerApp.last_results_df.empty:
            messagebox.showwarning("Aucune donnée", "Veuillez d'abord lancer une analyse.")
            return
            
        path = filedialog.asksaveasfilename(
            title="Enregistrer les résultats",
            defaultextension=".txt",
            filetypes=(("Fichier Texte", "*.txt"), ("Fichier CSV", "*.csv"))
        )
        
        if not path:
            return
            
        try:
            if path.lower().endswith('.txt'):
                timeline_content = self.result_text.get("1.0", tk.END)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(timeline_content)
                messagebox.showinfo("Export réussi", f"Timeline sauvegardée dans {path}")
            elif path.lower().endswith('.csv'):
                PowerlogAnalyzerApp.last_results_df.to_csv(path, index=False, encoding='utf-8-sig')
                messagebox.showinfo("Export réussi", f"Données CSV sauvegardées dans {path}")
        except Exception as e:
            messagebox.showerror("Erreur d'export", f"Une erreur est survenue: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PowerlogAnalyzerApp(root)
    root.mainloop()
