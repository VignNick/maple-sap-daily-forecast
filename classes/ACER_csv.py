import pandas as pd
import re
from datetime import datetime, timedelta
import os

'''
ACER_csv.py, par Nicolas Vigneault janvier 2026
Ce script lit un fichier Excel contenant des données de volume journalier, Brix, nombre d'entaille et système,
puis reconstruit un fichier CSV propre avec les colonnes suivantes : date, system, volume, nb entaille, volume entaille, brix et source.

Le script est conçu pour être robuste face aux variations de noms de colonnes et aux formats de date, y compris les dates juliennes.
Il peut traiter plusieurs feuilles dans un fichier Excel et gère les erreurs de manière conviviale.

-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
USAGE: python ACER_csv.py
    - Le script demandera le chemin du fichier Excel à traiter.
    - Le script demandera le chemin du fichier CSV de sortie.

Alternative: ACER_csv.main(input_path="chemin/vers/fichier.xlsx", output_path="chemin/vers/sortie.csv", verbose=True)
'''

def convert_julian_date(julian_value, year):
    '''Convertit une valeur de jour Julien (1-365) en datetime.date, de manière sécurisée.'''
    try:
        base = datetime(int(year), 1, 1)
        return base + timedelta(days=int(julian_value) - 1)
    except Exception:
        return pd.NaT

def main(input_path: str = None, output_path: str = None, verbose: bool = False):
    def verbose_print(message):
        if verbose:
            print(message)

    while input_path is None:
        _in = input("Entrer le chemin du fichier Excel: ").strip()
        if os.path.isfile(_in):
            if _in.lower().endswith(('.xls', '.xlsx')):
                input_path = _in
            else:
                print("⚠️ Le fichier doit être un fichier Excel avec une extension .xls ou .xlsx. Veuillez réessayer.")
        else:
            print("⚠️ Le fichier n'existe pas. Veuillez réessayer.")

    try:
        all_sheets = pd.read_excel(input_path, sheet_name=None)
    except Exception as e:
        raise RuntimeError(f"❌ Échec de la lecture du fichier Excel : {e}")

    verbose_print(f"📂 Chargé {len(all_sheets)} feuilles depuis '{input_path}'")

    cleaned_data = []
    for sheet_name, df in all_sheets.items():
        verbose_print(f"\n=== Traitement de la feuille : {sheet_name} ===")

        # Normaliser les noms de colonnes en supprimant les espaces superflus
        df.columns = df.columns.str.strip()

        # Trouver tous les suffixes basés sur 'Volume journalier'
        suffixes = []
        for col in df.columns:
            match = re.match(r"Volume\s*journalier.*?(\.\w+)?$", col)
            if match:
                suffix = match.group(1) if match.group(1) else ""
                suffixes.append(suffix)
        suffixes = sorted(set(suffixes))

        # Trouver la colonne de date dynamiquement
        date_col = next((c for c in df.columns if re.search(r"date|jour", c, re.IGNORECASE)), None)
        # Si aucune colonne de date trouvée, passer cette feuille
        if not date_col:
            verbose_print(f"⚠️ Aucune colonne de date trouvée — passage de la feuille '{sheet_name}'.")
            continue
        # Détecter si des dates juliennes sont utilisées en se basant sur le nom de la colonne
        if re.search(r"julien", date_col, re.IGNORECASE):
            verbose_print("⚠️ Format de date julienne détecté basé sur le nom de la colonne.")
            df[date_col] = df[date_col].map(lambda x: convert_julian_date(x, sheet_name))
            
        # Corriger la colonne de date en datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        # Vérifier que l'année de la date correspond au nom de la feuille
        if not all(df[date_col].dt.year == int(sheet_name)):
            print(f"⚠️ Les années de la colonne de date ne correspondent pas au nom de la feuille '{sheet_name}' — veuillez vérifier les données.")
            print(f"Les années trouvées dans la colonne de date : {df[date_col].dt.year.unique()}, voulez-vous les remplacer par '{sheet_name}' ? (y/n)")
            user_input = input().strip().lower()
            if user_input == 'y':
                df[date_col] = df[date_col].apply(lambda d: d.replace(year=int(sheet_name)) if pd.notnull(d) else d)
            else:
                verbose_print(f"⛔ Ignorer la feuille '{sheet_name}' en raison d'une incompatibilité d'année.")
                continue

        for suffix in suffixes:

            # Build patterns using the full suffix (including dot if present)
            vol_pattern = rf"^Volume\s*journalier.*?{re.escape(suffix)}$"
            brix_pattern = rf"^Brix{re.escape(suffix)}$"
            nb_pattern = rf"^Nb\s*entaille.*?{re.escape(suffix)}$"
            sys_pattern = rf"^Système{re.escape(suffix)}$"

            # Match actual columns
            vol_col = next((c for c in df.columns if re.match(vol_pattern, c)), None)
            brix_col = next((c for c in df.columns if re.match(brix_pattern, c)), None)
            nb_col = next((c for c in df.columns if re.match(nb_pattern, c)), None)
            sys_col = next((c for c in df.columns if re.match(sys_pattern, c)), None)

            if not vol_col:
                verbose_print("⚠️ Aucune colonne de volume trouvée pour ce suffixe —> ignorer.")
                continue

            sub_df = pd.DataFrame({
                "date": df[date_col] if date_col else None,
                "system": df[sys_col] if sys_col else None,
                "volume": df[vol_col] if vol_col else None,
                "nb entaille": df[nb_col] if nb_col else None,
                "volume entaille": (df[vol_col] / df[nb_col]) if vol_col and nb_col else None,
                "brix": df[brix_col] if brix_col else None,
                "Source": input_path+"--"+sheet_name,
            })

            cleaned_data.append(sub_df)
    
    if not cleaned_data:
        verbose_print("No valid data found in any sheet.")
        return

    # Combiner et nettoyer
    final_df = pd.concat(cleaned_data, ignore_index=True)
    final_df.dropna(subset=["date", "volume", "brix", "nb entaille"], how="all", inplace=True)

    # Enregistrer en CSV
    if output_path is None:
        output_path = input(f"\nEntrez le chemin du fichier CSV de sortie (par défaut : ./cleaned_sap_data.csv) : ").strip() or "./cleaned_sap_data.csv"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final_df.to_csv(output_path, index=False)

    verbose_print(f"\n✅ Fichier CSV reconstruit enregistré sous '{output_path}'")

if __name__ == "__main__":
    main()