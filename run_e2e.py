import argparse
import sys
import time
import pandas as pd
from termcolor import colored

# Importiamo i moduli del nostro progetto
import data_handler
import model_engine

def run_prediction_test(year, race, driver, start_lap, laps_to_load, predict_laps):
    """
    Simula il flusso dell'app per un singolo pilota.
    Restituisce True se il test passa, False (e l'errore) se fallisce.
    """
    try:
        # 1. Caricamento Dati Gara
        r_session = data_handler.load_session(year, race, 'R')
        laps_df = data_handler.get_race_laps(r_session, driver)
        
        if laps_df.empty:
            return False, "Nessun dato gara disponibile per questo pilota."
            
        # 2. Pre-processing
        df_prep, le = model_engine.prepare_features(laps_df)
        df_train = df_prep[df_prep['LapNumber'] <= laps_to_load]
        
        if df_train.empty:
            return False, "Dati filtrati insufficienti per l'addestramento."
            
        # 3. Addestramento
        model = model_engine.train_pace_model(df_train)
        
        # 4. Setup Starting Lap
        if start_lap == 0:
            try:
                q_session = data_handler.load_session(year, race, 'Q')
                q_lap = data_handler.get_qualy_fastest_lap(q_session, driver)
            except Exception as e:
                return False, f"Impossibile caricare qualifiche: {e}"
            
            c_lap = 0
            c_tyre = 0
            c_comp = df_train.iloc[0]['Compound_encoded'] if not df_train.empty else 0
            c_stint = 1
            
            c_laptime = q_lap if q_lap else (df_train.iloc[0]['LapTime_sec'] if not df_train.empty else 90.0)
            avg_thr = df_train['AvgThrottle'].mean() if not df_train.empty else 0.0
            avg_brk = df_train['AvgBrake'].mean() if not df_train.empty else 0.0
        else:
            subset = df_train[df_train['LapNumber'] <= start_lap]
            if not subset.empty:
                last_known = subset.iloc[-1]
                c_lap = last_known['LapNumber']
                c_tyre = last_known['TyreLife']
                c_comp = last_known['Compound_encoded']
                c_stint = last_known['Stint']
                
                c_laptime = last_known['LapTime_sec']
                avg_thr = subset['AvgThrottle'].mean()
                avg_brk = subset['AvgBrake'].mean()
            else:
                return False, f"Il giro di partenza {start_lap} non è presente nei dati."

        # 5. Predizione
        preds = model_engine.predict_future_pace(
            model, current_lap=c_lap, current_tyre_life=c_tyre,
            current_compound_enc=c_comp, current_stint=c_stint, 
            current_laptime=c_laptime, avg_throttle=avg_thr, avg_brake=avg_brk,
            num_laps=predict_laps
        )
        
        # Validazioni
        if preds is None or preds.empty:
            return False, "Il modello non ha generato predizioni."
        
        if start_lap == 0 and not q_lap:
            return True, "OK (Senza Qualifica, usato Fallback)"
            
        return True, "OK"
        
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="F1 Telemetry E2E Test Suite")
    parser.add_argument('--year', type=int, help="Testa un anno specifico (default: tutti dal 2022 al 2026)")
    parser.add_argument('--race', type=str, help="Testa una gara specifica (default: tutte le gare dell'anno)")
    parser.add_argument('--driver', type=str, help="Testa un pilota specifico (default: tutti i piloti della gara)")
    parser.add_argument('--delay', type=int, default=5, help="Secondi di pausa tra le chiamate all'API (default: 5)")
    args = parser.parse_args()

    years_to_test = [args.year] if args.year else [2022, 2023, 2024, 2025, 2026]
    
    total_tests = 0
    passed_tests = 0
    
    print(colored("🏁 Avvio F1 Telemetry E2E Test Suite 🏁", "cyan", attrs=['bold']))
    print(colored(f"Per evitare blocchi 429 dall'API, verrà inserito un delay di {args.delay} secondi tra le chiamate.", "yellow"))
    
    for y in years_to_test:
        print(colored(f"\n>> Anno: {y}", "blue", attrs=['bold']))
        
        try:
            races = data_handler.get_available_races(y)
        except Exception as e:
            print(colored(f"ERRORE caricamento gare {y}: {e}", "red"))
            continue
            
        races_to_test = [args.race] if args.race else races
        
        for r in races_to_test:
            print(colored(f"  >> Gran Premio: {r}", "blue"))
            
            try:
                session_key = data_handler.load_session(y, r, 'R')
                drivers = data_handler.get_session_drivers(session_key)
                tot_laps = data_handler.get_race_total_laps(session_key)
            except Exception as e:
                print(colored(f"    ERRORE caricamento sessione {r}: {e}", "red"))
                continue
                
            drivers_to_test = [args.driver] if args.driver else drivers
            
            for d in drivers_to_test:
                # Test 1: Partenza da Qualifica (Lap 0)
                total_tests += 1
                sys.stdout.write(f"    [Test {total_tests}] {d} @ {r} {y} | Start: Qualy | Predict: {tot_laps} laps ... ")
                sys.stdout.flush()
                
                data_handler.NETWORK_HIT = False
                success, msg = run_prediction_test(
                    year=y, race=r, driver=d, 
                    start_lap=0, laps_to_load=tot_laps, predict_laps=tot_laps
                )
                source = "[API]" if data_handler.NETWORK_HIT else "[CACHE]"
                
                if success:
                    print(colored(f"PASSED {source}", "green"))
                    passed_tests += 1
                else:
                    print(colored(f"FAILED {source} ({msg})", "red"))
                    
                if data_handler.NETWORK_HIT:
                    time.sleep(args.delay) # Delay anti-ban solo se ha fatto richieste di rete
                
                if not success:
                    continue # Non testare il Lap 10 se mancano i dati di base (es. qualifica o gara falliti)
                
                # Test 2: Partenza da Giro 10
                if tot_laps > 10:
                    total_tests += 1
                    sys.stdout.write(f"    [Test {total_tests}] {d} @ {r} {y} | Start: Lap 10 | Predict: {tot_laps - 10} laps ... ")
                    sys.stdout.flush()
                    
                    data_handler.NETWORK_HIT = False
                    success, msg = run_prediction_test(
                        year=y, race=r, driver=d, 
                        start_lap=10, laps_to_load=tot_laps, predict_laps=tot_laps - 10
                    )
                    source1 = "[API]" if data_handler.NETWORK_HIT else "[CACHE]"
                    
                    if success:
                        print(colored(f"PASSED {source1}", "green"))
                        passed_tests += 1
                    else:
                        print(colored(f"FAILED {source1} ({msg})", "red"))
                        
                    if data_handler.NETWORK_HIT:
                        time.sleep(args.delay) # Delay anti-ban solo se ha fatto richieste di rete
                    
    print("\n" + "="*50)
    color = "green" if passed_tests == total_tests and total_tests > 0 else "red"
    print(colored(f"RISULTATI TEST: {passed_tests}/{total_tests} passati.", color, attrs=['bold']))
    print("="*50)
    
if __name__ == "__main__":
    main()
