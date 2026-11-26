```
ACERDataModule
│
├─ Données brutes CPU (vérité)
│   ├─ input_volume_lagged
│   ├─ input_temperature_lagged
│   ├─ target_volume
│   ├─ id_date_system
│
├─ Split indices
│   ├─ train_idx
│   └─ val_idx
│
├─ Normalization stats (calculées sur TRAIN)
│
├─ ACERDataset(train)  ← vue
└─ ACERDataset(val)    ← vue
```
```
Pipeline de données pour ACERDataset
-------------------------------------------------------
CPU
 ├─ input_volume_lagged          (float32, vérité)
 ├─ input_temperature_lagged     (float32, vérité)
 ├─ target_volume                (float32, vérité)
 │
 └─ stats de normalisation       (CPU, immuables)

GPU
 └─ cached_items
     ├─ concaténés
     ├─ normalisés
     ├─ castés (fp16 / bf16)
     └─ prêts pour le modèle
```
```
Timeline diagram:
    Time --> T-[n_lookback_days] ... T-2, T-1 |    T0   | T1, T2, ... T[horizon-1]
            |              Passé              | Présent |          Futur          |
            |---------------------------------|---------|-------------------------|
                    X (lagged inputs)         |       X (forecast inputs)
                                              |       Y (target outputs)
```