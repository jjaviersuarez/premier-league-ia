import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from imblearn.over_sampling import SMOTE

# Cargar el dataset
data = pd.read_csv('PremierLeague.csv')

# Seleccionar las características (X) y las etiquetas (y)
features = ['HomeTeam', 'AwayTeam', 'HomeTeamShots', 'AwayTeamShots', 
            'HomeTeamShotsOnTarget', 'AwayTeamShotsOnTarget', 
            'HomeTeamCorners', 'AwayTeamCorners', 
            'HomeTeamYellowCards', 'AwayTeamYellowCards',
            'B365HomeTeam', 'B365Draw', 'B365AwayTeam']

# Codificar las etiquetas de los equipos
label_encoder = LabelEncoder()
data['HomeTeam'] = label_encoder.fit_transform(data['HomeTeam'])
data['AwayTeam'] = label_encoder.transform(data['AwayTeam'])

# Imputar los valores faltantes (solo para columnas numéricas)
imputer = SimpleImputer(strategy='mean')
numeric_features = ['HomeTeamShots', 'AwayTeamShots', 
                    'HomeTeamShotsOnTarget', 'AwayTeamShotsOnTarget',
                    'HomeTeamCorners', 'AwayTeamCorners', 
                    'HomeTeamYellowCards', 'AwayTeamYellowCards',
                    'B365HomeTeam', 'B365Draw', 'B365AwayTeam']

data[numeric_features] = imputer.fit_transform(data[numeric_features])

# Separar las etiquetas para los goles y el resultado del partido
X = data[features]
y_goles_local = data['FullTimeHomeTeamGoals']
y_goles_visitante = data['FullTimeAwayTeamGoals']
y_resultado = data['FullTimeResult']  # 'H' para victoria local, 'D' para empate, 'A' para victoria visitante

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train_goles_local, y_test_goles_local = train_test_split(X, y_goles_local, test_size=0.2, random_state=42)
_, _, y_train_goles_visitante, y_test_goles_visitante = train_test_split(X, y_goles_visitante, test_size=0.2, random_state=42)
_, _, y_train_resultado, y_test_resultado = train_test_split(X, y_resultado, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar SMOTE para balancear las clases en el conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resultado_resampled = smote.fit_resample(X_train_scaled, y_train_resultado)

# Modelo de regresión para los goles del equipo local
regression_model_local = LinearRegression()
regression_model_local.fit(X_train_scaled, y_train_goles_local)

# Modelo de regresión para los goles del equipo visitante
regression_model_visitante = LinearRegression()
regression_model_visitante.fit(X_train_scaled, y_train_goles_visitante)

# Ajuste de hiperparámetros del modelo de clasificación
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

classification_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=classification_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resultado_resampled)

# Evaluación del modelo para la predicción de goles (MSE)
y_pred_goles_local = regression_model_local.predict(X_test_scaled)
y_pred_goles_visitante = regression_model_visitante.predict(X_test_scaled)
mse_local = mean_squared_error(y_test_goles_local, y_pred_goles_local)
mse_visitante = mean_squared_error(y_test_goles_visitante, y_pred_goles_visitante)

print("\n--- Evaluación del modelo ---")
print(f"Error cuadrático medio (MSE) Goles Local: {mse_local:.4f}")
print(f"Error cuadrático medio (MSE) Goles Visitante: {mse_visitante:.4f}\n")

# Evaluación del modelo de clasificación (precisión)
y_pred_resultado = grid_search.predict(X_test_scaled)
accuracy = accuracy_score(y_test_resultado, y_pred_resultado)
print(f"Precisión del modelo de clasificación (resultado): {accuracy:.4%}\n")
print("Mejores parámetros encontrados por GridSearchCV:", grid_search.best_params_)

# Predicción para un partido futuro usando datos previos
home_team = 'Liverpool'
away_team = 'Chelsea'

home_team_encoded = label_encoder.transform([home_team])[0]
away_team_encoded = label_encoder.transform([away_team])[0]

# Estimaciones basadas en el rendimiento reciente (usar datos que estén disponibles)
home_team_shots = 16  # Tiros del Local en sus últimos partidos
away_team_shots = 22   # Tiros del Visitante en sus últimos partidos
home_team_shots_on_target = 4  # Tiros a puerta del Aston Villa
away_team_shots_on_target = 8  # Tiros a puerta del Liverpool
home_team_corners = 8  # Tiros de esquina del Aston Villa
away_team_corners = 11    # Tiros de esquina del Liverpool
home_team_yellow_cards = 2  # Tarjetas amarillas del Aston Villa
away_team_yellow_cards = 6   # Tarjetas amarillas del Liverpool

# Crear un DataFrame para el partido futuro
partido_futuro = pd.DataFrame([[home_team_encoded, away_team_encoded, 
                                home_team_shots, away_team_shots, 
                                home_team_shots_on_target, away_team_shots_on_target, 
                                home_team_corners, away_team_corners, 
                                home_team_yellow_cards, away_team_yellow_cards, 
                                1.6, 4.5, 5.25]],  # Puedes ajustar estos últimos valores según tus datos
                              columns=features)

partido_futuro_scaled = scaler.transform(partido_futuro)

# Predicción de goles del futuro partido
goles_local_pred = regression_model_local.predict(partido_futuro_scaled)[0]
goles_visitante_pred = regression_model_visitante.predict(partido_futuro_scaled)[0]

# Predicción del resultado del futuro partido
resultado_pred = grid_search.predict(partido_futuro_scaled)[0]
resultado_str = 'Victoria Local' if resultado_pred == 'H' else 'Empate' if resultado_pred == 'D' else 'Victoria Visitante'

# Salida con mejor formato
print("\n--- Predicción para el partido ---")
print(f"Partido: {home_team} vs {away_team}")
print(f"Goles Predichos - {home_team}: {goles_local_pred:.2f}")
print(f"Goles Predichos - {away_team}: {goles_visitante_pred:.2f}")
print(f"Resultado Predicho: {resultado_str}\n")
