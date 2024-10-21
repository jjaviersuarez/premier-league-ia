import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, mean_squared_error
from tabulate import tabulate

# Cargar el dataset
df = pd.read_csv("PremierLeague.csv")

# Convertir a tipo numérico las columnas relevantes
df['B365HomeTeam'] = pd.to_numeric(df['B365HomeTeam'], errors='coerce')
df['B365Draw'] = pd.to_numeric(df['B365Draw'], errors='coerce')
df['B365AwayTeam'] = pd.to_numeric(df['B365AwayTeam'], errors='coerce')
df['B365Over2.5Goals'] = pd.to_numeric(df['B365Over2.5Goals'], errors='coerce')
df['B365Under2.5Goals'] = pd.to_numeric(df['B365Under2.5Goals'], errors='coerce')

# Columnas que podrían tener valores nulos
columns_with_na = ['FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals', 'B365HomeTeam', 'B365Draw', 'B365AwayTeam', 
                   'B365Over2.5Goals', 'B365Under2.5Goals', 'HomeTeamShots', 'AwayTeamShots',
                   'HomeTeamShotsOnTarget', 'AwayTeamShotsOnTarget', 'HomeTeamCorners', 'AwayTeamCorners',
                   'HomeTeamYellowCards', 'AwayTeamYellowCards', 'HomeTeamRedCards', 'AwayTeamRedCards',
                   'HalfTimeHomeTeamGoals', 'HalfTimeAwayTeamGoals']

# Llenar valores nulos con la media
df[columns_with_na] = df[columns_with_na].fillna(df[columns_with_na].mean())

# Dividir las columnas entre numéricas y categóricas
numerical_features = ['B365HomeTeam', 'B365Draw', 'B365AwayTeam', 'HomeTeamPoints', 'AwayTeamPoints',
                      'HomeTeamShots', 'AwayTeamShots', 'HomeTeamShotsOnTarget', 'AwayTeamShotsOnTarget',
                      'HomeTeamCorners', 'AwayTeamCorners', 'HomeTeamYellowCards', 'AwayTeamYellowCards',
                      'HomeTeamRedCards', 'AwayTeamRedCards', 'B365Over2.5Goals', 'B365Under2.5Goals',
                      'HalfTimeHomeTeamGoals', 'HalfTimeAwayTeamGoals']
categorical_features = ['HomeTeam', 'AwayTeam']

# Crear el preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Asegurarse de que las columnas que se quieren eliminar existan en el DataFrame
columns_to_drop = ["FullTimeResult", "FullTimeHomeTeamGoals", "FullTimeAwayTeamGoals"]
columns_to_drop = [col for col in columns_to_drop if col in df.columns]

# Preprocesar los datos
X_preprocessed = preprocessor.fit_transform(df.drop(columns=columns_to_drop, errors='ignore'))

# Convertir las clases 'FullTimeResult' a valores numéricos
label_encoder = LabelEncoder()

if 'FullTimeResult' in df.columns:
    y_result = label_encoder.fit_transform(df['FullTimeResult'])
else:
    raise ValueError("'FullTimeResult' no se encuentra en el DataFrame.")

# Separar los conjuntos de entrenamiento y prueba
X_train, X_test, y_train_result, y_test_result = train_test_split(X_preprocessed, y_result, test_size=0.2, random_state=42)

# Manejo del desbalanceo con SMOTE
smote = SMOTE(random_state=42)
X_train_res_smote, y_train_res_smote = smote.fit_resample(X_train, y_train_result)

# Modelo para predecir el resultado del partido
model_result = XGBClassifier(random_state=42)
model_result.fit(X_train_res_smote, y_train_res_smote)

# Separar en entrenamiento y prueba para goles
y_home_goals = df['FullTimeHomeTeamGoals']
y_away_goals = df['FullTimeAwayTeamGoals']
X_train_home_goals, X_test_home_goals, y_train_home_goals, y_test_home_goals = train_test_split(X_preprocessed, y_home_goals, test_size=0.2, random_state=42)
X_train_away_goals, X_test_away_goals, y_train_away_goals, y_test_away_goals = train_test_split(X_preprocessed, y_away_goals, test_size=0.2, random_state=42)

# Modelo de predicción de goles del equipo local
model_home_goals = XGBRegressor(random_state=42)
model_home_goals.fit(X_train_home_goals, y_train_home_goals)

# Modelo de predicción de goles del equipo visitante
model_away_goals = XGBRegressor(random_state=42)
model_away_goals.fit(X_train_away_goals, y_train_away_goals)

# Función mejorada para predecir partido futuro con los puntos de los equipos
def predecir_partido_futuro_mejorado(home_team, away_team, bet365_home, bet365_draw, bet365_away, home_team_points, away_team_points,
                                     home_shots, away_shots, home_shots_on_target, away_shots_on_target, home_corners, away_corners,
                                     home_yellow_cards, away_yellow_cards, home_red_cards, away_red_cards, halftime_home_goals, halftime_away_goals,
                                     b365_over_2_5_goals, b365_under_2_5_goals):
    # Crear un nuevo dataframe con los datos del partido futuro
    partido_futuro = pd.DataFrame({
        'HomeTeam': [home_team],
        'AwayTeam': [away_team],
        'B365HomeTeam': [bet365_home],
        'B365Draw': [bet365_draw],
        'B365AwayTeam': [bet365_away],
        'HomeTeamPoints': [home_team_points],
        'AwayTeamPoints': [away_team_points],
        'HomeTeamShots': [home_shots],
        'AwayTeamShots': [away_shots],
        'HomeTeamShotsOnTarget': [home_shots_on_target],
        'AwayTeamShotsOnTarget': [away_shots_on_target],
        'HomeTeamCorners': [home_corners],
        'AwayTeamCorners': [away_corners],
        'HomeTeamYellowCards': [home_yellow_cards],
        'AwayTeamYellowCards': [away_yellow_cards],
        'HomeTeamRedCards': [home_red_cards],
        'AwayTeamRedCards': [away_red_cards],
        'HalfTimeHomeTeamGoals': [halftime_home_goals],
        'HalfTimeAwayTeamGoals': [halftime_away_goals],
        'B365Over2.5Goals': [b365_over_2_5_goals],
        'B365Under2.5Goals': [b365_under_2_5_goals]
    })
    
    # Preprocesar el nuevo partido
    partido_preprocesado = preprocessor.transform(partido_futuro)
    
    # Predecir el resultado
    resultado_predicho = model_result.predict(partido_preprocesado)
    resultado_predicho_label = label_encoder.inverse_transform([resultado_predicho[0]])  # Convertir de nuevo a etiquetas originales
    
    # Predecir goles del equipo local y visitante
    goles_local_predichos = model_home_goals.predict(partido_preprocesado)
    goles_visitante_predichos = model_away_goals.predict(partido_preprocesado)

    # Crear tabla para mostrar el resultado
    resultados = [
        ["Equipo Local", home_team],
        ["Equipo Visitante", away_team],
        ["Probabilidad Casa (B365)", bet365_home],
        ["Probabilidad Empate (B365)", bet365_draw],
        ["Probabilidad Visitante (B365)", bet365_away],
        ["Puntos del equipo local", home_team_points],
        ["Puntos del equipo visitante", away_team_points],
        ["Resultado Predicho", resultado_predicho_label[0]],
        ["Goles Local Predichos", round(goles_local_predichos[0], 2)],
        ["Goles Visitante Predichos", round(goles_visitante_predichos[0], 2)]
    ]
    
    print(tabulate(resultados, headers=["Descripción", "Valor"], tablefmt="fancy_grid"))

# Llamada de ejemplo con los nuevos parámetros
predecir_partido_futuro_mejorado(
    "Liverpool", "Chelsea", bet365_home=1.6, bet365_draw=4.5, bet365_away=5.25,
    home_team_points=45, away_team_points=38,
    home_shots=12, away_shots=8, home_shots_on_target=6, away_shots_on_target=3,
    home_corners=5, away_corners=2, home_yellow_cards=2, away_yellow_cards=3,
    home_red_cards=0, away_red_cards=1, halftime_home_goals=1, halftime_away_goals=0,
    b365_over_2_5_goals=1.8, b365_under_2_5_goals=2.1
)
