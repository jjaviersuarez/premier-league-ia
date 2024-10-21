import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, mean_squared_error

# Cargar el dataset
df = pd.read_csv("PremierLeague.csv")

# Ver los nombres de las columnas
print(df.columns)

# Convertir a tipo numérico las columnas relevantes
df['B365HomeTeam'] = pd.to_numeric(df['B365HomeTeam'], errors='coerce')
df['B365Draw'] = pd.to_numeric(df['B365Draw'], errors='coerce')
df['B365AwayTeam'] = pd.to_numeric(df['B365AwayTeam'], errors='coerce')

# Columnas que podrían tener valores nulos
columns_with_na = ['FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals', 'B365HomeTeam', 'B365Draw', 'B365AwayTeam']

# Llenar valores nulos con la media
df[columns_with_na] = df[columns_with_na].fillna(df[columns_with_na].mean())

# Dividir las columnas entre numéricas y categóricas
numerical_features = ['B365HomeTeam', 'B365Draw', 'B365AwayTeam']
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

# Asegurarse de que la columna 'FullTimeResult' exista antes de asignarla como variable objetivo
if 'FullTimeResult' in df.columns:
    y_result = label_encoder.fit_transform(df['FullTimeResult'])  # Convertir a clases numéricas
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

# Hacer predicciones
y_pred_result = model_result.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test_result, y_pred_result)
print(f'Accuracy para la predicción del resultado: {accuracy}')

# Para predecir goles, usaré otro modelo de regresión
y_home_goals = df['FullTimeHomeTeamGoals']
y_away_goals = df['FullTimeAwayTeamGoals']

# Separar en entrenamiento y prueba para goles
X_train_home_goals, X_test_home_goals, y_train_home_goals, y_test_home_goals = train_test_split(X_preprocessed, y_home_goals, test_size=0.2, random_state=42)
X_train_away_goals, X_test_away_goals, y_train_away_goals, y_test_away_goals = train_test_split(X_preprocessed, y_away_goals, test_size=0.2, random_state=42)

# Modelo de predicción de goles del equipo local
model_home_goals = XGBRegressor(random_state=42)
model_home_goals.fit(X_train_home_goals, y_train_home_goals)

# Modelo de predicción de goles del equipo visitante
model_away_goals = XGBRegressor(random_state=42)
model_away_goals.fit(X_train_away_goals, y_train_away_goals)

# Hacer predicciones de goles
y_pred_home_goals = model_home_goals.predict(X_test_home_goals)
y_pred_away_goals = model_away_goals.predict(X_test_away_goals)

# Evaluar el rendimiento de los modelos de goles
mse_home_goals = mean_squared_error(y_test_home_goals, y_pred_home_goals)
mse_away_goals = mean_squared_error(y_test_away_goals, y_pred_away_goals)

print(f'MSE para predicción de goles del equipo local: {mse_home_goals}')
print(f'MSE para predicción de goles del equipo visitante: {mse_away_goals}')

# Función para predecir partido futuro
def predecir_partido_futuro(home_team, away_team, bet365_home, bet365_draw, bet365_away):
    # Crear un nuevo dataframe con los datos del partido futuro
    partido_futuro = pd.DataFrame({
        'HomeTeam': [home_team],
        'AwayTeam': [away_team],
        'B365HomeTeam': [bet365_home],
        'B365Draw': [bet365_draw],
        'B365AwayTeam': [bet365_away]
    })
    
    # Preprocesar el nuevo partido
    partido_preprocesado = preprocessor.transform(partido_futuro)
    
    # Predecir el resultado
    resultado_predicho = model_result.predict(partido_preprocesado)
    resultado_predicho_label = label_encoder.inverse_transform([resultado_predicho[0]])  # Convertir de nuevo a etiquetas originales
    
    # Predecir goles del equipo local y visitante
    goles_local_predichos = model_home_goals.predict(partido_preprocesado)
    goles_visitante_predichos = model_away_goals.predict(partido_preprocesado)
    
    print(f'Resultado predicho: {resultado_predicho_label[0]}')
    print(f'Goles del equipo local predichos: {goles_local_predichos[0]}')
    print(f'Goles del equipo visitante predichos: {goles_visitante_predichos[0]}')

# Ejemplo de uso
predecir_partido_futuro("Liverpool", "Chelsea", bet365_home=1.6, bet365_draw=4.5, bet365_away=5.25)
