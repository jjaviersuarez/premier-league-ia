# Predicción de Resultados de Partidos de la Premier League

Este proyecto utiliza **machine learning** para predecir los resultados de partidos de la Premier League, así como los goles anotados por el equipo local y visitante. El modelo está basado en **XGBoost**, con la ayuda de características como estadísticas de los equipos (goles, tiros, tarjetas) y cuotas de apuestas de Bet365. El objetivo es ayudar a anticipar el resultado final de los partidos con mayor precisión.

## Descripción del Código

El código se basa en los siguientes pasos principales:

1. **Carga de Datos**: Se carga un dataset de la Premier League que contiene información sobre partidos pasados, como equipos, goles, estadísticas de rendimiento, y cuotas de apuestas.
   
2. **Preprocesamiento de Datos**: Se normalizan características numéricas como los puntos de los equipos, los tiros a puerta, las tarjetas amarillas, y se codifican categóricamente los nombres de los equipos.

3. **Entrenamiento del Modelo**:
   - Se usa **SMOTE** para manejar el desbalance de clases en los resultados.
   - Se entrena un modelo de **XGBoost** para predecir el resultado final del partido.
   - Se entrena otro modelo de **XGBoost** para predecir el número de goles del equipo local y visitante.

4. **Predicción de Partidos Futuros**: La función `predecir_partido_futuro_mejorado` permite ingresar información sobre un partido futuro (equipos, estadísticas, cuotas de apuestas) para predecir el resultado final y los goles.

5. **Precisión del Modelo**: Se calcula y muestra la precisión del modelo en el conjunto de prueba, ayudando a evaluar su rendimiento.

## Ejemplo de Uso

```python
predecir_partido_futuro_mejorado(
    "Liverpool", "Chelsea", bet365_home=1.6, bet365_draw=4.5, bet365_away=5.25,
    home_team_points=45, away_team_points=38,
    home_shots=12, away_shots=8, home_shots_on_target=6, away_shots_on_target=3,
    home_corners=5, away_corners=2, home_yellow_cards=2, away_yellow_cards=3,
    home_red_cards=0, away_red_cards=1, halftime_home_goals=1, halftime_away_goals=0,
    b365_over_2_5_goals=1.8, b365_under_2_5_goals=2.1
)
```

Este ejemplo predice el resultado de un partido entre Liverpool y Chelsea con base en la información proporcionada.

## Colaboración con ChatGPT

Este proyecto ha sido desarrollado con la ayuda de **ChatGPT**, lo cual me ha permitido mejorar el flujo de trabajo y optimizar el código para obtener mejores resultados en la predicción de partidos. La colaboración con ChatGPT ha sido clave en la implementación de técnicas de **machine learning** y en la automatización de la limpieza y procesamiento de datos.

## Sígueme en Mis Redes Sociales

Para más contenido de programación y machine learning, ¡sígueme en mis redes sociales!

- [YouTube: @shots_programacion](https://www.youtube.com/@shots_programacion)
- [Facebook: Shots Programación](https://www.facebook.com/shots.programacion/)
- [Instagram: @jenier_suarez](https://www.instagram.com/jenier_suarez/)

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/prediccion-partidos-premier-league.git
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Contribuciones

Si deseas contribuir a este proyecto, ¡siéntete libre de hacer un **pull request** o abrir un **issue**! Tu ayuda es bienvenida para seguir mejorando el modelo y las predicciones.
