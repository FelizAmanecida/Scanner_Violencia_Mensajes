![Superación de la Violencia Psicológica](images/recuperación_violencia_psicológica.png)

<p align="center">
# Detección automática de niveles de propensión a la violencia contra la mujer analizando expresiones escritas con técnicas de Natural Language Processing y Machine Learning

## [Tesis de Pregrado - Universidad ESAN, Lima, Perú, 2020]
#### Ingeniería en Tecnologías de la Información y Sistemas
#### Tereza Yallico Arias & Junior Fabián Arteaga


En Perú, 7 de cada 10 mujeres han pasado por violencia contra la mujer en algún momento de su vida. La más común es la Violencia Psicológica, que es también la más difícil de detectar, pues no deja huellas visibles, pero avanza hacia comportamientos más peligrosos a través del concepto psicológico del Ciclo de Violencia, que hace que la víctima normalice ese comportamiento y solo se romperá cuando sea  consciente del riesgo que corre en esa relación y tome acción al respecto. Darle ese aviso es el objetivo del modelo de Machine Learning desarrollado, pues analizando las expresiones virtuales escritas de su pareja clasifica su nivel de propensión a la violencia en 5 niveles (Bajo riesgo, Chantaje emocional, Celos, Humillaciones/Insultos, Amenazas/Posesividad)  llamando a la toma de acción oportuna. Para ello, se recolectaron 5250 registros que fueron etiquetados en los 5 niveles, teniendo 5 fuentes de datos: Twitter, Webs de apoyo psicológico, Registro de casos de la Línea 100, Videos Relacionados y Narraciones de potenciales víctimas; estos fueron preprocesados con técnicas de Natural Language Processing. Tras 396 experimentos, se obtuvieron los mejores resultados con la combinación de TF-IDF y los modelos de Machine Learning Naive Bayes y Support Vector Machine, con un 0.9266 y 0.9215 de Accuracy, respectivamente.

Keywords: Natural Language Processing, Machine Learning, Violencia contra la mujer, Violencia psicológica, Violencia de Pareja, Análisis de mensajes virtuales, Niveles de Violencia.

En este repositorio se ha cargado la data (solo la parte pública, por acuerdos de confidencialidad) utilizada para el entrenamiento del modelo clasificador de niveles de propensión a la violencia contra la mujer.

### Cuál es el problema?

![Ciclo de la Violencia contra la mujer](images/Ciclo_de_Violencia_Leonor_Walker.png)

El Ciclo de la Violencia contra la mujer tiene tres fases: La primera es la **Acumulación de Tensión** hay discusiones,descontento, él tiene repentinos cambios de humor e intenta hacer (de una forma no amable) "que ella haga lo que debe", ella ajusta su comportamiento para "evitar  peleas", excusandolo por "haber tenido un mal día", atribuyendolo a su personalidad o hasta culpándose a ella misma.  

Sin embargo llegan a la segunda fase, la más traumática, el **Episodio Agudo de Violencia**, aquí él explota, la insulta, la humilla, la amenaza, incluso puede llegar a golpearla. Para el agresor "ella ha recibido lo que merecía". Aquí ella intentará alejarse de él (es lo más natural), sin embargo es la fase 3 la que hace que sigan "dando vueltas" en este "círculo vicioso". 

En la **Fase de "Luna de Miel"**, para evitar que ella lo deje, él le rogará su perdón, le jurará que él va a cambiar, que nunca pasará de nuevo, que lo que pasó fue algo aislado. Se portará como al comienzo de la relación con palabras bonitas y detalles, hará TODO lo necesario para que ella lo perdone y acepte quedarse con él. Lamentablemente esta fase no dura mucho y si ella acepta volver a una relación así sin buscar ayuda por el comportamiento de él, lo más probable es que empiecen a dar vueltas en el ciclo una y otra vez, acumulando tensión,explotando y volviendo a dar otra vuelta. 

Lo peligroso de  dichas vueltas es que si una permanece iterando en el ciclo estos se vuelven cada vez más peligrosos (pasa del tipo de violencia psicológica a la física y/o sexual) y más cortos/frecuentes (si antes tomaba meses dar una vuelta completa, con el tiempo hasta se pueden dar varias vueltas en un día). Lo peor es que la víctima va *normalizando este comportamiento*, o hasta culpándose por él y seguirá modificando su comportamiento o excusándolo pues *niega estar en una situación de riesgo*, niega que le esté pasando a ella, porque es "algo que le pasa a todas las parejas", porque "los hombres son así" (estereotipos de género), por qué él le jura que no va a volver a pasar...

Solo hay una manera de salir del Ciclo de Violencia contra la mujer, y consta de dos pasos:
1. Darse cuenta de la situación de peligro que una corre
2. Tomar acción (buscar ayuda psicológica, apoyo legal, ponerse a salvo a una y a sus hijos)

Hasta que no sea conciente del riesgo que corre, la víctima lo seguirá negando y avanzará en dar vueltas en el ciclo de la violencia. Y no se trata solo de darse cuenta, sino también ser consciente del nivel de riesgo que se corre en esa relación, de qué tan lejos ya ha ido. El primer tipo de violencia en manifestarse es la psicológica, pero ¿Cómo podemos medir el nivel de violencia contra la mujer si la violencia psicológica no es visible? o sí? Acaso no son los mensajes virtuales un buen indicador de lo que piensa y siente su pareja hacia ella? Habrá un patrón reconocible por un algoritmo de Machine Learning?

Veamos...con ayuda de una de las psicólogas del Centro de Emergencia Mujer (Programa Nacional Contra la Violencia Familiar y Sexual, Perú) identificamos y validamos niveles de que un chico sea propenso a ser violento contra la mujer en base a lo que le escribe a su pareja. Observemos cada nivel y si se le puede reconocer en los mensajes. Nótese que a medida que los niveles van subiendo, el riesgo de ella en esa relación también.

#### NIVEL 1: CHANTAJE EMOCIONAL

![Nivel1](images/Nivel1_Chantaje_Emocional.png)

#### NIVEL 2: CELOS

![Nivel2](images/Nivel2_Celos.png)

#### NIVEL 3: INSULTOS/HUMILLACIONES

![NIvel3](images/Nivel3_Insultos_Humillaciones.png)

#### NIVEL 4: AMENAZAS/POSESIVIDAD

![Nivel4](images/Nivel4_Amenazas_Posesividad.png)

Los niveles están planteados para clasificar el avance de la violencia psicológica, como PREVENCIÓN, pues desde aquí se pasará a la violencia Física/Sexual, ya con consecuencias mucho más graves que incluso desembocan en feminicidios.
Durante mucho tiempo el Machine Learning ha permitido clasificar/predecir con mucha certeza en tópicos como diagnóstico de cáncer en base a imágenes, proyecciones de precios en base a históricos, análisis de sentimientos para evaluar la satisfacción del cliente en base a sus comentarios. Sin embargo, raras veces se le ha utilizado para prevenir la Violencia contra la mujer. Ese es el objetivo de este algoritmo. Detectar a tiempo, para actuar a tiempo. 

Entonces...la pregunta es: ¿Podemos entrenar un algoritmo que ayude a identificar niveles de propensión a la violencia contra la mujer (en su fase inicial, la psicológica) analizando los mensajes virtuales (chats) de su pareja para que le dé aviso a la usuaria del riesgo que corre en esa relación? 

### El Algoritmo...

A continuación se visualiza el diagrama de para la elaboración del algoritmo según las 5 etapas del desarrollo de un modelo de Machine Learning, en la *Recolección de Datos* se expondrán las fuentes de datos y cuántos registros fueron extraídos de cada una. En el *Preprocesamiento* se identificará cada una de las técnicas (manuales y automáticas) que se le aplicó al texto base para "limpiarlo".
En la fase de *Extracción de características* se exponen las técnicas a utilizar para representar el texto de los mensajes virtuales como vectores numéricos, donde se puedan probar algoritmos para clasificarlos en niveles. En la fase de *Modelado* se probarán distintos algoritmos para lograr la clasificación de la forma más certera, siendo esta medida en la fase de *Evaluación del modelo*. 


![Metodología](images/Diagrama_Metodología.png)


### Etapas de desarrollo del modelo:

### 1. Recolección de datos


![Metodología](images/Tabla_Fuentes_Dato.png)


![Metodología](images/Cantidad_Datos_Por_Fuente.PNG)


### 2. Preprocesamiento


### 3. Extracción de Características
### 4. Modelado
### 5. Evaluación del modelo

</p>