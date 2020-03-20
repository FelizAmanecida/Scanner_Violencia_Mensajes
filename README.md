![Superación de la Violencia Psicológica](https://github.com/TerezaYallicoArias/Scanner_Violencia_Mensajes/blob/master/recuperación_violencia_psicológica.png)

#               "Detección automática de niveles de propensión a la violencia contra la mujer analizando expresiones escritas con técnicas de Natural Language Processing y Machine Learning"

##                               [Tesis de Pregrado - Universidad ESAN, Lima, Perú, 2020]
####                                    Ingeniería en Tecnologías de la Información y Sistemas
####                                   Tereza Yallico Arias & Junior Fabián Arteaga


En Perú, 7 de cada 10 mujeres han pasado por violencia contra la mujer en algún momento de su vida. La más común es la Violencia Psicológica, que es también la más difícil de detectar, pues no deja huellas visibles, pero avanza hacia comportamientos más peligrosos a través del concepto psicológico del Ciclo de Violencia, que hace que la víctima normalice ese comportamiento y solo se romperá cuando sea  consciente del riesgo que corre en esa relación y tome acción al respecto. Darle ese aviso es el objetivo del modelo de Machine Learning desarrollado, pues analizando las expresiones virtuales escritas de su pareja clasifica su nivel de propensión a la violencia en 5 niveles (Bajo riesgo, Chantaje emocional, Celos, Humillaciones/Insultos, Amenazas/Posesividad)  llamando a la toma de acción oportuna. Para ello, se recolectaron 5250 registros que fueron etiquetados en los 5 niveles, teniendo 5 fuentes de datos: Twitter, Webs de apoyo psicológico, Registro de casos de la Línea 100, Videos Relacionados y Narraciones de potenciales víctimas; estos fueron preprocesados con técnicas de Natural Language Processing. Tras 396 experimentos, se obtuvieron los mejores resultados con la combinación de TF-IDF y los modelos de Machine Learning Naive Bayes y Support Vector Machine, con un 0.9266 y 0.9215 de Accuracy, respectivamente.

En este repositorio se ha cargado la data (solo la parte pública, por acuerdos de confidencialidad) utilizada para el entrenamiento del modelo clasificador de niveles de propensión a la violencia contra la mujer.

### Cuál es el problema?
![Ciclo de la Violencia contra la mujer] 
(https://github.com/TerezaYallicoArias/Scanner_Violencia_Mensajes/blob/master/Ciclo de Violencia Leonor Walker.png)

El Ciclo de la Violencia contra la mujer tiene tres fases: La primera es la *Acumulación de Tensión* hay discusiones,descontento, conflictos, él siempre está molesto, ella ajusta su comportamiento para evitar "provocar peleas". 

Sin embargo llegan a la segunda fase, la más traumática, el *Episodio Agudo de Violencia*, aquí él explota, la insulta, la humilla, la amenaza, incluso puede llegar a golpearla. Aquí ella intentará alejarse de él (es lo más natural), sin embargo es la fase 3 la que hace que sigan "dando vueltas" en este "círculo vicioso". 

En la *Fase de "Luna de Miel"*, para evitar que ella lo deje, él le rogará su perdón, le jurará que él va a cambiar, que nunca pasará de nuevo, que lo que pasó fue algo aislado. Se portará como al comienzo de la relación con palabras bonitas y detalles, hará TODO lo necesario para que ella lo perdone y acepte quedarse con él. Lamentablemente esta fase no dura mucho y si ella acepta volver a una relación así sin buscar ayuda por el comportamiento de él, lo más probable es que empiecen a dar vueltas en el ciclo una y otra vez, acumulando tensión,explotando y volviendo a dar otra vuelta. 

Lo peligroso de  dichas vueltas es que si una permanece iterando en el ciclo estos se vuelven cada vez más peligrosos (pasa del tipo de violencia psicológica a la física y/o sexual) y más cortos/frecuentes (si antes tomaba meses dar una vuelta completa, con el tiempo hasta se pueden dar varias vueltas en un día). Lo peor es que la víctima va normalizando este comportamiento, por los estereotipos de género, porque lo atribuyen a su caracter, hasta hay quien se hecha la culpa "por haberlo provocado".
