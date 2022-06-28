# Proyecto GPU
Realizado por Anabel Díaz Labrador
## Introducción
Se pide la realización de un histograma de un vector V de un número elevado N (de orden de millones) de elementos aleatorios. El histograma consiste en un vector H que tiene M elementos (siendo recomendable 8) que representan las "cajas" del histograma. En cada caja se cuenta el número de veces en la que ha aparecido un elemento concreto del vector V, para simplificar esta asignación se realiza utilizando el módulo de M, para que sea directamente incrementable en el vector H.

El objetivo de esta actividad es programar utilizando recursos de la GPU para conocer de manera más cercana su arquitectura y su gran capacidad para optimizar problemas que se puedan solucionar usando computación paralela. Para ello se ha programado en CUDA que es una plataforma de computación paralela y una interfaz de programación de aplicaciones que permite que el software use ciertos tipos de unidades de procesamiento de gráficos.

Se tienen varias versiones:
- Implementación base: Se crean tantos hilos como elementos haya en el vector V y cada uno se encarga de incrementar en uno de forma atómica el vector H.
- Implementación con reducción: Se crean tantos histogramas de tamaño M como bloques se hayan creado para el kernel de incremento. Luego usando otro kernel, mediante el algoritmo de reducción, sumamos los histogramas en uno solo.


## Implementación base
El objetivo de esta implementación es cumplir con el objetivo de la manera más intuitiva y simple posible para tener un punto de referencia para comparar con las versiones posteriores.

Para realizarlo se ha implementado en un kernel cuya función es, teniendo un número de hilos y bloques que permita leer el vector V con un hilo por cada elemento, incrementar la "caja" correspondiente de un histograma concreto de tamaño M.

## Implemetación con reducción
El objetivo de esta implementación es reducir drásticamente el número de operaciones atómicas en la misma posición de memoria.

Para conseguirlo se ha dividido la operación en dos fases:
- La creación de multiples histogramas
- La unión de los histogramas en uno solo.

He hecho que el tamaño total del vector de histogramas locales sea el número de bloques creado por grid multiplicado por M porque es una manera óptima de paralelizar la cantidad de hilos que se usan en el vector V (cada hilo va a un elemento diferente del vector) y al mismo tiempo tener un número razonable de sumas atómicas.

En esta implementación es importante que el tamaño del vector V sea una potencia de 2 además del número de bloques y el número de hilos para el correcto funcionamiento del algoritmo de reducción.

Se ha utilizado este algoritmo porque es una forma de paralelizar totalmente la unión de histogramas, con la desventaja de que hay que ir sincronizando los hilos en cada iteración del algoritmo.
