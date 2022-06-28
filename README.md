# Proyecto GPU
Realizado por Anabel Díaz Labrador
## Introducción
Se pide la realización de un histograma de un vector V de un número elevado N (de orden de millones) de elementos aleatorios. El histograma consiste en un vector H que tiene M elementos (siendo recomendable 8) que representan las "cajas" del histograma. En cada caja se cuenta el número de veces en la que ha aparecido un elemento concreto del vector V, para simplificar esta asignación se realiza utilizando el módulo de M, para que sea directamente incrementable en el vector H.

El objetivo de esta actividad es programar utilizando recursos de la GPU para conocer de manera más cercana su arquitectura y su gran capacidad para optimizar problemas que se puedan solucionar usando computación paralela. Para ello se ha programado en CUDA que es una plataforma de computación paralela y una interfaz de programación de aplicaciones que permite que el software use ciertos tipos de unidades de procesamiento de gráficos.

Se tienen varias versiones:
- Implementación base: Se crean tantos hilos como elementos haya en el vector V y cada uno se encarga de incrementar en uno de forma atómica el vector H.
- Implementación con reducción: Se crean tantos histogramas de tamaño M como bloques se hayan creado para el kernel de incremento con un número arbitrario de hilos. Luego usando otro kernel, mediante el algoritmo de reducción, sumamos los histogramas en uno solo.
- Implementación con reducción con menor uso de operaciones atómicas: Es igual que el anterior pero se crean tantos histogramas de tamaño M con un número de hilos menor al utilizado en el anterior.


## Comparativa de tiempos
El támaño del vector para todas las pruebas ha sido $1048576 = 2^{20}$


| Versión del código         | Nº Hilos | Nº Bloques | Tiempo obtenido kernel 1 | Tiempo obtenido kernel 2 | Tiempo total   |
| :------------------------: | :------: | :--------: | :----------------------: | :----------------------: | :------------: |
| **Versión base**           | $512$    | $2048$     | $0,265934$               | -                        | $0,26594$      |
| **Versión reducción**      | $512$    | $2048$     | $0,036224$               | $0,007176$               | $0,04276$      |
| **Versión Unified Memory** | $512$    | $2048$     | $1,903072$               | $0,205568$               | $2,10864$      |


## Implementación base
El objetivo de esta implementación es cumplir con el objetivo de la manera más intuitiva y simple posible para tener un punto de referencia para comparar con las versiones posteriores.

Se han tenido que crear variables alojadas en el host (CPU) y otras alojadas en el device (GPU) que se inicializarán en el host, se copiarán en el device, luego se modificarán esos datos usando paralelismo (y así comprobar la mejoría del rendimiento con respecto a otros códigos con hilos que hemos realizado en el pasado), estos datos se copiarán de vuelta del device al host y se comprobará el tiempo que ha tomado.

Se ha realizado un kernel (código ejecutado en la GPU) cuya función es, teniendo un número de hilos y de bloques dado, permitir leer el vector V con un hilo por cada elemento de este e incrementar la "caja" correspondiente de un histograma único de tamaño M.

Disminuir o aumentar el número de hilos en este caso no hace cambios significativos.

## Implemetación con reducción
El objetivo de esta implementación es reducir el número de operaciones atómicas en la misma posición de memoria.

Para conseguirlo se ha dividido la operación en dos fases:
- La creación de multiples histogramas
- La unión de los histogramas en uno solo.

He hecho que el tamaño total del vector de histogramas locales sea el número de bloques creado por grid multiplicado por M porque es una manera óptima de paralelizar la cantidad de hilos que se usan en el vector V (cada hilo va a un elemento diferente del vector) y al mismo tiempo tener un número razonable de sumas atómicas.

En esta implementación es importante que el tamaño del vector V sea una potencia de 2 además del número de bloques y el número de hilos para el correcto funcionamiento del algoritmo de reducción, ya que de ello depende el tamaño del vector de histogramas locales y este es el que se va a reducir.

Se ha utilizado este algoritmo porque es una forma de paralelizar totalmente la unión de histogramas, con la desventaja de que hay que ir sincronizando los hilos en cada iteración del algoritmo y que necesitamos, como se dijo antes, que el tamaño del vector al que se le va a hacer reducción sea potencia de 2.


## Implementación de Unified Memory
El objetivo de esta implementación es hacer que el código sea aún más amigable que la versión base. En este caso no es necesario reservar memoria para la CPU y luego reservar otra vez para la GPU sino que lo que tenemos es un único espacio de direcciones de memoria accesible desde la CPU o GPU. Asignar memoria unificada es tan simple como reemplazar las llamadas a malloc() con llamadas a cudaMallocManaged().

Este cambio no pretende mejorar el rendimiento, ya que esta memoria unificada es inequivocamente más lenta que hasta incluso la de la versión base.

Es una muy buena implementación para personas que se están empezando a familiarizar con la programación con CUDA y puede ser útil para depurar el código al poder acceder en todo momento los datos sin tener que volver a copiarlos de la GPU a la CPU.






