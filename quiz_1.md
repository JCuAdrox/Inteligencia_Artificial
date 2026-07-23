# Actividad de Clase: Analizando Agentes de IA con Hugging Face Spaces

**Nombre del estudiante:** 
Jerónimo Zapata Cuadros

## Objetivo

Explorar aplicaciones reales de Inteligencia Artificial en **Hugging
Face Spaces** y analizarlas desde la perspectiva de los **agentes
racionales**.

Al finalizar la actividad, los estudiantes deberán ser capaces de:

-   Identificar los componentes **PEAS** de un agente.
-   Clasificar las propiedades del entorno.
-   Proponer qué tipo de programa de agente podría implementarse detrás
    del sistema.
-   Justificar sus respuestas.

------------------------------------------------------------------------

## Instrucciones

1.  Ingresen a **https://huggingface.co/spaces**.
2.  Exploren diferentes Spaces.
3.  Seleccionen uno que les parezca interesante.
4.  Interactúen con el sistema durante algunos minutos.
5.  Completen la siguiente ficha de análisis.

------------------------------------------------------------------------

# Ficha de análisis

## 1. Nombre del Space

**Nombre:**
Omni Editor 2.0
**Enlace:**
https://huggingface.co/spaces/selfit-camera/Omni-Image-Editor
------------------------------------------------------------------------

## 2. ¿Qué hace el agente?
El agente es capaz de modificar imagenes, generar texto a base de imagenes
o videos, remover marcas de agua y combinar dos o más imagenes.


------------------------------------------------------------------------

## 3. Análisis PEAS

  Elemento          Respuesta
  ----------------- ----------------------------------------------------
  **Performance**   ¿Qué significa que el agente haga bien su trabajo?
  Se verifica el número de errores cometidos por imagen. Es decir, que 
  tanto cumple con todas las intrucciones que le asigna el usuario y 
  qué tan buena interpretacion tiene de estas.
  
  **Environment**   ¿Con qué interactúa el agente?
  El agente interactua con diferentes tableros en el entorno, también
  con las imagenes y las instrucciones del usuario.
  
  **Actuators**     ¿Qué acciones produce?
  Procesa la imagen, luego verifica las instrucciones que posteriormente 
  se le aplican a la imagen y por ultimo retorna el resultado.
  
  **Sensors**       ¿Qué información recibe como entrada?
  Dependiendo de la herramienta que se utilice, recibe diferentes 
  intrucciones como: Las imagenes cargadas por el usuario, y Las dimensiones que debe tener la imagen de retorno. Por último, recibe el prompt.
------------------------------------------------------------------------

## 4. Clasificación del entorno

Complete la siguiente tabla y justifique brevemente cada respuesta.

  Propiedad      Clasificación     Justificación
  -------------- ----------------- ---------------
  Observable     Total             El agente tiene acceso a la imagen completa, el prompt del usuario y todos los pixeles de la imagen. 
  
  Determinista   No                El agente puede llegar a diferentes resultados partiendo del mismo prompt e imagen.
  
  Episódico      Sí / No           Si el usuario realiza una única edición, la decisión depende de solo esa imagen y ese propmt,entonces sería episódico. Pero, si el usuario realizar varias ediciones consecutivas sobre la                                    misma imagen, la interacción completa podría verse como secuencial. 
  
  Estático       Sí                La imagenes no cambian por si solas y el entorno permanece igual hasta que el agente termine.
  
  Discreto       No                Las imágenes contienen millones de valores de píxeles (intensidades y colores), por lo que el espacio de estados es continuo o muy cercano a continuo.
  
  Conocido       Sí                El sistema conoce cómo procesar una imagen mediante el modelo de IA con el que fue entrenado. Además, sabe que operaciones puede realizar.      

------------------------------------------------------------------------

## 5. ¿Qué tipo de programa de agente creen que es?

Seleccione la opción que consideren más adecuada y explique por qué.
-   Agente con aprendizaje
  
**Justificación:**

El modelo esta entrenado con grandes conjuntos de datos de imágenes y texto. Gracias a ese entrenamiento, el agente aprende patrones, relaciones entre imágenes e instrucciones en lenguaje natural, y cómo realizar diferentes tipos de edición.

Cuando el usuario escribe un prompt como "Reemplaza el cielo por un atardecer." el agente no sigue una regla fija programada manualmente. En cambio, utiliza el conocimiento adquirido durante el entrenamiento para interpretar la instrucción y generar una edición coherente.

> **Importante:** No existe una única respuesta correcta. Lo importante
> es justificar la elección a partir del comportamiento observado.

------------------------------------------------------------------------

# Discusión en clase

Después de las presentaciones, discutiremos preguntas como:

-   ¿Dos Spaces diferentes pueden compartir el mismo tipo de entorno? 
Sí, dos aplicaciones diferentes pueden operar en entornos con las mismas características, aunque realicen tareas distintas.

-   ¿Es posible saber con certeza qué tipo de agente implementa un Space
    únicamente observándolo?
No, desde el exterior solo podemos observar el comportamiento del sistema, pero no cómo está implementado.
    
-   ¿Qué diferencia existe entre el comportamiento observable de un
    agente y su implementación interna?
La diferencia es que el comportamiento observable es lo que el usuario puede ver, mientras que la implentación interna es cómo se consigue ese resultado.
------------------------------------------------------------------------

# Reto adicional

Encuentre un Space que pueda clasificarse como:

1.  **Totalmente observable, determinista y episódico.**

**Nombre del space:**
PaddleOCR-VL Online Demo

**Enlace:**
https://huggingface.co/spaces/PaddlePaddle/PaddleOCR-VL_Online_Demo

**Justificación:**

**Totalmente observable:** El agente recibe la imagen completa del documento antes de procesarla. No existe información relevante oculta para realizar el reconocimiento del texto.

**Determinista:** Si se utiliza la misma imagen y la misma configuración del modelo, el resultado será el mismo en cada ejecución.

**Episódico:** Cada documento se procesa de manera independiente. El reconocimiento realizado sobre una imagen no afecta el procesamiento de la siguiente.

2.  **Parcialmente observable, estocástico y secuencial.**

**Nombre del space:**
Experience reality through an RFT observer agent’s perspective

**Enlace:**
https://huggingface.co/spaces/RFTSystems/Agents_Perspective

**Justificación:**

**Parcialmente observable:** Cada agente solo puede ver lo que está dentro de su campo de visión, los muros bloquean la visibilidad y no puede observar todo el entorno.

**Estocástico:** Aunque el entorno puede configurarse de forma reproducible, en modos automáticos el comportamiento depende de las acciones de otro agente y de la evolución del entorno, por lo que el resultado no está completamente determinado desde la perspectiva de un agente.

**Secuencial:** Cada decisión influye en los estados futuros. Moverse en una dirección cambia la situación del siguiente instante y afecta las decisiones posteriores.

Justifique su respuesta.

------------------------------------------------------------------------

# Rúbrica (10 puntos)

| Criterio | Puntos |
|-----------|:------:|
| Descripción correcta del Space | 2 |
| Identificación de PEAS | 3 |
| Clasificación del entorno | 3 |
| Justificación del tipo de agente | 2 |
| **Total** | **10** |

------------------------------------------------------------------------
