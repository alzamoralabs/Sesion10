# ReflexAgent.py
# Implementación de un agente de reflejo simple
######################## SIMPLE REFLEX AGENT ###################################################

import random
import ollama

# Reglas de condición-acción
reglas = {
    "temperatura < 18": "encender calefacción",
    "temperatura > 24": "apagar calefacción",
    "presión < 30": "activar bomba",
    "presión > 70": "desactivar bomba"
}

# Función para evaluar condiciones simples
def cumple_condicion(percepcion, condicion):
    variable, operador, valor = condicion.split()
    valor = float(valor)
    entrada = percepcion.get(variable)

    if entrada is None:
        return False

    if operador == "<":
        return entrada < valor
    elif operador == ">":
        return entrada > valor
    elif operador == "==":
        return entrada == valor
    else:
        return False

# Simple Reflex Agent
def agente_reflejo_simple(percepcion):
    for condicion, accion in reglas.items():
        if cumple_condicion(percepcion, condicion):
            print(f"Acción local: {accion}")
            return accion

    print("Sin acción necesaria.")
    return "sin acción"

# Ejemplo de uso
percepcion_actual = {
    "temperatura": random.randint(10, 35), #15 in Celsius
    "presión": random.randint(20, 80) #45 in PSI
}
print(f"Percepción actual: {percepcion_actual}")
accion_tomada = agente_reflejo_simple(percepcion_actual)

######################## LLM AS A JUDGE ###################################################

# Validación de la acción usando Ollama con el modelo llama3.2
def validar_accion_ollama(percepcion, accion):
    prompt = (
        f"Percepción: {percepcion}\n"
        f"Acción tomada: {accion}\n"
        "¿Es esta acción apropiada según las reglas dadas? Responde 'sí' o 'no' y explica brevemente."
    )
    respuesta = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    print("Validación Ollama:", respuesta['message']['content'])

# Invocación de la validación
validar_accion_ollama(percepcion_actual, accion_tomada)