from langchain_ollama.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class ModelBasedReflexAgent:
    def __init__(self, model_name="llama3.2", memory_buffer_size=5):
        # Inicializa el modelo Ollama localmente
        self.llm = Ollama(model=model_name)
        # Inicializa el buffer de memoria para guardar estados del ambiente
        self.memory = ConversationBufferMemory(k=memory_buffer_size)
        # Define el prompt para el agente
        self.prompt = PromptTemplate(
            input_variables=["state", "history"],
            template=(
                "Estado actual del ambiente: {state}\n"
                "Historial de estados previos: {history}\n"
                "¿Cuál es la mejor acción reflexiva a tomar ahora?"
            )
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

    def perceive_and_act(self, state):
        # Guarda el estado actual en la memoria
        self.memory.save_context({"input": state}, {})
        # Obtiene el historial de estados
        history = self.memory.buffer_as_str
        # Invoca el modelo para decidir la acción
        action = self.chain.run(state=state, history=history)
        return action

# Ejemplo de uso:
if __name__ == "__main__":
    agent = ModelBasedReflexAgent()
    estados = [
        "La habitación está sucia",
        "La habitación está limpia",
        "La habitación tiene una ventana abierta"
    ]
    for estado in estados:
        accion = agent.perceive_and_act(estado)
        print(f"Estado: {estado} -> Acción: {accion}")