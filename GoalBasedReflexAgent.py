from langchain_community.llms import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Inicializa el modelo Ollama local
llm = OllamaLLM(model="llama3.2")

# Crea una memoria de buffer de contexto
memory = ConversationBufferMemory()

# Define el agente basado en objetivos (goal-based reflex agent)
class GoalBasedReflexAgent:
    def __init__(self, llm, memory, goal):
        self.llm = llm
        self.memory = memory
        self.goal = goal
        self.chain = ConversationChain(llm=self.llm, memory=self.memory)

    def perceive_and_act(self, observation):
        prompt = f"Goal: {self.goal}\nObservation: {observation}\nWhat should I do next?"
        response = self.chain.run(prompt)
        return response

# Ejemplo de uso
if __name__ == "__main__":
    goal = "Ayudar al usuario a planificar un viaje."
    agent = GoalBasedReflexAgent(llm, memory, goal)

    while True:
        obs = input("Usuario: ")
        if obs.lower() in ["salir", "exit", "quit"]:
            break
        action = agent.perceive_and_act(obs)
        print("Agente:", action)