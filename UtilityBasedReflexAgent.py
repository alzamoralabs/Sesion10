from langchain.llms import OllamaLLM
from langchain.prompts import PromptTemplate

class UtilityBasedReflexAgent:
    def __init__(self, model="llama3.2"):
        self.llm = OllamaLLM(model=model)  # Usa Ollama local por defecto con modelo llama3.2
        self.prompt_template = PromptTemplate(
            input_variables=["percept"],
            template="Dado el siguiente percept: {percept}\n¿Qué acción maximiza la utilidad?"
        )

    def perceive_and_act(self, percept):
        prompt = self.prompt_template.format(percept=percept)
        action = self.llm(prompt)
        return action.strip()

if __name__ == "__main__":
    agent = UtilityBasedReflexAgent()
    percept = "El agente ve un obstáculo enfrente y una puerta a la derecha."
    action = agent.perceive_and_act(percept)
    print(f"Percepto: {percept}")
    print(f"Acción sugerida: {action}")