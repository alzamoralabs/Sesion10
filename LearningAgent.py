from langchain_ollama.llms import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class CriticAgent:
    def __init__(self, model_name="llama3:latest"):
        self.llm = OllamaLLM(model=model_name)
        self.critic_prompt = PromptTemplate(
            input_variables=["observation", "suggested_action"],
            template=(
                "Eres un agente crítico. Observa el ambiente:\n"
                "Observación: {observation}\n"
                "Acción sugerida: {suggested_action}\n"
                "¿La acción es adecuada? Si no, sugiere una mejor acción."
            )
        )
        self.critic_chain = LLMChain(llm=self.llm, prompt=self.critic_prompt)

    def critique(self, observation, suggested_action):
        return self.critic_chain.run({
            "observation": observation,
            "suggested_action": suggested_action
        })


class PerceptiveLearningAgent:
    def __init__(self, model_name="llama3.2"):
        self.llm = OllamaLLM(model=model_name)
        self.critic = CriticAgent(model_name=model_name)
        self.perceive_prompt = PromptTemplate(
            input_variables=["observation"],
            template=(
                "Eres un agente inteligente. Observa el siguiente ambiente:\n"
                "Observación: {observation}\n"
                "¿Qué acción deberías tomar para maximizar el aprendizaje?"
            )
        )
        self.learn_prompt = PromptTemplate(
            input_variables=["observation", "action", "feedback"],
            template=(
                "Has observado: {observation}\n"
                "Tomaste la acción: {action}\n"
                "Recibiste la retroalimentación: {feedback}\n"
                "¿Qué aprendiste y cómo ajustarías tus acciones futuras?"
            )
        )
        self.perceive_chain = LLMChain(llm=self.llm, prompt=self.perceive_prompt)
        self.learn_chain = LLMChain(llm=self.llm, prompt=self.learn_prompt)

    def perceive(self, observation):
        suggested_action = self.perceive_chain.run({"observation": observation})
        critique = self.critic.critique(observation, suggested_action)
        # Si el crítico sugiere una mejor acción, la usamos; si no, usamos la sugerida
        if "mejor acción" in critique.lower() or "sugiero" in critique.lower():
            return critique
        return suggested_action

    def learn(self, observation, action, feedback):
        return self.learn_chain.run({
            "observation": observation,
            "action": action,
            "feedback": feedback
        })

if __name__ == "__main__":
    agent = PerceptiveLearningAgent(model_name="llama3.2")
    observacion = "El ambiente muestra una imagen con un animal de cuatro patas y orejas puntiagudas."
    accion = agent.perceive(observacion)
    print(observacion)
    print("Acción sugerida:", accion)
    retroalimentacion = "La acción fue parcialmente correcta, el animal era un gato."
    aprendizaje = agent.learn(observacion, accion, retroalimentacion)
    print("Aprendizaje del agente:", aprendizaje)