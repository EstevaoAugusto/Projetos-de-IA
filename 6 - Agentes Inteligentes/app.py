import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("ERRO: Chave de API não encontrada. Verifique o arquivo .env")
    exit()

config_list_gemini = [
    {
        "model": "gemini-2.5-flash",
        "api_key": API_KEY,
        "api_type": "google"
    }
]

llm_config = {
    "config_list": config_list_gemini,
    "temperature": 0.7,
}

# --- CRIAÇÃO DOS AGENTES ---

# 1. Agente Usuário (Admin)
user_proxy = UserProxyAgent(
    name="Admin",
    system_message="Um administrador humano que propõe a tarefa e encerra quando satisfeito.",
    code_execution_config=False,
    human_input_mode="NEVER", # "NEVER" automatiza tudo para o vídeo ficar fluido
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

# 2. Agente Redator
redator = AssistantAgent(
    name="Redator",
    llm_config=llm_config,
    system_message="""
    Você é um redator de marketing criativo.
    Escreva textos curtos e inovadores sobre tecnologia.
    Se o Editor pedir mudanças, refaça o texto acatando as sugestões.
    Não use emojis em excesso.
    """
)

# 3. Agente Editor
editor = AssistantAgent(
    name="Editor",
    llm_config=llm_config,
    system_message="""
    Você é um editor sênior rigoroso.
    Analise o texto do Redator.
    Critérios: O texto deve ser claro, objetivo e inspirador.
    - Se o texto estiver ruim: Diga EXATAMENTE o que mudar (ex: "Muito longo", "Tom muito informal").
    - Se o texto estiver excelente: Responda APENAS com a palavra "TERMINATE".
    """
)

# --- EXECUÇÃO (CHAT EM GRUPO) ---

# Cria o grupo com os 3 participantes
groupchat = GroupChat(
    agents=[user_proxy, redator, editor], 
    messages=[], 
    max_round=8 # Limita a conversa para não ficar infinita
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

print("--- Iniciando a Agência de Marketing com Gemini ---")

# O Admin inicia a conversa pedindo o trabalho
task = "Escreva um post curto para LinkedIn sobre 'Quais são boas práticas de programação?'."

user_proxy.initiate_chat(
    manager,
    message=task
)