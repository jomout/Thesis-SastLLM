from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from sastllm.configs import get_logger

logger = get_logger(__name__)


FUNCTIONALITY_PROMPT = """
You are a code functionality analysis expert. Analyze the given code snippets and summarize what each chunk functionally does.

Rules:
- Summarize the functionality into clear, semantic-leveled, goal-oriented actions (e.g., "validate input", "decrypt data", "write file").
- Focus on the intended behavior of the code, not the specific instructions.
- Do not mention registers, variables, opcodes, or syntax-level details.
- Use imperative sentences.
- Each action must be one short sentence (max 20 words).
- Separate functionalities with semicolons.
- Always output one line per chunk.
- If no meaningful functionality is found, write "None" after the chunk number.
- Do not add commentary or explanations.

Output format:
<chunk_number>: <functionality 1>; <functionality 2>; <functionality 3>

Examples:
1: Load configuration file; Decrypt payload; Execute decrypted code
2: Generate random token; Connect to remote server; Send authentication request
3: None

Now analyze the following code snippets and output only in this format:
{code_snippets}
"""


class FunctionalityAnalyzer:
    """
    Analyzes potentially malicious source code snippets using an LLM to generate 
    concise, structured reports describing each chunk's observable functionality.
    
    Attributes:
        llm (BaseChatModel): The language model used for analysis.
    """
    
    def __init__(self, llm) -> None:
        """
        Initializes the MalwareFunctionalityAnalyzer with the provided language model.
        
        Args:
            llm (BaseChatModel): An LLM compatible with LangChain, such as OpenAI's or Anthropic's chat models.
        """
        logger.debug("Initializing FunctionalityAnalyzer.")
        
        self.llm = llm
        self.functionality_prompt = ChatPromptTemplate.from_messages([
            ("user", FUNCTIONALITY_PROMPT)
        ])
        self.output_parser = StrOutputParser()
        
        # Create a LangChain chain to combine the prompt, LLM, and output parser.
        self.functionality_chain = self.functionality_prompt | self.llm | self.output_parser
        
        logger.debug("FunctionalityAnalyzer initialized.")
        

    def analyze(self, input: str) -> str:
        """
        Analyzes the provided code snippets and returns a formatted report on 
        their observable functionalities.
        
        Args:
            input (str): One or more code snippets to analyze.
                Each snippet can be separated as preferred (e.g., with comments or newlines).
        
        Returns:
            str: A concise, chunk-indexed analysis report following the prescribed format.
        
        Example output:
            1: Load configuration file; Decrypt payload; Execute decrypted code
            2: Generate random token; Connect to remote server; Send authentication request
            3: None
        """
        logger.debug("Analyzing code snippets with LLM.")
        answer = self.functionality_chain.invoke({"code_snippets": input})
        logger.debug("Functionality analysis completed.")
        return answer
