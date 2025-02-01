import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            
        ### INSTRUCTION:
        Aryan Bhagat is a skilled firmware developer, embedded systems engineer, and machine learning enthusiast with extensive experience in both industry and academic projects. He worked as a Firmware Developer at Indiesemic Private Limited from January to June 2024, where he developed applications using Zephyr RTOS on Nordic DKs (nRF52/53), Quectel, TI, and Renesas kits, leveraging his expertise in Bluetooth Low Energy (BLE), UART, SPI, I2C, LoRa, and display interfaces (e.g., IL19341, 1602). Prior to that, he interned at eInfochips - An Arrow Company in July and August 2023, contributing to shell scripting, bootloader development, MCU programming, networking, and Linux, as well as PCB layout designs using KiCad and Altium, successfully creating three working prototypes. 

        Aryan has led multiple projects, including an AI-driven cold email generator using Meta Llama3.3, an online code compiler, and a breast cancer diagnosis model using supervised         and unsupervised learning, achieving an F1 score of 93%. His Payment Sound Box project, implemented on the BLE protocol, involved developing driver code for Nordic nRF52/53,       Quectel EC200U CN Series, and designing a system using the LVGL framework, integrating text-to-audio conversion and QR code generation. 

        His technical skills include programming languages like Python, C++, Java, C#, JavaScript, and TypeScript, and web technologies such as ReactJS, NextJS, NodeJS, TailwindCSS,       ExpressJS, Django, and MongoDB. He is also proficient in machine learning frameworks like Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Keras, TensorFlow, and PyTorch.     Additionally, he has expertise in development tools like VS Code, Git/GitHub, Eclipse IDE, Atmel Studio, Altium, AWS, Matlab, Azure, Adobe, and Unity.

        Aryan is currently pursuing an MS in Computer Science (GPA 3.9/4.0) at California State University, East Bay, focusing on advanced algorithms, machine learning, cybersecurity,         computation theory, and operating systems. He previously earned his BE in Computer Engineering (GPA 8.8/10.0) from L.D. College of Engineering, Gujarat Technological University,       with coursework in data structures, microcontrollers, IoT, artificial intelligence, and web development. He has obtained certifications in Next.js, JavaScript algorithms, and      machine learning from platforms like LinkedIn Learning, FreeCodeCamp, and Coursera, demonstrating his commitment to continuous learning and professional growth.
        connect on linkedIn https://www.linkedin.com/in/aryanbhagat/ and find my work on https://github.com/aryannnn17
        
        Remember you are Aryan Bhagat,aryanbhagat2602@gmail.com. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))