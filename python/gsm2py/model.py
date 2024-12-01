from pydantic import BaseModel
from pydantic import Field
from openai import OpenAI
import json
from dill.source import getsource
__CLIENT__ = None

class GSMProblem(BaseModel):
    pycode: str = Field(description='the python code input by user to generate the gsm problem')
    question: str = Field(description='the math problem being generated')
    IC: str = Field(description='the irrelevant context which was added as noise when present or empty if no noise was added')
    analysis: str = Field(description='analysis indicating whether the gsm problem generated satisfies all the directives')
    valid: bool = Field(description='the result of the consistency check')

    def __str__(self):
        # Pretty print the fields
        pretty_dict = self.dict()
        pretty_dict["pycode"] = "\n" + self.pycode  # Add a newline for better formatting
        formatted_output = json.dumps(pretty_dict, indent=4)
        # Replace escaped characters in pycode
        formatted_output = formatted_output.replace('\\n', '\n').replace('\\"', '"')
        return formatted_output

class GSMResponse(BaseModel):
    question: str = Field(description='the math problem being answered')
    reasoning: str = Field(description='the reasoning leading to the answer offered')
    answer: str = Field(description='the answer to the math problem')
    pycode: str = Field(description='the python code generated to represent the math problem')
    IC: str = Field(description='if some irrelevant context was detected put it there')

    def __str__(self):
        # Pretty print the fields
        pretty_dict = self.dict()
        pretty_dict["pycode"] = "\n" + self.pycode  # Add a newline for better formatting
        formatted_output = json.dumps(pretty_dict, indent=4)
        # Replace escaped characters in pycode
        formatted_output = formatted_output.replace('\\n', '\n').replace('\\"', '"')
        return formatted_output


class GSMResponseFE(BaseModel):
    generic: str = Field(description='if the user did not wanted to test functional equivalence with a model python function just put the response in this field')
    reasoning: str = Field(description='reasoning justifying why response code was found functionally equivalent or not with the model provided ')
    pytests : str = Field(description='if functional equivalence was requested place unit tests checking the functional equivalence here')

    def __str__(self):
        # Pretty print the fields
        pretty_dict = self.dict()
        pretty_dict["pytests"] = "\n" + self.pytests  # Add a newline for better formatting
        formatted_output = json.dumps(pretty_dict, indent=4)
        # Replace escaped characters in pycode
        formatted_output = formatted_output.replace('\\n', '\n').replace('\\"', '"')
        return formatted_output

def get_client():
    global __CLIENT__
    if __CLIENT__ is None:
        __CLIENT__=OpenAI()

    return __CLIENT__

def query(model,messages, format=None, client=get_client()):
    if format is not None:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=format,
        )
        return completion.choices[0].message.parsed

    completion=client.chat.completions.create(
        model=model,
        messages=messages)
    return completion.choices[0].message.content


def message(role,content):
    return {"role": role, "content": content}


class Model:
    def __init__(self, model_name, enc_dir, dec_dir):
        self.model_name = model_name
        self.enc_dir=message("system",enc_dir)
        self.dec_dir=message('system',dec_dir)

    def enc(self,mod):
        mod_src=getsource(mod)
        msgs=[
            self.enc_dir,
            message('user',f'please generate 2 problems for the function\n {mod_src}')
        ]
        return query(self.model_name, msgs, format=GSMProblem)

    def dec(self,gsm,mod=None,followup=None,followup2=None):
        def resp(msgs,format):
            return query(self.model_name,msgs,format=format)
        question=f'please solve the following grade school mathematics problem \n "{gsm}"'
        msgs=[
            self.dec_dir,
            #message('assistant','Got it! I will follow the outlined process for solving grade school mathematics problems.\n Please provide a mathematics problem, and I will generate the solution in the format specified.'),
            message('user',question)
        ]
        res={'Q1':question,
             'A1':resp(msgs, format=GSMResponse)}
        if followup is None and mod is not None:
            mod_src = getsource(mod)
            followup=f'Assume you are allowed to perform some simple mapping between function arguments. Is your function sol functionally equivalent to:\n {mod_src} \n. Write a list of 3 unit tests to check the functional equivalent under the mapping you identified. Make sure to name all the input variables of the functions sol and mod when you write the unit test in order to avoid confusion in the ordering of the input variables'

        if followup is not None:
            msgs.append(message('assistant',str(res['A1'])))
            msgs.append(message('user',followup))
            res['Q2']=followup
            res['A2']=resp(msgs,format=GSMResponseFE)

        if followup2 is not None:
            msgs.append(message('assistant',str(res['A2'])))
            msgs.append(message('user',followup2))
            res['Q3']=followup2
            res['A3']=resp(msgs,format=None)

        return res
