from pydantic import BaseModel
from pydantic import Field
from typing import List
from openai import OpenAI
import json
from dill.source import getsource
__CLIENT__ = None

def str2fn(code,name):
    local_namespace = {}
    exec(code, globals())
    return globals()[name]

import ast

def execute_pycode(pycode: str, testcase: str) -> float:
    """
    Executes the given Python code and test case, returning the computed answer.
    """
    # Define a local namespace for execution
    local_namespace = {}

    # Execute the Python code to define the function
    try:
        exec(pycode, globals(), local_namespace)
    except Exception as e:
        raise ValueError(f"Error in executing pycode: {e}")

    # Execute the test case to compute the answer
    try:
        exec(testcase, globals(), local_namespace)
    except Exception as e:
        raise ValueError(f"Error in executing testcase: {e}")

    # Extract the answer
    computed_answer = local_namespace.get("answer")
    if computed_answer is None:
        raise ValueError("The testcase must assign the result to a variable called 'answer'.")

    return computed_answer

# Mock function call handler
def tool_function_call(messages):
    # Extract the tool request (pycode and testcase)
    for message in messages:
        if message.get("role") == "function_call":
            tool_name = message["content"]["tool_name"]
            tool_args = message["content"]["arguments"]

            if tool_name == "execute_pycode":
                pycode = tool_args["pycode"]
                testcase = tool_args["testcase"]

                # Execute the tool and return the answer
                return execute_pycode(pycode, testcase)

# Example GPT integration with a tool
def query_with_tool(model, messages, tool_function):
    # Simulate a response from the LLM that calls the function
    tool_request = {
        "role": "function_call",
        "content": {
            "tool_name": "execute_pycode",
            "arguments": {
                "pycode": messages[-1]["pycode"],
                "testcase": messages[-1]["testcase"],
            },
        },
    }

    # Simulate the LLM delegating to the tool
    answer = tool_function([tool_request])
    return answer


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

    def mod(self):
        return str2fn(self.pycode,'mod')

class GSMResponse(BaseModel):
    gsm: str = Field(description='the grade school math problem (GSM) which was formulated by the user')
    reasoning: str = Field(description='the reasoning behind the answer you offered to the gsm problem')
    clues: List[str] = Field(description='the list of clues identified in the GSM leading to the solution')
    pycode: str = Field(description='the python definition of a function called sol generated to represent the solution to the math problem. This field should purely contain a python function definition. The function arguments should correspond to the various variables defined in the gsm')
    answer: float = Field(description='the final answer to the math problem calculated by calling the function sol with the relevant value for the input variables. Answer is expressed as a numerical value')
    IC: str = Field(description='if some irrelevant context was detected put it there')
    testcase: str = Field(description='the function call to sol that inputs for each variable of the function the concrete value extracted from the gsm problem. The testcase should be a string looking like "answer=sol(var1=...,var2=...,var3=...,...)" where each varX corresponds to a sepcific variable and ... correspond to the concrete values')

    def __str__(self):
        # Pretty print the fields
        pretty_dict = self.dict()
        pretty_dict["pycode"] = "\n" + self.pycode  # Add a newline for better formatting
        formatted_output = json.dumps(pretty_dict, indent=4)
        # Replace escaped characters in pycode
        formatted_output = formatted_output.replace('\\n', '\n').replace('\\"', '"')
        return formatted_output

    def sol(self):
        return str2fn(self.pycode,'sol')

    def validate(self):
        """
        Validates the 'answer' field by executing the 'pycode' and 'testcase'.
        If there's a mismatch, it updates the 'answer' field with the correct value.
        """
        # Define a local namespace to execute the code
        local_namespace = {}

        # Execute the function definition from pycode
        try:
            exec(self.pycode, globals(), local_namespace)
        except Exception as e:
            raise ValueError(f"Error in executing pycode: {e}")

        # Extract the function object
        sol = local_namespace.get("sol")
        if sol is None:
            raise ValueError("The pycode field must define a function called 'sol'.")

        # Execute the testcase
        try:
            exec(self.testcase, globals(), local_namespace)
        except Exception as e:
            raise ValueError(f"Error in executing testcase: {e}")

        # Extract the computed answer
        computed_answer = local_namespace.get("answer")
        if computed_answer is None:
            raise ValueError("The testcase must assign the result to a variable called 'answer'.")

        # Compare and update the answer field if necessary
        if self.answer != computed_answer:
            print(f"Warning: Answer field ({self.answer}) does not match computed answer ({computed_answer}). Updating...")
            #self.answer = computed_answer
            return False

        return True

class GSMResponseTA(GSMResponse):
    review_reasoning: str = Field(description='Offer 1 point for each correct deduction and remove 1 point for any mistake identified. Offer final score for the reasoning section')
    review_clues: str = Field(description='Offer a score 1 point for each correctly identified clue and remove 1 point for any clue missed. Offer eventually the total score for the clues section')
    review_code: str = Field(description='Offer a score of 1 point if you think the code computes the correct solution and -1 if not')
    review_ic: str = Field(description='Offer a score of 1 point if you think the solution identified properly any irrelevant context. Remove one point if the problem failed to identify the irrelevant context and remove 2 points if it incorrectly marked some important clue as irrelevant')

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
    def __init__(self, model_name, enc_dir, dec_dir, ta_dir):
        self.model_name = model_name
        self.enc_dir=message("system",enc_dir)
        self.dec_dir=message('system',dec_dir)
        self.ta_dir=message('system',ta_dir)

    def enc(self,mod):
        mod_src=getsource(mod)
        msgs=[
            self.enc_dir,
            message('user',f'please generate 2 problems for the function\n {mod_src}')
        ]
        return query(self.model_name, msgs, format=GSMProblem)

    def dec(self,gsm,mod=None,followup=None,followup2=None,ta=False):
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

        if ta:
            to_review=str(res['A1']) # the full structured response
            ta_msgs=[
                self.ta_dir,
                message('user',to_review)
            ]
            res['Q1b']=to_review
            res['A1b']=resp(ta_msgs, format=GSMResponseTA)

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
