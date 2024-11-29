from openai import OpenAI
from dill.source import getsource
__CLIENT__ = None

def get_client():
    global __CLIENT__
    if __CLIENT__ is None:
        __CLIENT__=OpenAI()
    else:
        return __CLIENT__


def query(model,messages, client=get_client()):
    completion=get_client().chat.completions.create(
        model=model,
        messages=messages)
    return completion.choices[0].message


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
        return query(self.model_name,msgs)

    def dec(self,gsm,mod=None,followup=None,followup2=None):
        question=f'please solve the following grade school mathematics problem \n "{gsm}"'
        msgs=[
            self.dec_dir,
            #message('assistant','Got it! I will follow the outlined process for solving grade school mathematics problems.\n Please provide a mathematics problem, and I will generate the solution in the format specified.'),
            message('user',question)
        ]
        res={'Q1':question,
             'A1':query(self.model_name,msgs).content}
        if followup is None and mod is not None:
            mod_src = getsource(mod)
            followup=f'Assume you are allowed to perform some simple mapping between function arguments. Is your function sol functionally equivalent to:\n {mod_src} \n. Write a list of 3 unit tests to check the functional equivalent under the mapping you identified. Make sure to name all the input variables of the functions sol and mod when you write the unit test in order to avoid confusion in the ordering of the input variables'

        if followup is not None:
            msgs.append(message('assistant',res['A1']))
            msgs.append(message('user',followup))
            res['Q2']=followup
            res['A2']=query(self.model_name,msgs).content

        if followup2 is not None:
            msgs.append(message('assistant',res['A2']))
            msgs.append(message('user',followup2))
            res['Q3']=followup2
            res['A3']=query(self.model_name,msgs).content

        return res
