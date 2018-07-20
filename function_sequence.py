import numpy as np




class function_seguence:


    def __init__(self,function_dict,data_flow_key,input_dict):
        '''

        input dictionary:   {"input variable name":variable,
                            "name":variable,
                            "name":variable,
                            "name":variable,
                            "name":variable}

        function_dict:     {"function name": fucntion...


        data_flow_list:     ["function name",["function variable in place 1","function variable in place 2"...]

                            *functions should be in order of execution



        '''

        self.__initialize_dicts(function_dict,data_flow_key,input_dict)
        self.outputs=np.zeros(len(data_flow_key),dtype=np.object)




    def __initialize_dicts(self, function_dict,data_flow_key,input_dict):


        ordered_function_calls = np.zeros((len(data_flow_key)), dtype=np.object)


        self.ordered_list=data_flow_key
        self.function_dict=function_dict
        self.data_flow_key=data_flow_key
        self.input_dict=input_dict
        self.step=0



    def run(self):

        for entry in self.ordered_list:



            args=self.__fetch_args(entry)

            if entry[0]=="out":
                return args

            else:
                function=self.function_dict[entry[0]]
                output=function(*args)

                self.outputs[self.step]=output

                self.step+=1



    def __fetch_args(self,entry):
        input_key_list = entry[1]

        args = np.zeros(len(input_key_list), dtype=object)


        n = 0
        for key in input_key_list:

            if key.startswith("out_"):

                for i in range(0, np.shape(self.outputs)[0]):
                    if key.endswith(str(i)):
                        number = i
                        break
                    else:
                        number = 0
                args[n] = self.outputs[number-1]

            else:

                args[n] = self.input_dict[key]

            n += 1
        args=args.tolist()

        if len(args)==1:
            return args[0]
        else:
            return args







'''
def nothing():
    return


functions={"add":np.add,
           "multiply":np.multiply,
           "nothing":nothing}


flow=[["add",["one","two"]],
      ["multiply",["one","out_1"]],
      ["add",["one","out_2"]],
      ["multiply",["out_3","out_3"]],
      ["nothing",[]],
      ["out",["out_4"]]]

inputs={"one":1,
        "two":2,
        "three":3}



seq=function_seguence(functions,flow,inputs)

print(seq.run())
'''
