import os                                                                                                                                                                                   
MP_ID = int(os.getenv("MP_ID", "0"))                                                                                                                                                        
MP_TOTAL = int(os.getenv("MP_TOTAL", "1"))                                                                                                                                                  

inputs = range(1000)
def f(x):
    print(x)
inputs = list(inputs[MP_ID::MP_TOTAL])   