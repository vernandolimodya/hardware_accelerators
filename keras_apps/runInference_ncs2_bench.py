# -*- coding: utf-8 -*-
"""
@author: Vernando Limodya

This code runs the inference of the angle model with the Neural Compute Stick with Myriad X VPU.
Uses the OpenVINO framework, meaning we need to convert our Tensorflow Lite models
into .bin and .xml files with the Model Optimizer from the OpenVINO Development framework.

"""
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
# Importing the IECore from OpenVINO Runtime # 
from openvino.inference_engine import IECore

class evalSession():

    def __init__(self, model_name, session_name="evalSession", generated_inputs=1e4):
        # Set the path for the .xml and .bin files for the model #
        # as well as the input file directory #

        self.generated_inputs = int(generated_inputs)
        self.session_name = session_name
        self.model_name = model_name
        self.input_size = 32
        self.filters = 64
        self.model_float_ops = int(pd.read_csv('vpu_models/' + self.model_name + "/" + "op_stat.csv")['flops'].sum())
        # self.model_float_ops = (self.input_size*self.input_size*1*self.filters*int(self.model_name.split("_")[1])*int(self.model_name.split("_")[1]))*2

        # Load the Inference Engine #
        self.ie_core = IECore()
        self.ie_core.set_config(config={'NUM_STREAMS':'1', 'INFERENCE_NUM_THREADS': '1'}, device_name="CPU")
        self.ie_core.set_config(config={'MYRIAD_ENABLE_HW_ACCELERATION':'YES'}, device_name="MYRIAD")

    def inference(self, batch_size=1, device='CPU', verbose=False):
        
        dir_name = 'vpu_models/' + self.model_name + "/"
        # Read the model files and load the network # 
        # This is analog to creating an Interpreter #
        if device == 'CPU':
            self.net = self.ie_core.read_network(model= dir_name + "ir_model.xml", weights= dir_name + "ir_model.bin")
        elif device == 'MYRIAD':
            self.net = self.ie_core.read_network(model= dir_name + "ir_model_compressed.xml", weights= dir_name + "ir_model_compressed.bin")
        results_arr = np.empty((self.generated_inputs, 1))
        self.net.batch_size = batch_size
        
        if verbose:
            print("Loading the model into the " + device)
        
        start_load_time = time.time()
        # Load the network onto the Myriad device + pre-planning tensor allocations #
        exec_net = self.ie_core.load_network(network=self.net, device_name=device)
        stop_load_time = time.time()

        # Get the input and output layer names
        input_layers = list(self.net.input_info.keys())
        output_layers = list(self.net.outputs.keys())

        # input_data = np.array([np.random.randint(-128, 127, size=(self.input_size,self.input_size,1)).astype('float16') for z in range(self.generated_inputs)])
        input_data = np.array([np.load("inputs/numpy_fp32_concat/convertedInput_" + str(i) + ".npy").reshape((self.input_size,self.input_size,1)).astype('float32') for i in range(self.generated_inputs)])

        def create_batches(data, batch_size):
            for i in range(0, len(data), batch_size):
                yield data[i:i+batch_size]

        input_data = create_batches(input_data, batch_size)

        if verbose:
            print("Starting inference of " + str(self.generated_inputs) + " samples with batch size " + str(batch_size) + " ...")

        start_inference_time = time.time()

        iter = 0
        for batch in input_data:
            if iter % 1000 == 0:
                print(iter)
            # for the last batch make an exception -- fill this with zeros till the batch matches the batch size
            batch_len = len(batch)
            if (iter+1)*batch_size > self.generated_inputs:
                for _ in range((iter+1)*batch_size - self.generated_inputs):
                    batch = np.append(batch, [np.zeros((self.input_size, self.input_size, 1), dtype=np.float32)], axis=0)
            
            # Setting input + running the inference #
            results = exec_net.infer(inputs={input_layers[0]: batch[:,0]})

            # Retrieve the output data for each output layer #
            outputs_0 = results[output_layers[0]]

            # Since the output is in a form so that the i-th element of the output #
            # with 1.0 as a value corresponds to the angle i, therefore I thought of the #
            # workaround with the scalar dot product #
            pid = [np.argmax(_) for _ in outputs_0]

            for i in range(batch_len):
                results_arr[batch_size*iter + i, 0] = pid[i]

            iter += 1

        stop_inference_time = time.time()

        del exec_net # to be safe

        return {'model_load_time' : stop_load_time - start_load_time,
                'results_arr': results_arr, 
                'latency': (stop_inference_time - start_inference_time)/self.generated_inputs,
                'fps': self.generated_inputs/(stop_inference_time - start_inference_time),
                'flops': (self.model_float_ops*self.generated_inputs)/(stop_inference_time - start_inference_time)
                }

    def accuracy_degr(self,A,B):
        return np.equal(A,B).sum()/np.size(A)
    
    def difference(self,A,B):
        return np.linalg.norm(B-A)/np.linalg.norm(A) # the norm is the Frobenius norm
        # See lecture on PCA Analysis Stephan Guennemann
        # maybe for angle models could be a good metric

    def experiment(self, batch_sizes, reps=1, verbose=False):

        exp_data ={'batch_size': batch_sizes,
                    'diff_mean': [],
                    'diff_stdev': [],
                    'accuracy_degr_mean': [],
                    'accuracy_degr_stdev': [],
                    'latency_cpu_mean': [],
                    'latency_cpu_stdev': [],
                    'latency_vpu_mean': [],
                    'latency_vpu_stdev': [],
                    'fps_cpu_mean': [],
                    'fps_cpu_stdev': [],
                    'fps_vpu_mean': [],
                    'fps_vpu_stdev': [],
                    'flops_cpu_mean': [],
                    'flops_cpu_stdev': [],
                    'flops_vpu_mean': [],
                    'flops_vpu_stdev': [],
                    'speed_mult': [],
                    'model_load_time_cpu_mean': [],
                    'model_load_time_cpu_stdev': [],
                    'model_load_time_vpu_mean': [],
                    'model_load_time_vpu_stdev': []
                }

        for i in range(len(batch_sizes)):
            
            print("Evaluating for series size " + str(batch_sizes[i]) + " ... ")

            model_load_time_cpu = np.empty((reps,))
            latency_cpu = np.empty((reps,))
            fps_cpu = np.empty((reps,))
            flops_cpu = np.empty((reps,))

            model_load_time_vpu = np.empty((reps,))
            latency_vpu = np.empty((reps,))
            fps_vpu = np.empty((reps,))
            flops_vpu = np.empty((reps,))

            results_cpu = np.empty((reps, self.generated_inputs, 1))
            results_vpu = np.empty((reps, self.generated_inputs, 1))

            for _ in range(reps):

                print("Rep number : " + str(_))

                if verbose:
                    print("Evaluating for CPU ... ")

                cpu = self.inference(batch_size=batch_sizes[i], device='CPU', verbose=verbose)

                model_load_time_cpu[_] = cpu['model_load_time']
                results_cpu[_] = cpu['results_arr']
                latency_cpu[_] = cpu['latency']
                fps_cpu[_] = cpu['fps']
                flops_cpu[_] = cpu['flops']

                if verbose:
                    print("Evaluating for MYRIAD ... ")

                vpu = self.inference(batch_size=batch_sizes[i], device='MYRIAD', verbose=verbose)
                # vpu = {'model_load_time' : 1, 'results_arr': 1, 'latency': 1, 'fps': 1, 'flops': 1}

                model_load_time_vpu[_] = vpu['model_load_time']
                results_vpu[_] = vpu['results_arr']
                latency_vpu[_] = vpu['latency']
                fps_vpu[_] = vpu['fps']
                flops_vpu[_] = vpu['flops']

            # Processing results for one series size
            diff = np.empty((reps, reps))
            accuracy_degr = np.empty((reps, reps))
            for j in range(len(results_cpu)):
                for k in range(len(results_vpu)):
                    diff[j,k] = self.difference(results_cpu[j],results_vpu[k])
                    accuracy_degr[j,k] = self.accuracy_degr(results_cpu[j],results_vpu[k])

            exp_data['diff_mean'].append(np.mean(diff))
            exp_data['diff_stdev'].append(np.std(diff))

            exp_data['accuracy_degr_mean'].append(np.mean(accuracy_degr))
            exp_data['accuracy_degr_stdev'].append(np.std(accuracy_degr))

            exp_data['latency_cpu_mean'].append(np.mean(latency_cpu))
            exp_data['latency_cpu_stdev'].append(np.std(latency_cpu))
            exp_data['latency_vpu_mean'].append(np.mean(latency_vpu))
            exp_data['latency_vpu_stdev'].append(np.std(latency_vpu))

            exp_data['fps_cpu_mean'].append(np.mean(fps_cpu))
            exp_data['fps_cpu_stdev'].append(np.std(fps_cpu))
            exp_data['fps_vpu_mean'].append(np.mean(fps_vpu))
            exp_data['fps_vpu_stdev'].append(np.std(fps_vpu))

            exp_data['flops_cpu_mean'].append(np.mean(flops_cpu))
            exp_data['flops_cpu_stdev'].append(np.std(flops_cpu))
            exp_data['flops_vpu_mean'].append(np.mean(flops_vpu))
            exp_data['flops_vpu_stdev'].append(np.std(flops_vpu))

            exp_data['model_load_time_cpu_mean'].append(np.mean(model_load_time_cpu))
            exp_data['model_load_time_cpu_stdev'].append(np.std(model_load_time_cpu))
            exp_data['model_load_time_vpu_mean'].append(np.mean(model_load_time_vpu))
            exp_data['model_load_time_vpu_stdev'].append(np.std(model_load_time_vpu))

        exp_data['speed_mult'] = [ exp_data['fps_vpu_mean'][a]/exp_data['fps_cpu_mean'][a] for a in range(len(batch_sizes)) ]

        return pd.DataFrame.from_dict(exp_data)

    def plot_results(self, batch_sizes, reps=1, verbose=False, filename=None):
    
        if filename == None:
            filename = self.session_name + "_" + datetime.now().strftime("%Y_%m_%d-%H_%M")

        if not os.path.exists("output_files_vpu/"):
            os.mkdir("output_files_vpu/")

        if not os.path.exists("output_files_vpu/" + filename + "/"):
            os.mkdir("output_files_vpu/" + filename + "/")

        df_exp_data = self.experiment(batch_sizes=batch_sizes, reps=reps, verbose=verbose)
        df_exp_data.to_csv("output_files_vpu/" + filename + "/" + filename + ".csv")

if __name__=="__main__":
    
    batch_sizes = [1]

    model_names = []

    for i in ["DenseNet121", "MobileNet", "ResNet50", "ResNet50V2", "VGG16", "VGG19"]:
        model_names.append(i)
    
    for model_name in model_names:
        print(model_name)
        ex_fp32 = evalSession(model_name=model_name, generated_inputs=1000) # model_float_ops must be researched
        ex_fp32.plot_results(batch_sizes, reps=10, verbose=True, filename=model_name)
        del ex_fp32