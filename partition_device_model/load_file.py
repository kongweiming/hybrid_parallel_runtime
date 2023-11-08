import yaml
import sys
devices_infer_profile_path = "../PipeEdge-main/large_case_exam/5_case_large.yml"

devices_path = "./devices.yml"
models_profile_path = "../PipeEdge-main/models_large.yml"

def load_yaml(yaml_path):
    try:
        # 打开文件
        with open(yaml_path,"r",encoding="utf-8") as f:
            data=yaml.load(f,Loader=yaml.FullLoader)
            return data
    except:
        return None
#print(type(devices_infer_profile))
print(devices_infer_profile_path)
class model_layer:
    def __init__(self, layer_parameters_out, layer_mem_MB):
        self.layer_parameters_out = layer_parameters_out
        self.layer_mem_MB = layer_mem_MB

class devcies_profile:
    def __init__(self,device_name,mem_MB,bw_Mbps,layers_comp):
        self.device_name = device_name
        self.mem_MB = mem_MB
        self.bw_Mbps = bw_Mbps
        self.layers_comp = layers_comp

class schedule_ctx:
    def __init__(self, choose_device_group):
        self.parameters_in = None
        self.model_layers = []#每层模型的内存及输出大小
        self.dev_infos = []#devcies_profile
        self.dtype_size = None
        self.batch_size = None
        self.devices_infer_profile = load_yaml(devices_infer_profile_path)
        self.devices = load_yaml(devices_path)
        self.models_profile = load_yaml(models_profile_path)
        self.choose_device_group = choose_device_group

    def load_data(self):
        self.parameters_in  = self.models_profile["parameters_in"]
        #print(self.parameters_in)
        for i in range(self.models_profile["layers"]):
            self.model_layers.append(model_layer(self.models_profile["parameters_out"][i], self.models_profile["mem_MB"][i]))
        #print(len(self.devices))
        # for i in range(1,len(self.devices) + 1):
        for i in self.choose_device_group:
            self.dev_infos.append(devcies_profile(self.devices[i],self.devices_infer_profile[i]["mem_MB"], self.devices_infer_profile[i]["bw_Mbps"],
                                                  self.devices_infer_profile[i]["model_profiles"]["time_s"]))
        if(self.devices_infer_profile[i]["model_profiles"]["dtype"] == "torch.float32"):
            self.dtype_size = 4
            #print(self.dtype_size)
        self.batch_size = self.devices_infer_profile[i]["model_profiles"]["batch_size"]
        
        

# a = schedule_ctx()
# a.load_data()
# print(11111)   

# import struct

# value = 0.0 # 设置一个float32类型的变量
# packed = struct.pack('f', value) # 使用struct.pack()函数将float32变量打包成二进制数据
# size = len(packed) # 获取打包后的二进制数据长度，即float32类型所占用的字节大小
# print(size)
