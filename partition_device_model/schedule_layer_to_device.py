from load_file import schedule_ctx
import pdb
# #B = 3 #batch_nums 批次的个数
# B = smallest_multiple(5)
class dev_type_sched_stage:
  def __init__(self,dev_type_idx, layer_l, layer_r):
    self.dev_type_idx = dev_type_idx
    self.layer_l = layer_l
    self.layer_r = layer_r

class host_sched_stage:
    def __init__(self,host, layer_l, layer_r):
        self.host = host
        self.layer_l = layer_l
        self.layer_r = layer_r

def schedule_layer_to_device(ctx:schedule_ctx):
    num_layers = len(ctx.model_layers) #模型层数
    dev_types_count = [1] * len(ctx.dev_infos) #每种设备的个数的列表
    devices = ctx.dev_infos #每个设备的带宽，内存，模型运行时间的信息
    mask = 1
    for i in range(dev_types_count.__len__()):
        mask *= dev_types_count[i] + 1
    #print("mask: ", mask)
    # 初始化
    # 初始化结果三维数组
    h = [[[float('inf') for k in range(len(dev_types_count))] for j in range(mask)] for i in range(num_layers+1)]
    parent = [[[(-1, -1) for k in range(len(dev_types_count))] for j in range(mask)] for i in range(num_layers+1)]

    prefix_product = [0] * (len(dev_types_count) + 1)
    for i in range(dev_types_count.__len__() + 1):
        prefix_product[i] = 1 if (i == 0) else (prefix_product[i - 1] * (dev_types_count[i - 1] + 1))
        #1 2 4 8
    for i in range(dev_types_count.__len__()):
        h[0][0][i] = 0
    #最终的时间
    res = 1e60
    res_index = list(-1 for i in range(3))
    #res_index:tuple
    #初始化结束

    #算法开始
    for i in range(num_layers):
        for S in range(mask):
            for u in range(dev_types_count.__len__()):
                if h[i][S][u] > 1e59:
                    continue
                if (S // prefix_product[u] % (dev_types_count[u] + 1) == dev_types_count[u]):
                    continue
                for j in range(i + 1, num_layers + 1):
                    if (is_layers_fit(devices, u, ctx.model_layers, i + 1, j,
                    ctx.parameters_in, ctx.dtype_size, ctx.batch_size) == False):
                        continue#这几层所占用的总内存是否超过设备内存的限制
                    computation_time = compute_time(devices[u].layers_comp, i + 1, j)
                    if (j == num_layers) :#只有最后一步才会输出结果
                        cost = max(h[i][S][u], computation_time)
                        if cost < res:
                            res = cost
                            res_index[0] = i
                            res_index[1] = S
                            res_index[2] = u
                    for v in range(dev_types_count.__len__()):
                        if (S // prefix_product[v] % (dev_types_count[v] + 1) == dev_types_count[v]):
                            # v in fully used in S
                            continue
                        communication_time = comm_time(ctx.model_layers, j, ctx.dtype_size, ctx.batch_size, devices, u, v)
                        cost = max(h[i][S][u], max(computation_time, communication_time))
                        #print("h[j][S + prefix_product[u]][v]: ", j," , ", S + prefix_product[u], " , ", v)
                        if (cost < h[j][S + prefix_product[u]][v]):
                            h[j][S + prefix_product[u]][v] = cost
                            parent[j][S + prefix_product[u]][v] = (i, u)

    #算法结束
    print("Minimum time : ", res)
    if  res_index == list(-1 for i in range(3)):##出现bug
        return   
    layer, S, u = res_index
    # calculate the selected nodes
    stage_num = 0
    while (layer > 0):
        last_l:int 
        last_u:int
        stage_num += 1
        assert(parent[layer][S][u][0] >= 0)
        assert(parent[layer][S][u][1] >= 0)
        last_l, last_u = parent[layer][S][u]
        layer = last_l
        S -= prefix_product[last_u]
        u = last_u

    layer, S, u = res_index
    dev_type_sched = []
    dev_type_sched.append(dev_type_sched_stage(u, layer+1, num_layers))

    while (layer > 0):
        last_l:int 
        last_u:int
        stage_num -= 1
        assert(parent[layer][S][u][0] >= 0)
        assert(parent[layer][S][u][1] >= 0)
        last_l, last_u = parent[layer][S][u]
        dev_type_sched.append(dev_type_sched_stage(last_u, last_l + 1, layer))
        layer = last_l
        S -= prefix_product[last_u]
        u = last_u
    
    dev_type_sched = sorted(dev_type_sched, key=lambda x : x.layer_l)

    if res == 1e60:
        res = 100.0
    return dev_type_sched, res
    
def is_layers_fit(device_info:list, node:int, model_layers:list, layer_l:int, layer_r:int,
                        parameters_in:int, dtype_size:int, batch_size:int): #node 为 device种类
    mem_bytes_model = 0
    for i in range(layer_l-1, layer_r):
        mem_bytes_model += model_layers[i].layer_mem_MB * 1024 * 1024
    dat_bytes_in = parameters_in if(layer_l == 1) else layer_bytes_out(model_layers, layer_l - 1, dtype_size, batch_size)
    dat_bytes_out = layer_bytes_out(model_layers, layer_r, dtype_size, batch_size)
    mem_bytes_buffers = 0
    # if layer_l > 1:#有缓冲区的情况下，还要算进去缓冲区数据的内存
    #     mem_bytes_buffers += dat_bytes_in
    mem_bytes_buffers += dat_bytes_in + dat_bytes_out
    mem_bytes_req = mem_bytes_model + mem_bytes_buffers
    mem_bytes_avail = device_info[node].mem_MB * 1024 * 1024 #转为单位为B(字节)
    return mem_bytes_avail > mem_bytes_req # true为能够容下

def layer_bytes_out(model_layers:list, layer:int, dtype_size:int, batch_size:int):
    num_elements = model_layers[layer - 1].layer_parameters_out
    data_bytes = dtype_size * num_elements * batch_size
    return data_bytes


def compute_time(t_comp:list, layer_l:int, layer_r:int):
    time = 0
    for i in range(layer_l-1, layer_r):
        time += t_comp[i]
    return time

def comm_time(model_layers:list, layer_r:int, dtype_size:int, batch_size:int, devices:list, u:int, v:int):
     dat_bytes = layer_bytes_out(model_layers, layer_r, dtype_size, batch_size)
     mbits_sec = min(devices[u].bw_Mbps, devices[v].bw_Mbps)#u->v的通信带宽按照最小的算
     bytes_sec = mbits_sec * 1024 * 1024 / 8
     comm_time = (dat_bytes) / bytes_sec
     return comm_time


def schedule_types_to_hosts(dev_type_sched:list, device_info:list, slow_stage_time, b):
    
    host_sched = []
    every_stage_time = []
    if slow_stage_time == 10000.0:
        return [host_sched_stage("null-0", 0, 0)], slow_stage_time
    for node in dev_type_sched:
        host_sched.append(host_sched_stage(device_info[node.dev_type_idx].device_name, node.layer_l, node.layer_r))
        stage_time = 0
        for layer in range(node.layer_l - 1, node.layer_r):
            #pdb.set_trace()
            stage_time += device_info[node.dev_type_idx].layers_comp[layer]
        every_stage_time.append(stage_time)
    #print("dev_type_sched:", dev_type_sched)
    if len(every_stage_time) > 1:
        chunk = 8
        if slow_stage_time == max(every_stage_time):
            infer_time = sum(every_stage_time) + (chunk-1) * slow_stage_time
            infer_time = b * infer_time
            return host_sched, infer_time
        else:
            print(slow_stage_time)
            print(max(every_stage_time))
            raise ValueError("二者不相等")
    else:
        if slow_stage_time == max(every_stage_time):
            infer_time = sum(every_stage_time) + (b-1) * slow_stage_time
            return host_sched, infer_time
        else:
            print(slow_stage_time)
            print(max(every_stage_time))
            raise ValueError("二者不相等")



def print_host_sched(host_sched:list):
    for node in host_sched:
        print(node.host,": ","[",node.layer_l,",", node.layer_r, "]")





#test
# a = schedule_ctx()
# a.load_data()
# dev_type_sched_dev, res = schedule_layer_to_device(a)
# host_sched = schedule_types_to_hosts(dev_type_sched_dev, a.dev_infos)
# print_host_sched(host_sched)
# print("最小时间：", res)