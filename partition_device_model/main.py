from Stling import detail
from load_file import schedule_ctx
from schedule_layer_to_device import schedule_layer_to_device, schedule_types_to_hosts, print_host_sched, dev_type_sched_stage
from smallest_multiple import smallest_multiple
#y = 2
x = [1,2,3,4,5,6,7,8]#已知设备数量
B = smallest_multiple(len(x))##总批次（总任务量）
print("B:", B)
#print(detail(x,y))
optimal_time_list = []
optimal_group_list = []
model_device_part_method_1 = []
for y in range(1,len(x)+1):
    b = B // y#总批次分给每个组的微批次
    device_par_list = detail(x,y)#每一种结果
    final_time_list = []
    model_device_part_method_2 = []
    for device_partition_res in device_par_list:#枚举每一种结果
        group_time_res = []
        model_device_part_method_3 = []
        for group in device_partition_res:
            #print("group: ", group)
            a = schedule_ctx(group)
            a.load_data()
            sched_and_res = schedule_layer_to_device(a)
            #dev_type_sched_dev, res = schedule_layer_to_device(a)
            if sched_and_res is not None:
                dev_type_sched_dev, res = sched_and_res
            else:
                print("该组：", group, "设备内存不足，放弃")
                res = 10000.0
                dev_type_sched_dev = [dev_type_sched_stage(0,0,0)]
                #break
            host_sched, infer_time = schedule_types_to_hosts(dev_type_sched_dev, a.dev_infos, res, b)
            print_host_sched(host_sched)
            #group_time_res.append(res)
            group_time_res.append(infer_time)
            model_device_part_method_3.append(host_sched)   
            #print("每个小组的最小时间：", res)
            #print("--------------------------------------")
            #print(" ")
        # print("device_partition:  ", device_partition_res)
        final_time = max(group_time_res)
        model_device_part_method_2.append(model_device_part_method_3)
        print("该分配组：",device_partition_res," 最终的最优时间为： ", final_time)
        print("********************************")
        print(" ")
        final_time_list.append(final_time)
    all_group_minitime = min(final_time_list)
    device_parti_final_res = device_par_list[final_time_list.index(all_group_minitime)]
    model_device_part_method_1.append(model_device_part_method_2[final_time_list.index(all_group_minitime)])
    print("分为",y,"组的", "最终结果： ")
    print(device_parti_final_res)
    print("分为",y,"组的","最终时间： ",  all_group_minitime)
    #print(final_time_list)
    print("######################################################")
    optimal_time_list.append(all_group_minitime)
    optimal_group_list.append(device_parti_final_res)
optimal_time = min(optimal_time_list)
optimal_index = optimal_time_list.index(optimal_time)
model_device_part_method_optimal = model_device_part_method_1[optimal_index]#optimal_index
print("--------------------------------------")
print("最优组数：", optimal_index + 1)
print("组内设备分配的策略：", optimal_group_list[optimal_index])
print("最优策略所花费的时间：", optimal_time)
print("模型设备划分的最优分法为：")
for i, method in enumerate(model_device_part_method_optimal):
    print("第", i+1, "组：")
    print_host_sched(method)