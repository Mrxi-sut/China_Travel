import os
import sys
from json_repair import repair_json
from pathlib import Path
project_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if project_path not in sys.path:
    sys.path.append(project_path)

import json
from tqdm import tqdm
from copy import deepcopy
from chinatravel.agent.llms import Deepseek, GPT4o, Qwen, Mistral, GLM4Plus
from chinatravel.symbol_verification.concept_func import func_dict
from chinatravel.agent.tpc_agent.prompts import NL2SL_INSTRUCTION_V2, NL2SL_INSTRUCTION
from chinatravel.agent.nesy_agent.ast_checker import HardLogicPyChecker
from chinatravel.data.load_datasets import save_json_file, load_json_file

sys.path.append("../../../")
project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)
func_docs = """
(1) day_count(plan)
Docs: Get the number of days in the plan.
Return: int
(2) people_count(plan)
Docs: Get the number of people in the plan.
Return: int
(3) start_city(plan)
Docs: Get the start city of the plan.
Return: str
(4) target_city(plan)
Docs: Get the target city of the plan.
Return: str
(5) allactivities(plan)
Docs: Get all the activities in the plan.
Return: list of activities
(6) allactivities_count(plan)
Docs: Get the number of activities in the plan.
Return: int
(7) dayactivities(plan, day)
Docs: Get all the activities in the specific day [1, 2, 3, ...].
Return: list of activities
(8) activity_cost(activity)
Docs: Get the cost of specific activity without transport cost.
Return: float
(9) activity_position(activity)
Docs: Get the position name of specific activity.
Return: str
(10) activity_price(activity)
Docs: Get the price of specific activity. The price is price per person.
Return: float
(11) activity_type(activity)
Docs: Get the type of specific activity. ['breakfast', 'lunch', 'dinner', 'attraction', 'accommodation', 'train', 'airplane']
Return: str
(12) activity_tickets(activity)
Docs: Get the number of tickets needed for specific activity. ['attraction', 'train', 'airplane']
Return: int
(13) activity_transports(activity)
Docs: Get the transport information of specific activity.
Return: list of dict
(14) activity_start_time(activity)
Docs: Get the start time of specific activity.
Return: str
(15) activity_end_time(activity)
Docs: Get the end time of specific activity.
Return: str
(16) activity_time(activity)
Docs: Get the duration of specific activity.
Return: int (minutes)
(17) innercity_transport_cost(transports)
Docs: Get the total cost of innercity transport.
Return: float
(18) poi_recommend_time(city, poi):
Docs: Get the recommend time of specific poi in the city. Only support attractions now.
Return: int (minutes)
(19) poi_distance(city, poi1, poi2):
Docs: Get the distance between two pois in the city.
Return: float (km)
(20) innercity_transport_price(transports)
Docs: Get the price of innercity transport. The price is price per person.
Return: float
(21) innercity_transport_distance(transports)
Docs: Get the distance of innercity transport.
Return: float (km)
(22) metro_tickets(transports)
Docs: Get the number of metro tickets if the type of transport is metro.
Return: int
(23) taxi_cars(transports)
Docs: Get the number of taxi cars if the type of transport is taxi. The number of taxi cars is `(people_count(plan) + 3) // 4`.
Return: int
(24) room_count(activity)
Docs: Get the number of rooms of accommodation activity.
Return: int
(25) room_type(activity)
Docs: Get the type of room of accommodation activity.
1 for single room, 2 for double room. Must be 1 or 2. Never use "大床房" or "双床房" or other words but 1 or 2.
Return: int
(26) restaurant_type(activity, target_city)
Docs: Get the type of restaurant's cuisine in the target city. The return value must be in ['云南菜', '西藏菜', '东北菜', '烧烤', '亚洲菜', '粤菜', '西北菜', '闽菜', '客家菜', '快餐简餐', '川菜', '台湾菜', '其他', '清真菜', '小吃', '西餐', '素食', '日本料理', '江浙菜', '湖北菜', '东南亚菜', '湘菜', '北京菜', '韩国料理', '海鲜', '中东料理', '融合菜', '茶馆/茶室', '酒吧/酒馆', '创意菜', '自助餐', '咖啡店', '本帮菜', '徽菜', '拉美料理', '鲁菜', '新疆菜', '农家菜', '海南菜', '火锅', '面包甜点', '其他中餐'].
Return: str
(27) attraction_type(activity, target_city)
Docs: Get the type of attraction in the target city. The return value must be in ['博物馆/纪念馆', '美术馆/艺术馆', '红色景点', '自然风光', '人文景观', '大学校园', '历史古迹', '游乐园/体育娱乐', '图书馆', '园林', '其它', '文化旅游区', '公园', '商业街区'].
Return: str
(28) accommodation_type(activity, target_city)
Docs: Get the feature of accommodation in the target city to judge whether it's feature meets the user's requirement. The return value must be in ['儿童俱乐部', '空气净化器', '山景房', '私汤房', '四合院', '温泉', '湖畔美居', '电竞酒店', '温泉泡汤', '行政酒廊', '充电桩', '设计师酒店', '民宿', '湖景房', '动人夜景', '行李寄存', '中式庭院', '桌球室', '私人泳池', '钓鱼', '迷人海景', '园林建筑', '老洋房', '儿童泳池', '历史名宅', '棋牌室', '智能客控', '情侣房', '小而美', '特色 住宿', '茶室', '亲子主题房', '多功能厅', '洗衣房', '客栈', '自营亲子房', '停车场', 'Boss推荐', '江河景房', '日光浴场', '自营影音房', '厨房', '空调', '网红泳池', '别墅', '免费停车', '洗衣服务', '窗外好景', '酒店公寓', '会议厅', '家庭房', '24小时前台', '商务中心', '提前入园', '农家乐', '智能马桶', '美食酒店', 'SPA', '拍照出片', '海景房', '泳池', '影音房', '管家服务', '穿梭机场班车', '桑拿', '机器人服务', '儿童乐园', '健身室', '洗衣机', '自营舒睡房', '宠物友好', '电竞房', '位置超好', '套房'].
Return: str
(29) innercity_transport_type(transports)
Docs: Get the type of innercity transport. The return value must be in ['metro', 'taxi', 'walk'].
Return: str
(30) innercity_transport_start_time(transports)
Docs: Get the start time of innercity transport.
Return: str
(31) innercity_transport_end_time(transports)
Docs: Get the end time of innercity transport.
Return: str
(32) intercity_transport_type(activity)
Docs: Get the type of intercity transport. The return value must be in ['train', 'airplane'].
Return: str
(33) innercity_transport_time(transports)
Docs: Get the duration of innercity transport.
Return: int (minutes)
(34) intercity_transport_origin(activity)
Docs: Get the origin city of intercity transport.
Return: str
(35) intercity_transport_destination(activity)
Docs: Get the destination city of intercity transport.
Return: str
(36) len(collection)
Docs: Get the length (number of elements) of a collection.
Return: int
"""

sl_trans_prompt = (
    """
We offer some functions below, try to translate the constraints in nature language into python code and output them in json list format.
variables:
(1) plan: a dict of the generated plan with information of the specific plan.

functions:"""
    + func_docs
    + """
You need to response in the following format:
[
    "python code block 1",
    "python code block 2",
    ...
]

Not all the constraints need to be translated into python code. Ignore them if they can not be translated into legal python code.
!!! Only `plan` variable can be used directly in the python code. Others must be defined in the python code use the functions we offer above. !!! Pay attention to the return TYPE of functions!!!
For most case, for exist constraints, you can set `result=False` at the beginning of the code, and then set `result=True` if the condition is satisfied. For all constraints, you can set `result=True` at the beginning of the code, and then set `result=False` if the condition is not satisfied.

### Attention!!!
If you find some pesucode in the nature language constraints is not defined in the functions we offer above, you must translate them into python block code with the functions we offer above. Usually, for attractions and restaurants, if the required one exists, the requirement is satisfied. However, for accommodation, people usually stay in the same hotel for the whole trip, so we need check all the accommodation activities in the plan. 
###

if you find some error in nature language constraints, you need to fix them in the code block. if {'自然景观'} <= spot_type, you need to change it to '自然风光' in the code block as we offer above. As the same, if {'大学'} <= spot_type, you need to change it to '大学校园' and '繁华的商业街' to '商业街区' in the code block. Also for restaurant_type and accommodation_type.
### IMPORTANT RULES FOR OR CONDITIONS:
If the constraint contains "or" (logical OR), you MUST combine them into ONE code block using Python's `or` operator.
- First compute each condition as separate variables
- Then combine them with `or` operator
- DO NOT create separate code blocks for each part of the OR condition
- DO NOT use if-else logic to simulate OR

Example for OR condition:
Nature language: "({'A','B'}<=attraction_names) or (innercity_transport_cost<=100)"
Correct approach:
"attraction_names_set = set()
for activity in allactivities(plan):
    if activity_type(activity) == 'attraction':
        attraction_names_set.add(activity_position(activity))
condition1 = ({'A','B'}<=attraction_names_set)
innercity_cost = 0
for activity in allactivities(plan):
    innercity_cost += innercity_transport_cost(activity_transports(activity))
condition2 = (innercity_cost<=100)
result = (condition1 or condition2)"

Incorrect approaches (DO NOT use these):
- Creating separate code blocks for each condition
- Using if-else statements to check conditions sequentially
- Writing complex nested logic

Example:
nature_language:
days==2
people_number==3
total_cost<=3000
tickets==3
room_type==2
{'北京菜'}<=food_type
intercity_transport_go=={'train'}
intercity_transport_back=={'train'}
spot_type<={'自然风光'}
{'智能客控'}<=hotel_feature
hotel_price<=500
{'北京全聚德(前门店)'} <= restaurant_names
food_price<=100
transport_type<={'metro', 'taxi'}
{'故宫博物院'}<=attraction_names
taxi_cars==1
attraction_price<=0
food_cost<=300
attraction_cost<=200
hotel_cost<=1000
{'上海外滩万达瑞华酒店', '上海木林之畔度假屋'}<=hotel_names
intercity_transport_cost<=4000
innercity_transport_cost<=3000
distance>2.82
start_time{清平古墟影视文化园}<=08:20
end_time{三郎日料•烧肉酒场(文晖店)}>=17:50
accommodation_distance{客家土楼(福建)}<=10.41
time_range{茅家埠景区,08:50-10:20}
activity_time{滨海文化公园}>=90
answer:
[
"result=(day_count(plan)==2)",
"result=(people_count(plan)==3)",
"total_cost = 0
for activity in allactivities(plan):
    total_cost += activity_cost(activity)
    if activity_transports(activity) is not None:
        total_cost += innercity_transport_cost(activity_transports(activity))
result = (total_cost <= 3000)",
"result=True
for activity in allactivities(plan):
  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!=people_count(plan): 
    result=False
  if innercity_transport_type(activity_transports(activity))=='metro' and metro_tickets(activity_transports(activity))!=people_count(plan): 
    result=False",
"result=True
for activity in allactivities(plan):
  if innercity_transport_type(activity_transports(activity))=='taxi' and taxi_cars(activity_transports(activity))!=(people_count(plan)+3)//4: 
    result=False",
"result=True
for activity in allactivities(plan):
  if activity_type(activity)=='accommodation' and room_type(activity)!=2: result=False
  if activity_type(activity)=='accommodation' and room_count(activity)!=(people_count(plan)+room_type(activity)-1)//room_type(activity): result=False
  if activity_type(activity)=='accommodation' and activity_price(activity)>500: result=False",
"restaurant_type_set = set()
for activity in allactivities(plan):
  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:
    rest_type = restaurant_type(activity, target_city(plan))
    if rest_type is not None:
      restaurant_type_set.add(restaurant_type)
result=({'北京菜'}<=restaurant_type_set)",

"attraction_type_set = set()
for activity in allactivities(plan):
  if activity_type(activity)=='attraction':
    attr_type = attraction_type(activity, target_city(plan))
    if attr_type is not None:
      attraction_type_set.add(attraction_type)
result=(attraction_type_set.isdisjoint({'自然风光'}))",
"accommodation_type_set=set()
for activity in allactivities(plan):
  if activity_type(activity)=='accommodation': 
    acco_type = accommodation_type(activity, target_city(plan))
    if acco_type is not None:
      accommodation_type_set.add(acco_type)
result=({'智能客控'}<=accommodation_type_set)",
"intercity_transport_go_set = set()
intercity_transport = plan['itinerary'][0]['activities'][0]['type']
if intercity_transport is not None:
  intercity_transport_go_set.add(intercity_transport)
result = (intercity_transport_go_set == {'train'})
",
"intercity_transport_back_set = set()
activities = allactivities(plan)
for activity in activities:
  if activity_type(activity) in ['train', 'airplane']:
    if is_back_trip(activity, plan):
      activity_type = activity_type(activity)
      if activity_type is not None:
        intercity_transport_back_set.add(activity_type)
result = (intercity_transport_back_set == {'train'})
",
"restaurant_names_set = set()
for activity in allactivities(plan):
  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:
    restaurant_names_set.add(activity_position(activity))
    result=({'北京全聚德(前门店)'}<=restaurant_names_set)",
"result=True
for activity in allactivities(plan):
  if activity_type(activity) in ['breakfast', 'lunch', 'dinner'] and activity_price(activity)>100: 
    result=False",
"innercity_transport_set = set()
for activity in allactivities(plan):
  transport = innercity_transport_type(activity_transports(activity))
    if transport is not None:
        innercity_transport_set.add(transport)
result=(innercity_transport_set.isdisjoint({'metro', 'taxi'}))",

"attraction_names_set = set()
for activity in allactivities(plan):
  if activity_type(activity)=='attraction':
    attraction_names_set.add(activity_position(activity))
    result=({'故宫博物院'}<=attraction_names_set)",
"result=True
for activity in allactivities(plan):
  if activity_type(activity)=='attraction' and activity_price(activity)>0: 
    result=False",
"food_cost=0
for activity in allactivities(plan):
  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:
    food_cost+=activity_cost(activity)
result=(food_cost<=300)",

"attraction_cost=0
for activity in allactivities(plan):
  if activity_type(activity)=='attraction':
    attraction_cost+=activity_price(activity)
result=(attraction_cost<=200)",

"hotel_cost=0
for activity in allactivities(plan):
  if activity_type(activity)=='accommodation':
    hotel_cost += room_count(activity) * activity_price(activity)
result=(hotel_cost<=1000)",
"result=False
accommodation_names = set()
for activity in allactivities(plan):
  if activity_type(activity)=='accommodation':
    accommodation_names.add(activity_position(activity))
if {'上海外滩万达瑞华酒店'} <= accommodation_names or {'上海木林之畔度假屋'} <= accommodation_names:
  result=True",
  
"intercity_cost = 0
for activity in allactivities(plan):
  if activity_type(activity) in ['train', 'airplane']:
    intercity_cost += activity_cost(activity)
result = (intercity_cost <= 4000)
",
"innercity_cost = 0
for activity in allactivities(plan):
  innercity_cost += innercity_transport_cost(activity_transports(activity))
result = (innercity_cost <= 3000)
",
"result=True
for activity in allactivities(plan):
  if activity_transports(activity) and innercity_transport_distance(activity_transports(activity))>2.82 and 
  innercity_transport_type(activity_transports(activity)) != 'taxi':
    result=False
    break",
"result=False
  for activity in allactivities(plan):
    if activity_position(activity)=='清平古墟影视文化园':
      if activity_start_time(activity)<='08:20':
        result=True",
"result=False
  for activity in allactivities(plan):
    if activity_position(activity)=='三郎日料•烧肉酒场(文晖店)':
      if activity_end_time(activity)>='17:50':
        result=True",
"result=False
accommodation_position=''
for activity in allactivities(plan):
  if activity_type(activity)=='accommodation': 
    accommodation_position=activity_position(activity)
result=(poi_distance(target_city(plan), '客家土楼(福建)', accommodation_position)<=10.41)",
"result = False
for activity in allactivities(plan):
    if activity_position(activity) == '茅家埠景区':
        if activity_start_time(activity) >= '08:50' and activity_end_time(activity) <= '10:20':
            result = True",
"result=False
for activity in allactivities(plan):
  if activity_position(activity)=='滨海文化公园':
    if activity_time(activity)>=90:
      result=True",
]"""
)

reflect_prompt = (
    """
we offer some functions below, try to reflect on the python code block and fix them and output in the same format.
[
"python code block 1",
"python code block 2",
...
]
We offer functions below:"""
    + func_docs
    + """
Try to fix the error in the code block and output them in json list format. 
The attractions_type, restaurants_type, and accommodations_type must be in the list we offer above. You must trans the original type to !!!a similar one!!! we offer if the original type is not in the list we offer above. For example '购物街' to '商业街区' and '本地特色菜' usually refers to the local cuisine in the city. 
For return value of activity_position(activity), it will be checked by whether the position is in the database. You need to trans it to a similar one if it is rufused with you own knowledge. For example, '故宫' to '故宫博物院', 'A near B' may be 'A(B店)' or 'A（B店）' or other similar ones.
Also hotel_names should be checked by activity_position(activity), not accommodation_type(activity, target_city(plan)) and so for other names.
Usually, for attractions and restaurants, if the required one exists, the requirement is satisfied. However, for accommodation, people usually stay in the same hotel for the whole trip, so we need check all the accommodation activities in the plan. Either change the function or value to make the code block correct.
You must output the whole code block. Including those constraints that are correct.
The original code block is:
"""
)

import os
from pathlib import Path


# def load_example_plans(example_plans_dir=None):
#     """加载示例计划文件（跨平台兼容版本）"""
#     # 设置默认路径（使用绝对路径）
#     if example_plans_dir is None:
#         # 获取当前文件所在目录的父目录（nesy_agent）
#         current_dir = Path(__file__).parent
#         # 构建默认路径（推荐使用Path对象）
#         example_plans_dir = current_dir / "plan_for_check"
#     else:
#         # 如果传入了路径参数，转换为Path对象
#         example_plans_dir = Path(example_plans_dir)
#
#     # 确保目录存在
#     if not example_plans_dir.exists():
#         raise FileNotFoundError(
#             f"示例计划目录不存在: {example_plans_dir}\n"
#             "请确保目录已创建并包含JSON计划文件"
#         )
#
#     plan_for_test = {}
#
#     # 使用glob查找所有JSON文件
#     plan_files = list(example_plans_dir.glob("*.json"))
#
#     for file in plan_files:
#         try:
#             with open(file, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 # 更安全的文件名解析
#                 day_num = int(file.stem.split("day")[-1])  # 从"plan_day1.json"中提取1
#                 plan_for_test[day_num] = data
#         except (json.JSONDecodeError, ValueError) as e:
#             print(f"警告: 跳过无效文件 {file.name}: {str(e)}")
#             continue
#
#     return plan_for_test
def load_example_plans(example_plans_dir=Path(__file__).resolve().parent.parent / "nesy_agent" / "plan_for_check"):
    plan_for_test = {}
    plan_files = os.listdir(example_plans_dir)
    plan_files = [plan_file for plan_file in plan_files if plan_file.endswith(".json")]
    for file in plan_files:
        with open(os.path.join(example_plans_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            plan_for_test[int(file.split("day")[1].split(".")[0])] = data
    return plan_for_test


EXAMPLE_PLANS = load_example_plans()


def get_first_list_in_str(json_str):
    # 使用栈得到第一个合法的list
    json_str = repair_json(json_str, ensure_ascii=False)
    st = 0
    # print(json_str)
    while st < len(json_str) and json_str[st] != "[":
        st += 1
    json_str = json_str[st:]
    stack = []
    for i, c in enumerate(json_str):
        if c == "[":
            stack.append(i)
        elif c == "]":
            stack.pop()
            if not stack:
                res = json_str[: i + 1]
                return res
    return "[]"


def nl2sl_step1(query, backbone_llm):

    nature_language = query["nature_language"]
    hard_logic_py = query["hard_logic_py"]
    start_city = query["start_city"]
    target_city = query["target_city"]
    messages = [{"role": "user", "content": NL2SL_INSTRUCTION.format(nature_language, hard_logic_py)}]
    # print(messages[0]["content"])
    query_ = backbone_llm(messages, one_line=False, json_mode=True)
    
    try:
        l_ptr = query_.find("{")
        r_ptr = query_.rfind("}")
        if l_ptr != -1 and r_ptr != -1:
            query_ = query_[l_ptr: r_ptr + 1]
        
        query_ = json.loads(query_)
        for key in query_:
            query[key] = query_[key]
        query["start_city"] = start_city
        query["target_city"] = target_city
    except Exception as e:
        query["hard_logic"] = []
        return query
    return query


def nl2sl_step2(query, backbone_llm):
    try:
        query["hard_logic"] = [str(hl) for hl in query["hard_logic"]]
        hard_logic = "\n".join(query["hard_logic"])
    except Exception as e:
        query["hard_logic"] = []
        query["hard_logic_py"] = []
        return query
    messages = [
        {
            "role": "user",
            "content": sl_trans_prompt
            + hard_logic
            + "\n The query is: \n"
            + query["nature_language"]
            + "\nanswer:\n",
        }
    ]
    # print(messages[0]["content"])
    hard_logic_py = backbone_llm(messages, one_line=False, json_mode=True)
    # l_ptr = hard_logic_py.find("[")
    # r_ptr = hard_logic_py.rfind("]")
    # if l_ptr != -1 and r_ptr != -1:
    #     hard_logic_py = hard_logic_py[l_ptr : r_ptr + 1]
    hard_logic_py = get_first_list_in_str(hard_logic_py)
    # print(hard_logic_py)
    try:
        query["hard_logic_py"] = json.loads(hard_logic_py)
    except Exception as e:
        query["error_hard_logic_py"] = hard_logic_py
        query["hard_logic_py"] = []
    query["hard_logic_py"] = [str(item) for item in query["hard_logic_py"]]
    query["hard_logic_py"] = list(set(query["hard_logic_py"]))
    return query


def check(query):
    run_error_list = []
    run_error_idx = []
    hard_logic_py = query["hard_logic_py"]
    
    if query["days"] not in EXAMPLE_PLANS:
        print("Error: days should be in [1, 2, 3, 4, 5, 6]")
        return [], []
    
    example_plan = EXAMPLE_PLANS[query["days"]]
    for idx, constraint in enumerate(hard_logic_py):
        vars_dict = deepcopy(func_dict)
        vars_dict["plan"] = example_plan
        try:
            # Evaluate the constraint in a safe manner
            exec(
                constraint,
                {
                    "__builtins__": {
                        "set": set,
                    }
                },
                vars_dict,
            )
        except Exception as e:
            if str(e) not in [
                "Failed to create Point instance from string: unknown format.",
            ]:
                run_error_list.append(str(e))
                run_error_idx.append(idx)
    return run_error_list, run_error_idx


def reflect_info(query, checker: HardLogicPyChecker):
    hard_logic_py = query["hard_logic_py"]

    run_error_list, run_error_idx = check(query)
    if len(run_error_list):
        return run_error_list, run_error_idx, [], []
    value_error_list = [checker.check(constraint)[0] for constraint in hard_logic_py]
    value_error_idx = [idx for idx, item in enumerate(value_error_list) if len(item)]
    value_error_list = [item for sublist in value_error_list for item in sublist]
    return run_error_list, run_error_idx, value_error_list, value_error_idx


def reflect(query, backbone_llm, run_error_list, value_error_list):

    content = (
        reflect_prompt
        + str(query["hard_logic_py"])
        + "The error is: "
        + "\n".join(run_error_list)
        + "\n".join(value_error_list)
        + "\nThe query is: \n"
        + query["nature_language"]
        + "\nanswer:\n"
    )
    # print(content)
    messages = [{"role": "user", "content": content}]
    res = backbone_llm(messages, one_line=False, json_mode=True)
    # l_ptr = res.find("[")
    # r_ptr = res.rfind("]")
    # if l_ptr != -1 and r_ptr != -1:
    #     res = res[l_ptr : r_ptr + 1]
    res = get_first_list_in_str(res)
    # print(res)
    try:
        query["hard_logic_py"] = json.loads(res)
    except Exception as e:
        query["error_hard_logic_py"] = res
        query["hard_logic_py"] = []
    query["hard_logic_py"] = [str(item) for item in query["hard_logic_py"]]
    # print(query["hard_logic_py"])
    return query, len(run_error_list + value_error_list) == 0


def nl2sl_step3(query, backbone_llm, checker, max_trails=5):

    cnt = 0
    query["reflect_info"] = []
    query["hard_logic_py_ood"] = []
    value_error_idx = []
    run_error_idx = []
    while cnt < max_trails:
        run_error_list, run_error_idx, value_error_list, value_error_idx = reflect_info(
            query, checker
        )
        query["reflect_info"].append(
            {
                "cnt": cnt,
                "run_error_list": run_error_list,
                "value_error_list": value_error_list,
                "hard_logic_py": query["hard_logic_py"],
            }
        )
        flag = len(run_error_list + value_error_list) == 0
        if flag:
            break
        query, _ = reflect(query, backbone_llm, run_error_list, value_error_list)
        query["hard_logic_py"] = list(set(query["hard_logic_py"]))

        # if "OOD!!!" in query["hard_logic_py"]:
        #     run_error_list, run_error_idx, value_error_list, value_error_idx = (
        #         reflect_info(query, checker)
        #     )
        #     query["ood"] = True
        #     ood_idx = list(set(run_error_idx + value_error_idx))
        #     for idx in ood_idx:
        #         query["hard_logic_py_ood"].append(query["hard_logic_py"][idx])
        #     for ood_logic in query["hard_logic_py_ood"]:
        #         query["hard_logic_py"].remove(ood_logic)
        #     return query

        cnt += 1
    query["reflect_cnt"] = cnt
    run_error_list, run_error_idx, value_error_list, value_error_idx = reflect_info(
        query, checker
    )
    query["reflect_info"].append(
        {
            "cnt": cnt,
            "run_error_list": run_error_list,
            "value_error_list": value_error_list,
            "hard_logic_py": query["hard_logic_py"],
        }
    )
    error_indices = set(run_error_list + value_error_list)
    query["hard_logic_py"] = [
        val
        for idx, val in enumerate(query["hard_logic_py"])
        if idx not in error_indices
    ]
    # ood_idx = list(set(run_error_idx + value_error_idx))
    # if len(ood_idx):
    #     query["ood"] = True
    #     for idx in ood_idx:
    #         query["hard_logic_py_ood"].append(query["hard_logic_py"][idx])
    #     for ood_logic in query["hard_logic_py_ood"]:
    #         query["hard_logic_py"].remove(ood_logic)
    return query


def nl2sl(query, backbone_llm, checker, cache_dir="cache_hybrid"):
    file_path = os.path.join(
        project_path,
        cache_dir,
        "translation_{}_reflect".format(backbone_llm.name),
        "{}.json".format(query["uid"]),
    )
    if os.path.exists(file_path):
        query = load_json_file(file_path)
        return query
    city_list = [
        "上海",
        "北京",
        "深圳",
        "广州",
        "重庆",
        "成都",
        "杭州",
        "武汉",
        "南京",
        "苏州",
    ]
    if query["target_city"] not in city_list or query["start_city"] not in city_list:
        query["hard_logic"] = []
        query["hard_logic_py"] = []
        query["ood"] = True
        return query
    query = nl2sl_step1(query, backbone_llm)
    query = nl2sl_step2(query, backbone_llm)
    query = nl2sl_step3(query, backbone_llm, checker)

    save_json_file(query, file_path)
    return query


def nl2sl_reflect(query, backbone_llm):
    city_list = [
        "上海",
        "北京",
        "深圳",
        "广州",
        "重庆",
        "成都",
        "杭州",
        "武汉",
        "南京",
        "苏州",
    ]
    if "target_city" in query and "start_city" in query:
        if query["target_city"] not in city_list or query["start_city"] not in city_list:
            query["hard_logic"] = []
            query["hard_logic_py"] = []
            query["ood"] = True
            return query
    query = nl2sl_step1(query, backbone_llm)

    return query


def run(splits: str = "tpc_phase1", backbone_llm=None, need_check=False):
    """增强版的运行函数，包含完整错误处理"""
    # 初始化路径 - 修改为您的实际路径
    cache_root = "cache_reflect_v1"
    splits_file = os.path.join(project_root_path, "chinatravel", "evaluation", "default_splits",
                               "{}.txt".format(splits))
    data_folder = os.path.join(project_root_path, "chinatravel", "data", "tpc_phase2_data")
    # 1. 验证分割文件
    if not os.path.exists(splits_file):
        available_splits = [
            f.split(".")[0] for f in os.listdir(os.path.dirname(splits_file))
            if f.endswith(".txt")
        ]
        raise FileNotFoundError(
            f"分割文件 '{splits}' 不存在\n"
            f"可用分割: {', '.join(available_splits)}"
        )

    # 2. 加载所有查询数据
    query_dict = {}

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"数据目录不存在: {data_folder}")

    print(f"正在从 {data_folder} 加载数据...")
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".json"):
                uid = os.path.splitext(file)[0]
                try:
                    query_dict[uid] = load_json_file(os.path.join(root, file))
                except Exception as e:
                    print(f"警告: 跳过文件 {file} (错误: {str(e)})")

    # 3. 验证分割文件中的UID
    with open(splits_file, "r", encoding="utf-8") as f:
        query_uid_list = [line.strip() for line in f if line.strip()]

    missing_uids = [uid for uid in query_uid_list if uid not in query_dict]
    if missing_uids:
        sample_missing = "\n".join(missing_uids[:3])
        raise KeyError(
            f"{len(missing_uids)} 个UID在数据中不存在\n"
            f"示例:\n{sample_missing}\n"
            f"可能原因:\n"
            f"1. 数据文件命名不符合 <uid>.json 格式\n"
            f"2. 部分数据文件损坏\n"
            f"3. 分割文件包含无效UID"
        )

    # 4. 准备查询列表
    query_list = [query_dict[uid] for uid in query_uid_list]
    del query_dict  # 释放内存

    # 5. 处理缓存目录
    cache_dir = os.path.join(
        os.path.dirname(data_folder),  # 使用数据目录的父目录
        cache_root,
        f"translation_{backbone_llm.name}_reflect"
    )
    os.makedirs(cache_dir, exist_ok=True)

    # 6. 处理每个查询
    city_list = ["上海", "北京", "深圳", "广州", "重庆", "成都", "杭州", "武汉", "南京", "苏州"]

    for query in tqdm(query_list, desc="处理查询"):
        try:
            # 检查查询是否包含必要字段
            if "target_city" not in query or "start_city" not in query:
                query.update({
                    "hard_logic": [],
                    "hard_logic_py": [],
                    "ood": True
                })
                save_json_file(
                    query,
                    os.path.join(cache_dir, f"{query.get('uid', 'unknown')}.json")
                )
                continue

            # 跳过非目标城市
            if query["target_city"] not in city_list or query["start_city"] not in city_list:
                query.update({
                    "hard_logic": [],
                    "hard_logic_py": [],
                    "ood": True
                })
                save_json_file(
                    query,
                    os.path.join(cache_dir, f"{query['uid']}.json")
                )
                continue

            # 正常处理逻辑
            checker = HardLogicPyChecker(query["target_city"])
            query = nl2sl(query, backbone_llm, checker, cache_dir=cache_root)

            if need_check and not len(check(query)[0]):
                print(f"验证失败: {query['uid']}")

        except Exception as e:
            print(f"处理查询 {query.get('uid', '未知')} 时出错: {str(e)}")
            continue

if __name__ == "__main__":
    splits_list = ["tpc_phase1"]
    # splits_list = ["easy_1209", "multi_cons"]
    # splits_list = ["cost", "food", "attraction", "hotel", "transport"]
    # splits_list = ["attraction"]
    llm = Deepseek("sk-fa3c6e12204d46f0b00616ab1c2d205e")
    for splits in splits_list:
        run(splits=splits, backbone_llm=llm)
