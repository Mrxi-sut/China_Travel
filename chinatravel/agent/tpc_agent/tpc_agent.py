import json
import sys
import os
sys.path.append("./../../../")
project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from chinatravel.agent.tpc_agent.utils import (
    TimeOutError,
)
from chinatravel.agent.utils import Logger
from chinatravel.symbol_verification.commonsense_constraint import (
    func_commonsense_constraints,
)
from chinatravel.symbol_verification.preference import evaluate_preference_py
from chinatravel.agent.tpc_agent.nl2sl_hybrid import nl2sl_reflect
from copy import deepcopy
from chinatravel.agent.tpc_agent.tpc_llm import TPCLLM
import argparse
import re
import time

import pandas as pd


import numpy as np

from geopy.distance import geodesic
from typing import List
from chinatravel.environment.tools.poi.apis import Poi
from chinatravel.environment.tools.transportation.apis import Transportation, find_nearest_station
project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)
from chinatravel.agent.tpc_agent.utils import (
    time_compare_if_earlier_equal,
    add_time_delta,
)
from chinatravel.data.load_datasets import load_json_file, save_json_file
from chinatravel.symbol_verification.hard_constraint import (
    get_symbolic_concepts,
    evaluate_constraints_py,
)


sys.path.append("./../../../")
project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)


from chinatravel.agent.base import BaseAgent

class TPCAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="TPC", **kwargs)
        self.max_steps = kwargs.get('max_steps', 0)
        self.debug = kwargs.get("debug", False)
        self.memory = {}
        self.TIME_CUT = 60 * 5

        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, "cache")

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        self.least_plan_schema, self.least_plan_comm = None, None
        self.method = "TPC"
        print("cache dir:", self.cache_dir)

        if not os.path.exists(
                os.path.join(self.cache_dir, self.method + "_" + self.backbone_llm.name)
        ):
            os.makedirs(
                os.path.join(self.cache_dir, self.method + "_" + self.backbone_llm.name)
            )
        self.search_width = kwargs.get("search_width", 100)
        self.preference_search = False
        self.prompt_upd = True
        self.poi = Poi()
        self.transport = Transportation()

    def ranking_intercity_transport_go(self, go_info, query, required_budget):
        transport_info = go_info.copy().reset_index(drop=True)
        transport_info["orig_index"] = transport_info.index
        filtered = transport_info[~transport_info['Cost'].isna()].copy()
        transport_type = None
        budget = None
        if 'hard_logic' in query:
            def parse_constraint(c: str):
                nonlocal filtered, budget
                c = c.strip()
                if c.startswith("intercity_transport_go==") or c.startswith("intercity_transport=="):
                    transport_str = c.split("==")[1].strip()
                    transport_type = eval(transport_str)
                    if isinstance(transport_type, set):
                        transport_type = list(transport_type)[0]
                    if transport_type == 'train':
                        filtered = filtered[~filtered['TrainID'].isna()]
                    elif transport_type == 'airplane':
                        filtered = filtered[~filtered['FlightID'].isna()]
                elif c.endswith("==intercity_transport_go"):
                    transport_str = c.split("==")[0].strip()
                    transport_type = eval(transport_str)
                    if isinstance(transport_type, set):
                        transport_type = list(transport_type)[0]
                    if transport_type == 'train':
                        filtered = filtered[~filtered['TrainID'].isna()]
                    elif transport_type == 'airplane':
                        filtered = filtered[~filtered['FlightID'].isna()]
                elif c.startswith("intercity_transport_cost<="):
                    budget_str = c.split("<=")[1].strip()
                    budget = float(budget_str)
            for constraint in query['hard_logic']:
                if not isinstance(constraint, str):
                    continue
                if ' or ' in constraint:
                    parts = [p.strip(' ()') for p in constraint.split(' or ')]
                    for part in parts:
                        if ' and ' in part:
                            p = [p.strip() for p in part.split(' and ')]
                            for pp in p:
                                pp = pp.strip()
                                if pp.startswith('('):
                                    pp = pp[1:].strip()
                                if pp.endswith(')'):
                                    pp = pp[:-1].strip()
                                parse_constraint(pp)
                        else:
                            part = part.strip()
                            if part.startswith('('):
                                part = part[1:].strip()
                            if part.endswith(')'):
                                part = part[:-1].strip()
                            parse_constraint(part)
                else:
                    parse_constraint(constraint)
        if budget is not None:
            people = query.get('people_number', 1)
            filtered = filtered[filtered['Cost'] * people * 2 <= budget]  # 往返费用
        def is_valid_time_interval(begin_time, end_time):
            """检查开始时间是否小于结束时间"""
            try:
                begin_minutes = time_to_minutes(begin_time)
                end_minutes = time_to_minutes(end_time)
                return begin_minutes < end_minutes
            except:
                return False  # 时间格式无效的也排除
        def time_to_minutes(time_str):
            try:
                hours, minutes = map(int, time_str.split(':'))
                return hours * 60 + minutes
            except:
                return float('inf')  # 无效时间排最后
        # 应用时间有效性过滤
        # valid_time_mask = [
        #     is_valid_time_interval(begin, end)
        #     for begin, end in zip(filtered["BeginTime"], filtered["EndTime"])
        # ]
        # filtered = filtered[valid_time_mask]
        if filtered.empty:
            filtered = transport_info.copy()
        # 排序逻辑
        def is_after_5am(t):
            try:
                return not time_compare_if_earlier_equal(t, "05:00")
            except:
                return False

        good_time_flags = transport_info["EndTime"].apply(is_after_5am).tolist()
        cost_list = transport_info["Cost"].tolist()
        end_time_list = transport_info["EndTime"].tolist()

        end_time_minutes = [time_to_minutes(t) for t in end_time_list]

        if required_budget is not None:
            sort_keys = list(zip(
                [not flag for flag in good_time_flags],
                cost_list,
                end_time_minutes
            ))
        else:
            sort_keys = list(zip(
                [not flag for flag in good_time_flags],
                end_time_minutes
            ))
        indexed_items = list(zip(sort_keys, transport_info["orig_index"]))
        sorted_items = sorted(indexed_items, key=lambda x: x[0])
        sorted_indices = [idx for (_, idx) in sorted_items]
        final_indices = [idx for idx in sorted_indices if idx in filtered["orig_index"].values]
        return np.array(final_indices)
    def ranking_intercity_transport_back(self, back_info, query, required_budget):
        transport_info = back_info.copy().reset_index(drop=True)
        transport_info["orig_index"] = transport_info.index
        filtered = transport_info[~transport_info['Cost'].isna()].copy()
        transport_type = None
        budget = None
        if 'hard_logic' in query:
            def parse_constraint(c: str):
                nonlocal filtered, budget
                c = c.strip()
                if c.startswith("intercity_transport_back==") or c.startswith("intercity_transport=="):
                    transport_str = c.split("==")[1].strip()
                    transport_type = eval(transport_str)
                    if isinstance(transport_type, set):
                        transport_type = list(transport_type)[0]
                    if transport_type == 'train':
                        filtered = filtered[~filtered['TrainID'].isna()]
                    elif transport_type == 'airplane':
                        filtered = filtered[~filtered['FlightID'].isna()]
                elif c.endswith("==intercity_transport_back"):
                    transport_str = c.split("==")[0].strip()
                    transport_type = eval(transport_str)
                    if isinstance(transport_type, set):
                        transport_type = list(transport_type)[0]
                    if transport_type == 'train':
                        filtered = filtered[~filtered['TrainID'].isna()]
                    elif transport_type == 'airplane':
                        filtered = filtered[~filtered['FlightID'].isna()]
                elif c.startswith("intercity_transport_cost<="):
                    budget_str = c.split("<=")[1].strip()
                    budget = float(budget_str)
            for constraint in query['hard_logic']:
                if not isinstance(constraint, str):
                    continue
                if ' or ' in constraint:
                    parts = [p.strip(' ()') for p in constraint.split(' or ')]
                    for part in parts:
                        if ' and ' in part:
                            p = [p.strip() for p in part.split(' and ')]
                            for pp in p:
                                pp = pp.strip()
                                if pp.startswith('('):
                                    pp = pp[1:].strip()
                                if pp.endswith(')'):
                                    pp = pp[:-1].strip()
                                parse_constraint(pp)
                        else:
                            part = part.strip()
                            if part.startswith('('):
                                part = part[1:].strip()
                            if part.endswith(')'):
                                part = part[:-1].strip()
                            parse_constraint(part)
                else:
                    parse_constraint(constraint)
        if budget is not None:
            people = query.get('people_number', 1)
            filtered = filtered[
                (filtered['Cost'] * 2) * people <= budget  # 往返费用
                ]
        if filtered.empty:
            filtered = transport_info.copy()
        # 排序逻辑 - 返程偏好晚班机
        def time_to_minutes(t):
            """将时间字符串转换为分钟数，用于排序"""
            try:
                hours, minutes = map(int, t.split(':'))
                return hours * 60 + minutes
            except:
                return 0
        # 提取排序要素
        time_minutes = transport_info["BeginTime"].apply(time_to_minutes).tolist()
        duration_list = transport_info["Duration"].tolist()
        cost_list = transport_info["Cost"].tolist()
        # 构建排序键 - 返程越晚越好
        if required_budget is not None:
            sort_keys = list(zip(
                [-t for t in time_minutes],  # 负值实现时间越晚排名越前
                cost_list  # 价格低的优先
            ))
        else:
            sort_keys = list(zip(
                [-t for t in time_minutes],  # 负值实现时间越晚排名越前
                duration_list  # 时长短的优先
            ))
        indexed_items = list(zip(sort_keys, transport_info["orig_index"]))
        sorted_items = sorted(indexed_items, key=lambda x: x[0])
        sorted_indices = [idx for (_, idx) in sorted_items]
        final_indices = [idx for idx in sorted_indices if idx in filtered["orig_index"].values]
        return np.array(final_indices)
    def select_and_add_breakfast(
            self, plan, poi_plan, current_day, current_time, current_position
    ):
        # have breakfast at hotel
        plan[current_day]["activities"] = self.add_poi(
            plan[current_day]["activities"],
            poi_plan["accommodation"]["name"],
            "breakfast",
            0,
            0,
            "07:00",
            "07:30",
            innercity_transports=[],
        )
        return plan
    def select_next_poi_type(
            self,
            candidates_type,
            plan,
            poi_plan,
            current_day,
            current_time,
            current_position,
            have_lunch_today,
            have_dinner_today,
    ):
        # 1. 检查是否是最后一天且需要返程
        if current_day == self.query["days"] - 1:
            if time_compare_if_earlier_equal(
                    poi_plan["back_transport"]["BeginTime"],
                    add_time_delta(current_time, 180),
            ) or (time_compare_if_earlier_equal(
                "00:00",
                add_time_delta(current_time, 180),
            ) and time_compare_if_earlier_equal(
                "21:00",
                current_time,
            )):
                return "back-intercity-transport", ["back-intercity-transport"], current_time
        # 2. 深夜时段强制返回酒店 (22:00-02:00)
        if (time_compare_if_earlier_equal("22:00", current_time) or
                (time_compare_if_earlier_equal("00:00", current_time) and
                 time_compare_if_earlier_equal(current_time, "02:00"))):
            if current_day == self.query["days"] - 1:
                return "back-intercity-transport", ["back-intercity-transport"], current_time
            if "hotel" in candidates_type:
                return "hotel", ["hotel"], current_time
            # 如果没有酒店选项，仍然返回酒店类型，调用方需要处理这种情况
            return "hotel", ["hotel"]
        if current_day == 0 and (time_compare_if_earlier_equal(current_time, "07:30")):
            return "attraction", candidates_type, "07:30"
        # 3. 初始化用餐状态和候选类型
        haved_lunch_today, haved_dinner_today = have_lunch_today, have_dinner_today
        candidates_type = ["attraction"]
        # 4. 添加用餐选项
        if (not haved_lunch_today) and time_compare_if_earlier_equal(current_time, "12:30"):
            candidates_type.append("lunch")
        if (not haved_dinner_today) and time_compare_if_earlier_equal(current_time, "19:00"):
            candidates_type.append("dinner")
        # 5. 如果不是最后一天，添加酒店选项
        if "accommodation" in poi_plan and current_day < self.query["days"] - 1:
            candidates_type.append("hotel")
        # 6. 检查是否太晚需要回酒店 (22:00之后，且剩余时间不足2小时)
        if (
                time_compare_if_earlier_equal("22:00", add_time_delta(current_time, 120))
                and "hotel" in candidates_type
        ):
            return "hotel", ["hotel"], current_time
        # 7. 午餐时间判断
        if ("lunch" in candidates_type) and (
                time_compare_if_earlier_equal("11:00", add_time_delta(current_time, 40))
                or time_compare_if_earlier_equal("12:40", add_time_delta(current_time, 120))
        ):
            return "lunch", ["lunch"], current_time
        # 8. 晚餐时间判断
        if ("dinner" in candidates_type) and (
                time_compare_if_earlier_equal("17:00", add_time_delta(current_time, 40))
                or time_compare_if_earlier_equal("19:00", add_time_delta(current_time, 120))
        ):
            return "dinner", ["dinner"], current_time
        # 9. 默认返回景点
        return "attraction", candidates_type, current_time

    def ranking_hotel(self, hotel_info, query) -> np.ndarray:
        if hotel_info is None or hotel_info.empty:
            return np.array([], dtype=int)

        # 初始化约束集合
        must_include_features = set()  # 用 <= 语法
        must_exclude_features = set()  # 用 <= 语法（反向）
        required_hotels = set()
        hotel_budget, hotel_price = None, None
        room_type, rooms = None, None
        distance_constraint = None
        target_poi = None
        total_cost_limit = self.total_cost_limit
        # 解析 hard_logic 约束
        if 'hard_logic' in query:
            def parse_constraint(c: str):
                nonlocal must_include_features, must_exclude_features
                nonlocal required_hotels, hotel_budget, hotel_price, room_type, rooms
                nonlocal distance_constraint, target_poi
                try:
                    if '{' in c and '}<=hotel_feature' in c:
                        features = eval(c.split('<=hotel_feature')[0].strip())
                        must_include_features.update(features)
                    elif 'hotel_feature<={' in c and '}' in c:
                        features = eval(c.split('hotel_feature<=')[1].strip())
                        must_exclude_features.update(features)
                    elif 'hotel_names' in c and '<=' in c:
                        hotels_str = c.split('<=hotel_names')[0].strip()
                        required_hotels.update(eval(hotels_str))
                    elif c.strip().startswith("hotel_cost<="):
                        hotel_budget = float(c.split("hotel_cost<=")[1].strip())
                    elif c.strip().startswith("hotel_price<="):
                        hotel_price = float(c.split("hotel_price<=")[1].strip())
                    elif c.strip().startswith("room_type=="):
                        room_type = int(c.split("room_type==")[1].strip())
                    elif c.strip().startswith("rooms=="):
                        rooms = int(c.split("rooms==")[1].strip())
                    elif 'accommodation_distance{' in constraint and '<=' in constraint:
                        match = re.search(r'accommodation_distance\{([^}]+)\}\s*<=\s*([\d.]+)', constraint)
                        if match:
                            target_poi = match.group(1).strip()
                            distance_str = match.group(2).strip()
                            target_poi = target_poi.strip("'\"")
                            distance_constraint = float(distance_str)
                            print(f"解析住宿距离约束: {target_poi} 距离不超过 {distance_constraint}")
                except:
                    pass

            for constraint in query['hard_logic']:
                if not isinstance(constraint, str):
                    continue
                if ' or ' in constraint:
                    parts = [p.strip(' ()') for p in constraint.split(' or ')]
                    for part in parts:
                        parse_constraint(part)
                else:
                    constraint = constraint.strip()
                    if constraint.startswith('('):
                        constraint = constraint[1:].strip()
                    if constraint.endswith(')'):
                        constraint = constraint[:-1].strip()
                    parse_constraint(constraint)

        filtered = hotel_info.copy()

        if required_hotels:
            def match_hotel(name, required_list):
                name_str = str(name).lower()
                for hotel in required_list:
                    if hotel.lower() in name_str:
                        return True
                return False

            mask = filtered['name'].apply(lambda x: match_hotel(x, required_hotels))
            filtered = filtered[mask]

        # 应用特征筛选（必须包含）
        if must_include_features:
            filtered = filtered[filtered['featurehoteltype'].isin(must_include_features)]
        # 应用特征筛选（必须排除）
        if must_exclude_features:
            filtered = filtered[~filtered['featurehoteltype'].isin(must_exclude_features)]

        if distance_constraint is not None and target_poi is not None:
            # 使用您提供的坐标计算函数
            city = query['target_city']
            # 获取目标POI的坐标
            poi_coord = self.poi.search(city, target_poi)
            if isinstance(poi_coord, str):
                print(f"Warning: Cannot find coordinates for {target_poi} in {city}")
            else:
                # 计算每个酒店到POI的距离
                valid_hotels = []
                for idx, row in filtered.iterrows():
                    hotel_name = row['name']
                    # 获取酒店坐标（假设酒店信息中包含坐标或可以通过名称查询）
                    hotel_coord = self.poi.search(city, hotel_name)

                    if isinstance(hotel_coord, str):
                        # 如果找不到酒店坐标，跳过该酒店
                        continue

                    # 计算距离
                    try:
                        from geopy.distance import geodesic
                        distance = geodesic(poi_coord, hotel_coord).kilometers

                        if distance <= distance_constraint:
                            valid_hotels.append(idx)
                    except Exception as e:
                        print(f"Error calculating distance for {hotel_name}: {e}")
                        continue
                # 只保留距离在约束范围内的酒店
                filtered = filtered.loc[valid_hotels]

        if hotel_budget is not None:
            days = query['days'] - 1
            people = query['people_number']
            # 计算总成本: 天数 * 单价 * ceil(人数 / 每间床可容纳人数)
            cost = days * filtered['price'] * np.ceil(people / filtered['numbed'])
            filtered = filtered[cost <= hotel_budget]

        if hotel_price is not None:
            filtered = filtered[filtered['price'] <= hotel_price]

        if rooms is not None and room_type is None:
            if query['people_number'] == rooms:
                room_type = 1
            else:
                room_type = 2

        if room_type is not None:
            filtered = filtered[filtered['numbed'] == room_type]

        self.suggested_hotels_from_query = filtered['name'].tolist()
        self.ranking_hotel_flag = True

        # 获取完整酒店信息
        name_to_index = {hotel_info.iloc[i]["name"]: i for i in range(len(hotel_info))}
        suggested_hotels = hotel_info[hotel_info["name"].isin(self.suggested_hotels_from_query)]
        suggested_sorted = suggested_hotels.sort_values(by="price")
        suggested_indices = [name_to_index[name] for name in suggested_sorted["name"]]
        remaining_hotels = hotel_info[~hotel_info["name"].isin(self.suggested_hotels_from_query)]
        remaining_sorted = remaining_hotels.sort_values(by="price")
        remaining_indices = [name_to_index[name] for name in remaining_sorted["name"]]

        # 合并最终排序
        final_ranking = suggested_indices + remaining_indices

        return np.array(final_ranking, dtype=int)

    def ranking_attractions(
            self,
            plan,
            poi_plan,
            current_day,
            current_time,
            current_position,
            intercity_with_hotel_cost,
    ) -> np.ndarray:
        # 获取景点信息
        attr_info = self.memory["attractions"][["name", "type", "opentime", "endtime", "price", "recommendmintime"]]

        if not hasattr(self, "_hard_logic_cache"):
            must_include_names, exclude_names = set(), set()
            must_include_types, exclude_types = set(), set()
            price_limit, cost_limit = float('inf'), float('inf')

            hard_logic = self.query.get('hard_logic', [])

            def parse_single_constraint(constraint: str):
                nonlocal must_include_names, exclude_names
                nonlocal must_include_types, exclude_types
                nonlocal price_limit, cost_limit
                try:
                    if '<=attraction_names' in constraint:
                        names = eval(constraint.split('<=')[0].strip())
                        must_include_names.update(names)
                    elif 'attraction_names<=' in constraint:
                        names = eval(constraint.split('<=')[1].strip())
                        exclude_names.update(names)
                    elif '<=spot_type' in constraint:
                        types = eval(constraint.split('<=')[0].strip())
                        must_include_types.update(types)
                    elif 'spot_type<=' in constraint:
                        types = eval(constraint.split('<=')[1].strip())
                        exclude_types.update(types)
                    elif 'attraction_price<=' in constraint:
                        price_limit = min(price_limit, float(constraint.split('<=')[1].strip()))
                    elif 'attraction_cost<=' in constraint:
                        cost_limit = min(cost_limit, float(constraint.split('<=')[1].strip()))
                except Exception as e:
                    pass  # 可以加 logging

            for constraint in hard_logic:
                if ' or ' in constraint:
                    # 先拆分 OR 条件，再分别解析
                    parts = [p.strip(' ()') for p in constraint.split(' or ')]
                    for part in parts:
                        parse_single_constraint(part)
                else:
                    constraint = constraint.strip()
                    if constraint.startswith('('):
                        constraint = constraint[1:].strip()
                    if constraint.endswith(')'):
                        constraint = constraint[:-1].strip()
                    parse_single_constraint(constraint)

            # 缓存解析结果
            self._hard_logic_cache = {
                "must_include_names": must_include_names,
                "exclude_names": exclude_names,
                "must_include_types": must_include_types,
                "exclude_types": exclude_types,
                "price_limit": price_limit,
                "cost_limit": cost_limit,
            }

        must_include_names = self._hard_logic_cache["must_include_names"]
        exclude_names = self._hard_logic_cache["exclude_names"]
        must_include_types = self._hard_logic_cache["must_include_types"]
        exclude_types = self._hard_logic_cache["exclude_types"]
        price_limit = self._hard_logic_cache["price_limit"]
        cost_limit = self._hard_logic_cache["cost_limit"]
        all_cost_limit = self.total_cost_limit
        start_time_constraints = self.start_time_constraints
        end_time_constraints = self.end_time_constraints
        time_range_constraints = self.time_range_constraints
        activity_time_constraints = self.activity_time_constraints
        activity_order_constraints = self.activity_order_constraints
        people = self.query.get('people_number', 1)
        # 统计当前 plan 中已安排景点的成本
        current_cost = 0.0
        for dayplan in plan:
            for act in dayplan.get("activities", []):
                if act.get("type") == "attraction":
                    current_cost += act.get("cost", 0)
        num_attractions = len(attr_info)
        attr_dist = []
        valid_indices = []
        for i in range(num_attractions):
            row = attr_info.iloc[i]
            name, type_, opentime, endtime, price, recommendmintime = row["name"], row["type"], row["opentime"], row[
                "endtime"], row[
                "price"], row["recommendmintime"]
            if not (time_compare_if_earlier_equal(opentime, current_time) and time_compare_if_earlier_equal(
                    current_time, endtime)):
                continue
            # 名称过滤
            if name in exclude_names:
                continue
            # 类型过滤
            if type_ in exclude_types:
                continue
            # 价格过滤
            if price > price_limit:
                continue
            # 总成本过滤
            total_cost = current_cost + price * people
            all_total_cost = total_cost + self.cur_all_cost
            if total_cost > cost_limit:
                continue
            if all_cost_limit is not None and all_total_cost > all_cost_limit:
                continue
            # 收集交通距离
            transports_sel, cost = self.collect_innercity_transport(
                self.query["target_city"],
                current_position,
                name,
                current_time,
                "taxi",
                True,
            )
            if isinstance(transports_sel, str) or not transports_sel:
                continue
            if not all(isinstance(t, dict) and "distance" in t for t in transports_sel):
                continue
            valid_indices.append(i)
            attr_dist.append(transports_sel[0]["distance"])
        combined_scores = []
        for i in range(len(attr_info)):
            row = attr_info.iloc[i]
            name = row["name"]
            # 默认无穷大的分数，便于排在后面
            score = float('inf')

            # 只有 valid 景点才计算真实的 combined_score
            if i in valid_indices:
                price = row["price"]
                price_norm = (price - attr_info.iloc[valid_indices]["price"].min()) / (
                        attr_info.iloc[valid_indices]["price"].max() - attr_info.iloc[valid_indices][
                    "price"].min() + 1e-8
                )
                dist = attr_dist[valid_indices.index(i)]
                dist_norm = (dist - np.min(attr_dist)) / (np.max(attr_dist) - np.min(attr_dist) + 1e-8)
                # score = 0.5 * price_norm + 0.5 * dist_norm
                score = dist
            combined_scores.append(score)

        # 添加 combined_score 到 attr_info 中（注意不要修改原数据）
        attr_info = attr_info.copy()
        attr_info["combined_score"] = combined_scores

        if not valid_indices:
            # 对所有景点按 combined_score 排序
            all_sorted = attr_info.sort_values("combined_score")
            return np.array(all_sorted.index.tolist(), dtype=int)

        time_constrained_indices = []
        for i in valid_indices:
            name = attr_info.iloc[i]["name"]
            if ((name in start_time_constraints or name in end_time_constraints or name in time_range_constraints
                 or name in activity_time_constraints or (
                         activity_order_constraints and name == activity_order_constraints[0]))
                    and name not in self.attraction_names_visiting):
                time_constrained_indices.append(i)
        prioritized_indices = time_constrained_indices

        # 划分 valid / not valid
        valid_set = set(valid_indices)

        # 修改点1：获取价格和距离的排名
        valid_prices = attr_info.iloc[list(valid_set)]["price"].rank()  # price 排名
        valid_distances = pd.Series(
            [attr_dist[valid_indices.index(i)] for i in valid_set],
            index=valid_set  # 显式指定索引
        ).rank()  # distance 排名

        # 修改点2：准备排序要素（模仿示例代码的模式）
        sort_keys = []
        for i in range(len(attr_info)):
            if i in valid_set:
                try:
                    price_rank = valid_prices.loc[i]
                    dist_rank = valid_distances.loc[i]

                    if all_cost_limit is not None or cost_limit is not None:
                        # 当有总成本限制时：先按价格排名，再按距离排名
                        sort_keys.append((
                            price_rank,  # 主要按价格排序（价格越低越好）
                            dist_rank  # 价格相同时按距离排序（距离越短越好）
                        ))
                    else:
                        # 没有总成本限制时：只按距离排序
                        sort_keys.append((
                            dist_rank,  # 主要按距离排序
                            0  # 辅助键，保持结构一致
                        ))
                except KeyError:
                    # 无效数据，给很大的值确保排在后面
                    sort_keys.append((float('inf'), float('inf')))
            else:
                # 无效景点，给很大的值确保排在后面
                sort_keys.append((float('inf'), float('inf')))

        # 修改点3：为了保持后续代码兼容性，更新 combined_score（使用距离排名）
        combined_scores = []
        for i in range(len(attr_info)):
            if i in valid_set:
                try:
                    dist_rank = valid_distances.loc[i]
                    combined_scores.append(dist_rank)
                except KeyError:
                    combined_scores.append(float('inf'))
            else:
                combined_scores.append(float('inf'))

        attr_info["combined_score"] = combined_scores

        valid_rows = attr_info.iloc[list(valid_set)]
        not_valid_rows = attr_info.drop(index=list(valid_set))

        # 排序
        valid_sorted = valid_rows.sort_values("combined_score")
        not_valid_sorted = not_valid_rows.sort_values("combined_score")

        # 构建最终排序，优先 must_include_names 和 must_include_types 中的项
        added_names = set()
        # 有时间约束的
        # must_include
        for t in self.spot_type_visiting:
            if t in must_include_types:
                must_include_types.remove(t)
        valid_candidates = []
        for i in prioritized_indices:
            row = attr_info.iloc[i]
            if row["name"] not in added_names:
                valid_candidates.append(row["name"])
                added_names.add(row["name"])

        # must_include 名称匹配
        for _, row in valid_sorted.iterrows():
            row_name = row["name"]
            is_must_include = False
            for must_name in must_include_names:
                if must_name in row_name:
                    is_must_include = True
                    break
            if is_must_include and row_name not in self.attraction_names_visiting and row_name not in added_names:
                valid_candidates.append(row_name)
                added_names.add(row_name)

        # must_include 类型匹配
        for _, row in valid_sorted.iterrows():
            if row["type"] in must_include_types and row["type"] not in self.spot_type_visiting and row[
                "name"] not in added_names:
                valid_candidates.append(row["name"])
                added_names.add(row["name"])

        not_valid_candidates = []
        for _, row in valid_sorted.iterrows():
            if row["name"] not in added_names:
                not_valid_candidates.append(row["name"])
        for _, row in not_valid_sorted.iterrows():
            if row["name"] not in added_names:
                not_valid_candidates.append(row["name"])

        # 转为索引形式
        name_to_index = {attr_info.iloc[i]["name"]: i for i in range(len(attr_info))}

        if all_cost_limit is not None:
            # 对 valid 景点进行多条件排序
            valid_indices = [name_to_index[name] for name in valid_candidates]
            valid_indices_sorted = sorted(
                valid_indices,
                key=lambda x: sort_keys[x]  # 使用预先计算的排序键
            )

            # 对 not valid 景点进行多条件排序
            not_valid_indices = [name_to_index[name] for name in not_valid_candidates]
            not_valid_indices_sorted = sorted(
                not_valid_indices,
                key=lambda x: sort_keys[x]  # 使用预先计算的排序键
            )

            # 合并：valid 在前，not valid 在后
            final_index_ranking = valid_indices_sorted + not_valid_indices_sorted
        else:
            # 原来的逻辑
            final_ranking = valid_candidates + not_valid_candidates
            final_index_ranking = [name_to_index[name] for name in final_ranking]

        return np.array(final_index_ranking, dtype=int)

    def ranking_restaurants(
            self,
            plan,
            poi_plan,
            current_day,
            current_time,
            current_position,
            intercity_with_hotel_cost,
            trans_type='taxi'
    ) -> np.ndarray:
        # 获取景点信息
        res_info = self.memory["restaurants"][["name", "cuisine", "price", "opentime", "endtime", "recommendedfood"]]
        if not hasattr(self, "_hard_logic_cache_restaurants"):
            must_include_names, exclude_names = set(), set()
            must_include_types, exclude_types = set(), set()
            price_limit, cost_limit = float('inf'), float('inf')

            hard_logic = self.query.get('hard_logic', [])

            def parse_single_constraint(constraint: str):
                nonlocal must_include_names, exclude_names
                nonlocal must_include_types, exclude_types
                nonlocal price_limit, cost_limit
                try:
                    if '<=restaurant_names' in constraint:
                        names = eval(constraint.split('<=')[0].strip())
                        must_include_names.update(names)
                    elif 'restaurant_names<=' in constraint:
                        names = eval(constraint.split('<=')[1].strip())
                        exclude_names.update(names)
                    elif '<=food_type' in constraint:
                        types = eval(constraint.split('<=')[0].strip())
                        must_include_types.update(types)
                    elif 'food_type<=' in constraint:
                        types = eval(constraint.split('<=')[1].strip())
                        exclude_types.update(types)
                    elif 'food_price<=' in constraint:
                        price_limit = min(price_limit, float(constraint.split('<=')[1].strip()))
                    elif 'food_cost<=' in constraint:
                        cost_limit = min(cost_limit, float(constraint.split('<=')[1].strip()))
                except Exception:
                    pass  # 可以加 logging 方便调试

            for constraint in hard_logic:
                if ' or ' in constraint:
                    # 先拆分 OR 条件，再分别解析
                    parts = [p.strip(' ()') for p in constraint.split(' or ')]
                    for part in parts:
                        parse_single_constraint(part)
                else:
                    constraint = constraint.strip()
                    if constraint.startswith('('):
                        constraint = constraint[1:].strip()
                    if constraint.endswith(')'):
                        constraint = constraint[:-1].strip()
                    parse_single_constraint(constraint)

            self._hard_logic_cache_restaurants = {
                "must_include_names": must_include_names,
                "exclude_names": exclude_names,
                "must_include_types": must_include_types,
                "exclude_types": exclude_types,
                "price_limit": price_limit,
                "cost_limit": cost_limit,
            }

        must_include_names = self._hard_logic_cache_restaurants["must_include_names"]
        exclude_names = self._hard_logic_cache_restaurants["exclude_names"]
        must_include_types = self._hard_logic_cache_restaurants["must_include_types"]
        exclude_types = self._hard_logic_cache_restaurants["exclude_types"]
        price_limit = self._hard_logic_cache_restaurants["price_limit"]
        cost_limit = self._hard_logic_cache_restaurants["cost_limit"]
        all_cost_limit = self.total_cost_limit
        start_time_constraints = self.start_time_constraints
        end_time_constraints = self.end_time_constraints
        time_range_constraints = self.time_range_constraints
        activity_time_constraints = self.activity_time_constraints
        activity_order_constraints = self.activity_order_constraints
        people = self.query.get('people_number', 1)
        current_cost = 0.0
        for dayplan in plan:
            for act in dayplan.get("activities", []):
                if act.get("type") in ["lunch", "dinner"]:
                    current_cost += act.get("cost", 0)
        num_res = len(res_info)
        res_dist = []
        valid_indices = []
        for i in range(num_res):
            row = res_info.iloc[i]
            name, type_, opentime, endtime, price = row["name"], row["cuisine"], row["opentime"], row["endtime"], row[
                "price"]
            # 时间限制
            if time_compare_if_earlier_equal(endtime, current_time):
                continue
            # 名称过滤
            if name in exclude_names:
                continue
            # 类型过滤
            if type_ in exclude_types:
                continue
            # 价格过滤
            if price > price_limit:
                continue
            # 总成本过滤
            total_cost = current_cost + price * people
            all_total_cost = total_cost + self.cur_all_cost
            if total_cost > cost_limit:
                continue
            if all_cost_limit is not None and all_total_cost > all_cost_limit:
                continue
            # 收集交通距离
            transports_sel, cost = self.collect_innercity_transport(
                self.query["target_city"],
                current_position,
                name,
                current_time,
                trans_type,
                True,
            )
            if isinstance(transports_sel, str) or not transports_sel:
                continue
            if not all(isinstance(t, dict) and "distance" in t for t in transports_sel):
                continue
            valid_indices.append(i)
            res_dist.append(transports_sel[0]["distance"])

        # 计算所有餐厅的 combined_score
        combined_scores = []
        for i in range(len(res_info)):
            row = res_info.iloc[i]
            score = float('inf')  # 默认值（无效数据）

            if i in valid_indices:
                # 对于有效餐厅，计算真实的 combined_score
                try:
                    # 使用距离作为评分标准（与景点函数保持一致）
                    dist = res_dist[valid_indices.index(i)]
                    score = dist
                except (IndexError, KeyError):
                    score = float('inf')

            combined_scores.append(score)

        # 添加 combined_score 列（不修改原始数据）
        res_info = res_info.copy()
        res_info["combined_score"] = combined_scores

        # 如果 valid_indices 为空，直接返回按 combined_score 排序的所有餐厅
        if not valid_indices:
            # 对所有餐厅按 combined_score 排序
            all_sorted = res_info.sort_values("combined_score")
            return np.array(all_sorted.index.tolist(), dtype=int)

        # 以下是原有的处理逻辑（当 valid_indices 不为空时）
        time_constrained_indices = []
        for i in valid_indices:
            name = res_info.iloc[i]["name"]
            if (name in start_time_constraints or name in end_time_constraints or name in time_range_constraints
                    or name in activity_time_constraints or (
                            activity_order_constraints and name == activity_order_constraints[0])):
                time_constrained_indices.append(i)

        prioritized_indices = time_constrained_indices

        valid_set = set(valid_indices)
        valid_prices = res_info.iloc[list(valid_set)]["price"].rank()  # price 排名
        valid_distances = pd.Series(
            [res_dist[valid_indices.index(i)] for i in valid_set],
            index=valid_set  # 显式指定索引
        ).rank()  # distance 排名

        # 重新计算 combined_scores（使用排名而不是原始距离）
        combined_scores = []
        for i in range(len(res_info)):
            row = res_info.iloc[i]
            score = float('inf')  # 默认值（无效数据）

            if i in valid_set:
                # 安全获取排名值
                try:
                    price_rank = valid_prices.loc[i]
                    dist_rank = valid_distances.loc[i]
                    # score = (price_rank + dist_rank) / 2
                    score = dist_rank
                except KeyError:  # 如果索引不存在
                    score = float('inf')

            combined_scores.append(score)

        # 更新 combined_score 列
        res_info["combined_score"] = combined_scores

        # 划分 valid / not valid
        valid_rows = res_info.iloc[list(valid_set)]
        not_valid_rows = res_info.drop(index=list(valid_set))

        # 检查是否有成本限制
        has_cost_limit = (all_cost_limit is not None) or (cost_limit < float('inf'))

        # 排序逻辑：如果有成本限制，按价格排序；否则按 combined_score 排序
        if has_cost_limit:
            valid_sorted = valid_rows.sort_values("price")
            not_valid_sorted = not_valid_rows.sort_values("price")
        else:
            valid_sorted = valid_rows.sort_values("combined_score")
            not_valid_sorted = not_valid_rows.sort_values("combined_score")

        # 构建最终排序
        final_ranking = []
        added_names = set()
        for i in prioritized_indices:
            row = res_info.iloc[i]
            if row["name"] not in added_names:
                final_ranking.append(row["name"])
                added_names.add(row["name"])
        for _, row in valid_sorted.iterrows():
            if row["name"] in must_include_names and row["name"] not in self.restaurant_names_visiting:
                final_ranking.append(row["name"])
                added_names.add(row["name"])
        for _, row in valid_sorted.iterrows():
            if row["cuisine"] in must_include_types and row["cuisine"] not in self.food_type_visiting:
                final_ranking.append(row["name"])
                added_names.add(row["name"])
        for _, row in valid_sorted.iterrows():
            if row["name"] not in added_names:
                final_ranking.append(row["name"])
                added_names.add(row["name"])

        for _, row in not_valid_sorted.iterrows():
            if row["name"] not in added_names:
                final_ranking.append(row["name"])

        # 转换为索引形式返回
        name_to_index = {res_info.iloc[i]["name"]: i for i in range(len(res_info))}
        final_index_ranking = [name_to_index[name] for name in final_ranking]

        return np.array(final_index_ranking, dtype=int)

    def calculate_taxi_cost(self, distance: float) -> float:
        """计算出租车费用"""
        if distance <= 1.8:
            return 11.0
        elif distance <= 10:
            return 11.0 + (distance - 1.8) * 3.5
        else:
            return 11.0 + (10 - 1.8) * 3.5 + (distance - 10) * 4.5

    def calculate_metro_cost(self, distance: float) -> int:
        """计算地铁费用"""
        if distance <= 4:
            return 2
        elif distance <= 9:
            return 3
        elif distance <= 14:
            return 4
        elif distance <= 21:
            return 5
        elif distance <= 28:
            return 6
        elif distance <= 37:
            return 7
        elif distance <= 48:
            return 8
        elif distance <= 61:
            return 9
        else:
            extra_distance = distance - 61
            extra_cost = (extra_distance + 14) // 15
            return 9 + extra_cost
    def ranking_innercity_transport(
            self, current_position: str, target_position: str, current_time: str
    ) -> List[str]:
        if not hasattr(self, "_cached_distance"):
            distance = None
            transport_types = None
            def parse_constraint(c: str):
                nonlocal distance
                try:
                    if c.strip().startswith("distance>"):
                        distance = float(c.split("distance>")[1].strip())
                    elif "transport_type<=" in c:
                        transport_str = c.split("<=")[1].strip()
                        if transport_str.startswith("{") and transport_str.endswith("}"):
                            transport_types_str = transport_str[1:-1].replace("'", "").replace('"', "").split(", ")
                            transport_types = set(transport_types_str)
                except:
                    pass

            if "hard_logic" in self.query:
                for constraint in self.query["hard_logic"]:
                    if not isinstance(constraint, str):
                        continue
                    if " or " in constraint:
                        parts = [p.strip(" ()") for p in constraint.split(" or ")]
                        for part in parts:
                            parse_constraint(part)
                    else:
                        parse_constraint(constraint)
            self._cached_distance = distance
            self._cached_transport_types = transport_types
        try:
            # 1. 获取坐标位置
            city = self.query["target_city"]
            start_coord = self.poi.search(city, current_position)
            end_coord = self.poi.search(city, target_position)

            if isinstance(start_coord, str) or isinstance(end_coord, str):
                ranking = ["metro", "taxi", "walk"]  # 默认排序
            else:
                # 2. 计算距离(公里)
                distance = geodesic(start_coord, end_coord).kilometers

                # 3. 解析当前时间
                self.city_list = ["shanghai", "beijing", "shenzhen", "guangzhou", "chongqing",
                                  "suzhou", "chengdu", "hangzhou", "wuhan", "nanjing"]
                self.city_cn_map = {en: cn for en, cn in zip(self.city_list,
                                                             ["上海", "北京", "深圳", "广州", "重庆", "苏州", "成都",
                                                              "杭州",
                                                              "武汉", "南京"])}
                self.city_en_map = {cn: en for en, cn in self.city_cn_map.items()}
                # 3. 解析当前时间
                hour = int(current_time.split(":")[0])
                city_en = self.city_en_map.get(city, city)
                start_station, walk1_dist = find_nearest_station(start_coord,
                                                                 self.transport.city_stations_dict[city_en])
                end_station, walk2_dist = find_nearest_station(end_coord, self.transport.city_stations_dict[city_en])
                start_station_name = start_station['name']
                end_station_name = end_station['name']

                # 4. 基础排序规则
                if self._cached_distance is None:
                    if distance < 1.5:  # 短距离步行优先
                        ranking = ["walk", "metro", "taxi"]
                    elif 1 <= distance <= 5:  # 中等距离
                        if start_station_name == end_station_name:
                            ranking = ["taxi", "walk", "metro"]
                        else:
                            # 新增条件：当步行总距离超过2公里时，taxi优先
                            if walk1_dist + walk2_dist > 2:
                                ranking = ["taxi", "metro", "walk"]
                            else:
                                ranking = ["metro", "taxi", "walk"]
                    else:  # 长距离
                        if self.innercity_cost_limit is None:
                            ranking = ["taxi", "metro", "walk"]
                        else:
                            ranking = ["metro", "taxi", "walk"]

                    # 5. 时间因素调整
                    # 晚高峰地铁优先
                    if 17 <= hour < 19 and start_station_name != end_station_name:
                        if "metro" in ranking:
                            ranking.remove("metro")
                            ranking.insert(0, "metro")
                    # 深夜出租车优先
                    elif 22 <= hour or hour < 6:
                        if "taxi" in ranking:
                            ranking.remove("taxi")
                            ranking.insert(0, "taxi")
                    # 6. 价格因素调整(同等条件下)
                    # 如果地铁和出租车都在选项中，且距离<10km时比较价格
                    if "metro" in ranking and "taxi" in ranking and (
                            10 > distance > 2) and start_station_name != end_station_name:
                        metro_cost = self.calculate_metro_cost(distance)
                        taxi_cost = self.calculate_taxi_cost(distance)
                        if taxi_cost < metro_cost:
                            ranking.remove("taxi")
                            ranking.insert(0, "taxi")
                else:
                    if distance > self._cached_distance:
                        ranking = ["taxi"]
                    else:
                        if start_station_name == end_station_name:
                            ranking = ["walk", "metro", "taxi"]
                        else:
                            ranking = ["metro", "taxi", "walk"]

            # 7. 应用交通方式约束
            if hasattr(self, "_cached_transport_types") and self._cached_transport_types is not None:
                # 过滤掉不在允许列表中的交通方式
                filtered_ranking = [transport for transport in ranking if transport in self._cached_transport_types]

                # 如果过滤后为空，使用默认排序并再次过滤
                if not filtered_ranking:
                    default_ranking = ["metro", "taxi", "walk"]
                    filtered_ranking = [transport for transport in default_ranking if
                                        transport in self._cached_transport_types]

                # 如果仍然为空，返回第一个允许的交通方式（如果有的话）
                if not filtered_ranking and self._cached_transport_types:
                    filtered_ranking = [list(self._cached_transport_types)[0]]

                ranking = filtered_ranking

                # 打印调试信息
                if ranking != filtered_ranking:
                    print(f"应用交通方式约束: 原始排序 {ranking} -> 过滤后 {filtered_ranking}")

            return ranking

        except Exception as e:
            print(f"交通排序规则计算错误: {str(e)}")
            # 在异常情况下也应用约束
            default_ranking = ["metro", "taxi", "walk"]
            if hasattr(self, "_cached_transport_types") and self._cached_transport_types is not None:
                default_ranking = [transport for transport in default_ranking if
                                   transport in self._cached_transport_types]
            return default_ranking
    def time_difference_minutes(self, start_time, end_time):
        """
        计算两个时间之间的分钟差
        Args:
            start_time: 开始时间，格式 "HH:MM"
            end_time: 结束时间，格式 "HH:MM"
        Returns:
            int: 分钟差（如果end_time在start_time之前，返回负数）
        """
        # 将时间字符串转换为分钟数
        def time_to_minutes(time_str):
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes

        start_minutes = time_to_minutes(start_time)
        end_minutes = time_to_minutes(end_time)

        # 处理跨天的情况（如果结束时间在开始时间之前，假设是第二天）
        if end_minutes < start_minutes:
            end_minutes += 24 * 60  # 加上一天的分钟数

        return end_minutes - start_minutes
    def select_poi_time(self, plan, poi_plan, current_day, start_time, poi_name, poi_type, recommended_visit_time,
                        recommendmax_time, have_lunch, have_dinner):
        activity_time_constraints = self.activity_time_constraints
        if poi_name in activity_time_constraints:
            return activity_time_constraints[poi_name]
        else:
            ret_time = recommended_visit_time
            if int(recommended_visit_time) == 0:
                ret_time = int(recommendmax_time / 2)
            end_time = add_time_delta(start_time, int(ret_time))
            if time_compare_if_earlier_equal("11:00", end_time) and time_compare_if_earlier_equal(start_time, "10:30"):
                ret_time = 30
            if time_compare_if_earlier_equal("17:00", end_time) and time_compare_if_earlier_equal(start_time, "16:30"):
                ret_time = 30
            if time_compare_if_earlier_equal("21:00", end_time) and time_compare_if_earlier_equal(start_time, "20:30"):
                ret_time = 30
            if not have_lunch and time_compare_if_earlier_equal("11:00", start_time):
                ret_time = 30
            if not have_dinner and time_compare_if_earlier_equal("17:00", start_time):
                ret_time = 30
            if current_day == self.query["days"] - 1 and "back_transport" in poi_plan:
                # 使用推荐最小时间或更保守的时间
                ret_time = min(ret_time, 60)
                # 检查时间是否足够
                estimated_end_time = add_time_delta(start_time, int(ret_time))
                transport_buffer = 90  # 交通+缓冲时间
                # 如果预估结束时间太晚，进一步缩短游玩时间
                latest_allowed_time = add_time_delta(poi_plan["back_transport"]["BeginTime"], -int(transport_buffer))
                if time_compare_if_earlier_equal(latest_allowed_time, estimated_end_time):
                    # 计算最大允许游玩时间
                    if time_compare_if_earlier_equal(latest_allowed_time, start_time):
                        return 0
                    max_allowed_time = self.time_difference_minutes(start_time, latest_allowed_time)
                    if max_allowed_time > 30:  # 至少保留30分钟游玩时间
                        ret_time = max_allowed_time - 10  # 留出10分钟缓冲
                    else:
                        ret_time = 0  # 时间不足，跳过该景点
            return ret_time
    def decide_rooms(self, query):
        rooms, room_type = None, None
        if 'hard_logic' in query:
            def parse_room_constraint(c: str):
                nonlocal room_type, rooms
                try:
                    if c.strip().startswith("room_type=="):
                        room_type = int(c.split("room_type==")[1].strip())
                    elif c.strip().startswith("rooms=="):
                        rooms = int(c.split("rooms==")[1].strip())
                except:
                    pass

            for constraint in query['hard_logic']:
                if not isinstance(constraint, str):
                    continue
                if " or " in constraint:
                    parts = [p.strip(" ()") for p in constraint.split(" or ")]
                    for part in parts:
                        parse_room_constraint(part)
                else:
                    parse_room_constraint(constraint)
        return rooms, room_type

    def get_final_transport_ranking(self, current_position, poi_plan_name, current_time):
        # 获取两种排序结果
        transports_ranking = self.ranking_innercity_transport(current_position, poi_plan_name,
                                                              current_time)
        transports_ranking.append("walk")
        transports_ranking.append("taxi")
        transports_ranking.append("metro")
        return transports_ranking
    def reset(self):
        pass

    def translate_nl2sl(self, query, load_cache=False):

        llm_method = "translation_tpc_llm2_reflect"
        if not os.path.exists(os.path.join(self.cache_dir, llm_method)):
            os.makedirs(os.path.join(self.cache_dir, llm_method))

        file_path = os.path.join(
            self.cache_dir, llm_method, "{}.json".format(query["uid"])
        )

        print(file_path)

        if load_cache and os.path.exists(file_path):
            query = load_json_file(file_path)
        else:
            query["hard_logic_py"] = "\n".join(self.load_query_py(query["uid"]))
            query = nl2sl_reflect(query, self.backbone_llm)
            if "error" in query:
                query["hard_logic_py"] = {}
            save_json_file(query, file_path)

        return query

    def add_intercity_transport(
            self, activities, intercity_info, innercity_transports=[], tickets=1
    ):
        activity_i = {
            "start_time": intercity_info["BeginTime"],
            "end_time": intercity_info["EndTime"],
            "start": intercity_info["From"],
            "end": intercity_info["To"],
            "price": intercity_info["Cost"],
            "cost": intercity_info["Cost"] * tickets,
            "tickets": tickets,
            "transports": innercity_transports,
        }
        if not pd.isna(intercity_info["TrainID"]):
            activity_i["TrainID"] = intercity_info["TrainID"]
            activity_i["type"] = "train"
        elif not pd.isna(intercity_info["FlightID"]):
            activity_i["FlightID"] = intercity_info["FlightID"]
            activity_i["type"] = "airplane"

        activities.append(activity_i)
        return activities

    def add_poi(
            self,
            activities,
            position,
            poi_type,
            price,
            cost,
            start_time,
            end_time,
            innercity_transports,
    ):
        activity_i = {
            "position": position,
            "type": poi_type,
            "price": price,
            "cost": cost,
            "start_time": start_time,
            "end_time": end_time,
            "transports": innercity_transports,
        }

        activities.append(activity_i)
        return activities

    def add_accommodation(
            self,
            current_plan,
            hotel_sel,
            current_day,
            arrived_time,
            required_rooms,
            transports_sel,
    ):

        current_plan[current_day]["activities"] = self.add_poi(
            activities=current_plan[current_day]["activities"],
            position=hotel_sel["name"],
            poi_type="accommodation",
            price=int(hotel_sel["price"]),
            cost=int(hotel_sel["price"]) * required_rooms,
            start_time=arrived_time,
            end_time="24:00",
            innercity_transports=transports_sel,
        )
        current_plan[current_day]["activities"][-1]["room_type"] = hotel_sel["numbed"]
        current_plan[current_day]["activities"][-1]["rooms"] = required_rooms

        return current_plan

    def add_restaurant(
            self, current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel
    ):

        # 开放时间
        opentime, endtime = (
            poi_sel["opentime"],
            poi_sel["endtime"],
        )

        # it is closed ...
        if time_compare_if_earlier_equal(endtime, arrived_time):
            raise Exception("Add POI error")
        if time_compare_if_earlier_equal(arrived_time, opentime):
            act_start_time = opentime
        else:
            act_start_time = arrived_time

        if poi_type == "lunch" and time_compare_if_earlier_equal(
                act_start_time, "11:00"
        ):
            act_start_time = "11:00"
        if poi_type == "lunch" and time_compare_if_earlier_equal(endtime, "11:00"):
            raise Exception("Add POI error")

        if poi_type == "dinner" and time_compare_if_earlier_equal(
                act_start_time, "17:00"
        ):
            act_start_time = "17:00"
        if poi_type == "dinner" and time_compare_if_earlier_equal(endtime, "17:00"):
            raise Exception("Add POI error")

        if poi_type == "lunch" and time_compare_if_earlier_equal(
                "13:00", act_start_time
        ):
            raise Exception("Add POI error")
        if poi_type == "dinner" and time_compare_if_earlier_equal(
                "20:00", act_start_time
        ):
            raise Exception("Add POI error")

        poi_time = 60
        act_end_time = add_time_delta(act_start_time, poi_time)
        if time_compare_if_earlier_equal(endtime, act_end_time):
            act_end_time = endtime

        tmp_plan = deepcopy(current_plan)
        tmp_plan[current_day]["activities"] = self.add_poi(
            activities=tmp_plan[current_day]["activities"],
            position=poi_sel["name"],
            poi_type=poi_type,
            price=int(poi_sel["price"]),
            cost=int(poi_sel["price"]) * self.query["people_number"],
            start_time=act_start_time,
            end_time=act_end_time,
            innercity_transports=transports_sel,
        )
        return tmp_plan

    def add_attraction(
            self, current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel
    ):

        # 开放时间
        opentime, endtime = (
            poi_sel["opentime"],
            poi_sel["endtime"],
        )

        # it is closed ...

        opentime, endtime = poi_sel["opentime"], poi_sel["endtime"]
        # it is closed ...
        if time_compare_if_earlier_equal(endtime, arrived_time):
            raise Exception("Add POI error")

        if time_compare_if_earlier_equal(arrived_time, opentime):
            act_start_time = opentime
        else:
            act_start_time = arrived_time

        poi_time = 90
        act_end_time = add_time_delta(act_start_time, poi_time)
        if time_compare_if_earlier_equal(endtime, act_end_time):
            act_end_time = endtime

        tmp_plan = deepcopy(current_plan)
        tmp_plan[current_day]["activities"] = self.add_poi(
            activities=tmp_plan[current_day]["activities"],
            position=poi_sel["name"],
            poi_type=poi_type,
            price=int(poi_sel["price"]),
            cost=int(poi_sel["price"]) * self.query["people_number"],
            start_time=act_start_time,
            end_time=act_end_time,
            innercity_transports=transports_sel,
        )
        tmp_plan[current_day]["activities"][-1]["tickets"] = self.query["people_number"]
        return tmp_plan
    def constraints_validation(self, query, plan, poi_plan):

        self.constraints_validation_count += 1

        res_plan = {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": plan,
        }
        print("validate the plan [for query {}]: ".format(query["uid"]))
        print(res_plan)

        self.least_plan_schema = deepcopy(res_plan)

        bool_result = func_commonsense_constraints(query, res_plan, verbose=True)

        # if not bool_result:
        #     exit(0)

        if bool_result:
            self.commonsense_pass_count += 1

        try:
            extracted_vars = get_symbolic_concepts(query, res_plan, need_ood=False)

        except:
            extracted_vars = None

        print(extracted_vars)

        logical_result = evaluate_constraints_py(query["hard_logic_py"], res_plan, verbose=True)

        print(logical_result)

        logical_pass = True
        for idx, item in enumerate(logical_result):
            logical_pass = logical_pass and item

            if item:
                print(query["hard_logic_py"][idx], "passed!")
            else:

                print(query["hard_logic_py"][idx], "failed...")
        if bool_result and np.sum(logical_result) > self.least_plan_logical_pass:
            self.least_plan_comm = deepcopy(res_plan)
            self.least_plan_logical_pass = np.sum(logical_result)
        # if logical_result:
        #     print("Logical passed!")

        if logical_pass:
            self.logical_pass_count += 1

        bool_result = bool_result and logical_pass

        if bool_result:
            print("\n Pass! \n")
            self.all_constraints_pass += 1

            if self.least_plan_logic is None:
                self.least_plan_logic = res_plan

            if self.preference_search:
                # self.least_plan_logic = res_plan
                try:
                    if self.query["preference_opt"] == "maximize":

                        res = evaluate_preference_py([(self.query["preference_opt"], self.query["preference_concept"],
                                                       self.query["preference_code"])], res_plan)[0]
                        print(self.query["preference_concept"], res)

                        # print(res, self.least_plan_logic_pvalue)
                        if res != -1 and res > self.least_plan_logic_pvalue:
                            print("preference value [{}]: {} -> {} \n update plan".format(
                                self.query["preference_concept"], self.least_plan_logic_pvalue, res))
                            self.least_plan_logic_pvalue = res
                            self.least_plan_logic = deepcopy(res_plan)


                    elif self.query["preference_opt"] == "minimize":
                        res = evaluate_preference_py([(self.query["preference_opt"], self.query["preference_concept"],
                                                       self.query["preference_code"])], res_plan)[0]
                        print(self.query["preference_concept"], res)

                        # print(res, self.least_plan_logic_pvalue)
                        if res != -1 and res < self.least_plan_logic_pvalue:
                            print("preference value [{}]: {} -> {} \n update plan".format(
                                self.query["preference_concept"], self.least_plan_logic_pvalue, res))
                            self.least_plan_logic_pvalue = res
                            self.least_plan_logic = deepcopy(res_plan)

                    else:
                        raise ValueError("Invalid preference_opt")
                    print(self.least_plan_logic)
                except Exception as e:
                    print(e)
                    print(self.query["preference_code"])
        else:
            print("\n Failed \n")
            self.failed.append(query["uid"])
        # plan = res_plan

        # print(result)
        # exit(0)

        if self.preference_search:
            return False, plan

        if bool_result:
            # res_plan["search_time_sec"] = time.time() - self.time_before_search
            # res_plan["llm_inference_time_sec"] = self.llm_inference_time_count
            return True, res_plan
        else:
            return False, plan

    def symbolic_search(self, symoblic_query):
        # print(symoblic_query)
        if (symoblic_query["target_city"] in self.env.support_cities) and (
                symoblic_query["start_city"] in self.env.support_cities
        ):
            pass
        else:
            return False, {
                "error_info": f"Unsupported cities {symoblic_query['start_city']} -> {symoblic_query['target_city']}."}

        self.memory["accommodations"] = self.collect_poi_info_all(
            symoblic_query["target_city"], "accommodation"
        )
        self.memory["attractions"] = self.collect_poi_info_all(
            symoblic_query["target_city"], "attraction"
        )
        self.memory["restaurants"] = self.collect_poi_info_all(
            symoblic_query["target_city"], "restaurant"
        )
        # print(symoblic_query)
        self.query = symoblic_query
        success, plan = self.generate_plan_with_search(symoblic_query)

        print(success, plan)

        return success, plan

    from typing import List, Dict, Tuple

    def collect_innercity_transport(
            self,
            city: str,
            start: str,
            end: str,
            start_time: str,
            trans_type: str,
            check: bool = False
    ) -> Tuple[List[Dict], float]:
        if start == end:
            return [], 0.0

        if trans_type.lower() not in ["taxi", "walk", "metro"]:
            return [], 0.0

        try:
            # 构造调用语句
            call_str = f'goto("{city}", "{start}", "{end}", "{start_time}", "{trans_type}")'
            response = self.env(call_str)

            if not response or not hasattr(response, "_data"):
                return [], 0.0

            info = response._data
            if not isinstance(info, list) or not info:
                return [], 0.0

            processed_info = []
            total_cost = 0.0

            for item in info:
                if not isinstance(item, dict):
                    continue

                price = float(item.get("cost", 0))
                distance = float(item.get("distance", 0))

                standardized_item = {
                    "start": item.get("start", start),
                    "end": item.get("end", end),
                    "mode": item.get("mode", trans_type),
                    "start_time": item.get("start_time", start_time),
                    "end_time": item.get("end_time", ""),
                    "price": price,
                    "distance": distance,
                    "cost": price  # 先设为原始票价，后面根据交通方式修改
                }

                if standardized_item["mode"] == "taxi":
                    cars = int((self.query["people_number"] - 1) / 4) + 1
                    for constraint in self.query['hard_logic']:
                        if isinstance(constraint, str) and constraint.strip().startswith("taxi_cars=="):
                            try:
                                cars = int(constraint.split("taxi_cars==")[1].strip())
                                break
                            except:
                                pass
                    standardized_item["cars"] = cars
                    standardized_item["cost"] = price * cars

                elif standardized_item["mode"] == "metro":
                    tickets = self.query.get("people_number", 1)
                    standardized_item["tickets"] = tickets
                    standardized_item["cost"] = price * tickets

                total_cost += standardized_item["cost"]

                # 成本超过限制
                if self.innercity_cost_limit != None:
                    if total_cost + self.cur_innercity_cost > self.innercity_cost_limit and not check:
                        return [], 0.0

                processed_info.append(standardized_item)

            return processed_info, total_cost

        except Exception as e:
            print(f"[ERROR] 市内交通查询异常 - 城市:{city} 起点:{start} 终点:{end}: {str(e)}")
            return [], 0.0

    def collect_intercity_transport(self, source_city, target_city, trans_type):
        trans_info = pd.DataFrame()
        info_return = self.env(
            "intercity_transport_select('{start_city}', '{end_city}', '{intercity_type}')".format(
                start_city=source_city, end_city=target_city, intercity_type=trans_type
            )
        )
        # 检查是否成功获取数据
        if not info_return["success"]:
            return trans_info  # 返回空 DataFrame

        # 确保返回的数据是 DataFrame
        trans_info = info_return["data"]
        if trans_info is None:  # 首先检查是否为 None
            return pd.DataFrame()

        if not isinstance(trans_info, pd.DataFrame):
            try:
                trans_info = pd.DataFrame(trans_info)  # 尝试转换
            except Exception as e:
                print(f"数据转换DataFrame失败: {str(e)}")
                return pd.DataFrame()

        # 循环翻页获取剩余数据
        while True:
            next_page_return = self.env("next_page()")
            if not next_page_return["success"] or len(next_page_return["data"]) == 0:
                break

            info_i = next_page_return["data"]
            if info_i is None:  # 检查分页数据是否为 None
                continue

            if not isinstance(info_i, pd.DataFrame):
                try:
                    info_i = pd.DataFrame(info_i)
                except Exception as e:
                    print(f"分页数据转换DataFrame失败: {str(e)}")
                    continue

            trans_info = pd.concat([trans_info, info_i], axis=0, ignore_index=True)

        return trans_info

    def collect_poi_info_all(self, city, poi_type):
        if poi_type == "accommodation":
            func_name = "accommodations_select"
        elif poi_type == "attraction":
            func_name = "attractions_select"
        elif poi_type == "restaurant":
            func_name = "restaurants_select"
        else:
            raise NotImplementedError

        poi_info = self.env(
            "{func}('{city}', 'name', lambda x: True)".format(func=func_name, city=city)
        )["data"]
        # print(poi_info)
        while True:
            info_i = self.env("next_page()")["data"]
            if len(info_i) == 0:
                break
            else:
                poi_info = pd.concat([poi_info, info_i], axis=0, ignore_index=True)

        # print(poi_info)
        return poi_info
    def dinner_poi(self, query, poi_plan, plan, current_time, current_position, current_day):
        """
        安排晚餐活动
        """
        ranking_idx = self.ranking_restaurants(
            plan, poi_plan, current_day, current_time, current_position, self.intercity_with_hotel_cost
        )
        current_cost = 0.0
        for dayplan in plan:
            for act in dayplan.get("activities", []):
                if act.get("type") in ["lunch", "dinner"]:
                    current_cost += act.get("cost", 0)
        for sea_i, r_i in enumerate(ranking_idx):
            if r_i in self.restaurants_visiting:
                continue
            if self.search_width is not None and sea_i >= self.search_width:
                break
            poi_sel = self.memory["restaurants"].iloc[r_i]
            cost_limit = self._hard_logic_cache_restaurants["cost_limit"]
            if cost_limit and current_cost + self.query.get('people_number', 1) * poi_sel["price"] > cost_limit:
                continue
            transports_ranking = self.get_final_transport_ranking(
                current_position, poi_sel["name"], current_time
            )
            for trans_type_sel in transports_ranking:
                transports_sel, cost = self.collect_innercity_transport(
                    query["target_city"], current_position, poi_sel["name"],
                    current_time, trans_type_sel
                )
                if self.total_cost_limit is not None and (self.cur_all_cost + poi_sel["price"] * query["people_number"] + cost) > self.total_cost_limit:
                    continue
                if len(transports_sel) == 0:
                    continue
                if not isinstance(transports_sel, list):
                    continue

                arrived_time = transports_sel[-1]["end_time"] if transports_sel else current_time
                if poi_sel['name'] in self.start_time_constraints:
                    required_start_time = self.start_time_constraints[poi_sel['name']]
                    if time_compare_if_earlier_equal(required_start_time, arrived_time):
                        continue
                opentime, endtime = poi_sel["opentime"], poi_sel["endtime"]
                if poi_sel['name'] in self.time_range_constraints:
                    if (time_compare_if_earlier_equal(arrived_time, self.time_range_constraints[poi_sel['name']]['min_start'])
                            or time_compare_if_earlier_equal(self.time_range_constraints[poi_sel['name']]['max_end'], arrived_time)):
                        continue
                if (time_compare_if_earlier_equal(endtime, arrived_time) or
                        time_compare_if_earlier_equal("18:30", arrived_time)):
                    continue
                parts = arrived_time.split(":")
                if int(parts[0]) > 24:
                    continue
                act_end_time = add_time_delta(arrived_time, 60)  # 默认用餐1小时
                if poi_sel['name'] in self.time_range_constraints:
                    required_end_time = self.time_range_constraints[poi_sel['name']]['max_end']
                    if time_compare_if_earlier_equal(required_end_time, act_end_time):
                        act_end_time = required_end_time
                if poi_sel['name'] in self.end_time_constraints:
                    required_end_time = self.end_time_constraints[poi_sel['name']]
                    if time_compare_if_earlier_equal(act_end_time, required_end_time):
                        act_end_time = required_end_time
                if time_compare_if_earlier_equal(endtime, act_end_time):
                    continue

                plan = self.add_restaurant(plan, "dinner", poi_sel, current_day, arrived_time, transports_sel)
                self.cur_innercity_cost += cost
                self.cur_all_cost += (poi_sel["price"] * self.query["people_number"] + cost)
                if self.activity_order_constraints and poi_sel['name'] == self.activity_order_constraints[0]:
                    self.activity_order_constraints.pop(0)
                self.restaurants_visiting.append(r_i)
                self.food_type_visiting.append(poi_sel["cuisine"])
                print(f"安排晚餐：{poi_sel['name']}，时间：{arrived_time} - {act_end_time}")
                return plan, act_end_time, poi_sel["name"]

        return plan, current_time, current_position

    def back_hotel(self, query, poi_plan, plan, current_time, current_position, current_day):
        """
        返回酒店过夜
        """
        hotel_sel = poi_plan["accommodation"]
        transports_ranking = self.get_final_transport_ranking(
            current_position, hotel_sel["name"], current_time
        )
        for trans_type_sel in transports_ranking:
            transports_sel, cost = self.collect_innercity_transport(
                query["target_city"], current_position, hotel_sel["name"],
                current_time, trans_type_sel
            )
            if self.total_cost_limit is not None and (self.cur_all_cost + cost) > self.total_cost_limit:
                continue
            if len(transports_sel) == 0:
                continue
            if not isinstance(transports_sel, list):
                continue

            arrived_time = transports_sel[-1]["end_time"] if transports_sel else current_time
            plan = self.add_accommodation(
                plan, hotel_sel, current_day, arrived_time, self.required_rooms, transports_sel
            )
            self.cur_innercity_cost += cost
            self.cur_all_cost += cost
            print(f"安排回酒店：{hotel_sel['name']}，时间：{arrived_time}")
            return plan, "00:00", hotel_sel["name"]  # 第二天从00:00开始

        return plan, current_time, current_position

    def breakfast_poi(self, query, poi_plan, plan, current_time, current_position, current_day):
        """
        在酒店用早餐
        """
        # 确保计划中有当前天的数据
        if len(plan) <= current_day:
            plan.append({"day": current_day + 1, "activities": []})

        if current_time == "00:00":
            plan = self.select_and_add_breakfast(plan, poi_plan, current_day, current_time, current_position)
            if plan[current_day]["activities"]:  # 检查是否有活动被添加
                new_time = plan[current_day]["activities"][-1]["end_time"]
                return plan, new_time, current_position
        return plan, current_time, current_position

    def attraction_poi(self, query, poi_plan, plan, current_time, current_position, current_day):
        """
        安排一个景点活动，修复时间逻辑
        """
        ranking_idx = self.ranking_attractions(
            plan, poi_plan, current_day, current_time, current_position, self.intercity_with_hotel_cost
        )
        for sea_i, r_i in enumerate(ranking_idx):
            if self.search_width is not None and sea_i >= self.search_width:
                break

            attr_idx = r_i
            if attr_idx in self.attractions_visiting:
                continue

            poi_sel = self.memory["attractions"].iloc[attr_idx]

            # 检查景点开放时间
            opentime, endtime = poi_sel["opentime"], poi_sel["endtime"]
            if not time_compare_if_earlier_equal(opentime, current_time) or \
                    not time_compare_if_earlier_equal(current_time, endtime):
                continue  # 景点未开放

            transports_ranking = self.get_final_transport_ranking(
                current_position, poi_sel["name"], current_time
            )

            for trans_type_sel in transports_ranking:
                transports_sel, cost = self.collect_innercity_transport(
                    query["target_city"], current_position, poi_sel["name"],
                    current_time, trans_type_sel
                )
                if self.total_cost_limit is not None and (self.cur_all_cost + poi_sel["price"] * query["people_number"] + cost) > self.total_cost_limit:
                    continue
                if not isinstance(transports_sel, list) or len(transports_sel) == 0:
                    continue

                # 获取真实的到达时间
                arrived_time = transports_sel[-1]["end_time"]
                if poi_sel['name'] in self.start_time_constraints:
                    required_start_time = self.start_time_constraints[poi_sel['name']]
                    if time_compare_if_earlier_equal(required_start_time, arrived_time):
                        continue
                if poi_sel['name'] in self.time_range_constraints:
                    if (time_compare_if_earlier_equal(arrived_time, self.time_range_constraints[poi_sel['name']]['min_start'])
                            or time_compare_if_earlier_equal(self.time_range_constraints[poi_sel['name']]['max_end'], arrived_time)):
                        continue
                if (time_compare_if_earlier_equal(current_time, "10:30") and
                    time_compare_if_earlier_equal("11:00", arrived_time)):
                    continue
                if (time_compare_if_earlier_equal(current_time, "16:30") and
                    time_compare_if_earlier_equal("17:00", arrived_time)):
                    continue
                if (time_compare_if_earlier_equal(current_time, "21:00") and
                    time_compare_if_earlier_equal("21:30", arrived_time)):
                    continue
                if (time_compare_if_earlier_equal("00:00", arrived_time) and
                        time_compare_if_earlier_equal(arrived_time, "07:00")):
                    continue
                if current_day == query["days"] - 1 and "back_transport" in poi_plan:
                    back_time = poi_plan["back_transport"]["BeginTime"]
                    latest_activity_end = add_time_delta(back_time, -121)
                    if not time_compare_if_earlier_equal(arrived_time, latest_activity_end):
                        continue
                # 验证时间逻辑：到达时间不能晚于当前时间（倒退）
                if not time_compare_if_earlier_equal(current_time, arrived_time):
                    continue

                # 确保到达时间在开放时间内
                if not time_compare_if_earlier_equal(opentime, arrived_time) or \
                        not time_compare_if_earlier_equal(arrived_time, endtime):
                    continue
                act_start_time = arrived_time
                haved_lunch_today, haved_dinner_today = True, True
                if time_compare_if_earlier_equal(current_time,"10:30"):
                    haved_lunch_today = False
                if time_compare_if_earlier_equal(current_time,"16:30"):
                    haved_dinner_today = False
                parts = act_start_time.split(':')
                if int(parts[0]) > 25:
                    continue
                poi_time = self.select_poi_time(
                    plan, poi_plan, current_day, act_start_time,poi_sel["name"], "attraction",
                    recommended_visit_time=poi_sel["recommendmintime"] * 60,
                    recommendmax_time=poi_sel["recommendmaxtime"] * 60,
                    have_lunch=haved_lunch_today,
                    have_dinner=haved_dinner_today, )
                if poi_time == 0 or poi_time > poi_sel["recommendmaxtime"] * 30:
                    continue
                if (time_compare_if_earlier_equal(current_time, "10:30") and
                    time_compare_if_earlier_equal("11:00", add_time_delta(act_start_time, int(poi_time)))):
                    act_end_time = "11:00"
                elif (time_compare_if_earlier_equal(current_time, "16:30") and
                    time_compare_if_earlier_equal("17:00", add_time_delta(act_start_time, int(poi_time)))):
                    act_end_time = "17:00"
                elif (time_compare_if_earlier_equal(current_time, "21:00") and
                    time_compare_if_earlier_equal("21:30", add_time_delta(act_start_time, int(poi_time)))):
                    act_end_time = "21:30"
                else:
                    act_end_time = add_time_delta(act_start_time, int(poi_time))
                if poi_sel['name'] in self.time_range_constraints:
                    required_end_time = self.time_range_constraints[poi_sel['name']]['max_end']
                    if time_compare_if_earlier_equal(required_end_time, act_end_time):
                        act_end_time = required_end_time
                if poi_sel['name'] in self.end_time_constraints:
                    required_end_time = self.end_time_constraints[poi_sel['name']]
                    if (time_compare_if_earlier_equal(act_end_time, required_end_time) and
                            (self.time_difference_minutes(act_end_time, required_end_time) < 60)):
                        act_end_time = required_end_time
                    else:
                        continue
                # 检查结束时间是否在景点关闭前
                if time_compare_if_earlier_equal(endtime, act_end_time) or time_compare_if_earlier_equal(act_end_time, "07:00"):
                    act_end_time = endtime
                if act_start_time == act_end_time:
                    continue
                # # 最后一天检查返程时间
                # if current_day == query["days"] - 1 and "back_transport" in poi_plan:
                #     back_time = poi_plan["back_transport"]["BeginTime"]
                #     # 需要提前90分钟到达车站
                #     latest_activity_end = add_time_delta(back_time, -90)
                #     if not time_compare_if_earlier_equal(act_end_time, latest_activity_end):
                #         continue
                # 添加景点活动
                plan[current_day]["activities"] = self.add_poi(
                    plan[current_day]["activities"], poi_sel["name"], "attraction",
                    poi_sel["price"], poi_sel["price"] * query["people_number"],
                    act_start_time, act_end_time, transports_sel
                )
                plan[current_day]["activities"][-1]["tickets"] = query["people_number"]
                if self.activity_order_constraints and poi_sel['name'] == self.activity_order_constraints[0]:
                    self.activity_order_constraints.pop(0)
                self.cur_innercity_cost += cost
                self.cur_all_cost += (poi_sel["price"] * query["people_number"] + cost)
                self.attractions_visiting.append(attr_idx)
                self.spot_type_visiting.append(poi_sel["type"])
                self.attraction_names_visiting.append(poi_sel["name"])

                print(f"安排景点：{poi_sel['name']}，时间：{act_start_time} - {act_end_time}")
                return plan, act_end_time, poi_sel["name"], True

        return plan, current_time, current_position, False

    def lunch_poi(self, query, poi_plan, plan, current_time, current_position, current_day):
        ranking_idx = self.ranking_restaurants(
            plan, poi_plan, current_day, current_time, current_position, self.intercity_with_hotel_cost
        )
        current_cost = 0.0
        for dayplan in plan:
            for act in dayplan.get("activities", []):
                if act.get("type") in ["lunch", "dinner"]:
                    current_cost += act.get("cost", 0)
        for sea_i, r_i in enumerate(ranking_idx):
            if r_i in self.restaurants_visiting:
                continue
            if self.search_width is not None and sea_i >= self.search_width:
                break
            poi_sel = self.memory["restaurants"].iloc[r_i]
            cost_limit = self._hard_logic_cache_restaurants["cost_limit"]
            if cost_limit and current_cost + self.query.get('people_number', 1) * poi_sel["price"] > cost_limit:
                continue
            # 检查餐厅营业时间
            opentime, endtime = poi_sel["opentime"], poi_sel["endtime"]
            if not time_compare_if_earlier_equal("10:30", endtime) or \
                    not time_compare_if_earlier_equal(opentime, "13:00"):
                continue

            transports_ranking = self.get_final_transport_ranking(
                current_position, poi_sel["name"], current_time
            )

            for trans_type_sel in transports_ranking:
                transports_sel, cost = self.collect_innercity_transport(
                    query["target_city"], current_position, poi_sel["name"],
                    current_time, trans_type_sel
                )
                if self.total_cost_limit is not None and (self.cur_all_cost + poi_sel["price"] * query["people_number"] + cost) > self.total_cost_limit:
                    continue
                if len(transports_sel) == 0:
                    continue
                if not isinstance(transports_sel, list):
                    continue

                arrived_time = transports_sel[-1]["end_time"] if transports_sel else current_time
                if poi_sel['name'] in self.start_time_constraints:
                    required_start_time = self.start_time_constraints[poi_sel['name']]
                    if time_compare_if_earlier_equal(required_start_time, arrived_time):
                        continue
                if poi_sel['name'] in self.time_range_constraints:
                    if (time_compare_if_earlier_equal(arrived_time, self.time_range_constraints[poi_sel['name']]['min_start'])
                            or time_compare_if_earlier_equal(self.time_range_constraints[poi_sel['name']]['max_end'], arrived_time)):
                        continue
                # 时间验证
                if not time_compare_if_earlier_equal(current_time, arrived_time):
                    continue

                if not time_compare_if_earlier_equal("10:30", arrived_time) or \
                        not time_compare_if_earlier_equal(arrived_time, "13:00"):
                    continue

                act_start_time = arrived_time
                parts = arrived_time.split(":")
                if int(parts[0]) > 24:
                    continue
                act_end_time = add_time_delta(act_start_time, 60)  # 用餐1小时
                if poi_sel['name'] in self.time_range_constraints:
                    required_end_time = self.time_range_constraints[poi_sel['name']]['max_end']
                    if time_compare_if_earlier_equal(required_end_time, act_end_time):
                        act_end_time = required_end_time
                if poi_sel['name'] in self.end_time_constraints:
                    required_end_time = self.end_time_constraints[poi_sel['name']]
                    if time_compare_if_earlier_equal(act_end_time, required_end_time):
                        act_end_time = required_end_time
                try:
                    plan = self.add_restaurant(plan, "lunch", poi_sel, current_day, act_start_time, transports_sel)
                    self.restaurants_visiting.append(r_i)
                    if self.activity_order_constraints and poi_sel['name'] == self.activity_order_constraints[0]:
                        self.activity_order_constraints.pop(0)
                    self.food_type_visiting.append(poi_sel["cuisine"])
                    self.cur_innercity_cost += cost
                    self.cur_all_cost += (poi_sel["price"] * self.query["people_number"] + cost)
                    print(f"安排午餐：{poi_sel['name']}，时间：{act_start_time} - {act_end_time}")
                    return plan, act_end_time, poi_sel["name"]
                except Exception as e:
                    print(f"添加餐厅失败：{e}")
                    continue

        return plan, current_time, current_position

    def back_home(self, query, poi_plan, plan, current_time, current_position, current_day):
        """
        安排返程，确保时间逻辑正确
        """
        back_transport = poi_plan["back_transport"]
        back_time = back_transport["BeginTime"]

        # 需要提前到达车站
        latest_departure_time = add_time_delta(back_time, -90)  # 提前90分钟

        if not time_compare_if_earlier_equal(current_time, latest_departure_time):
            print(f"返程时间不足：当前时间 {current_time}，最晚出发时间 {latest_departure_time}")
            return plan, current_time, current_position

        transports_ranking = self.get_final_transport_ranking(
            current_position, back_transport["From"], current_time)

        for trans_type_sel in transports_ranking:
            transports_sel, cost = self.collect_innercity_transport(
                query["target_city"], current_position, back_transport["From"],
                current_time, trans_type_sel
            )
            if self.total_cost_limit is not None and (self.cur_all_cost + cost) > self.total_cost_limit:
                continue
            if not isinstance(transports_sel, list) or len(transports_sel) == 0:
                continue

            arrived_time = transports_sel[-1]["end_time"]
            # 验证到达时间早于发车时间
            if not time_compare_if_earlier_equal(arrived_time, back_time):
                print(f"到达时间晚于发车时间：{arrived_time} > {back_time}")
                continue

            plan[current_day]["activities"] = self.add_intercity_transport(
                plan[current_day]["activities"], back_transport,
                transports_sel, query["people_number"]
            )
            self.cur_innercity_cost += cost
            self.cur_all_cost += cost
            print(f"安排返程：{back_transport['From']} -> {back_transport['To']}，时间：{back_time}")
            return plan, back_transport["EndTime"], back_transport["To"], True

        return plan, current_time, current_position, False

    def max_time(self, time1, time2):
        """返回两个时间中较晚的一个"""
        if time_compare_if_earlier_equal(time1, time2):
            return time2
        else:
            return time1

    def linner_poi(self, query, poi_plan, plan, current_time, current_position):
        # 初始化计划结构
        if not plan:
            plan = [{"day": i + 1, "activities": []} for i in range(query["days"])]
        print("安排去程交通...")
        plan[0]["activities"] = self.add_intercity_transport(
            plan[0]["activities"],
            poi_plan["go_transport"],
            innercity_transports=[],
            tickets=self.query["people_number"],
        )
        current_time = poi_plan["go_transport"]["EndTime"]
        current_position = poi_plan["go_transport"]["To"]
        print(f"到达目的地：{current_position}，时间：{current_time}")
        for current_day in range(query["days"]):
            print(f"正在安排第 {current_day + 1} 天的行程...")
            have_dinner = False
            have_lunch = False
            jump = False
            # 最后一天的逻辑
            if current_day == query["days"] - 1 and current_time != "":
                # 确保计划中有当前天的数据
                if len(plan) <= current_day:
                    plan.append({"day": current_day + 1, "activities": []})

                print(f"第 {current_day + 1} 天（最后一天），当前位置：{current_position}，时间：{current_time}")

                # 获取返程时间
                back_time = poi_plan["back_transport"]["BeginTime"]
                print(f"返程时间：{back_time}，需要提前90分钟到达车站")

                # 需要提前到达车站的最晚活动结束时间
                latest_activity_time = add_time_delta(back_time, -91)

                # 早餐
                if current_day > 0:
                    print("安排早餐...")
                    plan, current_time, current_position = self.breakfast_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )

                # 检查时间是否允许继续安排活动
                if not time_compare_if_earlier_equal(current_time, latest_activity_time):
                    print("时间不足，直接安排返程")
                    ret = False
                    while not ret:
                        plan, current_time, current_position, ret = self.back_home(
                            query, poi_plan, plan, current_time, current_position, current_day
                        )
                        if not ret:
                            if plan[current_day]["activities"]:
                                ret = plan[current_day]["activities"].pop()
                                self.cur_all_cost -= ret["cost"]
                                transports = ret["transports"]
                                if len(transports) == 3:
                                    self.cur_innercity_cost -= transports[1]["cost"]
                                    self.cur_all_cost -= transports[1]["cost"]
                                elif len(transports) == 1:
                                    self.cur_innercity_cost -= transports[0]["cost"]
                                    self.cur_all_cost -= transports[0]["cost"]
                                if plan[current_day]["activities"]:
                                    current_time = plan[current_day]["activities"][-1]["end_time"]
                                    current_position = plan[current_day]["activities"][-1].get('position', '')
                                else:
                                    # 处理空列表情况
                                    current_time = "07:30"
                                    current_position = poi_plan["accommodation"]["name"]
                    break

                # 上午景点（10:30前，且不超过最晚活动时间）
                while (time_compare_if_earlier_equal(current_time, "10:30") and
                       time_compare_if_earlier_equal(current_time, latest_activity_time)):
                    print(f"安排上午景点，当前时间：{current_time}，最晚活动时间：{latest_activity_time}")
                    plan, new_time, new_position, ret = self.attraction_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )
                    if not ret:
                        break
                    # 检查新时间是否超过限制
                    if not time_compare_if_earlier_equal(new_time, latest_activity_time):
                        print(f"景点结束时间 {new_time} 超过最晚活动时间 {latest_activity_time}，跳过该景点")
                        # 回退更改
                        if plan[current_day]["activities"]:
                            plan[current_day]["activities"].pop()
                            jump = True
                        break

                    current_time, current_position = new_time, new_position

                # 检查是否还有时间安排午餐
                if time_compare_if_earlier_equal(current_time, latest_activity_time) and not jump:
                    print("安排午餐...")
                    have_lunch = True
                    plan, new_time, new_position = self.lunch_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )

                    # 检查午餐时间是否合理
                    if time_compare_if_earlier_equal(new_time, latest_activity_time):
                        current_time, current_position = new_time, new_position
                    else:
                        print("午餐时间过长，跳过午餐")
                        if plan[current_day]["activities"]:
                            plan[current_day]["activities"].pop()
                            have_lunch = False
                            jump = True

                # 下午景点（16:30前，且不超过最晚活动时间）
                while (time_compare_if_earlier_equal(current_time, "16:30") and
                       time_compare_if_earlier_equal(current_time, latest_activity_time) and have_lunch) and not jump:
                    print(f"安排下午景点，当前时间：{current_time}，最晚活动时间：{latest_activity_time}")
                    plan, new_time, new_position, ret = self.attraction_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )
                    if not ret:
                        break
                    # 检查新时间是否超过限制
                    if not time_compare_if_earlier_equal(new_time, latest_activity_time):
                        print(f"景点结束时间 {new_time} 超过最晚活动时间 {latest_activity_time}，跳过该景点")
                        if plan[current_day]["activities"]:
                            plan[current_day]["activities"].pop()
                            jump = True
                        break

                    current_time, current_position = new_time, new_position

                # 检查是否还有时间安排晚餐（至少需要1小时）
                dinner_end_time = add_time_delta(current_time, 60)  # 预估晚餐1小时
                if time_compare_if_earlier_equal(dinner_end_time, latest_activity_time) and not jump:
                    print("安排晚餐...")
                    have_dinner = True
                    plan, new_time, new_position = self.dinner_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )

                    if time_compare_if_earlier_equal(new_time, latest_activity_time):
                        current_time, current_position = new_time, new_position
                    else:
                        print("晚餐时间过长，跳过晚餐")
                        if plan[current_day]["activities"]:
                            plan[current_day]["activities"].pop()
                            have_dinner = False
                            jump = True

                # 晚上景点（21:00前，且不超过最晚活动时间）
                while (time_compare_if_earlier_equal(current_time, "21:00") and
                       time_compare_if_earlier_equal(current_time, latest_activity_time) and have_dinner) and not jump:
                    print(f"安排晚上景点，当前时间：{current_time}，最晚活动时间：{latest_activity_time}")
                    plan, new_time, new_position, ret = self.attraction_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )
                    if not ret:
                        break
                    # 检查新时间是否超过限制
                    if not time_compare_if_earlier_equal(new_time, latest_activity_time):
                        print(f"景点结束时间 {new_time} 超过最晚活动时间 {latest_activity_time}，跳过该景点")
                        if plan[current_day]["activities"]:
                            plan[current_day]["activities"].pop()
                            jump = True
                        break

                    current_time, current_position = new_time, new_position

                # 安排返程
                print("安排返程...")
                ret = False
                while not ret:
                    plan, current_time, current_position, ret = self.back_home(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )
                    if not ret:
                        if plan[current_day]["activities"]:
                            re = plan[current_day]["activities"].pop()
                            self.cur_all_cost -= re["cost"]
                            transports = re["transports"]
                            if len(transports) == 3:
                                self.cur_innercity_cost -= transports[1]["cost"]
                                self.cur_all_cost -= transports[1]["cost"]
                            elif len(transports) == 1:
                                self.cur_innercity_cost -= transports[0]["cost"]
                                self.cur_all_cost -= transports[0]["cost"]
                            if plan[current_day]["activities"]:
                                current_time = plan[current_day]["activities"][-1]["end_time"]
                                current_position = plan[current_day]["activities"][-1].get('position', '')
                            else:
                                # 处理空列表情况
                                current_time = "07:30"
                                current_position = poi_plan["accommodation"]["name"]

            elif current_day == 0:
                # 第一天，安排去程交通
                # if (time_compare_if_earlier_equal("00:00", current_time) and
                #         time_compare_if_earlier_equal(current_time, "07:00")):
                #
                #     plan, current_time, current_position = self.breakfast_poi(
                #         query, poi_plan, plan, current_time, current_position, current_day
                #     )
                # 上午景点（10:30前）
                while time_compare_if_earlier_equal(current_time, "10:30"):
                    print(f"安排上午景点，当前时间：{current_time}")
                    plan, current_time, current_position, ret = self.attraction_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )
                    if not ret:
                        break
                if time_compare_if_earlier_equal(current_time, "12:30"):
                    print("安排午餐...")
                    have_lunch = True
                    plan, current_time, current_position = self.lunch_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )

                while time_compare_if_earlier_equal(current_time, "16:30"):
                    print(f"安排下午景点，当前时间：{current_time}")
                    plan, current_time, current_position, ret = self.attraction_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )
                    if not ret:
                        break
                # 晚餐
                if time_compare_if_earlier_equal(current_time, "18:00"):
                    print("安排晚餐...")
                    have_dinner = True
                    plan, current_time, current_position = self.dinner_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )
                while time_compare_if_earlier_equal(current_time, "21:00"):
                    print(f"安排晚上景点，当前时间：{current_time}")
                    plan, current_time, current_position, ret = self.attraction_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )
                    if not ret:
                        break
                plan, current_time, current_position = self.back_hotel(
                    query, poi_plan, plan, current_time, current_position, current_day
                )
            elif current_day > 0 and current_day<query["days"]:
                # 确保计划中有当前天的数据
                if len(plan) <= current_day:
                    plan.append({"day": current_day + 1, "activities": []})

                print(f"第 {current_day + 1} 天，当前位置：{current_position}，时间：{current_time}")

                # 早餐
                print("安排早餐...")
                plan, current_time, current_position = self.breakfast_poi(
                    query, poi_plan, plan, current_time, current_position, current_day
                )

                # 上午景点（11:00前）
                while time_compare_if_earlier_equal(current_time, "10:30"):
                    print(f"安排上午景点，当前时间：{current_time}")
                    plan, current_time, current_position, ret = self.attraction_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )
                    if not ret:
                        break
                # 午餐
                print("安排午餐...")
                plan, current_time, current_position = self.lunch_poi(
                    query, poi_plan, plan, current_time, current_position, current_day
                )

                # 下午景点（16:30前）
                while time_compare_if_earlier_equal(current_time, "16:30"):
                    print(f"安排下午景点，当前时间：{current_time}")
                    plan, current_time, current_position, ret = self.attraction_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )
                    if not ret:
                        break
                # 晚餐
                print("安排晚餐...")
                plan, current_time, current_position = self.dinner_poi(
                    query, poi_plan, plan, current_time, current_position, current_day
                )
                # 不是最后一天，晚上景点（21:00前）然后回酒店
                while time_compare_if_earlier_equal(current_time, "21:00"):
                    print(f"安排晚上景点，当前时间：{current_time}")
                    plan, current_time, current_position, ret = self.attraction_poi(
                        query, poi_plan, plan, current_time, current_position, current_day
                    )
                    if not ret:
                        break

                plan, current_time, current_position = self.back_hotel(
                    query, poi_plan, plan, current_time, current_position, current_day
                )
        # 验证最终计划
        print("行程安排完成，开始验证约束...")
        # res_bool, res_plan = self.constraints_validation(query, plan, poi_plan)
        # if res_bool:
        #     print("约束验证通过！")
        #     return True, res_plan
        # else:
        #     print("约束验证失败！")
        #     return False, plan
        res_plan = {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": plan,
        }
        return True, res_plan
    def generate_plan_with_search(self, query):

        source_city = query["start_city"]
        target_city = query["target_city"]

        print(source_city, "->", target_city)

        train_go = self.collect_intercity_transport(source_city, target_city, "train")
        train_back = self.collect_intercity_transport(target_city, source_city, "train")

        flight_go = self.collect_intercity_transport(
            source_city, target_city, "airplane"
        )
        flight_back = self.collect_intercity_transport(
            target_city, source_city, "airplane"
        )
        flight_go_num = 0 if flight_go is None else flight_go.shape[0]
        train_go_num = 0 if train_go is None else train_go.shape[0]
        flight_back_num = 0 if flight_back is None else flight_back.shape[0]
        train_back_num = 0 if train_back is None else train_back.shape[0]

        go_list = [df for df in [train_go, flight_go] if df is not None]
        go_info = pd.concat(go_list, axis=0) if go_list else pd.DataFrame()

        back_list = [df for df in [train_back, flight_back] if df is not None]
        back_info = pd.concat(back_list, axis=0) if back_list else pd.DataFrame()
        self.time_before_search = time.time()
        self.llm_inference_time_count = 0
        # reset the cache before searching
        poi_plan = {}
        self.restaurants_visiting = []
        self.attractions_visiting = []
        self.food_type_visiting = []
        self.spot_type_visiting = []
        self.attraction_names_visiting = []
        self.restaurant_names_visiting = []
        self.ranking_attractions_flag = False
        self.ranking_restaurants_flag = False
        self.ranking_hotel_flag = False
        self.llm_rec_format_error = 0
        self.llm_rec_count = 0
        self.search_nodes = 0
        self.backtrack_count = 0

        self.constraints_validation_count = 0
        self.commonsense_pass_count = 0
        self.logical_pass_count = 0
        self.all_constraints_pass = 0
        self.least_plan_schema, self.least_plan_comm, self.least_plan_logic = None, None, None
        self.least_plan_logical_pass = -1
        self.cur_innercity_cost = 0.
        self.innercity_cost_limit = None
        self.cur_all_cost = 0.
        self.total_cost_limit = None
        self.start_time_constraints = {}
        self.end_time_constraints = {}
        self.time_range_constraints = {}
        self.activity_time_constraints = {}
        self.activity_order_constraints = []
        self.failed = []
        def parse_single_constraint(constraint: str):
            """
            解析单个约束条件
            """
            try:
                if 'innercity_transport_cost<=' in constraint:
                    try:
                        self.innercity_cost_limit = float(constraint.split('<=')[1].strip())
                    except:
                        pass

                elif 'total_cost<=' in constraint or 'cost<=' in constraint:
                    if constraint.split('<=')[0].strip() == "cost" or constraint.split('<=')[0].strip() == "total_cost":
                        cost = constraint.split('<=')[1].strip()
                        self.total_cost_limit = float(cost.strip('() '))
                elif 'start_time{' in constraint and '<=' in constraint:
                    match = re.search(r'start_time\{([^}]+)\}\s*<=\s*([\d:]+)', constraint)
                    if match:
                        venue = match.group(1).strip()
                        time_str = match.group(2).strip()
                        # 清理时间字符串
                        time_str = time_str.strip('() ')
                        # 清理venue中可能存在的引号
                        venue = venue.strip("'\"")
                        self.start_time_constraints[venue] = time_str
                        print(f"解析开始时间约束: {venue} 不晚于 {time_str} 到达")
                elif 'end_time{' in constraint and '>=' in constraint:
                    # 格式: end_time{三郎日料•烧肉酒场(文晖店)}>=17:50
                    match = re.search(r'end_time\{([^}]+)\}\s*>=\s*([\d:]+)', constraint)
                    if match:
                        venue = match.group(1).strip()
                        time_str = match.group(2).strip()
                        # 清理时间字符串
                        time_str = time_str.strip('() ')
                        # 清理venue中可能存在的引号
                        venue = venue.strip("'\"")
                        self.end_time_constraints[venue] = time_str
                        print(f"解析结束时间约束: {venue} 不早于 {time_str} 结束")
                elif 'time_range{' in constraint:
                    # 格式: time_range{茅家埠景区,08:50-10:20}
                    match = re.search(r'time_range\{([^,]+)[,，]\s*([\d:]+)-([\d:]+)\}', constraint)
                    if match:
                        venue = match.group(1).strip()
                        start_time = match.group(2).strip()
                        end_time = match.group(3).strip()
                        # 清理venue中可能存在的引号
                        venue = venue.strip("'\"")
                        # 存储时间范围约束
                        self.time_range_constraints[venue] = {
                            'min_start': start_time,  # 最早开始时间 >=
                            'max_end': end_time  # 最晚结束时间 <=
                        }
                        print(f"解析时间范围约束: {venue} 在 {start_time} 到 {end_time} 之间")
                elif 'activity_time{' in constraint and '>=' in constraint:
                    match = re.search(r'activity_time\{([^}]+)\}\s*>=\s*(\d+)', constraint)
                    if match:
                        venue = match.group(1).strip()
                        duration_str = match.group(2).strip()
                        # 清理venue中可能存在的引号
                        venue = venue.strip("'\"")
                        # 转换为整数
                        try:
                            duration = int(duration_str)
                            self.activity_time_constraints[venue] = duration
                            print(f"解析活动时间约束: {venue} 不少于 {duration} 分钟")
                        except ValueError:
                            print(f"活动时间格式错误: {duration_str}")
                elif 'activity_order{' in constraint and '-' in constraint:
                    # 匹配 activity_order{A-B}
                    match = re.search(r'activity_order\{([^}]+)\}', constraint)
                    if match:
                        order_str = match.group(1).strip()
                        # 按 '-' 分割活动
                        venues = order_str.split('-')
                        if len(venues) == 2:
                            venue_a = venues[0].strip()
                            venue_b = venues[1].strip()
                            # 清理venue中可能存在的引号
                            venue_a = venue_a.strip("'\"")
                            venue_b = venue_b.strip("'\"")
                            # 存储顺序约束：A 在 B 之前
                            self.activity_order_constraints.append((venue_a, venue_b))
                            print(f"解析顺序约束: {venue_a} 在 {venue_b} 之前")
            except Exception as e:
                print(f"解析约束条件时出错: {constraint}, 错误: {e}")

        # 主处理逻辑
        for constraint in query['hard_logic']:
            if not isinstance(constraint, str):
                continue

            if ' or ' in constraint:
                # 处理 OR 条件：拆分后分别解析
                parts = [p.strip() for p in constraint.split(' or ')]
                for part in parts:
                    # 清理每个部分的外层括号
                    part = part.strip()
                    if part.startswith('('):
                        part = part[1:].strip()
                    if part.endswith(')'):
                        part = part[:-1].strip()
                    parse_single_constraint(part)
            else:
                constraint = constraint.strip()
                if constraint.startswith('('):
                    constraint = constraint[1:].strip()
                if constraint.endswith(')'):
                    constraint = constraint[:-1].strip()
                parse_single_constraint(constraint)

        ranking_go = self.ranking_intercity_transport_go(go_info, query, self.total_cost_limit)
        ranking_hotel = self.ranking_hotel(self.memory["accommodations"], query)
        query_room_number, query_room_type = self.decide_rooms(query)

        for go_i in ranking_go:
            go_info_i = go_info.iloc[go_i]
            poi_plan["go_transport"] = go_info_i
            ranking_back = self.ranking_intercity_transport_back(
                back_info, query, self.total_cost_limit
            )
            for back_i in ranking_back:
                back_info_i = back_info.iloc[back_i]
                poi_plan["back_transport"] = back_info_i
                if query["days"] > 1:
                    for hotel_i in ranking_hotel:
                        poi_plan["accommodation"] = self.memory["accommodations"].iloc[hotel_i]
                        room_type = poi_plan["accommodation"]["numbed"]
                        required_rooms = (int((query["people_number"] - 1) / 2) + 1)
                        if query_room_type != None and query_room_type != room_type:
                            print("room_type not match, backtrack...")
                            continue
                        if query_room_number != None:
                            required_rooms = query_room_number
                        if query_room_number != None and query_room_type != None:
                            pass
                        self.required_rooms = required_rooms
                        self.intercity_with_hotel_cost = (
                                                                 poi_plan["go_transport"]["Cost"]
                                                                 + poi_plan["back_transport"]["Cost"]
                                                         ) * query["people_number"] + poi_plan["accommodation"][
                                                             "price"
                                                         ] * required_rooms * (
                                                                 query["days"] - 1
                                                         )
                        self.cur_all_cost = self.intercity_with_hotel_cost
                        if (
                                self.total_cost_limit != None
                                and self.total_cost_limit - self.intercity_with_hotel_cost
                                <= self.query["people_number"]
                                * (self.query["days"] - 1)
                                * 30
                        ):
                            self.backtrack_count += 1
                            if self.backtrack_count >= 3:
                                self.total_cost_limit += 30 * self.query["people_number"]*(self.query["days"] - 1)
                            continue
                        print("search: ...")
                        try:
                            success, plan = self.linner_poi(
                                query,
                                poi_plan,
                                plan=[],
                                current_time="",
                                current_position=""
                            )
                        except TimeOutError as e:
                            print("TimeOutError")
                            return False, {"error_info": "TimeOutError"}

                        if success:
                            return True, plan
                        else:
                            return False, {"error_info": "No solution found."}
                            # if time.time() > self.time_before_search + self.TIME_CUT:
                            #     print("Searching TIME OUT !!!")
                            #     return False, {"error_info": "TimeOutError"}
                            # self.backtrack_count += 1
                            # print("search failed given the intercity-transport and hotels, backtrack...")

                else:
                    if time_compare_if_earlier_equal(
                            poi_plan["back_transport"]["BeginTime"],
                            poi_plan["go_transport"]["EndTime"],
                    ):
                        self.backtrack_count += 1
                        print("back_transport BeginTime earlier than go_transport EndTime, backtrack...")
                        continue

                    self.intercity_with_hotel_cost = (
                                                             poi_plan["go_transport"]["Cost"]
                                                             + poi_plan["back_transport"]["Cost"]
                                                     ) * query["people_number"]
                    print("search: ...")
                    try:
                        success, plan = self.linner_poi(
                            query,
                            poi_plan,
                            plan=[],
                            current_time="",
                            current_position=""
                        )
                    except TimeOutError as e:
                        print("TimeOutError")
                        return False, {"error_info": "No solution found."}

                    print(success, plan)
                    if success:
                        return True, plan
                    else:
                        # return False, {"error_info": "No solution found."}
                        if time.time() > self.time_before_search + self.TIME_CUT:
                            print("Searching TIME OUT !!!")
                            return False, {"error_info": "TimeOutError"}
                        self.backtrack_count += 1
                        print("search failed given the intercity-transport and hotels, backtrack...")

        return False, {"error_info": "No solution found."}
    def run(self, query, prob_idx, load_cache=False, oralce_translation=False):
        method_name = self.method + "_" + self.backbone_llm.name
        self.log_dir = os.path.join(self.cache_dir, method_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # sys.stdout = Logger(
        #     "{}/{}.log".format(
        #         self.log_dir, query["uid"]
        #     ),
        #     sys.stdout,
        #     self.debug,
        # )
        # sys.stderr = Logger(
        #     "{}/{}.error".format(
        #         self.log_dir, query["uid"]
        #     ),
        #     sys.stderr,
        #     self.debug,
        # )

        self.backbone_llm.input_token_count = 0
        self.backbone_llm.output_token_count = 0
        self.backbone_llm.input_token_maxx = 0
        if not oralce_translation:
            query = self.translate_nl2sl(query, load_cache=True)

        succ, plan = self.symbolic_search(query)

        if succ:
            plan_out = plan
        else:
            if self.least_plan_logic is not None:
                plan_out = self.least_plan_logic
                print("The least plan with logic constraints: ", plan_out)
                succ = True

            elif self.least_plan_comm is not None:
                plan_out = self.least_plan_comm
            elif self.least_plan_schema is not None:
                plan_out = self.least_plan_schema
            else:
                plan_out = {}

        return succ, plan_out

    def load_query_py(self, uid, verbose=False):
        data_dir = os.path.join(project_root_path, "data")

        dir_list = os.listdir(data_dir)
        for dir_i in dir_list:
            dir_ii = os.path.join(data_dir, dir_i)
            if os.path.isdir(dir_ii):
                file_list = os.listdir(dir_ii)

                for file_i in file_list:
                    query_id = file_i.split(".")[0]
                    if query_id == uid:
                        data_i = json.load(
                            open(os.path.join(dir_ii, file_i), encoding="utf-8")
                        )
                        if "hard_logic_py" in data_i:
                            return data_i["hard_logic_py"]
                        else:
                            return []

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="argparse testing")
    parser.add_argument(
        "--splits",
        "-l",
        type=str,
        default="tpc_phase1",
        help="query subset",
    )
    parser.add_argument("--index", "-i", type=int, default=None, help="query index")
    parser.add_argument(
        "--skip-exist", "-sk", type=int, default=1, help="skip if the plan exists"
    )
    parser.add_argument(
        "--llm", "-m", type=str, default="tpc_llm", choices=["deepseek", "tpc_llm", "gpt-4o", "glm4-plus"]
    )
    parser.add_argument(
        "--oracle_translation",
        action="store_true",
        help="Set this flag to enable oracle translation.",
    )
    args = parser.parse_args()

    from chinatravel.data.load_datasets import load_query
    from chinatravel.environment.world_env import WorldEnv
    from chinatravel.agent.tpc_agent.tpc_llm import TPCLLM
    env = WorldEnv()

    query_index, query_data = load_query(args)
    # print(query_index, query_data)
    print(len(query_index), "samples")
    if args.index is not None:
        query_index = [query_index[args.index]]
    if args.llm == "deepseek":
        llm = TPCLLM("sk-fa3c6e12204d46f0b00616ab1c2d205e")
    elif args.llm == "tpc_llm":
        llm = TPCLLM("abc123")
    else:
        llm = TPCLLM("sk-fa3c6e12204d46f0b00616ab1c2d205e")
    method = "TPC"

    method = method + "_" + args.llm

    os.environ["OPENAI_API_KEY"] = ""
    F = []
    cache_dir = os.path.join(project_root_path, "agent", "tpc_agent", "cache")

    res_dir = os.path.join(project_root_path, "agent", "tpc_agent", "fresults2", method)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    print(res_dir)

    white_list = []

    succ_count, eval_count = 0, 0

    # 在循环开始前创建失败记录文件
    failed_file = os.path.join(res_dir, "failed_records.txt")

    for i, data_idx in enumerate(query_index):
        if args.skip_exist and os.path.exists(
                os.path.join(res_dir, f"{data_idx}.json")
        ):
            continue
        if i in white_list:
            continue

        eval_count += 1

        symbolic_input = query_data[data_idx]
        agent = TPCAgent(
            env=env, backbone_llm=llm, cache_dir=cache_dir, search_width=100, debug=True
        )
        succ, plan = agent.run(symbolic_input, load_cache=True)

        if succ:
            succ_count += 1
        if agent.failed:
            F.append(agent.failed[0])
            # 将失败记录写入文件
            with open(failed_file, 'a', encoding='utf-8') as f:
                f.write(f"Data Index: {data_idx}, Failed Reason: {agent.failed[0]}\n")
        save_json_file(
            json_data=plan, file_path=os.path.join(res_dir, f"{data_idx}.json")
        )
    print(F)
