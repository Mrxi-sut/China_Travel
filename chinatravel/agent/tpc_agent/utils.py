import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class TimeOutError(Exception):
    def __init__(self, message="Searching TIME OUT !!!"):
        self.message = message
        super().__init__(self.message)


def time_compare_if_earlier_equal(time_1, time_2):
    """
    安全比较两个时间字符串（HH:MM格式），自动规范化超范围时间

    参数:
        time_1: 第一个时间字符串（允许为空）
        time_2: 第二个时间字符串（允许为空）

    返回:
        bool:
            - 当两个时间都有效时，返回 time1 <= time2
            - 当任一时间为空时，返回 False

    异常:
        当时间格式非法（无法解析为数字）时抛出ValueError
    """

    def _normalize_time(time_str):
        """规范化时间字符串，处理超范围值"""
        if not time_str:  # 空字符串直接返回None
            return None

        try:
            # 分割时间字符串
            parts = time_str.split(':')
            if len(parts) != 2:
                raise ValueError(f"时间格式应为HH:MM，实际得到: '{time_str}'")

            # 转换为整数并规范化
            hours = int(parts[0]) % 25  # 小时取模24
            minutes = int(parts[1]) % 61  # 分钟取模60
            return hours * 60 + minutes
        except ValueError:
            raise ValueError(f"时间包含非数字字符: '{time_str}'")

    try:
        time1 = _normalize_time(time_1)
        time2 = _normalize_time(time_2)

        # 处理空时间情况
        if time1 is None or time2 is None:
            return False

        return time1 <= time2
    except ValueError as e:
        print(f"时间比较错误: {str(e)}")
        raise


def add_time_delta(time1: str, time_delta: int) -> str:
    """
    给时间字符串增加指定分钟数，返回新的时间字符串(HH:MM格式)

    参数:
        time1: 基础时间字符串(HH:MM格式)
        time_delta: 要增加的分钟数

    返回:
        新的时间字符串(HH:MM格式)

    异常:
        ValueError: 当时间格式无效时抛出
    """

    # 输入验证
    if not time1:
        raise ValueError("时间字符串不能为空")

    try:
        if time1 == "24:00":
            time1 = "23:59"
        # 分割时间字符串
        parts = time1.split(':')
        if len(parts) != 2:
            raise ValueError(f"时间格式应为HH:MM，实际得到: '{time1}'")

        # 转换为整数
        hour = int(parts[0])
        minu = int(parts[1])

        # 验证时间范围
        if not (0 <= hour <= 23 and 0 <= minu <= 59):
            raise ValueError(f"时间值超出范围: '{time1}'")

        # 计算新时间
        total_minutes = hour * 60 + minu + time_delta
        if total_minutes < 0:
            total_minutes = 0  # 处理负时间情况

        hour_new = (total_minutes // 60) % 24
        min_new = total_minutes % 60

        # 格式化输出
        return f"{hour_new:02d}:{min_new:02d}"

    except ValueError as e:
        raise ValueError(f"无效的时间格式: '{time1}' - {str(e)}")
# def add_time_delta(time1, time_delta):
#
#     hour, minu = int(time1.split(":")[0]), int(time1.split(":")[1])
#
#     min_new = minu + time_delta
#
#     if min_new >= 60:
#         hour_new = hour + int(min_new / 60)
#         min_new = min_new % 60
#     else:
#         hour_new = hour
#
#     if hour_new < 10:
#         time_new = "0" + str(hour_new) + ":"
#     else:
#         time_new = str(hour_new) + ":"
#     if min_new < 10:
#
#         time_new = time_new + "0" + str(min_new)
#     else:
#         time_new = time_new + str(min_new)
#
#     return time_new

def calc_cost_from_itinerary_wo_intercity(itinerary, people_number):
    total_cost = 0
    for day in itinerary:
        for activity in day["activities"]:
            
            for transport in activity.get("transports", []):
                
                mode = transport["mode"]
                if mode=='taxi':
                    if 'cars' in transport.keys():
                        total_cost += transport.get('cars',0)*transport.get("cost", 0)
                    else:
                        total_cost += transport.get('tickets',0)*transport.get("cost", 0)
                if mode=='metro':
                    total_cost += transport.get('tickets',0)*transport.get("cost", 0)
                
            
            # if activity["type"] == "airplane":
            #     total_cost += activity.get('tickets',0)*activity.get("cost", 0)
            
            # if activity["type"] == "train":
            #     total_cost += activity.get('tickets',0)*activity.get("cost", 0)

            if activity["type"] == "breakfest" or activity["type"] == "lunch" or activity["type"] == "dinner":
                total_cost += activity.get('cost',0)*people_number
            
            # if activity["type"] == "accommodation":
            #     total_cost += activity.get('rooms',0)*activity.get("cost", 0)

            if activity["type"] == "attraction":
                total_cost += activity.get('tickets',0)*activity.get("cost", 0)
    return total_cost

def mmr_algorithm(name_list,score,lambda_value=0.3):
    selected_indices = []
    remaining_indices = list(range(len(name_list)))
        
    tfidf_vectorizer = TfidfVectorizer()

    while len(selected_indices) < len(name_list):
        if len(selected_indices) == 0:
            mmr_scores = np.ones(len(name_list))
        else:
            selected_names = [name.split()[0] for name in name_list[selected_indices]]
            remaining_names = [name.split()[0] for name in name_list[remaining_indices]]
            
            tfidf_matrix = tfidf_vectorizer.fit_transform(np.concatenate((selected_names, remaining_names)))
            similarity_matrix = cosine_similarity(tfidf_matrix)

            selected_similarities = similarity_matrix[:len(selected_names), len(selected_names):]
            remaining_similarities = similarity_matrix[len(selected_names):, len(selected_names):]

            mmr_scores = lambda_value*score[remaining_indices] - (1 - lambda_value) * np.max(selected_similarities, axis=0)

        max_index = np.argmax(mmr_scores)
        selected_indices.append(remaining_indices[max_index])
        del remaining_indices[max_index]

    return mmr_scores
