# -*- coding: utf-8 -*-

NEXT_POI_TYPE_INSTRUCTION = """ 
   You are a travel planning assistant. 
   The user's requirements are: {}. 
   Current travel plans are: {}. 
   Today is {}, current time is {}, current location is {}, and POI_type_list is {}. 
   Select the next POI type based on the user's needs and the current itinerary. 
   Please answer in the following format.
   Thought: [Your reason]
   Type: [type in POI_type_list]
    """

INTERCITY_TRANSPORT_GO_INSTRUCTION = """
    You are a travel planning assistant. 
    The user's requirements are: {user_requirements}. 
    Now let's plan the journey from the start city to the target city.
    The available transport options are: 
    {transport_info} 
    Your task is to rank all available intercity transport options based on the user's needs and the provided transport information. Consider the following factors:
    1. User preferences (e.g., type, comfort, cost, speed). 
    2. Availability and reliability of the transport options.

    Please provide a ranking list of all transport options based on the user's preferences. 
    
    For trains, include the TrainID.
    For flights, include the FlightID. 

    Your response should follow this format:
    
    Thought: [Your reasoning for selecting the transport options]
    IDList: [List of all IDs ranked by preference, where each ID is either a TrainID or FlightID, formatted as a Python list. The list should contain a maximum of 30 elements.]
    """

INTERCITY_TRANSPORT_BACK_INSTRUCTION = """
    You are a travel planning assistant. 
    The user's requirements are: {user_requirements}. 
    Now let's plan the return journey from the target city back to the start city.
    The available transport options are:
    {transport_info} 
    
    Additionally, here is the transport information for the start-to-target journey:
    {selected_go_info}

    Your task is to rank all available intercity transport options for the return journey based on the user's needs and the provided transport information. Consider the following factors:
    1. User preferences (e.g., type, comfort, cost, speed).
    2. Availability and reliability of the transport options.
    3. Consistency with the start-to-target journey (e.g., using the same mode of transport if preferred).
    4. Ensuring sufficient time for sightseeing and leisure activities in the target city. 
    
    Please provide a ranking list of all transport options based on the user's preferences. 
    
    For trains, include the TrainID.
    For flights, include the FlightID. 
    
    Your response should follow this format:
    
    Thought: [Your reasoning for ranking the transport options]
    IDList: [List of all IDs ranked by preference, where each ID is either a TrainID or FlightID, formatted as a Python list. The list should contain a maximum of 30 elements.]
    """
HOTEL_RANKING_INSTRUCTION = """
You are a travel planning assistant. 
The user's requirements are: {user_requirements}. 
Now let's select a suitable hotel in the target city.
The available hotel options are:
{hotel_info}

Your task is to rank all available hotel options based on the user's needs and the provided hotel information. Consider the following factors in this order of importance:
1. **User-specified preferred hotels** — if the user has clearly named a preferred hotel, and it is available, it must be ranked at the top.
2. **Hotel features and comfort**, especially scenic views, family-friendly options, and unique amenities.
3. **Number of beds per room (numbed)** — prioritize hotels that match the user's intended number of occupants per room. For example, if the user is traveling with 2 people per room, numbed=2 should be preferred.
4. **Price per night per room**, considering the total trip duration and number of rooms. Ensure that the total hotel cost fits reasonably within the user’s total budget (considering food and transportation).
5. **Proximity to key attractions or areas the user is interested in (e.g., near Jinji Lake, city center)**.

**Important**:
- If the user has specified a preferred hotel name, **that hotel should always be ranked first if it is available**.
- If the preferred hotel is not available, rank the most similar alternatives highest, considering location, number of beds, and comfort.
- Ensure the selected hotels align with the number of people and trip duration — total cost should not exceed budget constraints.

Please provide a ranking list of the hotel options based on the user's preferences.

Your response should follow this format:

Thought: [Your reasoning for ranking the hotel options]
HotelNameList: [List of all hotel names ranked by preference, formatted as a Python list]

Example:
Thought: The user prefers Hotel A, which is available and fits their budget. Hotel B and Hotel C also offer double beds and are near the desired area, making them good alternatives.
HotelNameList: ["Hotel A", "Hotel B", "Hotel C", ...]
"""



# HOTEL_RANKING_INSTRUCTION = """
#     You are a travel planning assistant.
#     The user's requirements are: {user_requirements}.
#     Now let's select a suitable hotel in the target city.
#     The available hotel options are:
#     {hotel_info}
#
#     Your task is to rank all available hotel options based on the user's needs and the provided hotel information. Consider the following factors:
#     1. User preferences (e.g., comfort, cost, location).
#     2. Hotel features.
#     3. Room price per night.
#     4. Number of beds per room (numbed=2 for double beds, numbed=1 for single beds).
#     5. Proximity to key attractions or points of interest in the target city.
#
#     Additionally, keep in mind that the user's budget is allocated across multiple expenses, including intercity transportation and daily meals. Ensure that the hotel recommendations fit within the remaining budget constraints after accounting for these costs. Note that the price provided for each hotel is the cost per night per room. If the user has provided a specific budget requirement, ensure that the total cost of the hotel stay, including intercity transportation and daily meals, does not exceed this budget. Leave sufficient space in the budget for daily meals and other travel expenses.
#
#     Please provide a ranking list of all hotel options based on the user's preferences.
#
#     For each hotel, include the name.
#
#     Your response should follow this format:
#
#     Thought: [Your reasoning for ranking the hotel options]
#     HotelNameList: [List of all Hotel name ranked by preference, formatted as a Python list]
#
#     Example:
#     Thought: Based on the user's preference for comfort and proximity to key attractions, the hotels are ranked as follows:
#     HotelNameList: ["hotel1", "hotel2", ...]
#
#     """
ROOMS_PLANNING_INSTRUCTION = """
    You are a travel planning assistant.
    The user's requirements are: {user_requirements}.
    The hard_logic are:{hard_logic}.

    Your task is to extract the following information from the user's requirements and the hard_logic:
    1. Number of rooms requested.
    2. Number of beds per room.

    Room types and corresponding bed counts:
    - Single bed room: 1 bed
    - Double bed room: 2 beds
    - Twin room: 2 beds
    - King bed room: 1 bed
    
    Auto-Calculation (if missing):
       - If rooms ≠ -1 but beds = -1:
         → beds = ceil(people_number / rooms)
       - If beds ≠ -1 but rooms = -1:
         → rooms = ceil(people_number / beds)
       - Use people_number from hard_logic (e.g., "people_number==4")
    If the user's requirements and the hard_logic do not specify the number of rooms or the type of room, default to -1 for both values.
    Your response should follow this format:
    Thought: [Your reasoning for extracting the information]
    RoomInfo: [Number of rooms, Number of beds per room]
    """

BUDGETS_INSTRUCTION = """
    You are a travel planning assistant.
    The user's requirements are: {user_requirements}.
    The hard_logic are: {hard_logic}.

Your task is to extract the total budget information **only if** it is clearly specified in the hard_logic in the form of `total_cost = [value]` or `total_cost <= [value]`.

- If such a statement is present in the hard_logic, extract the numeric value as the budget.
- If there is no such expression about `total_cost`, return nothing.

Ignore any other budget-related information in the natural language if it's not explicitly included in hard_logic.

Output format (if applicable):

Budget: [Extracted budget as a number]
"""

INNERCITY_TRANSPORTS_SELECTION_INSTRUCTION = """
    You are a travel planning assistant. 
    The user's requirements are: {user_requirements}. 

    Your task is to extract the preferred mode of urban transportation from the user's requirements and rank the following transportation options based on user preferences:
    1. Metro
    2. Taxi
    3. Walk

    IMPORTANT RULES:
    - If the user explicitly mentions NOT wanting a specific mode (e.g., "不希望打车", "不要taxi", "avoid taxi"), completely EXCLUDE that mode from the ranking
    - If the user specifies a preferred mode, prioritize that mode and include only compatible options
    - If no specific preferences are mentioned, use the default ranking: ["metro", "taxi", "walk"]
    - The final ranking list does NOT need to contain all three options - only include modes that match user preferences

    Your response should follow this format:

    Thought: [Your reasoning for ranking the transportation options, including any exclusions based on user preferences]
    TransportRanking: [List of transportation options ranked by preference, formatted as a Python list. Only include modes that are acceptable to the user.]
    """
# INNERCITY_TRANSPORTS_SELECTION_INSTRUCTION = """
#     You are a travel planning assistant.
#     The user's requirements are: {user_requirements}.
#
#     Your task is to extract the preferred mode of urban transportation from the user's requirements and rank the following transportation options based on user preferences:
#     1. Metro
#     2. Taxi
#     3. Walk
#
#     The user's requirements may specify a preferred mode of transportation or provide hints about their preferences. If a specific mode is mentioned, only include that mode in the ranking. If no specific mode is mentioned, rank the options based on common sense and typical user preferences, with the following default ranking: ["metro", "taxi", "walk"]
#
#     Your response should follow this format:
#
#     Thought: [Your reasoning for ranking the transportation options]
#     TransportRanking: [List of transportation options ranked by preference, formatted as a Python list]
#     """
ATTRACTION_RANKING_INSTRUCTION = """
    You are a travel planning assistant. 
    The user's requirements are: {user_requirements}. 
    The attraction info is:
    {attraction_info}
    The past cost for intercity transportation and hotel accommodations is: {past_cost}.
    
    Your task is to select and rank attractions based on the user's needs and the provided attraction information. Consider the following factors:
    1. Attraction name
    2. Attraction type
    3. Location
    4. Recommended duration
    
    Additionally, keep in mind that the user's budget is allocated across multiple expenses, including intercity 
    transportation and hotel accommodations. Ensure that the attraction recommendations fit within the remaining budget
    constraints after accounting for the past cost.
    
    For each day, recommend at least 8 attractions, combining attractions for all days together. To ensure a 
    comprehensive list, consider a larger pool of candidates and prioritize diversity in attraction type and location.
    
    Your response should follow this format:
    
    Thought: [Your reasoning for ranking the attractions]
    AttractionNameList: [List of attraction names ranked by preference, formatted as a Python list]

    Example:
    Thought: Based on the user's preference for historical sites and natural attractions, the attractions are ranked as follows:
    AttractionNameList: ["Attraction1", "Attraction2", ...]
    """
RESTAURANT_RANKING_INSTRUCTION = """
You are a travel planning assistant.
The user's requirements are: {user_requirements}.
The restaurant info is:{restaurant_info}
Additional context:
- The total cost already spent on intercity transportation and hotel accommodations is: {past_cost} (in CNY).
- The user's remaining budget must cover food expenses for the entire trip (3 people × 3 meals × {days} days).
- The price range provided for each restaurant represents the average cost per person per meal.

Your task:
Based on the user’s travel preferences and the restaurant data, select and rank restaurants that best meet the user's needs. Your recommendation should balance:
1. Restaurant name and reputation
2. Cuisine type and variety
3. Recommended dishes (highlight local/unique/featured items)
4. Price per person, ensuring affordability within the remaining food budget

Instructions:
- Recommend a total of at least 6 different restaurants (these can be distributed across multiple days)
- Prioritize restaurants located near or within commercial/shopping districts, if possible
- Ensure the selections do not exceed the total food budget after subtracting the past cost
- Focus on a mix of experience (e.g., fine dining + affordable gems) if the budget allows
    
Your response should follow this format:
Thought: [Your reasoning for ranking the restaurants]
RestaurantNameList: [List of restaurant names ranked by preference, formatted as a Python list]
"""

# RESTAURANT_RANKING_INSTRUCTION = """
#     You are a travel planning assistant.
#     The user's requirements are: {user_requirements}.
#     The restaurant info is:
#     {restaurant_info}
#     The past cost for intercity transportation and hotel accommodations is: {past_cost}.
#
#     Your task is to select and rank restaurants based on the user's needs and the provided restaurant information. Consider the following factors:
#     1. Restaurant name
#     2. Cuisine type
#     3. Price range
#     4. Recommended food
#
#     Additionally, keep in mind that the user's budget is allocated across multiple expenses, including intercity transportation and hotel accommodations. Ensure that the restaurant recommendations fit within the remaining budget constraints after accounting for the past cost.
#     Note that the price range provided for each restaurant is the average cost per person per meal, the remaining budget must cover the cost of three meals per day for {days} days.
#
#     For each day, recommend at least 6 restaurants, combining restaurants for all days together.
#
#     Your response should follow this format:
#
#     Thought: [Your reasoning for ranking the restaurants]
#     RestaurantNameList: [List of restaurant names ranked by preference, formatted as a Python list]
#     """


SELECT_POI_TIME_INSTRUCTION = """
    You are a travel planning assistant. 
    The user's requirements are: {user_requirements}. 
    Current travel plans are: {current_travel_plans}. 
    Today is {current_date}, current time is {current_time}, current visiting POI is {current_poi}, and its type is {poi_type}.
    The recommended visit time for the current POI is {recommended_visit_time} minutes.
    The maximum time is {recommendmax_time} minutes, and you must not exceed it.    
    If the parameter {activity_time} is not None and the {current_poi} is explicitly mentioned in {user_requirements} 
    together with a clear minimum stay requirement, then the visit time must be at least {activity_time} minutes. 
    If the {current_poi} is mentioned without any explicit minimum stay requirement, this rule does not apply.

    
    The user has the following time constraints:
    - Lunch time: 11:00-13:00
    - Dinner time: 17:00-20:00
    - Return to hotel by 22:00 (if not the last day of the trip)
    - If today is the last day of the trip, the return transport (train/flight) starts at {back_transport_time}.
              
    Your task is to select the time for the current POI based on the user's needs, current travel plans, and the provided information. Consider the following factors:
    1. User preferences
    2. Current travel plans
    3. POI type
    4. Recommended visit time for the current POI
    5. Time constraints for lunch, dinner, and return to hotel (if not the last day)
    6. If today is the last day, the return transport time
    
    The default value for the POI visit time is 90 minutes and can be adjusted based on the user's needs.
    
    Your response should follow this format:
    
    Thought: [Your reasoning for selecting the POI visit time]
    Time: [Time in minutes (Just INT value)]
    """

nl2sl_prompt = """
You need to extract the following 4 values from the natural language query:

- start_city: the departure city  
- target_city: the destination city  
- days: the number of travel days  
- people_number: number of travelers

Then, transform the query into a list of constraints named `hard_logic`. Each item must follow the format:

    variable operator value

For example: `"days==3"` means the trip lasts 3 days.

You can use the following 21 variables:

(1)  days: number of travel days  
     Format: `"days==n"`

(2)  people_number: number of travelers  
     Format: `"people_number==n"`

(3)  total_cost: overall trip budget  
     Format: `"total_cost<=n"`

(4)  tickets: number of tickets  
     Format: `"tickets==n"`

(5) rooms: an int value of the number of rooms the user needs to book.  
"rooms==n" means the user wants to book n rooms.  
Rules:  
- Only output the room number when the user explicitly specifies the quantity.  
- If the user doesn't clearly state how many rooms they need, do not write the rooms field.

(6) room_type: number of beds per room.  
Format: "room_type==n".  
Rules:  
- If the user specifies "single bed room" (单床房或大床房), set room_type=1.  
- If the user specifies "double bed room" (双床房或标间), set room_type=2, etc.  
- If not explicitly mentioned, leave `room_type` unspecified unless it can be inferred from context.

(7)  hotel_feature: hotel features  
    Format:
    For features the user wants: {"A", "B"}<=hotel_feature
    For features the user does not want: hotel_feature<={"A", "B"}
     Docs: Get the feature of accommodation in the target city. We only support ['儿童俱乐部', '空气净化器', '山景房', '私汤房', '四合院', 
     '温泉', '湖畔美居', '电竞酒店', '温泉泡汤', '行政酒廊', '充电桩', '设计师酒店', '民宿', '湖景房', '动人夜景', '行李寄存', '中式庭院', '桌球室', 
     '私人泳池', '钓鱼', '迷人海景', '园林建筑', '老洋房', '儿童泳池', '历史名宅', '棋牌室', '智能客控', '情侣房', '小而美', '特色 住宿', '茶室', 
     '亲子主题房', '多功能厅', '洗衣房', '客栈', '自营亲子房', '停车场', 'Boss推荐', '江河景房', '日光浴场', '自营影音房', '厨房', '空调', 
     '网红泳池', '别墅', '免费停车', '洗衣服务', '窗外好景', '酒店公寓', '会议厅', '家庭房', '24小时前台', '商务中心', '提前入园', '农家乐', 
     '智能马桶', '美食酒店', 'SPA', '拍照出片', '海景房', '泳池', '影音房', '管家服务', '穿梭机场班车', '桑拿', '机器人服务', '儿童乐园', 
     '健身室', '洗衣机', '自营舒睡房', '宠物友好', '电竞房', '位置超好', '套房'].

**Hotel Quality and Comfort Mapping:**
When users mention hotel quality preferences, automatically map to appropriate features:

**Comfort and Atmosphere Preferences:**
- "温馨一点", "住的舒服点", "住得比较的舒适些" → {"小而美", "窗外好景", "智能客控", "自营舒睡房", "动人夜景"}
- "希望酒店的位置比较安静" → {"窗外好景", "动人夜景", "园林建筑"} + hotel_feature<={"临街房", "闹市房"}

**Specific Brand Preferences:**
- "酒店想要住亚朵或者全季" → {"小而美", "智能客控", "自营舒睡房", "洗衣服务", "24小时前台"}
- 亚朵酒店特征: {"小而美", "智能客控", "自营舒睡房", "洗衣服务", "24小时前台", "健身室"}
- 全季酒店特征: {"小而美", "茶室", "智能客控", "自营舒睡房", "24小时前台"}

**Location Convenience Preferences:**
- "位置需要方便的地方", "位置超好" → {"位置超好", "免费停车", "穿梭机场班车", "停车场"}

**Budget and Value Preferences:**
- "住宿希望能够稍微好一点" → {"智能客控", "自营舒睡房", "洗衣服务", "24小时前台"}
- "住的地方想挑性价比高的" → {"小而美", "免费停车", "洗衣服务", "24小时前台"}

**Comfort Feature Clusters:**
**Basic Comfort**: {"空调", "24小时前台", "行李寄存", "免费停车"}
**Enhanced Comfort**: {"智能客控", "自营舒睡房", "洗衣服务", "窗外好景"}
**Premium Comfort**: {"行政酒廊", "管家服务", "SPA", "健身室", "泳池"}
**Family Comfort**: {"家庭房", "亲子主题房", "儿童乐园", "洗衣房"}

Rules:
Only apply if the feature is explicitly stated by the user.
Multiple features go in {} (e.g., {"温泉", "亲子主题房"}).
Never alter names (exact match required).
If both preferred and excluded features exist, list separately (e.g., {"泳池"}<=hotel_feature, hotel_feature<={"民宿"}).

**Special Mapping Rules:**
- "温馨一点" → {"小而美", "窗外好景", "智能客控"}
- "四星五星级酒店" → {"行政酒廊", "健身室", "泳池", "SPA"}
- "住宿希望能够稍微好一点" → {"智能客控", "自营舒睡房", "洗衣服务"}
- "酒店一般般就行" → {"小而美", "免费停车"} + hotel_feature<={"行政酒廊", "泳池"}
- "要住得比较的舒适些" → {"智能客控", "自营舒睡房", "窗外好景", "洗衣服务"}
- "酒店要好一点" → {"行政酒廊", "健身室", "泳池", "管家服务"}
- "酒店不需要住太好" → hotel_feature<={"行政酒廊", "泳池", "SPA", "管家服务"}
- "住的地方想挑性价比高的经济型酒店" → {"小而美", "免费停车", "洗衣服务"}
- "位置需要方便的地方" → {"位置超好", "免费停车"}
- "住宿便宜" → {"小而美", "免费停车"} + hotel_feature<={"行政酒廊", "泳池", "SPA"}
- "希望酒店的位置比较安静" → {"窗外好景", "动人夜景"}
- "住三星级以上" → {"健身室", "商务中心", "24小时前台"}
- "酒店想要住亚朵或者全季" → {"小而美", "智能客控", "自营舒睡房", "洗衣服务"}
- "住的舒服点" → {"智能客控", "自营舒睡房", "窗外好景", "洗衣服务"}
  
(8)  hotel_price: average nightly price  
     Format: `"hotel_price<=n"`
(9)  intercity_transport_go: transportation from origin to destination  
Format: `"intercity_transport=={'train'}"`  
If the user is flexible and can take either the 'train' OR the 'airplane' for the outbound journey, DO NOT include this constraint.
Only if the user specifies a strict preference for one specific mode (only 'train' or only 'airplane') for this leg of the trip,
use the format: "intercity_transport_go=={'train'}" or "intercity_transport_go=={'airplane'}".
Valid values: ['train', 'airplane']
(10)  intercity_transport_back: transportation form destination to origin  
Format: `"intercity_transport=={'train'}"`  
If the user is flexible and can take either the 'train' OR the 'airplane' for the return journey, DO NOT include this constraint.
Only if the user specifies a strict preference for one specific mode (only 'train' or only 'airplane') for this leg of the trip, 
use the format: "intercity_transport_back=={'train'}" or "intercity_transport_back=={'airplane'}".
Valid values: ['train', 'airplane']
(11) transport_type: in-city transport preferences  
     Format: `"transport_type<={'A'}"`  
     Valid values: ['metro', 'taxi', 'walk']

(12) spot_type: types of attractions the user wants to visit or avoid.
    Format:  
    - For attraction types the user **wants** to visit: `{"A", "B"}<=spot_type`  
    - For attraction types the user explicitly **does not want** to visit: `spot_type<={"A", "B"}`  
    Allowed values (must match exactly):  
    ['博物馆/纪念馆', '美术馆/艺术馆', '红色景点', '自然风光', '人文景观', '大学校园', '历史古迹', '游乐园/体育娱乐', '图书馆', '园林', '其它', '文化旅游区', '公园', '商业街区'].

**Educational and Scenic Spot Mapping:**
When users mention preferences for educational experiences or scenic spots, automatically map to the following attraction types:

**For "见见世面" (broaden horizons) or "教育意义" (educational value):**
- 博物馆/纪念馆 (Museums/Memorials - for cultural and historical education)
- 美术馆/艺术馆 (Art galleries - for artistic appreciation)
- 历史古迹 (Historical sites - for understanding history)
- 人文景观 (Human landscapes - for cultural experiences)

**For "大好河山" (beautiful landscapes) or "自然风光" (natural scenery):**
- 自然风光 (Natural scenery - mountains, rivers, landscapes)
- 公园 (Parks - well-maintained natural spaces)
- 园林 (Gardens - traditional Chinese gardens)
- 文化旅游区 (Cultural tourism areas - scenic spots with cultural elements)

**For Family with Children Considerations:**
- 游乐园/体育娱乐 (Amusement parks/Sports entertainment - for child-friendly activities)
- 公园 (Parks - suitable for family outings)
- 文化旅游区 (Cultural tourism areas - educational and enjoyable)

Rules:  
- Only use these formats if the attraction type is explicitly mentioned in user requirements.  
- Multiple types should be wrapped in curly braces `{}` and separated by commas.  
- If both wanted and unwanted types are provided, output both constraints separately.
- **Special Rule for Educational Preferences**: When users mention "见见世面" or educational purposes, automatically include: {"博物馆/纪念馆", "历史古迹", "人文景观"}<=spot_type
- **Special Rule for Scenic Preferences**: When users mention "大好河山" or natural beauty, automatically include: {"自然风光", "公园", "文化旅游区"}<=spot_type
- **Special Rule for Family Travel**: When traveling with children, automatically include: {"游乐园/体育娱乐", "公园"}<=spot_type

(13) attraction_names: names of attractions to visit or avoid.  
Format:  
- For attractions the user **wants** to visit: `{"A", "B"}<=attraction_names`  
- For attractions the user explicitly **does not want** to visit: `attraction_names<={"A", "B"}`  

**Keyword Extraction and Fuzzy Matching:**
When users mention specific interests that don't fit into standard spot_types, extract keywords and map to relevant attraction names:

**Common Interest Keywords Mapping:**
- "购物", "买东西", "逛街" → include shopping-related attraction names containing: {"商场", "购物中心", "商业街", "步行街", "市场"}
- "工艺品", "手工艺", "纪念品" → include craft-related attraction names containing: {"工艺", "手工艺", "文创", "纪念品", "艺术品"}
- "美食", "小吃", "特色菜" → include food-related attraction names containing: {"美食", "小吃", "餐饮", "食街"}
- "夜景", "夜游" → include night view attraction names containing: {"夜景", "夜游", "灯光秀"}
- "拍照", "打卡" → include photogenic attraction names containing: {"拍照", "打卡", "观景台"}
- "亲子", "儿童" → include family-friendly attraction names containing: {"儿童", "亲子", "乐园", "公园"}

**Keyword Extraction Rules:**
1. Extract prominent keywords from user requirements that indicate specific interests
2. Map keywords to relevant attraction name patterns
3. Include attractions whose names contain these keywords
4. Only apply when keywords don't clearly fit into existing spot_types

**Examples:**
- User: "要购物" → `{"商场", "购物中心", "商业街"}<=attraction_names`
- User: "带一些工艺品回去" → `{"工艺", "手工艺", "文创"}<=attraction_names`
- User: "想买特产和纪念品" → `{"特产", "纪念品", "工艺品"}<=attraction_names`
- User: "拍点好看的照片" → `{"拍照", "打卡", "观景台"}<=attraction_names`

Rules:  
- Only use these formats if the attraction name or clear keyword is explicitly mentioned in user requirements.  
- Multiple attractions should be wrapped in curly braces `{}` and separated by commas.  
- For keyword-based matching, use partial matching patterns rather than exact names
- If both wanted and unwanted attractions are provided, output both constraints separately.

**Special Keyword Processing:**
- When users mention interests that aren't covered by spot_types, extract 2-3 character keywords
- Map keywords to commonly associated attraction name patterns
- Use partial matching with Chinese keywords to capture relevant attractions

(14) restaurant_names: names of restaurants to visit or avoid
Format:
For restaurants the user wants to visit: {"A", "B"}<=restaurant_names
For restaurants the user does not want to visit: restaurant_names<={"A", "B"}

Rules:
Only use these formats if the restaurant name is explicitly mentioned in user requirements.
Multiple restaurants should be wrapped in {} and separated by commas (e.g., {"A", "B"}).
Preserve exact names (do not translate or modify).
If both wanted and unwanted restaurants are given, output separate constraints (e.g., {"A"}<=restaurant_names, restaurant_names<={"B"}).

(15) hotel_names: hotel name constraints  
Format:
For hotels the user wants to stay at: {"A"}<=hotel_names
For hotels the user does not want to stay at: hotel_names<={"A"}

Rules:
Only apply if the hotel name is explicitly stated by the user.
Multiple hotels go in {} (e.g., {"A", "B"}).
Never alter names (use exact spelling/case).
If preferences include both wanted and unwanted hotels, list separately (e.g., {"X"}<=hotel_names, hotel_names<={"Y"}).

(16) food_type: types of cuisine
Format:
For cuisines the user wants: {"A", "B"}<=food_type
For cuisines the user does not want: food_type<={"A", "B"}
Allowed values (must match exactly):
['云南菜', '西藏菜', '东北菜', '烧烤', '亚洲菜', '粤菜',
'西北菜', '闽菜', '客家菜', '快餐简餐', '川菜', '台湾菜', '其他', '清真菜', '小吃', '西餐', '素食', '日本料理', '江浙菜', '湖北菜', '东南亚菜',
'湘菜', '北京菜', '韩国料理', '海鲜', '中东料理', '融合菜', '茶馆/茶室', '酒吧/酒馆', '创意菜', '自助餐', '咖啡店', '本帮菜', '徽菜', '拉美料理',
'鲁菜', '新疆菜', '农家菜', '海南菜', '火锅', '面包甜点', '其他中餐'].
Spicy Cuisine Mapping:
When users mention preferences for "spicy" or "辣" food, automatically map to the following spicy cuisine types:
川菜 (Sichuan cuisine - famously spicy and numbing)
湘菜 (Hunan cuisine - known for fresh and spicy flavors)
火锅 (Hot pot - often with spicy broth)
韩国料理 (Korean cuisine - includes spicy dishes like kimchi and tteokbokki)
东南亚菜 (Southeast Asian cuisine - includes Thai, Vietnamese with spicy elements)
新疆菜 (Xinjiang cuisine - often includes spicy seasonings)
云南菜 (Yunnan cuisine - some dishes are spicy)

Non-Spicy Preference Handling:
When users mention "不吃太辣", "不要辣的", "avoid spicy food", or similar expressions indicating avoidance of spicy food, automatically exclude the main spicy cuisine types:
food_type<={"川菜", "湘菜", "韩国料理", "东南亚菜", "新疆菜"}

Sweet Cuisine Avoidance Handling:
When users mention "不能吃甜", "不要太甜", "avoid sweet food", or similar expressions indicating avoidance of sweet food, automatically exclude the following cuisine types known for sweeter flavors:
food_type<={"本帮菜", "江浙菜", "粤菜", "台湾菜", "面包甜点", "咖啡店", "茶馆/茶室", "融合菜"}
Note: 本帮菜 and 江浙菜 are particularly known for their sweet and savory flavor profiles

Local Cuisine Mapping:
When users mention "当地特色", "本地美食", "try local food", or use specific local flavor terms, automatically include the corresponding local cuisine:
For Beijing: {"北京菜"}<=food_type (including terms: "京味", "北京特色", "老北京味道")
For Shanghai: {"本帮菜"}<=food_type (including terms: "上海特色", "沪味")
For Sichuan: {"川菜"}<=food_type (including terms: "四川味道", "麻辣")
For Guangdong: {"粤菜"}<=food_type (including terms: "广东菜", "广府菜", "潮汕菜")
For Jiangsu/Zhejiang: {"江浙菜"}<=food_type (including terms: "江浙风味", "苏杭菜")
For Shandong: {"鲁菜"}<=food_type (including terms: "山东菜", "鲁味")
For Hunan: {"湘菜"}<=food_type (including terms: "湖南菜", "湘味")
For other cities, use the appropriate regional cuisine based on the destination
Regional Flavor Term Mapping:
When users mention specific regional flavor terms, map to corresponding cuisines:
"京味"/"北京特色"/"老北京" → 北京菜
"本帮"/"上海特色"/"沪味" → 本帮菜
"川味"/"麻辣"/"四川特色" → 川菜
"粤式"/"广式"/"广东特色" → 粤菜
"江浙风味"/"苏杭味道" → 江浙菜
"鲁味"/"山东特色" → 鲁菜
"湘味"/"湖南特色" → 湘菜
"徽味"/"安徽特色" → 徽菜
"闽味"/"福建特色" → 闽菜
Rules:
Only use these formats if the cuisine type is explicitly mentioned by the user.
Multiple types should be wrapped in {} (e.g., {"火锅", "川菜"}).
Preserve exact names (no translation/modification).
If both wanted and unwanted types are given, output separate constraints (e.g., {"川菜"}<=food_type, food_type<={"西餐"}).
Special Rule for Spicy Preferences: When users express preference for spicy food, automatically include: {"川菜", "湘菜", "火锅", "韩国料理", "东南亚菜"}<=food_type
Special Rule for Non-Spicy Preferences: When users express avoidance of spicy food, automatically exclude: food_type<={"川菜", "湘菜", "韩国料理", "东南亚菜", "新疆菜"}
Special Rule for Sweet Avoidance: When users express avoidance of sweet food, automatically exclude: food_type<={"本帮菜", "江浙菜", "粤菜", "台湾菜", "面包甜点", "咖啡店", "茶馆/茶室", "融合菜"}
Special Rule for Local Cuisine: When users want to try local specialties or use regional flavor terms, automatically include the appropriate regional cuisine based on target city and mentioned terms
Term Mapping Priority: When users mention both general preferences and specific regional terms, apply all relevant mappings
Conflicting Preferences: If users mention conflicting preferences (e.g., want local Shanghai food but avoid sweet), prioritize the avoidance constraint and exclude the conflicting cuisine types

(17) food_price: average food budget per meal  
     Format: `"food_price<=n"`

(18) taxi_cars: exact taxi count calculation  
Format: "taxi_cars==((people_number + 3) // 4)"  
Rule: Number of taxis must equal (people_number + 3) // 4  
Constraint: Use exactly this number of taxis - no more, no less.
Special Case: For 4 people, taxi count must be exactly 1

(19) attraction_price: per-attraction cost  
     Format: `"attraction_price<=n"`  
     For "free-only attractions", use `"attraction_price<=0"`

(20) activity_start_time_constraint: maximum start time for specific activity  
Format: `"start_time{清平古墟影视文化园}<=08:20"`
Rule: The activity at '清平古墟影视文化园' must start no later than 08:20  
Constraint: activity_start_time <= "08:20" for this specific venue only  
Time format: HH:MM (24-hour format)

(21) activity_end_time: minimum end time for specific activity  
Format: `"end_time{三郎日料•烧肉酒场(文晖店)}>=17:50"`  
Rule: The activity at '三郎日料•烧肉酒场(文晖店)' must end no earlier than 17:50  
Constraint: activity_end_time >= "17:50" for this specific venue only  
Time format: HH:MM (24-hour format)

(22) activity_time_range: specific time window for an activity
Format: `"time_range{茅家埠景区,08:50-10:20}"`
Rule: The activity at '茅家埠景区' must start no earlier than 08:50 AND end no later than 10:20
Constraint: activity_start_time >= "08:50" AND activity_end_time <= "10:20"

(23) activity_time: minimum duration for specific activity
    Format: "activity_time{滨海文化公园}>=90"
    Rule: The activity at '滨海文化公园' must last at least 90 minutes
    Constraint: activity_duration >= 90 minutes for this specific venue only
    Unit: minutes
(24) attraction_cost: total cost for attractions  
     Format: `"attraction_cost<=n"`  
     Calculated as: `sum of prices for all attraction activities`
(25) food_cost: total cost of food during the trip  
 Format: `"food_cost<=n"`  
 Calculated as: `sum of all breakfast/lunch/dinner activity prices`
**Special Interpretation for "吃一些平时吃不到的":**
When users mention wanting to "eat something they don't usually have" or "尝试平时吃不到的美食", this implies:
1. **Higher budget allocation** for unique dining experiences
2. **Premium and exotic cuisine types** should be prioritized
3. **Fine dining and specialty restaurants** over casual everyday options
**Budget Allocation Rules:**
- When users specify a food budget significantly higher than typical daily meals, interpret as desire for premium dining experiences
- For "平时吃不到的" experiences, focus on: 
  - Fine dining restaurants
  - Authentic regional specialty cuisines
  - Unique ethnic foods (e.g., 中东料理, 拉美料理, 创意菜)
  - High-end buffet and gourmet experiences
  - Traditional banquet-style dining
**Application Example:**
User: "准备花费7000元吃一些平时吃不到的"
→ Interpretation: User wants to allocate 7000元 for premium, unique dining experiences beyond everyday meals
→ Constraint: `"food_cost<=7000"` with emphasis on exotic, high-quality cuisine types

(26)  hotel_cost: total cost for accommodation  
 Format: `"hotel_cost<=n"`  
 Calculated as: `sum(room_count(activity) × activity_price(activity))`
(27) intercity_transport_cost: total budget constraint for all intercity transportation  
Format: "intercity_transport_cost<=1200.0"  
Calculation: Sum the cost of ALL intercity transport activities (both go and return trips) in the entire plan  
Constraint: The total cost of all intercity transportation must not exceed 1200.0  
Scope: Includes airplane tickets, train tickets, and any other intercity transport costs  
Time scope: Applies to the entire travel duration (all days)  
People factor: Cost should be calculated for all people (cost per person × number of people)
(28) innercity_transport_cost: total cost for all innercity transports (e.g., metro, taxi)  
Format: "innercity_transport_cost<=n"  
You should sum up the cost of all innercity transport modes (retrieved via activity_transports) in the plan.
(29)poi_distance: if the distance between two POIs exceeds X km, take a taxi. 
     Format: `"distance>X"`
(30)accommodation_distance: maximum distance from specific point of interest  
Format: "accommodation_distance{客家土楼(福建)}<=10.41"  
Rule: The accommodation must be located within 10.41 kilometers of 客家土楼(福建)  
Constraint: poi_distance(accommodation_position, 客家土楼(福建)) <= 10.41  
Unit: kilometers
(31) activity_order: specific order requirement between activities
Format: "activity_order{A-B}"
Rule: Activity A must occur BEFORE activity B
Constraint: activity_start_time(A) < activity_start_time(B) for the specified venues
Symbol: A-B means "A before B"
Rules:
- Only use when explicit order requirements are mentioned
- Use hyphen '-' to separate activities, first activity comes before second
- Use exact venue names as specified by the user
- Multiple order constraints should be listed separately
---

Rules:
    If the trip is only 1 day, ignore rooms, room_type, and other irrelevant constraints.
    If the query includes constraints not listed above, you can still include them in hard_logic.
    Output must be valid JSON.
    Pay attention to the format and examples above.
    
    Format rules for constraints:
    - SINGLE condition (no OR): DO NOT use parentheses
      Example: {'A','B'}<=restaurant_names
      Example: attraction_price<=100
      
    - MULTIPLE conditions with OR: wrap EACH condition in parentheses
      Example: ({'A','B'}<=restaurant_names) or (attraction_price<=100)
      Example: ({'公园'}<=attraction_type) or (attraction_price<=0) or (attraction_time<=60)
    
    - FORMAT COMPACTNESS: No spaces in constraints
      Correct: {'A','B'}<=restaurant_names
      Incorrect: { ' A ', ' B ' } <= restaurant_names
      Correct: days==3
      Incorrect: days == 3
      Correct: total_cost<=1000
      Incorrect: total_cost <= 1000
      Correct: ({'A'}<=restaurant_names) or (total_cost<=1000)
      Incorrect: ( { 'A' } <= restaurant_names ) or ( total_cost <= 1000 )
    Important: A single set condition like {'A','B'}<=variable is still a SINGLE condition and should NOT have parentheses.
"""

nl2sl_example = "Examples:\n"

nl2sl_example_1 = """
nature_language: 当前位置上海。我和女朋友打算去苏州玩两天，预算1300元，希望酒店每晚不超过500元，开一间单床房。请给我一个旅行规划。
Answer: {'start_city': "上海", 'target_city': "苏州", 'days': 2, 'people_number': 2, 'hard_logic':  ['days==2', 'people_number==2', 'cost<=1300', 'hotel_price<=500', 'tickets==2', 'rooms==1', 'room_type==1', 'taxi_cars==1']}
"""
nl2sl_example_2 = """
nature_language: 当前位置上海。我们三个人打算去北京玩两天，想去北京全聚德(前门店)吃饭，预算6000元，开两间双床房。请给我一个旅行规划。
Answer: {'start_city': "上海", 'target_city': "北京", 'days': 2, 'people_number': 3, 'hard_logic': ['days==2', 'people_number==3', 'cost<=6000', "{'北京全聚德(前门店)'} <= restaurant_names", 'tickets==3', 'rooms==2', 'taxi_cars==1','room_type==2']}
"""
nl2sl_example_3 = """
nature_language: 当前位置重庆。我一个人想去杭州玩2天，坐高铁（G），预算3000人民币，喜欢自然风光，住一间单床且有智能客控的酒店，人均每顿饭不超过100元，尽可能坐地铁，请给我一个旅行规划。
Answer: {'start_city': '成都', 'target_city': '杭州', 'days': 2, 'people_number': 1, 'hard_logic': ['days==2', 'people_number==1', 'cost<=3000', 'tickets==1', 'rooms==1', 'room_type==1', "intercity_transport_go=={'train'}", "{'自然风光'}<=spot_type", "{'智能客控'}<=hotel_feature", 'food_price<=100', "transport_type<={'metro'}" ]}
"""
nl2sl_example_4 = """
nature_language: 当前位置苏州。我和我的朋友想去北京玩3天，预算8000人民币，坐火车去，想吃北京菜，想去故宫博物院看看，住的酒店最好有管家服务。
Answer: {'start_city': '上海', 'target_city': '北京', 'days': 3, 'people_number': 2, 'hard_logic': ['days==3', 'people_number==2', 'cost<=8000', 'tickets==2', , 'taxi_cars==1', "intercity_transport_go=={'train'}", "{'北京菜'}<=food_type", "{'故宫博物院'}<=attraction_names", "{'管家服务'}<=hotel_feature"]}
"""
nl2sl_example_5 = """
nature_language: 我们2人，从成都出发，到苏州旅行2天，希望只游览免费景点
Answer: {'start_city': "成都", 'target_city': "苏州", 'days': 2, 'people_number': 2, 'hard_logic': ['days==2', 
    'people_number==2',
    'attraction_price<=0',
    'tickets==2',
    'taxi_cars==1']
}
"""
nl2sl_example_6 = """
nature_language:我们3人，从南京出发，到杭州旅行4天，要求如下：\n在用餐上的预算为2300.0
Answer:{'start_city': "南京", 'target_city': "杭州", 'days': 4, 'people_number': 3, 'hard_logic":
        ['days==4', 'people_number==3', 'food_cost<=2300.0', 'tickets==3', 'taxi_cars==1']}
"""
nl2sl_example_7 =""""nature_language": "我们4人，从南京出发，到武汉旅行3天，要求如下：\n不希望乘坐airplane前往目的地，不希望乘坐airplane返回",
    "hard_logic": [
        "days==3",
        "people_number==4",
        "intercity_transport_go=={'train'}",
        "intercity_transport_back=={'train'}",
        "tickets==4",
        "taxi_cars==1"
    ]
"""
nl2sl_example_8 = """
"nature_language": "我们5人，从深圳出发，到上海旅行2天，要求如下：\n希望游览大宁公园 和 观光夜市(威尼斯水城之夜) 和 上海乐高探索中心\n不希望尝试以下餐厅：上海外滩英迪格酒店·Quay江畔餐厅 和 耕海蟹将军(万象城店)",
    "hard_logic": [
        "days==2",
        "people_number==5",
        "tickets==5",
        "taxi_cars==2",
        "{\"大宁公园\", \"观光夜市(威尼斯水城之夜)\", \"上海乐高探索中心\"}<=attraction_names",
        "restaurant_names<={'上海外滩英迪格酒店·Quay江畔餐厅', '耕海蟹将军(万象城店)'}"
    ]
"""
class NL2SL_INSTRUCTION:
    def __init__(self):
        pass

    @classmethod
    def format(cls, nature_language,hard_logic_py):
        return (
            nl2sl_prompt
            + nl2sl_example
            + nl2sl_example_1
            + nl2sl_example_2
            + nl2sl_example_3
            + nl2sl_example_4
            + nl2sl_example_5
            + nl2sl_example_6
            + nl2sl_example_7
            + nl2sl_example_8
            + "\nExamples End."
            + "\nnature_language: "
            + nature_language
            + "\nlogical_constraints: "
            + hard_logic_py
            + "\n"
        )


nl2sl_prompt_v2 = """
You need to extract start_city, target_city, people_number, days from the nature language query and transform the nature language query to hard_logic. 
You need to extract the hard_logic from the nature language query and format them as python code. Each hard_logic should be a python block and the final result should be a boolean value.
We will offer you some atomic variables and funtions to help you transform the nature language query to hard_logic. You can combine them to form the hard_logic as long as they are legal python expressions.

!!! You must store the final result in the variable `result` so that we can get the final result from the variable `result`.!!!
!!! Note that the you must select activity with its type for some hard_logic.!!!

variables:
(1) plan: a dict of the generated plan with information of the specific plan.
functions:
(1) day_count(plan)
Docs: Get the number of days in the plan.
Return: int
(2) people_count(plan)
Docs: Get the number of people in the plan.
Return: int
(3) target_city(plan)
Docs: Get the target city of the plan.
Return: str
(4) allactivities(plan)
Docs: Get all the activities in the plan.
Return: list of activities
(5) activity_cost(activity)
Docs: Get the cost of specific activity without transport cost.
Return: float
(6) activity_position(activity)
Docs: Get the position name of specific activity.
Return: str
(7) activity_type(activity)
Docs: Get the type of specific activity. ['breakfast', 'lunch', 'dinner', 'attraction', 'accommodation', 'train', 'airplane']
Return: str
(8) activity_tickets(activity)
Docs: Get the number of tickets needed for specific activity. ['attraction', 'train', 'airplane']
Return: int
(9) activity_transports(activity)
Docs: Get the transport information of specific activity.
Return: list of dict
(10) activity_start_time(activity)
Docs: Get the start time of specific activity.
Return: str
(11) activity_end_time(activity)
Docs: Get the end time of specific activity.
Return: str
(12) innercity_transport_cost(transports)
Docs: Get the total cost of innercity transport.
Return: float
(13) metro_tickets(transports)
Docs: Get the number of metro tickets if the type of transport is metro.
Return: int
(14) taxi_cars(transports)
Docs: Get the number of taxi cars if the type of transport is taxi. We assume that the number of taxi cars is `(people_count(plan) + 3) // 4`.
Return: int
(15) room_count(activity)
Docs: Get the number of rooms of accommodation activity.
Return: int
(16) room_type(activity)
Docs: Get the type of room of accommodation activity. 1: 大床房, 2: 双床房
Return: int
(17) restaurant_type(activity, target_city)
Docs: Get the type of restaurant's cuisine in the target city. We only support ['云南菜', '西藏菜', '东北菜', '烧烤', '亚洲菜', '粤菜', '西北菜', '闽菜', '客家菜', '快餐简餐', '川菜', '台湾菜', '其他', '清真菜', '小吃', '西餐', '素食', '日本料理', '江浙菜', '湖北菜', '东南亚菜', '湘菜', '北京菜', '韩国料理', '海鲜', '中东料理', '融合菜', '茶馆/茶室', '酒吧/酒馆', '创意菜', '自助餐', '咖啡店', '本帮菜', '徽菜', '拉美料理', '鲁菜', '新疆菜', '农家菜', '海南菜', '火锅', '面包甜点', '其他中餐'].
Return: str
(18) attraction_type(activity, target_city)
Docs: Get the type of attraction in the target city. We only support ['博物馆/纪念馆', '美术馆/艺术馆', '红色景点', '自然风光', '人文景观', '大学校园', '历史古迹', '游乐园/体育娱乐', '图书馆', '园林', '其它', '文化旅游区', '公园', '商业街区'].
Return: str
(19) accommodation_type(activity, target_city)
Docs: Get the feature of accommodation in the target city. We only support ['儿童俱乐部', '空气净化器', '山景房', '私汤房', '四合院', '温泉', '湖畔美居', '电竞酒店', '温泉泡汤', '行政酒廊', '充电桩', '设计师酒店', '民宿', '湖景房', '动人夜景', '行李寄存', '中式庭院', '桌球室', '私人泳池', '钓鱼', '迷人海景', '园林建筑', '老洋房', '儿童泳池', '历史名宅', '棋牌室', '智能客控', '情侣房', '小而美', '特色 住宿', '茶室', '亲子主题房', '多功能厅', '洗衣房', '客栈', '自营亲子房', '停车场', 'Boss推荐', '江河景房', '日光浴场', '自营影音房', '厨房', '空调', '网红泳池', '别墅', '免费停车', '洗衣服务', '窗外好景', '酒店公寓', '会议厅', '家庭房', '24小时前台', '商务中心', '提前入园', '农家乐', '智能马桶', '美食酒店', 'SPA', '拍照出片', '海景房', '泳池', '影音房', '管家服务', '穿梭机场班车', '桑拿', '机器人服务', '儿童乐园', '健身室', '洗衣机', '自营舒睡房', '宠物友好', '电竞房', '位置超好', '套房'].
Return: str
(20) innercity_transport_type(transports)
Docs: Get the type of innercity transport. We only support ['metro', 'taxi', 'walk'].
Return: str
(21) innercity_transport_tickets(activity)
Docs: Get the number of tickets of innercity transport.
Return: int

response in json format as below:
"""

example_nl2sl_v2 = """
Example:

nature_language:
当前位置上海。我一个人想坐火车去杭州玩一天，预算1500人民币，请给我一个旅行规划。
answer:
{
"start_city": "上海",
"target_city": "杭州",
"days": 1,
"people_number": 1,
"hard_logic_py": ["result=(day_count(plan)==1)","result=(people_count(plan)==1)","total_cost=0 \nfor activity in allactivities(plan): total_cost+=activity_cost(activity)+innercity_transport_cost(activity_transports(activity))\nresult=(total_cost<=1500)","result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!=1: result=False\n  if innercity_transport_type(activity_transports(activity))=='metro'and metro_tickets(activity_transports(activity))!=1: result=False","result=True\nfor activity in allactivities(plan):\n  if innercity_transport_type(activity_transports(activity))=='taxi'and taxi_cars(activity_transports(activity))!=1: result=False","intercity_transport_set=set()\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['train', 'airplane']: intercity_transport_set.add(intercity_transport_type(activity))\nresult=(intercity_transport_set=={'train'})"],

}

nature_language:
当前位置广州。我们三个人想去成都玩3天，只坐地铁，住成都明悦大酒店，请给我们一个旅行规划。
answer:
{
"start_city": "广州",
"target_city": "成都",
"days": 3,
"people_number": 3,
"hard_logic_py": [
"result=(day_count(plan)==3)","result=(people_count(plan)==3)","result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!=3: result=False\n  if innercity_transport_type(activity_transports(activity))=='metro'and metro_tickets(activity_transports(activity))!=3: result=False","result=True\nfor activity in allactivities(plan):\n  if innercity_transport_type(activity_transports(activity))=='taxi'and taxi_cars(activity_transports(activity))!=1: result=False","accommodation_name_set=set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='accommodation': accommodation_name_set.add(activity_position(activity))\nresult=({'成都明悦大酒店'}<=accommodation_name_set)","innercity_transport_set=set()\nfor activity in allactivities(plan):\n  if activity_transports(activity)!=[]: innercity_transport_set.add(innercity_transport_type(activity_transports(activity)))\nresult=(innercity_transport_set<={'metro'})"],
}

nature_language:
当前位置上海。我和朋友计划去北京玩三天，预算6000元，市内交通只使用地铁，开一间单床房。请给我一个旅行规划。
answer:
{
"start_city": "上海",
"target_city": "北京",
"days": 3,
"people_number": 2,
"hard_logic_py": ["result=(day_count(plan)==3)","result=(people_count(plan)==2)","total_cost=0 \nfor activity in allactivities(plan): total_cost+=activity_cost(activity)+innercity_transport_cost(activity_transports(activity))\nresult=(total_cost<=6000)","result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!=2: result=False\n  if innercity_transport_type(activity_transports(activity))=='metro'and metro_tickets(activity_transports(activity))!=2: result=False","result=True\nfor activity in allactivities(plan):\n  if innercity_transport_type(activity_transports(activity))=='taxi'and taxi_cars(activity_transports(activity))!=1: result=False","result=True\nfor activity in allactivities(plan):\n  if activity_type(activity)=='accommodation' and room_count(activity)!=1: result=False\n  if activity_type(activity)=='accommodation' and room_type(activity)!=1: result=False","innercity_transport_set=set()\nfor activity in allactivities(plan):\n  if activity_transports(activity)!=[]: innercity_transport_set.add(innercity_transport_type(activity_transports(activity)))\nresult=(innercity_transport_set<={'metro'})"],
}
nature_language:
"""


class NL2SL_INSTRUCTION_V2:
    def __init__(self):
        pass

    @classmethod
    def format(cls, nature_language):
        nature_language = nature_language.strip().replace("\n", "")
        return nl2sl_prompt_v2 + example_nl2sl_v2 + nature_language + "\nanwser:"


if __name__ == "__main__":
    import os
    import sys

    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.append(root_path)
    sys.path.append(os.path.abspath(os.path.join(root_path, "..")))

    from chinatravel.agent.nesy_agent.prompts import NL2SL_INSTRUCTION_V2
    from chinatravel.agent.llms import Deepseek

    llm = Deepseek("sk-fa3c6e12204d46f0b00616ab1c2d205e")

    # nature_language = """当前位置广州。我和朋友两个人想去深圳玩3天，想吃八合里牛肉火锅(东园店)，请给我们一个旅行规划。"""
    # nature_language = """当前位置苏州。我两个人想去杭州玩2天，预算4000人民币，住一间大床房，期间打车，酒店最好有窗外好景，想去雷峰塔看一下，请给我一个旅行规划。"""
    # nature_language = """当前位置苏州。我和女朋友打算去上海玩两天，坐地铁，预算1300元，希望酒店每晚不超过500元，开一间单床房。请给我一个旅行规划。"""
    # nature_language = """"当前位置重庆。我一个人想去杭州玩2天，坐高铁（G），预算3000人民币，喜欢自然风光，住一间单床且有智能客控的酒店，人均每顿饭不超过100元，尽可能坐地铁，请给我一个旅行规划。"""
    nature_language = """"[当前位置上海, 目标位置苏州, 旅行人数2, 旅行天数3],我要带着我大儿子从上海出发去苏州玩三天，要轻快自由行，不要特种兵出行，希望参观免费景点。帮我规划一下。"""
    res = llm(
        [{"role": "user", "content": NL2SL_INSTRUCTION_V2.format(nature_language)}],
        json_mode=True,
        one_line=False,
    )
    print(res)
    # res_dict = eval(res)
    # for res_str in res_dict["hard_logic_py"]:
    #     print("-------------------")
    #     print(res_str)
    # print("-------------------")
