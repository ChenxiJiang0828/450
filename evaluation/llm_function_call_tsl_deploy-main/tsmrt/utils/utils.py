import json
import inspect
import pdb

def get_instance_all_properties(instance):
    # 获取instance对象中所有的属性
    property_dict = {}
    
    for property, value in dict(vars(instance)).items():
        if property in ['_version', '_owner', '_father_model', '_sub_model', '_profile', 'org_template']:
            continue
        if type(value) is not str: 
            value = value.value
        property_dict[property] = value
    return property_dict

def is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False

