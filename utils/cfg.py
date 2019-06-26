import yaml

def load_cfg(path):
    """
    加载配置
    """
    file = open(path)
    cfg = yaml.safe_load(file) #full_load()
    return cfg
'''
if __name__ == "__main__":
    path = '../configs/dehumidifier.yaml'
    test = load_cfg(path)
    print(test)
    print(test['mysql']['host'])
'''
