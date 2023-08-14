def help():
    print("깃허브 레포지트리를 참고해주세요!\n[Dev_Github](Layla-Focalors):https://github.com/layla-focalors\n기본 사용 메소드 : import layla-focalors.cyclegan as lfc")
    return None

def loadcuda():
    import platform
    print("해당 함수는 현재 지원하지 않습니다. 업데이트를 기다려주세요!")
    
def mkpreset(preset_name:str):
    import os 
    import sqlite3
    if os.path.exists("preset"):
        os.chdir("preset")
        conn = sqlite3.connect("preset.db")
    else:
        os.system("mkdir preset")
        os.chdir("preset")
        conn = sqlite3.connect("preset.db")
        conn.execute("CREATE TABLE preset (preset text, model_path text, webui text)")
    cur = conn.cursor()
    cur.execute(f"INSERT INTO preset preset VALUES {preset_name}, not_defined")
    
    
    