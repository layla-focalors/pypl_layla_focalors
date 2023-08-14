def help():
    print("깃허브 레포지트리를 참고해주세요!\n[Dev_Github](Layla-Focalors):https://github.com/layla-focalors\n기본 사용 메소드 : import layla-focalors.cyclegan as lfc")
    return None

def loadcuda():
    import platform
    print("해당 함수는 현재 지원하지 않습니다. 업데이트를 기다려주세요!")
    
def mkpreset(preset_name:str):
    import os 
    import sqlite3
    # package load
    if os.path.exists("preset"):
        os.chdir("preset")
        conn = sqlite3.connect("preset.db")
        cur = conn.cursor()
    else:
        os.system("mkdir preset")
        os.chdir("preset")
        conn = sqlite3.connect("preset.db")
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS preset (preset TEXT, model_path TEXT, webui INTEGER)")
    # if database == True, Bypass command, else, make database
    cur.execute(f"INSERT INTO preset(preset, model_path, webui) VALUES (?,?,?)",(preset_name, 'Notdefined', 1))
    # TestCode Init
    # print(cur.execute("SELECT * FROM preset").fetchall())
    
    
    