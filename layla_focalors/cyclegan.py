def help():
    print("깃허브 레포지트리를 참고해주세요!\n[Dev_Github](Layla-Focalors):https://github.com/layla-focalors\n기본 사용 메소드 : import layla-focalors.cyclegan as lfc")
    return None

def loadcuda():
    import platform
    print("해당 함수는 현재 지원하지 않습니다. 업데이트를 기다려주세요!")
    
def mkpreset(preset_name:str):
    import sqlite3
    conn = sqlite3.connect("preset.db")
    cur = conn.cursor()
    try:
        conn.execute("CREATE TABLE preset (preset_name text, model_path text, webui text)")
    except:
        # 오류가 나면 패키지가 있다고 간주
        pass
    
    