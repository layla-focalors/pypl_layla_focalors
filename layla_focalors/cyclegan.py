def help():
    print("깃허브 레포지트리를 참고해주세요!\n[Dev_Github](Layla-Focalors):https://github.com/layla-focalors\n기본 사용 메소드 : import layla-focalors.cyclegan as lfc")
    return None

def loadcuda():
    import platform
    print("해당 함수는 현재 지원하지 않습니다. 업데이트를 기다려주세요!")

def editpreset(preset_name:str):
    pass

# def web_edit_preset():
    # from fastapi import FastAPI
    # from fastapi.responses import StreamingResponse
    # from fastapi import Request
    # from fastapi.responses import HTMLResponse
    # from fastapi.templating import Jinja2Templates
    # from fastapi.staticfiles import StaticFiles
    # import pymysql
    # from pydantic import BaseModel
    # templates = Jinja2Templates(directory="templates")
    # app = FastAPI(docs_url="/documentation", redoc_url=None)
    # @app.get("/")
    # async def home(request: Request):
    #     return templates.TemplateResponse("index.html",{"request":request})
    # print("webui를 실행했습니다.")
    # 현재 지원하지 않는 기능입니다. / edit preset을 당분간 이용해주세요!

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
    print("프리셋 설정을 진행해주세요!")
    settings = ["model_path", "webui"]
    for i in range(len(settings)):
        if i == 0:
            print("모델의 경로를 지정하시겠습니까?")
            print("이미 존재하는 모델을 사용하는 경우 설정해주세요! ( 기본값 : Notdefined )")
            ses = input("모델의 경로를 지정할까요? (Y/N) : ")
            if ses == "Y" or ses == "y":
                model_path = input("모델의 경로를 입력해주세요\n")
                print("모델의 경로가 지정되었습니다. 수정하시려면 edit model을 사용해주세요!")
            elif ses == "N" or ses == "n":
                print("모델의 경로를 지정하지 않습니다.")
                model_path = "Undefined"
            else:
                print("잘못된 입력입니다. 모델의 경로를 지정하지 않습니다.!")
                model_path = "Undefined"
        elif i == 1:
            webui = input("웹UI를 사용하시겠습니까? ( 기본값 : Y ) : ")
            if webui == "Y" or webui == "y":
                webui_value = 1
            elif webui == "N" or webui == "n":
                webui_value = 0
            else:
                print("잘못된 입력입니다. 기본값이 적용됩니다.")
    print("-------------------------------------------------")
    print("설정을 완료했습니다. 다음과 같은 설정이 적용됩니다.")
    setting_dat = [model_path, webui_value]
    for i in range(len(settings)):
        print(f"Option {settings[i]} : {setting_dat[i]}")
    print("-------------------------------------------------")
    cur.execute(f"INSERT INTO preset(preset, model_path, webui) VALUES (?,?,?)",(preset_name, model_path, webui_value))
    # TestCode Init
    print(f"프리셋 {preset_name}가 생성되었습니다!")
    return None
    
def remove_all_preset():
    sas = input("[경고] 해당 작업은 등록된 모든 프리셋을 삭제합니다. 진행하시겠습니까? (Y/N) : ")
    if sas == "Y" or sas == "y":
        saa = input("해당 작업은 돌이킬 수 없습니다. 정말 진행하시겠습니까? (Y/N) : ")
        if saa == "Y" or saa == "y":
            print("모든 프리셋을 삭제합니다.")
            import os
            import shutil
            print(os.getcwd())
            try:
                shutil.rmtree("preset")
            except OSError as e:
                print(f"오류가 발생했습니다. 오류 : {e}")
            print("프리셋을 삭제했습니다.")
        elif saa == "N" or saa == "n":
            print("작업을 취소했습니다.")
        else:
            print("잘못된 입력입니다. 작업을 취소했습니다.")
    elif sas == "N" or sas == "n":
        print("작업을 취소했습니다.")
        return None
    else:
        print("잘못된 입력입니다. 작업을 취소했습니다.")
        return None
    return None
    
    